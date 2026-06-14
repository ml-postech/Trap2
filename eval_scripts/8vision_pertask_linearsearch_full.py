import time
from copy import deepcopy
import os
import math
import importlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import wandb
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch.cuda.amp import autocast
from torchvision import datasets, transforms
from peft.utils import set_peft_model_state_dict

from accuracies import get_vision_accuracies
from ft_handlers import GeneralHandler
from models.huggingface_clip import HFCLIPVisionModel
from task_merger import get_merge_handler
from utils import (
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    merge_args_into_task_merge_config,
    parse_eval_args,
    build_scaling_vector,
    recursively_getattr,
    prepare_data,
    set_seed,
)


def _load_full_model(base_model, path, device):
    model = deepcopy(base_model)
    try:
        sd = torch.load(path, map_location='cpu')
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {path}") from exc
    model.load_state_dict(sd, strict=False)
    return model.to(device)


def _save_merged_model_if_requested(args, merged_model, config_name, instance_params, eval_split):
    save_dir = getattr(args, "save_merged_model_dir", None)
    if not save_dir:
        return None
    if str(eval_split).lower() != "test":
        return None
    os.makedirs(save_dir, exist_ok=True)
    if getattr(args, "save_merged_model_name", None):
        file_name = args.save_merged_model_name
    else:
        stem = Path(config_name).stem
        merge_method = getattr(args, "merge_method", "merge")
        merge_space = getattr(args, "merge_space", "full")
        coeff = instance_params.get("scaling_coeffs", getattr(args, "scaling_coeffs", 1.0))
        file_name = f"{stem}_{merge_method}_{merge_space}_s{coeff}.pt"
    save_path = os.path.join(save_dir, file_name)
    torch.save({k: v.detach().cpu() for k, v in merged_model.state_dict().items()}, save_path)
    print(f"[Merge] Saved merged model to {save_path}")
    return save_path


def _build_full_experiment_config(raw_config, device):
    model_cfg = raw_config.get('model', {})
    if model_cfg.get('name') != 'hf_clip':
        raise ValueError("Full linearsearch supports only HF CLIP models.")
    base_model = HFCLIPVisionModel(
        model_name=model_cfg.get('base_type'),
        cache_dir=model_cfg.get('cachedir', 'data'),
        device=device,
    )
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = base_model.train_preprocess
        dataset_config['eval_preprocess'] = base_model.val_preprocess

    data = prepare_data(raw_config['dataset'], device=device)
    base_paths = model_cfg.get('bases', [])
    if not base_paths:
        raise ValueError("model.bases must contain full fine-tuned .pt checkpoints.")
    for path in base_paths:
        if not path.endswith('.pt'):
            raise ValueError(f"Expected .pt checkpoint path, got: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    finetuned_models = [
        _load_full_model(base_model, path, device='cpu')
        for path in base_paths
    ]

    return {
        'data': data,
        'models': {'bases': finetuned_models, 'new': base_model},
        'task_merge_config': raw_config['task_merge_config'],
        'param_handler': GeneralHandler,
        'model': model_cfg,
    }


def _is_state_dict(obj):
    return isinstance(obj, dict)


def _load_model_from_adapter_or_full(base_model, adapter_or_model, device_for_outlier):
    if _is_state_dict(adapter_or_model):
        model = deepcopy(base_model)
        adapter_name = None
        if hasattr(model.vision_model, 'peft_config') and model.vision_model.peft_config:
            adapter_name = next(iter(model.vision_model.peft_config.keys()))
        ft_sd = {}
        for k, v in adapter_or_model.items():
            if k.startswith('vision_model.'):
                ft_sd[k.replace('vision_model.', '', 1)] = v
            else:
                ft_sd[k] = v
        try:
            set_peft_model_state_dict(model.vision_model, ft_sd, adapter_name=adapter_name)
        except Exception as e:
            print(f"[full-eval] set_peft_model_state_dict failed ({e}); falling back to load_state_dict.")
            load_info = model.vision_model.load_state_dict(ft_sd, strict=False)
            if load_info.missing_keys or load_info.unexpected_keys:
                print(f"[full-eval] load_state_dict info -> missing: {load_info.missing_keys[:3]}{'...' if len(load_info.missing_keys)>3 else ''}, "
                      f"unexpected: {load_info.unexpected_keys[:3]}{'...' if len(load_info.unexpected_keys)>3 else ''}")
        model = model.to(device_for_outlier)
        model.eval()
        return model
    model = deepcopy(adapter_or_model)
    model = model.to(device_for_outlier)
    model.eval()
    return model


def _build_model_from_full_delta(base_model, ft_model, device):
    base = deepcopy(base_model)
    base_sd = base.state_dict()
    ft_sd = ft_model.state_dict()
    delta_sd = {}
    for k, v in base_sd.items():
        ft_v = ft_sd.get(k, None)
        if ft_v is None or ft_v.shape != v.shape:
            continue
        delta_sd[k] = ft_v.to(v.device) - v
    with torch.no_grad():
        for k, delta in delta_sd.items():
            base_sd[k].add_(delta)
    base.load_state_dict(base_sd, strict=False)
    base = base.to(device)
    base.eval()
    return base


def run_BIG_function(args):
    EVAL_TEST = True
    EVAL_SPLIT = getattr(args, 'eval_split', 'val')
    BIGSEED = 420
    premerge_split = getattr(args, 'premerge_split', 'val')

    if args.device is not None:
        if args.device >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(args.device)
            device = f'cuda:{args.device}'
        elif args.device >= 0 and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU despite --device request.")
            device = 'cpu'
        else:
            device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)

    # Get config
    config_name = args.config
    print("Config name : ", config_name)
    EARLY_STOPPING_STEPS = 10
    cfg_module = importlib.import_module(f"configs.{Path(config_name).stem}")
    raw_config = get_config_from_name(config_name, device=device)
    if args.wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=f"{config_name}_{args.merge_method}_{args.merge_space}_{args.representation}",
                   config={'raw_config': raw_config, 'args': vars(args)})
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = _build_full_experiment_config(raw_config, device=device)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    # Pass merge_on_gpu flag to merger config
    config['task_merge_config']['merge_on_gpu'] = args.merge_on_gpu
    dataset_names = [i['name'] for i in raw_config['dataset']]
    dataloaders = [i for i in config['data']]
    if args.early_stop_exclude:
        exclude_names = [n.strip() for n in args.early_stop_exclude.split(',') if n.strip()]
        early_stop_exclude = [n for n in exclude_names if n in dataset_names]
        missing = [n for n in exclude_names if n not in dataset_names]
        if missing:
            print(f"[WARNING] early_stop_exclude tasks not found in dataset list: {missing}")
    else:
        early_stop_exclude = []
    early_stop_include_idx = [i for i, name in enumerate(dataset_names) if name not in early_stop_exclude]
    if len(early_stop_include_idx) == 0:
        raise ValueError("All tasks were excluded from early stopping; provide at least one task to include.")
    model_type = config['model']['base_type']
    peft_type = 'full'
    rank = None
    fine_tuned_acc = get_vision_accuracies(model_type, peft_type=peft_type, rank=None, dataset_names=dataset_names)
    # Allow config override for baseline accuracies (e.g., LAION/MetaCLIP variants).
    override_norms = raw_config['model'].get('baseline_norms') or raw_config['model'].get('norms')
    if override_norms:
        if isinstance(override_norms, list):
            fine_tuned_acc = {dataset_names[i]: override_norms[i] for i in range(len(dataset_names))}
        elif isinstance(override_norms, dict):
            fine_tuned_acc = override_norms
        else:
            raise ValueError("baseline_norms/norms must be list or dict.")
    print(f'Finetuned Accs: {fine_tuned_acc}')
    eps = 1e-8

    def eval_model_on_all_datasets(model, tag, split=EVAL_SPLIT):
        """Evaluate a model on all datasets for the requested split."""
        model_results = {}
        model_results_norm = {}
        print(f"\n[{tag}] Evaluating on split: {split}")
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][split]
            acc = evaluate_cliphead(model.to(device), loader, class_vectors=all_clip_encodings[i].to(device), silent=True)
            model_results[dataset_names[i]] = acc * 100
            if fine_tuned_acc and fine_tuned_acc.get(dataset_names[i], None):
                norm_acc = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
                model_results_norm[dataset_names[i] + '_norm_acc'] = norm_acc
                print(f"[{tag}] {dataset_names[i]} accuracy: {np.round(acc * 100, 3)} | normalized: {np.round(norm_acc, 3)}")
            else:
                print(f"[{tag}] {dataset_names[i]} accuracy: {np.round(acc * 100, 3)}")
        avg_acc = np.mean(list(model_results.values()))
        if model_results_norm:
            avg_norm = np.mean(list(model_results_norm.values()))
            print(f"[{tag}] Average accuracy across datasets: {np.round(avg_acc, 3)} | normalized: {np.round(avg_norm, 3)}")
        else:
            print(f"[{tag}] Average accuracy across datasets: {np.round(avg_acc, 3)}")
        return model_results

    if args.run_premerge_evals:
        print("\nRunning pre-merge evaluations (zero-shot and single-LoRA baselines)...")
        # Zero-shot (no LoRA)
        zero_shot_model = deepcopy(config['models']['new'])
        # Ensure LoRA adapters are disabled for true zero-shot
        if hasattr(zero_shot_model, 'disable_adapters'):
            zero_shot_model.disable_adapters = True
        eval_model_on_all_datasets(zero_shot_model, tag='zero-shot', split=premerge_split)

        # Single full-ft evaluations: each finetuned model evaluated via base + delta
        for idx, ft_model in enumerate(config['models']['bases']):
            full_tag = f"full_{dataset_names[idx]}"
            delta_model = _build_model_from_full_delta(
                config['models']['new'], ft_model, device
            )
            eval_model_on_all_datasets(delta_model, tag=full_tag, split=premerge_split)
        print("Finished pre-merge evaluations.\n")

    # Parameters are tuned in the order specified in search_config
    default_params = {'scaling_coeffs': .6,
                      'topK': 30,
                      'cart_pruning_rank': 0.04,
                      'dare_pruning_coeffs': 1e-5,
                      }  # Default config
    order_of_processing_params = [
        'scaling_coeffs',
    ]
    search_config = {
        'scaling_coeffs': np.arange(0.1, 10.0, step=0.1),
        'topK': (np.arange(1, 11, step=1) * 10)[::-1],
        'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        'cart_pruning_rank': [0.04, 0.08, 0.16, 0.32]
    }

    if getattr(args, 'disable_search', False):
        order_of_processing_params = []
        default_params = {}
        search_config = {}
    else:
        if 'dare' in config['task_merge_config']['merge_method']:
            order_of_processing_params.append('dare_pruning_coeffs')
        if 'ties' in config['task_merge_config']['merge_method']:
            order_of_processing_params.append('topK')
        if 'cart' in config['task_merge_config']['merge_method']:
            order_of_processing_params.append('cart_pruning_rank')

    def merge_and_eval(merger, EVAL_SPLIT='val', instance_config=None, instance_params=None):
        if instance_params is None:
            instance_params = {}
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        # initialize merging function
        print('Creating Merge')
        t1 = time.time()
        merger.transform(instance_config)
        # set task scaling coefficients
        scaling = build_scaling_vector(
            instance_config['scaling_coeffs'],
            dataset_names,
        )
        merger.set_scaling_coeffs(scaling)
        merged_model = merger.merge(instance_config)
        _save_merged_model_if_requested(
            args, merged_model, config_name, instance_params, EVAL_SPLIT
        )
        t2 = time.time()
        print(f'Merging time: {t2 - t1:.2f} seconds')

        print('Evaluate Merged Model on Each Dataset')
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            if fine_tuned_acc and dataset_names[i] in fine_tuned_acc:
                norm_acc = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
                print(f"{dataset_names[i]} Normalized accuracy is {np.round(norm_acc, 3)}")
                all_results[dataset_names[i] + '_norm_acc'] = norm_acc
                if i in early_stop_include_idx:
                    avg_norm_accuracy += norm_acc
            else:
                print(f"{dataset_names[i]} Normalized accuracy skipped (no baseline)")
            if i in early_stop_include_idx:
                avg_accuracy += acc * 100
        avg_accuracy /= len(early_stop_include_idx)
        avg_norm_accuracy = avg_norm_accuracy / len(early_stop_include_idx) if fine_tuned_acc else 0.0
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        if fine_tuned_acc:
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        # Log the merge evaluation results to wandb
        if args.wandb:
            wandb.log({**all_results, "params": instance_params})
        return all_results

    with torch.no_grad():
        print(search_config)
        models = np.array([i for i in config['models']['bases']])

        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        merger = MergeClass(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        print(config['task_merge_config'])
        early_stopping = EARLY_STOPPING_STEPS
        if not order_of_processing_params:
            # Single run (no coefficient grid search)
            instance_params = deepcopy(default_params)
            config['task_merge_config'].update(instance_params)
            best_val_results = merge_and_eval(
                merger=merger,
                EVAL_SPLIT=EVAL_SPLIT,
                instance_config=config['task_merge_config'],
                instance_params=instance_params
            )
        else:
            for param in order_of_processing_params:
                best_val_results = {'Average_norm_acc': 0.0}
                for value in search_config[param]:
                    instance_params = deepcopy(default_params)
                    instance_params[param] = value
                    config['task_merge_config'].update(instance_params)
                    all_results = merge_and_eval(
                        merger=merger,
                        EVAL_SPLIT=EVAL_SPLIT,
                        instance_config=config['task_merge_config'],
                        instance_params=instance_params
                    )
                    if (all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']):
                        best_val_results = deepcopy(all_results)
                        early_stopping = EARLY_STOPPING_STEPS
                    else:
                        early_stopping -= 1
                        if (early_stopping == 0):
                            print("Early stopping")
                            break
                default_params[param] = best_val_results[param]

        if EVAL_TEST:
            # Evaluate on the test set with the best topK and scaling co-efficient
            print("Best params :", best_val_results)
            for key in search_config.keys():
                instance_params.update({key: best_val_results[key]})
            config['task_merge_config'].update(instance_params)
            test_result = merge_and_eval(
                merger=merger,
                EVAL_SPLIT='test',
                instance_config=config['task_merge_config'],
                instance_params=instance_params
            )
            dataset_order = dataset_names
            present = [ds for ds in dataset_order if f"{ds}_norm_acc" in test_result]
            test_results = " & ".join([f"{np.round(test_result[dataset+'_norm_acc'], 2):.2f}" for dataset in present])
            if 'Average_norm_acc' in test_result:
                test_results += f" & {np.round(test_result['Average_norm_acc'], 2):.2f} \\\\"
            else:
                test_results += " \\\\"
            print(f"Normalized Test results: {test_results}")
            print(test_result)
            # Save results to results.txt
            with open("results.txt", "a") as f:
                f.write(f"Args: {vars(args)}\n")
                f.write(f"Normalized Test results: {test_results}\n")
                f.write(f"Test result dict: {test_result}\n")
                f.write(f"Best parameters: {instance_params}\n\n")
            # Log final test results to wandb
            if args.wandb:

                wandb.log({"final_test": test_result, "best_parameters": instance_params})
    if args.wandb:
        # Finish the wandb run
        wandb.finish()


if __name__ == "__main__":
    args = parse_eval_args()
    if args.device is not None and args.device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
    run_BIG_function(args)
