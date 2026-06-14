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
from ft_handlers import LoRAHandler
from models.openclip_clip import OpenCLIPVisionModel

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "open_clip is required for ConvNeXt eval. Install with `pip install open_clip_torch`."
    ) from exc

from accuracies import get_vision_accuracies
from task_merger import get_merge_handler
from utils import (
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    merge_args_into_task_merge_config,
    parse_eval_args,
    prepare_experiment_config,
    build_scaling_vector,
    recursively_getattr,
    set_seed,
)


def _extract_lora_state_dict(adapter_sd):
    """Keep only LoRA params from a saved .pt state dict."""
    return {k: v for k, v in adapter_sd.items() if 'lora_A' in k or 'lora_B' in k}


def _normalize_lora_keys(ft_sd):
    """Normalize saved OpenCLIP LoRA keys to match model.visual state_dict keys."""
    normalized = {}
    for k, v in ft_sd.items():
        if k.startswith('vision_model.'):
            k = k.replace('vision_model.', '', 1)
        if k.startswith('model.visual.'):
            k = k.replace('model.visual.', '', 1)
        normalized[k] = v
    return normalized


def _apply_lora_delta_to_openclip(base_model, adapter_sd):
    """Apply LoRA delta weights directly to a base OpenCLIP model (no PEFT)."""
    handler = LoRAHandler(adapter_sd)
    deltas = handler.get_ft_parameters()
    sd = base_model.model.state_dict()
    for name, delta in deltas.items():
        if name.startswith('model.visual.base_model.model.'):
            name = name.replace('model.visual.base_model.model.', 'visual.', 1)
        elif name.startswith('model.visual.'):
            name = name.replace('model.visual.', 'visual.', 1)
        elif name.startswith('base_model.model.'):
            name = name.replace('base_model.model.', 'visual.', 1)
        key = name if name.endswith('.weight') else f"{name}.weight"
        if key not in sd:
            continue
        sd[key] = sd[key].cpu() + delta.to(sd[key].dtype).cpu()
    base_model.model.load_state_dict(sd, strict=False)
    return base_model


def run_BIG_function(args):
    EVAL_TEST = True
    # EVAL_SPLIT = 'test'
    EVAL_SPLIT = 'val'
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
    # Attach OpenCLIP preprocessors for dataset loaders.
    model_cfg = raw_config.get('model', {})
    model_name = model_cfg.get('openclip_model')
    pretrained = model_cfg.get('openclip_pretrained')
    precision = model_cfg.get('openclip_precision', 'fp32')
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision=precision,
        device='cpu',
        cache_dir=model_cfg.get('cachedir', ''),
    )
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = preprocess_train
        dataset_config['eval_preprocess'] = preprocess_val

    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    # ConvNeXt OpenCLIP: load adapters directly from .pt files.
    base_paths = raw_config['model'].get('bases', [])
    if not base_paths:
        raise ValueError("ConvNeXt linearsearch expects model.bases to be .pt files.")
    for base in base_paths:
        if not base.endswith('.pt'):
            raise ValueError(f"Expected .pt adapter path for ConvNeXt, got: {base}")
    config['models']['bases'] = [torch.load(path, map_location='cpu') for path in base_paths]
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
    model_type = config['model'].get('base_type', 'open_clip')
    rank = config['model']['ft_config'].get('r', None)
    peft_type = config['model']['ft_config'].get('type')
    fine_tuned_acc = {}
    # Allow optional config override for baseline accuracies.
    override_norms = raw_config['model'].get('baseline_norms') or raw_config['model'].get('norms')
    if override_norms:
        if isinstance(override_norms, list):
            fine_tuned_acc = {dataset_names[i]: override_norms[i] for i in range(len(dataset_names))}
        elif isinstance(override_norms, dict):
            fine_tuned_acc = override_norms
        else:
            raise ValueError("baseline_norms/norms must be list or dict.")
    use_norm = bool(fine_tuned_acc)
    if fine_tuned_acc:
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
            if fine_tuned_acc.get(dataset_names[i], None):
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

        # Single LoRA evaluations: each finetuned model evaluated on all datasets
        oc_base = OpenCLIPVisionModel(
            model_name=model_cfg.get('openclip_model'),
            pretrained=model_cfg.get('openclip_pretrained'),
            cache_dir=model_cfg.get('cachedir', ''),
            device=device,
            precision=model_cfg.get('openclip_precision', 'fp32'),
        )
        for idx, ft_model in enumerate(config['models']['bases']):
            lora_tag = f"lora_{dataset_names[idx]}"
            ft_model_copy = deepcopy(oc_base)
            ft_model_copy = _apply_lora_delta_to_openclip(ft_model_copy, ft_model)
            eval_model_on_all_datasets(ft_model_copy, tag=lora_tag, split=premerge_split)
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
        'scaling_coeffs': np.arange(0.1, 10, step=0.1),
        'topK': (np.arange(1, 11, step=1) * 10)[::-1],
        'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        'cart_pruning_rank': [0.04, 0.08, 0.16, 0.32]
    }

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
        t2 = time.time()
        print(f'Merging time: {t2 - t1:.2f} seconds')

        print('Evaluate Merged Model on Each Dataset')
        use_norm = bool(fine_tuned_acc)
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
            if use_norm and fine_tuned_acc.get(dataset_names[i]) is not None:
                print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc[dataset_names[i]] *100, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            if use_norm and fine_tuned_acc.get(dataset_names[i]) is not None:
                all_results[dataset_names[i] + '_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
            if i in early_stop_include_idx:
                avg_accuracy += acc * 100
                if use_norm and fine_tuned_acc.get(dataset_names[i]) is not None:
                    avg_norm_accuracy += (acc * 100) / fine_tuned_acc[dataset_names[i]] * 100
        avg_accuracy /= len(early_stop_include_idx)
        if use_norm:
            avg_norm_accuracy /= len(early_stop_include_idx)
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        if use_norm:
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        if use_norm:
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
                metric_key = 'Average_norm_acc' if use_norm else 'Average_acc'
                best_val_results = {metric_key: 0.0}
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
                    if (all_results[metric_key] >= best_val_results[metric_key]):
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
            if use_norm:
                test_results = " & ".join([f"{np.round(test_result[dataset+'_norm_acc'], 2):.2f}" for dataset in dataset_order]) + f" & {np.round(test_result['Average_norm_acc'], 2):.2f} \\\\"
                print(f"Normalized Test results: {test_results}")
            print(test_result)
            # Save results to results.txt
            with open("results.txt", "a") as f:
                f.write(f"Args: {vars(args)}\n")
                if use_norm:
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
    run_BIG_function(args)
