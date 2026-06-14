"""Merge and evaluate math adapters with linear hyperparameter search."""

import os
import time
from copy import deepcopy

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from task_merger import get_merge_handler
from utils import get_config_from_name, prepare_experiment_config, set_seed, parse_eval_args, merge_args_into_task_merge_config
from math_eval_utils import evaluate_math_generation

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

MATH_DATASETS = ['gsm8k', 'asdiv']


def run_math_eval(args):
    EVAL_SPLIT = getattr(args, 'eval_split', 'val')
    EVAL_TEST = True
    BIGSEED = 420
    EARLY_STOPPING_STEPS = 10
    DISABLE_SEARCH = getattr(args, 'disable_search', False)

    set_seed(BIGSEED)

    config_name = args.config
    print("Config name:", config_name)

    if args.device is not None and args.device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device = f'cuda:{args.device}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_local = device

    raw_config = get_config_from_name(config_name, device=device)
    if isinstance(raw_config.get('dataset'), list):
        for dataset_cfg in raw_config['dataset']:
            dataset_cfg['batch_size'] = 4
            dataset_cfg['num_workers'] = 4
    print(raw_config['task_merge_config'])

    config = prepare_experiment_config(raw_config)
    config['task_merge_config'] = merge_args_into_task_merge_config(config['task_merge_config'], args)
    config['task_merge_config']['merge_on_gpu'] = args.merge_on_gpu

    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    dataloaders = np.array([i for i in config['data']])

    # Load tokenizer for generation
    model_name = raw_config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Datasets: {list(dataset_names)}")

    # Grid search config
    default_params = {
        'scaling_coeffs': 0.3,
        'topK': 70,
        'cart_pruning_rank': 0.04,
        'dare_pruning_coeffs': 0.9,
    }
    order_of_processing_params = [
        'scaling_coeffs',
        'topK',
        'dare_pruning_coeffs',
        'cart_pruning_rank',
    ]
    search_config = {
        'scaling_coeffs': np.arange(0.1, 10.1, step=0.1),
        'topK': (np.arange(1, 11, step=1) * 10),
        'dare_pruning_coeffs': [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1e-5][::-1],
        'cart_pruning_rank': [0.04, 0.08, 0.16, 0.32],
    }

    def merge_and_eval(Merge, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print(f"EVAL_SPLIT: {EVAL_SPLIT}")
        print(f"Params: {instance_params}")
        all_results = deepcopy(instance_params)

        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)

        t0 = time.time()
        merged_model = Merge.merge(config['task_merge_config'])
        print(f"Merge time: {time.time() - t0:.1f}s")

        merged_model.config.pad_token_id = tokenizer.pad_token_id
        merged_model.config.use_cache = True  # Enable KV cache for generation

        merged_model = merged_model.to(device_local)

        avg_em = 0.0
        for i, loader_dict in enumerate(dataloaders):
            loader = loader_dict['test'][EVAL_SPLIT]
            result = evaluate_math_generation(
                merged_model, tokenizer, loader,
                max_new_tokens=getattr(args, 'max_new_tokens', 256),
            )
            em = result['em'] * 100
            print(f"  {dataset_names[i]}: EM={em:.2f}% ({result['correct']}/{result['total']})")
            all_results[dataset_names[i]] = em
            avg_em += em

        avg_em /= len(dataloaders)
        print(f"  Average EM: {avg_em:.2f}%")
        all_results['Average_em'] = avg_em

        # Move merged model off GPU
        if torch.cuda.is_available():
            merged_model.to('cpu')
            torch.cuda.empty_cache()

        return all_results

    # Build merger
    with torch.no_grad():
        lora_state_dicts = np.array([i for i in config['models']['bases']])
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            lora_state_dicts,
            pretrained_model=config['models']['new'],
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )

        if config['task_merge_config'].get('ingredients_path') is None or \
           not os.path.exists(config['task_merge_config'].get('ingredients_path', '')):
            Merge.transform(config['task_merge_config'])

        if DISABLE_SEARCH:
            # Single eval with CLI-supplied or default params, no search
            nosearch_params = deepcopy(default_params)
            for key in nosearch_params:
                cli_val = getattr(args, key, None)
                if cli_val is not None:
                    nosearch_params[key] = cli_val
            print(f"\nSearch disabled, running single eval with: {nosearch_params}")
            test_result = merge_and_eval(Merge, EVAL_SPLIT=EVAL_SPLIT, instance_params=nosearch_params)
            present = [d for d in MATH_DATASETS if d in test_result]
            test_results = " & ".join([f"{test_result[d]:.2f}" for d in present])
            test_results += f" & {test_result['Average_em']:.2f} \\\\"
            print(f"Test results (EM): {test_results}")
            print(test_result)
            return

        # Grid search
        print(f"\nStarting grid search...")
        early_stopping = EARLY_STOPPING_STEPS
        for param in order_of_processing_params:
            best_val_results = {'Average_em': 0.0}
            for value in search_config[param]:
                instance_params = deepcopy(default_params)
                instance_params[param] = value
                all_results = merge_and_eval(Merge, EVAL_SPLIT=EVAL_SPLIT, instance_params=instance_params)
                if all_results['Average_em'] >= best_val_results['Average_em']:
                    best_val_results = deepcopy(all_results)
                    early_stopping = EARLY_STOPPING_STEPS
                else:
                    early_stopping -= 1
                    if early_stopping <= 0:
                        print("Early stopping")
                        break
            default_params[param] = best_val_results[param]

        if EVAL_TEST:
            print(f"\nBest params: {best_val_results}")
            for key in search_config.keys():
                instance_params.update({key: best_val_results[key]})
            test_result = merge_and_eval(Merge, EVAL_SPLIT='test', instance_params=instance_params)

            present = [d for d in MATH_DATASETS if d in test_result]
            test_results = " & ".join([f"{test_result[d]:.2f}" for d in present])
            test_results += f" & {test_result['Average_em']:.2f} \\\\"
            print(f"Test results (EM): {test_results}")
            print(test_result)


if __name__ == "__main__":
    args = parse_eval_args()
    run_math_eval(args)
