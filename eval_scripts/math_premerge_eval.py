"""Evaluate zero-shot and each single math adapter before merging."""

import os
from copy import deepcopy

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import numpy as np
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_eval_utils import evaluate_math_generation
from utils import get_config_from_name, prepare_experiment_config, set_seed

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


def _model_dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _load_base_model(model_name, cache_dir):
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        cache_dir=cache_dir,
        torch_dtype=_model_dtype(),
        attn_implementation="eager",
    )


def _eval_model_on_all_datasets(model, tokenizer, dataloaders, dataset_names, split, max_new_tokens, tag):
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    model = model.to(model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu"))

    results = {}
    avg_em = 0.0
    for i, loader_dict in enumerate(dataloaders):
        loader = loader_dict["test"][split]
        result = evaluate_math_generation(
            model,
            tokenizer,
            loader,
            max_new_tokens=max_new_tokens,
        )
        em = result["em"] * 100
        print(f"[{tag}] {dataset_names[i]}: EM={em:.2f}% ({result['correct']}/{result['total']})")
        results[dataset_names[i]] = em
        avg_em += em
    avg_em /= len(dataloaders)
    results["Average_em"] = avg_em
    print(f"[{tag}] Average EM: {avg_em:.2f}%")
    return results


def run_premerge_eval(args):
    set_seed(420)

    if args.device is not None and args.device >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device = f"cuda:{args.device}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_config = get_config_from_name(args.config, device=device)
    if isinstance(raw_config.get("dataset"), list):
        for dataset_cfg in raw_config["dataset"]:
            dataset_cfg["batch_size"] = args.batch_size
            dataset_cfg["num_workers"] = min(args.num_workers, os.cpu_count() or args.num_workers)

    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([d["name"] for d in raw_config["dataset"]])
    dataloaders = np.array([d for d in config["data"]])

    model_name = raw_config["model"]["name"]
    cache_dir = raw_config["model"]["cachedir"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Config: {args.config}")
    print(f"Eval split: {args.eval_split}")
    print(f"Datasets: {list(dataset_names)}")

    # Zero-shot
    zero_shot_model = config["models"]["new"]
    _eval_model_on_all_datasets(
        zero_shot_model,
        tokenizer,
        dataloaders,
        dataset_names,
        args.eval_split,
        args.max_new_tokens,
        tag="zero-shot",
    )

    if torch.cuda.is_available():
        zero_shot_model.to("cpu")
        torch.cuda.empty_cache()

    # Single adapters
    for idx, base_path in enumerate(raw_config["model"]["bases"]):
        tag = f"adapter_{dataset_names[idx]}"
        print(f"\n[INFO] loading adapter for {dataset_names[idx]}: {base_path}")
        fresh_base = _load_base_model(model_name, cache_dir)
        model = PeftModel.from_pretrained(model=fresh_base, model_id=base_path)
        _eval_model_on_all_datasets(
            model,
            tokenizer,
            dataloaders,
            dataset_names,
            args.eval_split,
            args.max_new_tokens,
            tag=tag,
        )
        if torch.cuda.is_available():
            model.to("cpu")
            del model
            del fresh_base
            torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate zero-shot and single math adapters before merging")
    parser.add_argument("--config", type=str, required=True, help="Config name")
    parser.add_argument("--eval_split", type=str, default="test", choices=("val", "test"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    run_premerge_eval(args)
