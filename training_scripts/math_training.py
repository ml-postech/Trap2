import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from huggingface_hub import login

def hf_login():
    """Authenticate to Hugging Face for gated checkpoints (e.g. Llama-3.1-8B).

    Set HF_TOKEN or HUGGINGFACE_TOKEN, or run ``huggingface-cli login`` beforehand.
    Called at run time (not import) so ``--help`` never touches the network.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        login(token=token)

import torch
import transformers
import wandb
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from dataset.math_datasets import MathDataset
from math_eval_utils import evaluate_math_generation, evaluate_math_loss

transformers.utils.logging.set_verbosity(transformers.logging.ERROR)


MATH_TASKS = ("gsm8k", "asdiv")


def _get_model_dtype():
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def main(args):
    hf_login()
    CACHE_DIR = "data"
    MODEL_SAVE_DIR = args.save_dir
    MAX_STEPS = args.max_steps
    EVAL_AFTER_STEPS = args.eval_steps
    TASK = args.task
    LR = args.lr
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = args.grad_accum_steps
    MODEL_NAME_OR_PATH = args.model
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DTYPE = _get_model_dtype()

    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_dora=args.use_dora,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = MathDataset(
        task=TASK,
        model_name_or_path=MODEL_NAME_OR_PATH,
        batch_size=BATCH_SIZE,
        num_workers=min(os.cpu_count(), 8),
        max_length=args.max_length,
        val_fraction=0.1,
    )
    train_dataloader = dataset.train_loader
    val_dataloader = dataset.val_loader
    # For val loss monitoring, we need train-collated val data
    val_loss_loader = torch.utils.data.DataLoader(
        val_dataloader.dataset, shuffle=False,
        collate_fn=dataset._train_collate_fn,
        batch_size=BATCH_SIZE, num_workers=min(os.cpu_count(), 8),
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH, return_dict=True, cache_dir=CACHE_DIR,
        torch_dtype=MODEL_DTYPE, attn_implementation="eager",
    )
    model = get_peft_model(model, peft_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    print(f"Task: {TASK}")
    print(f"Model dtype: {MODEL_DTYPE}")
    print(model.print_trainable_parameters())

    if args.wandb:
        wandb.init(project=args.wandb_project, name=f"math_{TASK}_lr{LR}_baseline")

    optimizer = AdamW(params=model.parameters(), lr=LR)
    total_opt_steps = MAX_STEPS
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.02 * total_opt_steps),
        num_training_steps=total_opt_steps,
    )

    model = model.to(DEVICE)

    # Adapter sub-directory name includes lr and rank
    lr_str = f"{LR:.0e}" if LR < 1e-3 else f"{LR}"
    adapter_name = f"{TASK}_r{LORA_R}_lr{lr_str}"

    total_steps = 0
    optimizer_steps = 0
    early_stopping = 10
    best_val_loss = float("inf")
    train_window_loss = 0.0
    train_window_steps = 0
    epoch = 0
    done = False

    while not done:
        epoch += 1
        model.train()
        num_batches = len(train_dataloader)
        for step_in_epoch, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training Ep. {epoch}"), start=1
        ):
            total_steps += 1
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            train_window_loss += outputs.loss.item()
            train_window_steps += 1

            is_accum_boundary = (total_steps % GRAD_ACCUM_STEPS == 0) or (step_in_epoch == num_batches)
            if is_accum_boundary:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1

                if optimizer_steps % EVAL_AFTER_STEPS == 0:
                    model.eval()
                    val_loss = evaluate_math_loss(model, val_loss_loader)
                    print(f"epoch {epoch} step {optimizer_steps} val_loss: {val_loss:.4f}")

                    if args.wandb:
                        wandb.log({
                            "train/loss": train_window_loss / max(1, train_window_steps),
                            "val/loss": val_loss,
                            "epoch": epoch,
                            "step": optimizer_steps,
                        })
                        train_window_loss = 0.0
                        train_window_steps = 0

                    if val_loss < best_val_loss:
                        adapter_dir = Path(MODEL_SAVE_DIR, adapter_name)
                        adapter_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(adapter_dir)
                        best_val_loss = val_loss
                        early_stopping = 10
                    else:
                        early_stopping -= 1

                    if early_stopping == 0:
                        print("Early stopping")
                        done = True
                        break
                    model.train()

                if optimizer_steps >= MAX_STEPS:
                    print(f"Reached max_steps ({MAX_STEPS})")
                    done = True
                    break

    print("Training finished")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Load best checkpoint and evaluate EM on test set
    from peft import PeftModel as _PeftModel
    adapter_dir = Path(MODEL_SAVE_DIR, adapter_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH, return_dict=True, cache_dir=CACHE_DIR,
        torch_dtype=MODEL_DTYPE, attn_implementation="eager",
    )
    model = _PeftModel.from_pretrained(base_model, str(adapter_dir))
    model = model.to(DEVICE)
    model.eval()
    test_loader = dataset.test_loader
    result = evaluate_math_generation(model, tokenizer, test_loader, max_new_tokens=args.max_new_tokens)
    print(f"Test EM: {result['em']:.3%} ({result['correct']}/{result['total']})")

    if args.wandb:
        wandb.log({"test/em": result["em"]})


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train math baseline model")
    parser.add_argument("--task", type=str, choices=MATH_TASKS, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--save_dir", type=str, default="math_llama31_8b")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="trap2_math")
    args = parser.parse_args()

    main(args)
