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

import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.stateless import functional_call
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


def _has_nonfinite_grads(parameters):
    for param in parameters:
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return True
    return False


def _trap2_step(
    model,
    batch,
    lambda_reg,
    rand_alpha_min,
    rand_alpha_max,
    rand_alpha_weight,
    grad_scale=1.0,
    fixed_alphas=None,
):
    """TRAP2 step for CausalLM (no mask_class, no external criterion)."""
    named_params = list(model.named_parameters())
    base_param_dict = dict(named_params)
    base_buffer_dict = dict(model.named_buffers())
    lora_named_params = [(n, p) for (n, p) in named_params if ('lora' in n and p.requires_grad)]

    def _build_scaled_params(alpha):
        scaled = {}
        for name, param in lora_named_params:
            if 'lora_B' in name:
                scaled[name] = param * alpha
        return scaled

    def _forward_loss(param_overrides=None):
        if param_overrides:
            params = dict(base_param_dict)
            params.update(param_overrides)
            buffers = dict(base_buffer_dict)
            try:
                outputs = functional_call(model, {**params, **buffers}, (), batch, strict=False)
            except TypeError:
                outputs = functional_call(model, params, (), batch, buffers=buffers)
        else:
            outputs = model(**batch)
        return outputs.loss

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    loss_clean = _forward_loss()
    (loss_clean * grad_scale).backward()

    lora_params = [p for (n, p) in model.named_parameters() if ("lora" in n and p.requires_grad)]
    rand_losses = []
    if fixed_alphas is None:
        while True:
            alpha = float(np.random.uniform(rand_alpha_min, rand_alpha_max))
            if not (0.95 < alpha < 1.05):
                break
        sampled_alphas = [alpha]
    else:
        sampled_alphas = list(fixed_alphas)
    num_samples = max(1, len(sampled_alphas))

    for alpha in sampled_alphas:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        loss_alpha = _forward_loss(param_overrides=_build_scaled_params(alpha))
        if rand_alpha_weight == "inv":
            weight = 1.0 / alpha
        elif rand_alpha_weight == "inv_sqrt":
            weight = 1.0 / np.sqrt(alpha)
        else:
            weight = 1.0
        coeff = -lambda_reg * weight * grad_scale / num_samples
        grads = torch.autograd.grad(
            loss_alpha,
            lora_params,
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )
        with torch.no_grad():
            for p, g in zip(lora_params, grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                p.grad.add_(coeff * g)
        rand_losses.append((loss_alpha.detach() * weight).detach())

    with torch.no_grad():
        rand_loss_mean = torch.stack(rand_losses).mean().item() if rand_losses else 0.0
        penalty = lambda_reg * rand_loss_mean
        total_loss = loss_clean.item() - penalty

    return loss_clean.item(), rand_loss_mean, penalty, total_loss


def _alpha_sweep_loss(
    model,
    val_loss_loader,
    device,
    alpha_min,
    alpha_max,
    alpha_step,
    save_path=None,
):
    """Alpha sweep using CLM loss (fast, no generation)."""
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora" in n and p.requires_grad]
    backup = {n: p.detach().cpu().clone() for n, p in lora_params}
    alphas = []
    losses = []
    try:
        alpha = alpha_min
        while alpha <= alpha_max + 1e-9:
            with torch.no_grad():
                for name, param in lora_params:
                    if "lora_B" in name:
                        param.copy_(backup[name].to(device) * float(alpha))
                    else:
                        param.copy_(backup[name].to(device))

            loss_sum = 0.0
            total_tokens = 0
            with torch.no_grad():
                for batch in val_loss_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    num_tokens = (labels != -100).sum().item()
                    loss_sum += outputs.loss.item() * num_tokens
                    total_tokens += num_tokens
            avg_loss = loss_sum / max(1, total_tokens)
            alphas.append(float(alpha))
            losses.append(avg_loss)
            print(f"[alpha_sweep] alpha={alpha:.2f} loss={avg_loss:.6f}")
            alpha = round(alpha + alpha_step, 10)
    finally:
        with torch.no_grad():
            for name, param in lora_params:
                param.copy_(backup[name].to(device))
    if save_path is not None:
        try:
            import matplotlib.pyplot as plt
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots()
            ax.plot(alphas, losses, color="tab:red", marker="o", markersize=3)
            ax.set_xlabel("alpha")
            ax.set_ylabel("loss")
            ax.set_title("Alpha Sweep (Loss)")
            fig.tight_layout()
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"[alpha_sweep] saved plot: {save_path}")
        except Exception as e:
            print(f"[alpha_sweep] failed to save plot: {e}")
    return alphas, losses


def main(args):
    hf_login()
    CACHE_DIR = "data"
    MODEL_SAVE_DIR = args.save_dir
    MAX_STEPS = args.max_steps
    EVAL_AFTER_STEPS = args.eval_steps
    TASK = args.task
    LR = args.lr
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_STEPS = max(1, int(args.grad_accum_steps))
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

    # Val loader with train collator (for loss-based eval and alpha sweep)
    val_loss_loader = torch.utils.data.DataLoader(
        dataset.val_loader.dataset, shuffle=False,
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

    print(f"Task: {TASK} (TRAP2)")
    print(f"Model dtype: {MODEL_DTYPE}")
    print(model.print_trainable_parameters())

    optimizer = AdamW(params=model.parameters(), lr=LR)
    total_opt_steps = MAX_STEPS
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.02 * total_opt_steps),
        num_training_steps=total_opt_steps,
    )

    model = model.to(DEVICE)
    optimizer.zero_grad()

    # Adapter sub-directory name includes lr, rank and lambda
    lr_str = f"{LR:.0e}" if LR < 1e-3 else f"{LR}"
    adapter_name = f"{TASK}_r{LORA_R}_lr{lr_str}_lam{args.lambda_reg}"

    total_steps = 0
    optimizer_steps = 0
    early_stopping = 10
    best_val_loss = float("inf")
    fixed_alphas = None
    epoch = 0
    done = False

    while not done:
        epoch += 1
        model.train()
        for step_in_epoch, batch in enumerate(
            tqdm(train_dataloader, desc=f"Training Ep. {epoch}"), start=1
        ):
            total_steps += 1
            batch_device = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
                "labels": batch["labels"].to(DEVICE),
            }

            accum_index = (total_steps - 1) % GRAD_ACCUM_STEPS
            if accum_index == 0:
                while True:
                    alpha = float(np.random.uniform(args.rand_alpha_min, args.rand_alpha_max))
                    if not (0.95 < alpha < 1.05):
                        break
                fixed_alphas = [alpha]

            loss_clean, rand_loss_mean, penalty, total_loss = _trap2_step(
                model=model,
                batch=batch_device,
                lambda_reg=args.lambda_reg,
                rand_alpha_min=args.rand_alpha_min,
                rand_alpha_max=args.rand_alpha_max,
                rand_alpha_weight=args.rand_alpha_weight,
                grad_scale=1.0 / GRAD_ACCUM_STEPS,
                fixed_alphas=fixed_alphas,
            )

            did_optimizer_step = False
            if (total_steps % GRAD_ACCUM_STEPS == 0) or (step_in_epoch == len(train_dataloader)):
                if not np.isfinite(loss_clean) or not np.isfinite(rand_loss_mean) or not np.isfinite(penalty):
                    print(
                        f"Non-finite training stats at optimizer step boundary: "
                        f"clean={loss_clean}, rand={rand_loss_mean}, penalty={penalty}. "
                        "Skipping optimizer step."
                    )
                    optimizer.zero_grad()
                    fixed_alphas = None
                    continue

                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if _has_nonfinite_grads(trainable_params):
                    print("Detected non-finite gradients. Skipping optimizer step and clearing grads.")
                    optimizer.zero_grad()
                    fixed_alphas = None
                    continue

                grad_norm = clip_grad_norm_(trainable_params, args.max_grad_norm)
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    print(f"Detected non-finite grad norm ({grad_norm}). Skipping optimizer step.")
                    optimizer.zero_grad()
                    fixed_alphas = None
                    continue

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                did_optimizer_step = True

            if did_optimizer_step and optimizer_steps > 0 and optimizer_steps % EVAL_AFTER_STEPS == 0:
                model.eval()
                val_loss = evaluate_math_loss(model, val_loss_loader)
                if not np.isfinite(val_loss):
                    print(
                        f"Non-finite validation loss detected at epoch {epoch} step {optimizer_steps}: "
                        f"{val_loss}. Stopping training."
                    )
                    done = True
                    break
                print(f"epoch {epoch} step {optimizer_steps} val_loss: {val_loss:.4f} "
                      f"(clean={loss_clean:.4f}, rand={rand_loss_mean:.4f}, penalty={penalty:.4f})")

                if args.alpha_sweep:
                    sweep_path = Path(
                        MODEL_SAVE_DIR,
                        f"{adapter_name}_sweep_step{optimizer_steps}.png",
                    )
                    _alpha_sweep_loss(
                        model=model,
                        val_loss_loader=val_loss_loader,
                        device=DEVICE,
                        alpha_min=args.alpha_sweep_min,
                        alpha_max=args.alpha_sweep_max,
                        alpha_step=args.alpha_sweep_step,
                        save_path=sweep_path,
                    )

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

    # Final alpha sweep
    if args.alpha_sweep:
        sweep_path = Path(MODEL_SAVE_DIR, f"{adapter_name}_sweep_final.png")
        _alpha_sweep_loss(
            model=model,
            val_loss_loader=val_loss_loader,
            device=DEVICE,
            alpha_min=args.alpha_sweep_min,
            alpha_max=args.alpha_sweep_max,
            alpha_step=args.alpha_sweep_step,
            save_path=sweep_path,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train math model (TRAP2)")
    parser.add_argument("--task", type=str, choices=MATH_TASKS, required=True)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--save_dir", type=str, default="math_trap2")
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
    # TRAP2 scale-sampling args
    parser.add_argument("--rand_alpha_min", type=float, default=0.05)
    parser.add_argument("--rand_alpha_max", type=float, default=2.0)
    parser.add_argument("--rand_alpha_weight", type=str, default="inv_sqrt", choices=("inv", "inv_sqrt", "none"))
    parser.add_argument("--lambda_reg", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # Alpha sweep args
    parser.add_argument("--alpha_sweep", action="store_true")
    parser.add_argument("--alpha_sweep_min", type=float, default=-0.5)
    parser.add_argument("--alpha_sweep_max", type=float, default=2.0)
    parser.add_argument("--alpha_sweep_step", type=float, default=0.1)
    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="trap2_math")
    args = parser.parse_args()

    main(args)
