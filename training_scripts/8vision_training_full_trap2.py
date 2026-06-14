import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.stateless import functional_call
from tqdm import tqdm

import wandb
from models.huggingface_clip import HFCLIPVisionModel
from models.openclip_clip import OpenCLIPVisionModel
from utils import (
    CrossEntropyLoss,
    GradScaler,
    cosine_lr,
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    get_device,
    prepare_data,
    save_model,
    set_seed,
)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_validation_acc = 0.0
        self.best_test_acc = 0.0
        self.best_train_acc = 0.0

    def early_stop(self, performance_log, model, model_save_path):
        if performance_log['Val Acc'] > (self.max_validation_acc + self.min_delta):
            self.max_validation_acc = performance_log['Val Acc']
            self.best_test_acc = performance_log['Test Acc']
            self.best_train_acc = performance_log['Training Acc']
            save_model(model, model_save_path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def _resolve_device(device_override=None):
    if device_override is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(device_override, str):
        return device_override
    if isinstance(device_override, int):
        return f'cuda:{device_override}'
    raise ValueError(f"Invalid device override: {device_override}")


def _save_best_performance(path, model, training_config, early_stopper):
    with open(f"{Path(path).with_suffix('')}_best_performance.txt", 'w') as f:
        f.write(f"Early Stopping Counter: {early_stopper.counter}\n")
        f.write(f"Max Val Acc: {early_stopper.max_validation_acc}\n")
        f.write(f"Best Test Acc: {early_stopper.best_test_acc}\n")
        f.write(f"Best Train Acc: {early_stopper.best_train_acc}\n")
        f.write(f"Model Save Path: {path}\n")
        f.write(f"Model Name: {model.__class__.__name__}\n")
        f.write(f"Model Config: {training_config}\n")


def _to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
    return batch.to(device)


def _build_dataset_maps(raw_config, model, device):
    ds_list = raw_config.get('dataset') or raw_config.get('datasets')
    if ds_list is None:
        raise KeyError("Config must contain 'dataset' list.")

    for ds in ds_list:
        ds['train_preprocess'] = model.train_preprocess
        ds['eval_preprocess'] = model.val_preprocess

    data_loaders = prepare_data(ds_list, device=device)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in ds_list]
    dataset_names = np.array([i['name'] for i in ds_list])

    dataset_maps = {
        name: {'loaders': loaders, 'class_vectors': vectors.to(device)}
        for name, loaders, vectors in zip(dataset_names, data_loaders, all_clip_encodings)
    }
    return dataset_maps


def train_cliphead_full_trap2(
    model,
    train_loader,
    val_loader,
    test_loader,
    class_vectors,
    training_config,
    model_save_path,
):
    device = training_config['device']
    lr = training_config['lr']
    warm_up = training_config['warm_up']
    max_steps = training_config['max_steps']
    eval_freq = training_config['eval_freq']
    lambda_reg = training_config['lambda_reg']
    max_norm = training_config['max_norm']
    rand_alpha_min = training_config.get('rand_alpha_min', 0.05)
    rand_alpha_max = training_config.get('rand_alpha_max', 2.0)
    rand_alpha_weight = training_config.get('rand_alpha_weight', "inv")

    model = model.to(device)
    class_vectors = class_vectors.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=training_config['wd'],
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    scheduler = cosine_lr(optimizer, lr, warm_up, max_steps)
    scaler = GradScaler()
    criterion = CrossEntropyLoss(label_smoothing=training_config.get('label_smoothing', 0.0))
    early_stopper = EarlyStopper(
        patience=training_config['early_stopping_patience'],
        min_delta=training_config['early_stopping_min_delta'],
    )

    base_params = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def make_scaled_params(alpha):
        params = {}
        for name, p in model.named_parameters():
            w0 = base_params[name]
            params[name] = w0 + alpha * (p - w0)
        return params

    steps = 0
    loss_hist = []
    pbar = tqdm(total=max_steps)
    while steps < max_steps:
        model.train()
        for inputs, labels in train_loader:
            inputs = _to_device(inputs, device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

            enc = model(inputs)
            enc = enc / enc.norm(dim=-1, keepdim=True)
            logits = 100.0 * enc @ class_vectors.T
            loss_clean = criterion(logits, labels)

            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            while True:
                alpha = float(np.random.uniform(rand_alpha_min, rand_alpha_max))
                if not (0.95 < alpha < 1.05):
                    break
            params_alpha = make_scaled_params(alpha)
            enc_alpha = functional_call(model, {**params_alpha, **buffers}, (inputs,), strict=False)
            enc_alpha = enc_alpha / enc_alpha.norm(dim=-1, keepdim=True)
            logits_alpha = 100.0 * enc_alpha @ class_vectors.T
            loss_alpha = criterion(logits_alpha, labels)
            if rand_alpha_weight == "inv":
                weight = 1.0 / alpha
            elif rand_alpha_weight == "inv_sqrt":
                weight = 1.0 / np.sqrt(alpha)
            else:
                weight = 1.0
            rand_loss_mean = loss_alpha * weight
            penalty = lambda_reg * rand_loss_mean
            total_loss = loss_clean - penalty

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # returns pre-clip norm

            if steps % 50 == 0:
                print(f"[step {steps}] grad_norm(pre)={pre_norm.item():.4f}  "
                    f"max_norm={max_norm}  clipped={pre_norm.item() > max_norm}")
                print(f"[step {steps}] scale={scaler.get_scale()}")
                n_grad = sum((p.grad is not None) for p in model.parameters())
                print(f"[step {steps}] n_grad_params={n_grad}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler(steps)

            steps += 1
            pbar.update(1)
            pbar.set_description(f"Training epoch {steps // len(train_loader)}/{training_config['epochs']}")

            loss_hist.append((
                loss_clean.detach().item(),
                rand_loss_mean.detach().item(),
                penalty.detach().item(),
                total_loss.detach().item(),
            ))

            if steps % eval_freq == 0 or steps == max_steps:
                model.eval()
                train_acc, train_loss = evaluate_cliphead(
                    model, train_loader, class_vectors=class_vectors, return_loss=True
                )
                val_acc, val_loss = evaluate_cliphead(
                    model, val_loader, class_vectors=class_vectors, return_loss=True
                )
                test_acc, test_loss = evaluate_cliphead(
                    model, test_loader, class_vectors=class_vectors, return_loss=True
                )
                recent_losses = loss_hist[-eval_freq:]
                clean_losses = [x[0] for x in recent_losses]
                rand_losses_hist = [x[1] for x in recent_losses]
                penalties = [x[2] for x in recent_losses]
                total_losses = [x[3] for x in recent_losses]

                performance_log = {
                    "Training Acc": train_acc,
                    "Training Loss": train_loss,
                    "Val Acc": val_acc,
                    "Val Loss": val_loss,
                    "Test Acc": test_acc,
                    "Test Loss": test_loss,
                    "Clean Loss": float(np.mean(clean_losses)),
                    "RandAlpha Loss": float(np.mean(rand_losses_hist)),
                    "RandAlpha Penalty": float(np.mean(penalties)),
                    "Total Loss": float(np.mean(total_losses)),
                    "RandAlpha Min": rand_alpha_min,
                    "RandAlpha Max": rand_alpha_max,
                    "RandAlpha Weight": rand_alpha_weight,
                }

                print(f"\nSteps {steps}, Test Acc: {test_acc:.2%}, Test Loss: {test_loss:.3f}")
                print(f"Steps {steps}, Val Acc: {val_acc:.2%}, Val Loss: {val_loss:.3f}")
                print(f"Steps {steps}, Train Acc: {train_acc:.2%}, Train Loss: {train_loss:.3f}")
                print(
                    f"Steps {steps}, Clean Loss: {performance_log['Clean Loss']:.3f}, "
                    f"RandLoss: {performance_log['RandAlpha Loss']:.3f}, "
                    f"Penalty: {performance_log['RandAlpha Penalty']:.3f}, "
                    f"Total Loss: {performance_log['Total Loss']:.3f}"
                )
                if wandb.run is not None:
                    wandb.log(performance_log, step=steps)

                if early_stopper.early_stop(performance_log, model, model_save_path):
                    print("Early stopping")
                    _save_best_performance(model_save_path, model, training_config, early_stopper)
                    return model, test_acc, test_loss, val_acc, val_loss
                print(f"Early Stopping Counter: {early_stopper.counter}")

            if steps >= max_steps:
                break

    pbar.close()
    return model, performance_log.get('Test Acc', 0.0), performance_log.get('Test Loss', 0.0), performance_log.get('Val Acc', 0.0), performance_log.get('Val Loss', 0.0)


def train_functional(training_config, device_override=None):
    VIT_PATH = "openai/clip-vit-base-patch32"
    CACHE_DIR = 'data'
    CONFIG_NAME = training_config.get('config_name', '8vision_train')

    device = _resolve_device(device_override)
    training_config['device'] = device
    set_seed(training_config['seed'])

    if training_config.get('wandb', False):
        wandb.init(project=training_config.get('wandb_project', 'peft_merging'))

    base_cfg = get_config_from_name(CONFIG_NAME, training_config['device'])
    base_cfg.setdefault('model', {})
    base_cfg['model']['ft_config'] = {'type': 'full', 'r': 0}
    cfg_model = base_cfg.get('model', {})
    use_openclip = cfg_model.get('name') == 'open_clip'
    if cfg_model.get('base_type'):
        VIT_PATH = cfg_model['base_type']

    cli_target_dataset = training_config.get('dataset')
    for k, v in training_config.items():
        if k == 'dataset':
            continue
        if v is not None:
            base_cfg[k] = v

    training_config = base_cfg
    if cli_target_dataset is not None:
        training_config['target_dataset'] = cli_target_dataset

    if training_config.get('batch_size') is None:
        training_config['batch_size'] = training_config.get('default_batch_size', 32)

    if use_openclip:
        full_ptm = OpenCLIPVisionModel(
            model_name=cfg_model.get('openclip_model'),
            pretrained=cfg_model.get('openclip_pretrained'),
            cache_dir=cfg_model.get('cachedir', CACHE_DIR),
            device=device,
            precision=cfg_model.get('openclip_precision', 'fp32'),
            force_quick_gelu=True,
        )
    else:
        full_ptm = HFCLIPVisionModel(
            model_name=VIT_PATH,
            cache_dir=cfg_model.get('cachedir', CACHE_DIR),
            device=device,
        )

    print("Device : ", device)
    print("Config: ", CONFIG_NAME)
    if use_openclip:
        print("OpenCLIP model: ", f"{cfg_model.get('openclip_model')}-{cfg_model.get('openclip_pretrained')}")

    dataset_lookup = _build_dataset_maps(training_config, full_ptm, device)

    config_tag = f"{Path(CONFIG_NAME).stem}_full"
    model_save_dir = os.path.join("", f"full_trap2_{config_tag}_pat{training_config['early_stopping_patience']}")
    os.makedirs(model_save_dir, exist_ok=True)

    target_dataset = training_config.get('target_dataset')
    if target_dataset is None:
        ds_list = training_config.get('dataset') or training_config.get('datasets')
        target_dataset = ds_list[0]['name'] if isinstance(ds_list, list) else ds_list
    if target_dataset not in dataset_lookup:
        raise ValueError(f"Dataset {target_dataset} not found in config {CONFIG_NAME}.")

    target = dataset_lookup[target_dataset]

    weight_tag = training_config.get('rand_alpha_weight', 'inv')
    save_path = os.path.join(
        model_save_dir,
        f"{target_dataset}_lr_{training_config['lr']}_wd_{training_config['wd']}"
        f"_lam_{training_config['lambda_reg']}_alpha_{training_config.get('rand_alpha_min', 0.05)}"
        f"_{training_config.get('rand_alpha_max', 2.0)}"
        f"_w_{weight_tag}_full_trap2.pt",
    )

    print(f'Finetuning full model (TRAP2) on {target_dataset}')
    full_model = deepcopy(full_ptm).to(device)

    finetuned_model, test_acc, test_loss, val_acc, val_loss = train_cliphead_full_trap2(
        full_model,
        train_loader=target['loaders']['train']['full'],
        val_loader=target['loaders']['test']['val'],
        test_loader=target['loaders']['test']['test'],
        class_vectors=target['class_vectors'],
        training_config=training_config,
        model_save_path=save_path,
    )
    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--warm_up', type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--config', dest='config_name', type=str, default='8vision_train')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='peft_merging')
    parser.add_argument('--rand_alpha_min', type=float, default=0.05)
    parser.add_argument('--rand_alpha_max', type=float, default=2.0)
    parser.add_argument('--rand_alpha_weight', type=str, default="inv",
                        choices=["inv", "inv_sqrt", "none"])
    args = parser.parse_args()

    training_config = vars(args)
    training_config['num_workers'] = 8
    training_config['default_batch_size'] = 32
    training_config['use_target_val'] = 1

    loss = train_functional(training_config, device_override=args.device)
    print(loss)
