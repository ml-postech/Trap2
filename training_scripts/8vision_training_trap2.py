import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.stateless import functional_call
from tqdm import tqdm

import wandb
from models.huggingface_clip import HFLoRACLIPVisionModel
from models.openclip_clip import OpenCLIPLoRAVisionModel
from utils import (
    CrossEntropyLoss,
    GradScaler,
    LoraConfig,
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


def _trap2_step(
    model,
    inputs,
    labels,
    class_vectors,
    criterion,
    lambda_reg,
    max_norm,
    device,
    rand_alpha_min,
    rand_alpha_max,
    rand_alpha_weight,
):
    optimizer = model._optimizer
    scaler = model._scaler

    def _forward_loss(param_overrides=None):
        if param_overrides:
            params = dict(base_param_dict)
            params.update(param_overrides)
            buffers = dict(base_buffer_dict)
            try:
                enc = functional_call(model, {**params, **buffers}, (inputs,))
            except TypeError:
                enc = functional_call(model, params, (inputs,), buffers=buffers)
        else:
            enc = model(inputs)
        enc = enc / enc.norm(dim=-1, keepdim=True)
        logits = 100.0 * enc @ class_vectors.T
        return criterion(logits, labels)

    optimizer.zero_grad(set_to_none=True)
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

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # 1) clean
    loss_clean = _forward_loss()
    scaler.scale(loss_clean).backward()

    # 2) random alpha (single-sample Monte Carlo estimate; RNG restore)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    # Uniform sampling over [rand_alpha_min, rand_alpha_max], reject alpha in (0.95, 1.05).
    while True:
        alpha = float(np.random.uniform(rand_alpha_min, rand_alpha_max))
        if not (0.95 < alpha < 1.05):
            break
    loss_alpha = _forward_loss(param_overrides=_build_scaled_params(alpha))
    if rand_alpha_weight == "inv":
        weight = 1.0 / alpha
    elif rand_alpha_weight == "inv_sqrt":
        weight = 1.0 / np.sqrt(alpha)
    else:
        weight = 1.0
    scaler.scale((-lambda_reg * weight) * loss_alpha).backward()
    rand_loss_value = (loss_alpha.detach() * weight).detach()

    # logging values without keeping graph
    with torch.no_grad():
        rand_loss_mean = rand_loss_value.item()
        penalty = lambda_reg * rand_loss_mean
        total_loss = loss_clean - penalty

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()

    total_loss_val = total_loss.item()
    return loss_clean.item(), rand_loss_mean, penalty, total_loss_val


def train_cliphead_lora_trap2(
    lora_model,
    train_loader,
    val_loader,
    test_loader,
    class_vectors,
    training_config,
    model_save_path,
):
    lr = training_config['lr']
    batch_size = training_config['batch_size']
    warm_up = training_config['warm_up']
    max_steps = training_config['max_steps']
    eval_freq = training_config['eval_freq']
    lambda_reg = training_config['lambda_reg']
    max_norm = training_config['max_norm']
    rand_alpha_min = training_config.get('rand_alpha_min', 0.05)
    rand_alpha_max = training_config.get('rand_alpha_max', 2.0)
    rand_alpha_weight = training_config.get('rand_alpha_weight', "inv")
    device = training_config['device']
    lora_model = lora_model.to(device)
    class_vectors = class_vectors.to(device)

    optimizer = torch.optim.AdamW(
        lora_model.parameters(),
        lr=lr,
        weight_decay=training_config['wd'],
        betas=(0.9, 0.98),
        eps=1e-6,
    )
    lora_model._optimizer = optimizer
    lora_model._scaler = GradScaler()

    scheduler = cosine_lr(optimizer, lr, warm_up, max_steps)
    criterion = CrossEntropyLoss()
    early_stopper = EarlyStopper(patience=training_config['early_stopping_patience'],
                                 min_delta=training_config['early_stopping_min_delta'])

    steps = 0
    best_loss = float('inf')
    best_accuracy = 0.0
    loss_hist = []
    performance_log = {}

    pbar = tqdm(total=max_steps)
    while steps < max_steps:
        lora_model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            clean_loss, rand_loss_mean, penalty, total_loss = _trap2_step(
                model=lora_model,
                inputs=inputs,
                labels=labels,
                class_vectors=class_vectors,
                criterion=criterion,
                lambda_reg=lambda_reg,
                max_norm=max_norm,
                device=device,
                rand_alpha_min=rand_alpha_min,
                rand_alpha_max=rand_alpha_max,
                rand_alpha_weight=rand_alpha_weight,
            )

            scheduler(steps)
            steps += 1
            loss_hist.append((clean_loss, rand_loss_mean, penalty, total_loss))
            pbar.update(1)
            pbar.set_description(f"Training epoch {steps // len(train_loader)}/{training_config['epochs']}")

            if steps % eval_freq == 0 or steps == max_steps:
                lora_model.eval()

                train_acc, train_loss = evaluate_cliphead(lora_model, train_loader, class_vectors=class_vectors, return_loss=True)
                val_acc, val_loss = evaluate_cliphead(lora_model, val_loader, class_vectors=class_vectors, return_loss=True)
                test_acc, test_loss = evaluate_cliphead(lora_model, test_loader, class_vectors=class_vectors, return_loss=True)

                recent_losses = loss_hist[-eval_freq:]
                clean_losses = [x[0] for x in recent_losses]
                rand_losses = [x[1] for x in recent_losses]
                penalties = [x[2] for x in recent_losses]
                total_losses = [x[3] for x in recent_losses]

                performance_log = {
                    "Training Acc": train_acc,
                    "Training Loss": train_loss,
                    "Val Acc": val_acc,
                    "Val Loss": val_loss,
                    "Test Acc": test_acc,
                    "Test Loss": test_loss,
                    "Clean Loss": np.mean(clean_losses),
                    "RandAlpha Loss": np.mean(rand_losses),
                    "RandAlpha Penalty": np.mean(penalties),
                    "Total Loss": np.mean(total_losses),
                    "RandAlpha Min": rand_alpha_min,
                    "RandAlpha Max": rand_alpha_max,
                    "RandAlpha Weight": rand_alpha_weight,
                }
                print(f"Steps {steps}, Test Acc: {test_acc:.2%}, Test Loss: {test_loss:.3f}", end='')
                print(f"\nSteps {steps}, Val Acc: {val_acc:.2%}, Val Loss: {val_loss:.3f}")
                print(f"Steps {steps}, Train Acc: {train_acc:.2%}, Train Loss: {train_loss:.3f}")
                print(f"Steps {steps}, Clean Loss: {performance_log['Clean Loss']:.3f}, "
                      f"RandLoss: {performance_log['RandAlpha Loss']:.3f}, "
                      f"Penalty: {performance_log['RandAlpha Penalty']:.3f}, "
                      f"Total Loss: {performance_log['Total Loss']:.3f}")
                if wandb.run is not None:
                    wandb.log(performance_log, step=steps)

                if performance_log['Val Loss'] < best_loss or performance_log['Val Acc'] > best_accuracy:
                    best_loss = performance_log['Val Loss']
                    best_accuracy = performance_log['Val Acc']

                if early_stopper.early_stop(performance_log, lora_model, model_save_path):
                    print("Early stopping")
                    # Save best performance log
                    with open(f"{Path(model_save_path).with_suffix('')}_best_performance.txt", 'w') as f:
                        f.write(f"Early Stopping Counter: {early_stopper.counter}\n")
                        f.write(f"Max Val Acc: {early_stopper.max_validation_acc}\n")
                        f.write(f"Best Test Acc: {early_stopper.best_test_acc}\n")
                        f.write(f"Best Train Acc: {early_stopper.best_train_acc}\n")
                        f.write(f"Model Save Path: {model_save_path}\n")
                        f.write(f"Model Name: {lora_model.__class__.__name__}\n")
                        f.write(f"Model Config: {training_config}\n")
                    print(f"Early Stopping Counter: {early_stopper.counter}")
                    return lora_model, test_acc, test_loss, val_acc, val_loss
                print(f"Early Stopping Counter: {early_stopper.counter}")

            if steps >= max_steps:
                break

    pbar.close()
    return lora_model, performance_log.get('Test Acc', 0.0), performance_log.get('Test Loss', 0.0), performance_log.get('Val Acc', 0.0), performance_log.get('Val Loss', 0.0)


def _build_dataset_maps(raw_config, lora_model, device):
    ds_list = raw_config.get('dataset') or raw_config.get('datasets')
    if ds_list is None:
        raise KeyError("Config must contain 'dataset' list.")

    # Attach preprocess fns expected by prepare_data
    for ds in ds_list:
        ds['train_preprocess'] = lora_model.train_preprocess
        ds['eval_preprocess'] = lora_model.val_preprocess

    data_loaders = prepare_data(ds_list, device=device)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in ds_list]
    dataset_names = np.array([i['name'] for i in ds_list])

    dataset_maps = {
        name: {'loaders': loaders, 'class_vectors': vectors.to(device)}
        for name, loaders, vectors in zip(dataset_names, data_loaders, all_clip_encodings)
    }
    return dataset_maps


def train_functional(training_config, device_override=None):
    VIT_PATH = "openai/clip-vit-base-patch32"
    CACHE_DIR = 'data'
    MODEL_SAVE_DIR = ""
    CONFIG_NAME = training_config.get('config_name', '8vision_train')

    device = _resolve_device(device_override)
    training_config['device'] = device
    set_seed(training_config['seed'])

    if training_config.get('wandb', False):
        wandb.init(project=training_config.get('wandb_project', 'peft_merging'))

    base_cfg = get_config_from_name(CONFIG_NAME, training_config['device'])
    cfg_model = base_cfg.get('model', {})
    use_openclip = cfg_model.get('name') == 'open_clip'
    if cfg_model.get('base_type'):
        VIT_PATH = cfg_model['base_type']

    # Preserve CLI target dataset separately to avoid overwriting config dataset list
    cli_target_dataset = training_config.get('dataset')

    # CLI args override config values when provided (non-None), except dataset list
    for k, v in training_config.items():
        if k == 'dataset':
            continue
        if v is not None:
            base_cfg[k] = v

    training_config = base_cfg
    if cli_target_dataset is not None:
        training_config['target_dataset'] = cli_target_dataset
    if training_config.get('target_modules') is None:
        training_config['target_modules'] = cfg_model.get('ft_config', {}).get('target_modules')

    training_config['num_workers'] = training_config.get('num_workers', 8)
    if training_config.get('batch_size') is None:
        training_config['batch_size'] = training_config.get('default_batch_size', 32)

    lc = {
        'r': training_config['lora_rank'],
        'lora_alpha': training_config['lora_rank'],
        'target_modules': training_config.get(
            'target_modules',
            ["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
        'lora_dropout': 0.1,
        'bias': "none",
        'use_dora': training_config.get('use_dora', False),
    }

    if training_config['peft_type'] == 'lora':
        lora_config = LoraConfig(
            r=lc['r'],
            lora_alpha=lc['lora_alpha'],
            target_modules=lc['target_modules'],
            lora_dropout=lc['lora_dropout'],
            bias=lc['bias'],
            use_dora=lc['use_dora'],
        )

        if use_openclip:
            lora_ptm = OpenCLIPLoRAVisionModel(
                model_name=cfg_model.get('openclip_model'),
                pretrained=cfg_model.get('openclip_pretrained'),
                cache_dir=cfg_model.get('cachedir', CACHE_DIR),
                lora_config=lora_config.__dict__,
                device=device,
                precision=cfg_model.get('openclip_precision', 'fp32'),
            )
        else:
            lora_ptm = HFLoRACLIPVisionModel(
                model_name=VIT_PATH, cache_dir=CACHE_DIR, lora_config=lora_config.__dict__, device=device
            )
    else:
        raise ValueError("Unsupported peft_type")

    # Gradient checkpointing intentionally disabled (restore original behavior).

    print("Device : ", device)
    print("Config: ", CONFIG_NAME)
    if use_openclip:
        print("OpenCLIP model: ", f"{cfg_model.get('openclip_model')}-{cfg_model.get('openclip_pretrained')}")

    dataset_lookup = _build_dataset_maps(training_config, lora_ptm, device)

    config_tag = Path(CONFIG_NAME).stem
    peft_tag = "dora" if training_config.get('use_dora', False) else training_config.get('peft_type')
    model_save_dir = os.path.join("", f"{peft_tag}_rank{lc['r']}_{config_tag}")
    os.makedirs(model_save_dir, exist_ok=True)

    target_dataset = training_config.get('target_dataset')
    if target_dataset is None:
        ds_list = training_config.get('dataset') or training_config.get('datasets')
        target_dataset = ds_list[0]['name'] if isinstance(ds_list, list) else ds_list
    if target_dataset not in dataset_lookup:
        raise ValueError(f"Dataset {target_dataset} not found in config {CONFIG_NAME}.")

    target = dataset_lookup[target_dataset]

    bs_tag = training_config.get('batch_size')
    bs_suffix = f"_bs_{bs_tag}" if bs_tag is not None else ""
    weight_tag = training_config.get('rand_alpha_weight', 'inv')
    save_path = os.path.join(
        model_save_dir,
        f"{target_dataset}_lr_{training_config['lr']}_wd_{training_config['wd']}_rank_{training_config['lora_rank']}_lam_{training_config['lambda_reg']}_alpha_{training_config.get('rand_alpha_min', 0.05)}_{training_config.get('rand_alpha_max', 0.95)}_w_{weight_tag}{bs_suffix}_{peft_tag}_trap2.pt",
    )

    print(f'Finetuning {training_config.get("peft_type")} (TRAP2) on {target_dataset}')
    lora_model = deepcopy(lora_ptm).to(device)

    finetuned_model, test_acc, test_loss, val_acc, val_loss = train_cliphead_lora_trap2(
        lora_model,
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
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--warm_up', type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--use_dora', action='store_true')
    parser.add_argument('--config', dest='config_name', type=str, default='8vision_train')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='peft_merging')
    parser.add_argument('--rand_alpha_min', type=float, default=0.05)
    parser.add_argument('--rand_alpha_max', type=float, default=2.0)
    parser.add_argument('--rand_alpha_weight', type=str, default="inv",
                        choices=["inv", "inv_sqrt", "none"])
    args = parser.parse_args()

    training_config = vars(args)
    training_config['use_dora'] = args.use_dora
    training_config['peft_type'] = 'lora'
    training_config['num_workers'] = 8
    training_config['default_batch_size'] = 32
    training_config['use_target_val'] = 1

    loss = train_functional(training_config, device_override=args.device)
    print(loss)
