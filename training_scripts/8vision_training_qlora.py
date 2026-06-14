"""QLoRA baseline training for CLIP ViT vision tasks.

Identical to 8vision_training.py but loads the CLIP backbone in 4-bit
quantization (NF4) via bitsandbytes before applying LoRA adapters.

NOTE: QLoRA models are loaded on-device via device_map and must NOT be
deepcopy'd or .to(device)'d after construction.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from models.huggingface_clip_qlora import HFQLoRACLIPVisionModel
from utils import (
    CrossEntropyLoss,
    GradScaler,
    LoraConfig,
    cosine_lr,
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
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


def train_cliphead_qlora(model, train_loader, val_loader, test_loader,
                         class_vectors, remap_class_idxs=None,
                         training_config=None, model_save_path=None):
    epochs = training_config['epochs']
    max_steps = training_config['max_steps']
    device = training_config['device']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'],
                                  weight_decay=training_config['wd'])
    ne_iters = len(train_loader)
    scheduler = cosine_lr(optimizer, training_config['lr'],
                          training_config['warm_up'], max_steps)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=training_config.get('label_smoothing', 0.0))

    early_stopper = EarlyStopper(
        patience=training_config['early_stopping_patience'],
        min_delta=training_config['early_stopping_min_delta'],
    )
    end = False
    val_acc = 0.0
    pbar = tqdm(range(epochs), desc=f'QLoRA finetuning, prev acc: {val_acc}: ')

    for epoch in range(epochs):
        for i, (inputs, labels) in tqdm(enumerate(train_loader),
                                         desc=f"Training epoch {epoch + 1}/{epochs}"):
            model = model.train()
            step = i + epoch * ne_iters

            # max_steps guard — outside eval block
            if step >= max_steps:
                end = True
                break

            optimizer.zero_grad(set_to_none=True)
            encodings = model(inputs.to(device))
            normed_encodings = encodings / encodings.norm(dim=-1, keepdim=True)
            logits = (100.0 * normed_encodings @ class_vectors.T)

            if remap_class_idxs is not None:
                remapped_labels = remap_class_idxs[labels].to(device)
            else:
                remapped_labels = labels.to(device)
            loss = loss_fn(logits, remapped_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler(step)

            if (step + 1) % training_config['eval_freq'] == 0 or step + 1 >= max_steps:
                train_acc, train_loss = evaluate_cliphead(
                    model, train_loader, class_vectors=class_vectors,
                    remap_class_idxs=remap_class_idxs, return_loss=True)
                val_acc, val_loss = evaluate_cliphead(
                    model, val_loader, class_vectors=class_vectors,
                    remap_class_idxs=remap_class_idxs, return_loss=True)
                test_acc, test_loss = evaluate_cliphead(
                    model, test_loader, class_vectors=class_vectors,
                    remap_class_idxs=remap_class_idxs, return_loss=True)
                pbar.set_description(f'QLoRA finetuning, prev acc: {val_acc}: ')
                print(f'\nSteps {step}, Test Acc: {test_acc:.2%}, Test Loss: {test_loss:.3f}')
                print(f'Steps {step}, Val Acc: {val_acc:.2%}, Val Loss: {val_loss:.3f}')
                print(f'Steps {step}, Train Acc: {train_acc:.2%}, Train Loss: {train_loss:.3f}')
                performance_log = {
                    "Test Acc": test_acc, "Test Loss": test_loss,
                    "Training Loss": train_loss, "Training Acc": train_acc,
                    "Val Loss": val_loss, "Val Acc": val_acc,
                }
                wandb.log(performance_log, step=step)

                if early_stopper.early_stop(performance_log, model, model_save_path):
                    print(f"Early stopping at step: {step}")
                    with open(f"{Path(model_save_path).with_suffix('')}_best_performance.txt", 'w') as f:
                        f.write(f"Max Val Acc: {early_stopper.max_validation_acc}\n")
                        f.write(f"Best Test Acc: {early_stopper.best_test_acc}\n")
                        f.write(f"Best Train Acc: {early_stopper.best_train_acc}\n")
                        f.write(f"Model Save Path: {model_save_path}\n")
                    end = True
                    break
                print(f'Early Stopping Counter: {early_stopper.counter}')

        if end:
            break

    print("Ending Training")
    print(f" Val Acc @ best ckpt: {early_stopper.max_validation_acc:.2%}, "
          f"Test Acc @ best ckpt: {early_stopper.best_test_acc:.2%}, "
          f"Train Acc @ best ckpt: {early_stopper.best_train_acc:.2%}")
    wandb.log({
        "Val Acc @ best ckpt": early_stopper.max_validation_acc,
        "Test Acc @ best ckpt": early_stopper.best_test_acc,
        "Train Acc @ best ckpt": early_stopper.best_train_acc,
    })
    return model, test_acc, test_loss, val_acc, val_loss


def train_functional(training_config):
    VIT_PATH = "openai/clip-vit-base-patch32"
    CACHE_DIR = 'data'
    CONFIG_NAME = training_config.get('config_name', '8vision_train')

    device = training_config['device']
    set_seed(training_config['seed'])

    raw_config = get_config_from_name(CONFIG_NAME, device=device)
    cfg_model = raw_config.get('model', {})
    if cfg_model.get('base_type'):
        VIT_PATH = cfg_model['base_type']

    default_targets = cfg_model.get('ft_config', {}).get(
        'target_modules', ["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    lc = {
        'r': training_config['lora_rank'],
        'lora_alpha': training_config['lora_rank'],
        'target_modules': training_config.get('target_modules', default_targets),
        'lora_dropout': 0.1,
        'bias': "none",
        'use_dora': training_config.get('use_dora', False),
    }
    lora_config = LoraConfig(
        r=lc['r'],
        lora_alpha=lc['lora_alpha'],
        target_modules=lc['target_modules'],
        lora_dropout=lc['lora_dropout'],
        bias=lc['bias'],
        use_dora=lc['use_dora'],
    )

    model_name = VIT_PATH.split('/')[-1]
    wandb.init(
        project=training_config.get('wandb_project', 'peft_merging'),
        name=f"{model_name}_qlora_r{training_config['lora_rank']}_{training_config['dataset']}",
    )

    # QLoRA model: 4-bit quantized base + LoRA.
    # Already on-device via device_map; no .to() or deepcopy().
    lora_model = HFQLoRACLIPVisionModel(
        model_name=VIT_PATH,
        cache_dir=CACHE_DIR,
        lora_config=lora_config.__dict__,
        device=device,
    )

    print("Device:", device)
    print("Config:", CONFIG_NAME)
    print("Model:", VIT_PATH, "(QLoRA 4-bit)")

    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = lora_model.train_preprocess
        dataset_config['eval_preprocess'] = lora_model.val_preprocess

    data_loaders = prepare_data(raw_config['dataset'], device=device)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]

    config_tag = Path(CONFIG_NAME).stem
    model_save_dir = os.path.join("", f"qlora_rank{lc['r']}_{config_tag}")
    os.makedirs(model_save_dir, exist_ok=True)

    for dataset_name, loader_dict, class_vectors in zip(dataset_names, data_loaders, all_clip_encodings):
        if dataset_name != training_config['dataset']:
            continue
        class_vectors = class_vectors.to(device)

        save_path = os.path.join(
            model_save_dir,
            f"{training_config['dataset']}_lr_{training_config['lr']}_wd_{training_config['wd']}_rank_{training_config['lora_rank']}_qlora.pt",
        )
        print(f'QLoRA finetuning on {dataset_name}')

        finetuned_model, test_acc, test_loss, val_acc, val_loss = train_cliphead_qlora(
            lora_model,
            train_loader=loader_dict['train']['full'],
            val_loader=loader_dict['test']['val'],
            test_loader=loader_dict['test']['test'],
            class_vectors=class_vectors,
            training_config=training_config,
            model_save_path=save_path,
        )
    return val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP ViT with QLoRA (4-bit).")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--wd', type=float, default=1e-1)
    parser.add_argument('--lora_rank', type=int, default=16)
    parser.add_argument('--use_dora', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_freq', type=int, default=2000)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--warm_up', type=int, default=500)
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--config', dest='config_name', type=str, default='8vision_train')
    parser.add_argument('--wandb_project', type=str, default='peft_merging')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    training_config = vars(args)
    training_config['device'] = device

    loss = train_functional(training_config)
    print(loss)
