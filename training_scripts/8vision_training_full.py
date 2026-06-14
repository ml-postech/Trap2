import argparse
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import wandb
from models.huggingface_clip import HFCLIPVisionModel
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


def train_cliphead_full(
    model,
    train_loader,
    val_loader,
    test_loader,
    class_vectors,
    remap_class_idxs=None,
    eval_class_vectors=None,
    clip_mapper=None,
    training_config=None,
    model_save_path=None,
):
    """Train a cliphead model with full vision encoder finetuning."""
    epochs = training_config['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['lr'], weight_decay=training_config['wd'])
    ne_iters = len(train_loader)
    scheduler = cosine_lr(optimizer, training_config['lr'], training_config['warm_up'], training_config['max_steps'])
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss(label_smoothing=training_config['label_smoothing'])
    device = get_device(model)

    val_acc = 0.0
    pbar = tqdm(range(epochs), desc=f'finetuning, prev acc: {val_acc}: ')
    early_stopper = EarlyStopper(
        patience=training_config['early_stopping_patience'],
        min_delta=training_config['early_stopping_min_delta'],
    )
    end = False
    for epoch in range(epochs):
        for i, (inputs, labels) in tqdm(enumerate(train_loader), desc=f"Training epoch {epoch + 1}/{epochs}"):
            model = model.train()
            step = i + epoch * ne_iters
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

            if (step + 1) % training_config['eval_freq'] == 0:
                train_acc, train_loss = evaluate_cliphead(
                    model, train_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs, return_loss=True
                )
                val_acc, val_loss = evaluate_cliphead(
                    model, val_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs, return_loss=True
                )
                test_acc, test_loss = evaluate_cliphead(
                    model, test_loader, class_vectors=class_vectors, remap_class_idxs=remap_class_idxs, return_loss=True
                )
                pbar.set_description(f'finetuning, prev acc: {val_acc}: ')
                print(f'\nSteps {step}, Test Acc: {test_acc:.2%}, Test Loss: {test_loss:.3f}')
                print(f'Steps {step}, Val Acc: {val_acc:.2%}, Val Loss: {val_loss:.3f}')
                print(f'Steps {step}, Train Acc: {train_acc:.2%}, Train Loss: {train_loss:.3f}')
                performance_log = {
                    "Test Acc": test_acc,
                    "Test Loss": test_loss,
                    "Training Loss": train_loss,
                    "Training Acc": train_acc,
                    "Val Loss": val_loss,
                    "Val Acc": val_acc,
                }
                wandb.log(performance_log, step=step)

                if early_stopper.early_stop(performance_log, model, model_save_path):
                    print(f"Early stopping at : {step}")
                    end = True
                    with open(f"{Path(model_save_path).with_suffix('')}_best_performance.txt", 'w') as f:
                        f.write(f"Early Stopping Counter: {early_stopper.counter}\n")
                        f.write(f"Max Val Acc: {early_stopper.max_validation_acc}\n")
                        f.write(f"Best Test Acc: {early_stopper.best_test_acc}\n")
                        f.write(f"Best Train Acc: {early_stopper.best_train_acc}\n")
                        f.write(f"Model Save Path: {model_save_path}\n")
                        f.write(f"Model Name: {model.__class__.__name__}\n")
                        f.write(f"Model Config: {training_config}\n")
                    break
                print(f'Early Stopping Counter: {early_stopper.counter}')

                if step >= training_config['max_steps'] == 0:
                    end = True
                    break
        if end:
            break
    print("Ending Training")
    print(
        f" Val Acc @ best ckpt: {early_stopper.max_validation_acc:.2%}, "
        f"Test Acc @ best ckpt: {early_stopper.best_test_acc:.2%}, "
        f"Train Acc @ best ckpt: {early_stopper.best_train_acc:.2%}"
    )
    wandb.log({
        "Val Acc @ best ckpt": early_stopper.max_validation_acc,
        "Test Acc @ best ckpt": early_stopper.best_test_acc,
        "Train Acc @ best ckpt": early_stopper.best_train_acc,
    })

    return model, test_acc, test_loss, val_acc, val_loss


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def train_functional(training_config=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(CONFIG_NAME, device=device)

    if training_config is None:
        training_config = raw_config['training_config']
    print(training_config)

    model_base_type = raw_config.get('model', {}).get('base_type')
    vit_path = model_base_type or VIT_PATH
    model_name = vit_path.split('/')[-1]
    wandb.init(project="peft_merging", config=training_config, name=f"{model_name}_full_finetuning")

    full_ptm = HFCLIPVisionModel(
        model_name=vit_path,
        cache_dir=CACHE_DIR,
        device=device
    )

    print("Device : ", device)
    print("Config: ", CONFIG_NAME)
    print("Model base_type: ", vit_path)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])

    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = full_ptm.train_preprocess
        dataset_config['eval_preprocess'] = full_ptm.val_preprocess

    data_loaders = prepare_data(raw_config['dataset'], device=device)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]

    config_tag = Path(CONFIG_NAME).stem
    save_tag = training_config.get('save_tag')
    suffix = f"_{save_tag}" if save_tag else ""
    model_save_dir = os.path.join(MODEL_SAVE_DIR, f"full_rank_{config_tag}{suffix}")
    os.makedirs(model_save_dir, exist_ok=True)
    val_loss = 0
    for dataset_name, loader_dict, class_vectors in tqdm(zip(dataset_names, data_loaders, all_clip_encodings)):
        if dataset_name != training_config['dataset']:
            continue
        class_vectors = class_vectors.to(device)

        save_path = os.path.join(
            model_save_dir,
            f"{training_config['dataset']}_lr_{training_config['lr']}_wd_{training_config['wd']}_full.pt"
        )
        print(f'Finetuning full model on {dataset_name}')
        full_model = deepcopy(full_ptm)
        full_model = full_model.to(device)

        finetuned_model, test_acc, test_loss, val_acc, val_loss = train_cliphead_full(
            full_model,
            train_loader=loader_dict['train']['full'],
            val_loader=loader_dict['test']['val'],
            test_loader=loader_dict['test']['test'],
            class_vectors=class_vectors,
            training_config=training_config,
            model_save_path=save_path
        )
    return val_loss


if __name__ == "__main__":
    VIT_PATH = "openai/clip-vit-base-patch32"
    CACHE_DIR = 'data'                                      # Path to cache directory
    MODEL_SAVE_DIR = ""                                     # Path to save the model
    CONFIG_NAME = '8vision_train'                           # Training config file name
    training_config = {
        'early_stopping_patience': 5,
        'epochs': 2000,
        'max_steps': 100000,
        'eval_freq': 2000,
        'lr': 1e-5,
        'wd': 1e-1,
        'warm_up': 500,
        'label_smoothing': 0.0,
        'early_stopping_min_delta': 1e-3,
        'seed': 420,
    }
    set_seed(training_config['seed'])

    parser = argparse.ArgumentParser(description="Train a CLIP model with full finetuning.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset to train on.")
    parser.add_argument('--config', type=str, default=CONFIG_NAME, help="Config name to load.")
    parser.add_argument('--lr', type=float, default=training_config['lr'], help="Learning rate.")
    parser.add_argument('--wd', type=float, default=training_config['wd'], help="Weight decay.")
    parser.add_argument('--eval_freq', type=int, default=training_config['eval_freq'], help="Eval frequency (steps).")
    parser.add_argument('--save_tag', type=str, default=None, help="Optional suffix for save directory.")
    args = parser.parse_args()
    training_config['dataset'] = args.dataset
    training_config['lr'] = args.lr
    training_config['wd'] = args.wd
    training_config['eval_freq'] = args.eval_freq
    training_config['save_tag'] = args.save_tag
    CONFIG_NAME = args.config
    print(f"Training on dataset: {training_config['dataset']}")

    loss = train_functional(training_config)
    print(loss)
