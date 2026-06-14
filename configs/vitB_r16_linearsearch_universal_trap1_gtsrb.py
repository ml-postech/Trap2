import os
from configs.adapter_root import ADAPTER_ROOT

VIT_ARCH = 'ViT-B-32-CLIP'  # Model Architecture
MODEL_DIR = ''              # Model Directory
CACHE_DIR = 'data'              # Where to cache HF pretrained checkpoints
HEAD_DIR = 'data/heads'               # CLIP Head Directory

config = {
    'dataset': [
        {
            'name': 'stanford_cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'stanford_cars_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/cars_shuffled_idxs.pt')
        },
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'dtd_head.pt'),
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'eurosat',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'eurosat_head.pt'),
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'gtsrb_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/gtsrb_shuffled_idxs.pt')
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/mnist_shuffled_idxs.pt')
        },
        {
            'name': 'resisc45',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'resisc45_head.pt'),
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'fgvc_aircraft',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'fgvc_aircraft_head.pt'),
            'batch_size': 32,
            'num_workers': 8,
        },
        {
            'name': 'svhn',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'svhn_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/svhn_shuffled_idxs.pt')
        },

    ],
    'model': {
        'name': 'hf_clip',
        'base_type': "openai/clip-vit-base-patch32",
        'cachedir': CACHE_DIR,
        'bases': [
            os.path.join(ADAPTER_ROOT, "b32_lora_stanford_cars"),
            os.path.join(ADAPTER_ROOT, "b32_lora_dtd"),
            os.path.join(ADAPTER_ROOT, "b32_lora_eurosat"),
            os.path.join(ADAPTER_ROOT, "b32_lora_gtsrb_trap2"),
            os.path.join(ADAPTER_ROOT, "b32_lora_mnist"),
            os.path.join(ADAPTER_ROOT, "b32_lora_resisc45"),
            os.path.join(ADAPTER_ROOT, "b32_lora_fgvc_aircraft"),
            os.path.join(ADAPTER_ROOT, "b32_lora_svhn"),
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
        
        'baseline_norms': {
            'stanford_cars': 99.523,
            'dtd': 68.723,
            'eurosat': 98.259,
            'gtsrb':  98.416,
            'mnist': 99.087,
            'resisc45': 93.048,
            'fgvc_aircraft': 51.725,
            'svhn': 96.168,
        }
    },
    'task_merge_config': {
        'representation': 'matrix_per_layer',
        'merge_space': 'full',
        'merge_method': 'tv',
        'scaling_coeffs': 1.0,
        'isotropize': False,
        # compatible with full / knots / core spaces; override via CLI during search
    },
    'eval_type': 'clip',
}
