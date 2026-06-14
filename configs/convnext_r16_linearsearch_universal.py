import os
from configs.adapter_root import ADAPTER_ROOT

VIT_ARCH = 'ConvNeXt-Base-OpenCLIP'
MODEL_DIR = ''              # Model Directory
CACHE_DIR = 'data'
HEAD_DIR = 'data/heads'

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
        'name': 'open_clip',
        'openclip_model': 'convnext_base_w',
        'openclip_pretrained': 'laion2b_s13b_b82k',
        'openclip_precision': 'fp32',
        'cachedir': CACHE_DIR,
        'bases': [
            os.path.join(ADAPTER_ROOT, "convnext_lora_stanford_cars"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_dtd"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_eurosat"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_gtsrb"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_mnist"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_resisc45"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_fgvc_aircraft"),
            os.path.join(ADAPTER_ROOT, "convnext_lora_svhn"),
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["fc1", "fc2"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
        'baseline_norms': {
            'stanford_cars': 98.492,
            'dtd': 76.436,
            'eurosat': 98.926,
            'gtsrb': 99.268,
            'mnist': 99.325,
            'resisc45': 96.063,
            'fgvc_aircraft': 59.226,
            'svhn': 97.105,
        },
    },
    'task_merge_config': {
        'representation': 'matrix_per_layer',
        'merge_space': 'full',
        'merge_method': 'tv',
        'scaling_coeffs': 1.0,
        'isotropize': False,
    },
    'eval_type': 'clip',
}
