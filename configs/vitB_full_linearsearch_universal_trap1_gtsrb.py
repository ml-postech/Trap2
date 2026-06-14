import os
from configs.adapter_root import ADAPTER_ROOT

VIT_ARCH = 'ViT-B-32-CLIP'
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
        'name': 'hf_clip',
        'base_type': "openai/clip-vit-base-patch32",
        'cachedir': CACHE_DIR,
        'bases': [
            os.path.join(ADAPTER_ROOT, "b32_full_stanford_cars"),
            os.path.join(ADAPTER_ROOT, "b32_full_dtd"),
            os.path.join(ADAPTER_ROOT, "b32_full_eurosat"),
            os.path.join(ADAPTER_ROOT, "b32_full_gtsrb_trap2"),
            os.path.join(ADAPTER_ROOT, "b32_full_mnist"),
            os.path.join(ADAPTER_ROOT, "b32_full_resisc45"),
            os.path.join(ADAPTER_ROOT, "b32_full_fgvc_aircraft"),
            os.path.join(ADAPTER_ROOT, "b32_full_svhn"),
],
        'baseline_norms': {
            'stanford_cars': 99.798,
            'dtd': 62.766,
            'eurosat': 98.444,
            'gtsrb': 99.109,
            'mnist': 99.212,
            'resisc45': 93.317,
            'fgvc_aircraft': 49.625,
            'svhn': 96.706,
        },
        'ft_config': {
            'type': 'full',
            'r': 0,
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
