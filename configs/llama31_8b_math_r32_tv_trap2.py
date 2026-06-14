import os
from configs.adapter_root import ADAPTER_ROOT
"""Config for merging TRAP2-protected math adapters (8B, r=32)."""

CACHE_DIR = 'data'
INGREDIENTS_PATH = ""
PTM_PATH = ""

config = {
    'dataset': [
        {'name': 'gsm8k'},
        {'name': 'asdiv'},
    ],
    'model': {
        'name': 'meta-llama/Llama-3.1-8B',
        'ptm_path': PTM_PATH,
        'cachedir': CACHE_DIR,
        'bases': [
            os.path.join(ADAPTER_ROOT, "llama8b_lora_gsm8k_trap2"),
            os.path.join(ADAPTER_ROOT, "llama8b_lora_asdiv_trap2"),
        ],
        'ft_config': {
            'type': 'lora',
            'subtype': 'peft',
        },
        'peft_config': {
            'task_type': 'CAUSAL_LM',
            'inference_mode': True,
            'r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    },
    'task_merge_config': {
        'ingredients_path': INGREDIENTS_PATH,
        'representation': 'matrix_per_layer',
        'merge_space': 'full',
        'merge_method': 'tv',
        'scaling_coeffs': .3,
        'isotropize': False,
    },
    'eval_type': 'generation',
}
