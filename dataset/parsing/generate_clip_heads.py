import argparse
import os

import numpy as np
import torch
from dataset.templates import get_templates
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from utils import get_config_from_name, prepare_experiment_config


def build_classification_head(model, tokenizer, classnames, template, device):
    logit_scale = model.logit_scale

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            embeddings = []
            for t in template:
                tokenized_template = tokenizer(t(classname))
                tokenized_template = {k: torch.tensor(v).to(device).reshape(1, -1) for k, v in tokenized_template.items()}
                embedding = model.text_projection(model.text_model(**tokenized_template)[1])
                embeddings.append(embedding)
            embeddings = torch.concat(embeddings, dim=0)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    return zeroshot_weights


if __name__ == '__main__':
    # Defaults can be overridden via CLI args.
    vit_path = "openai/clip-vit-large-patch14"
    cache_dir = ""                              # Path to HF cache directory
    classification_heads_dir = "data/heads/ViT-L-14-CLIP"               # dir to save classification heads
    config_name = '8vision_train'               # 8 Vision dataset config name

    parser = argparse.ArgumentParser(description="Generate CLIP classification heads.")
    parser.add_argument("--model", default=vit_path, help="HF model ID for CLIP (text+vision).")
    parser.add_argument("--cache_dir", default=cache_dir, help="HF cache directory.")
    parser.add_argument("--heads_dir", default=classification_heads_dir, help="Directory to save head tensors.")
    parser.add_argument("--config", default=config_name, help="Config name to load datasets from.")
    args = parser.parse_args()

    vit_path = args.model
    cache_dir = args.cache_dir
    classification_heads_dir = args.heads_dir
    config_name = args.config

    model = CLIPModel.from_pretrained(vit_path, cache_dir=cache_dir)
    processor = CLIPProcessor.from_pretrained(vit_path, cache_dir=cache_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    # Add preprocessors
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = processor.image_processor
        dataset_config['eval_preprocess'] = processor.image_processor
    raw_config['task_merge_config'] = None
    config = prepare_experiment_config(raw_config)

    dataset_names = np.array([i['name'] for i in raw_config['dataset']])

    os.makedirs(classification_heads_dir, exist_ok=True)

    language_encoder = model.text_model.eval().to(device)
    for dataset_name, loader_dict in tqdm(zip(dataset_names, config['data'])):
        print(f'On {dataset_name}')
        template = get_templates(dataset_name)
        clip_encodings = build_classification_head(
            model.eval().to(device), processor.tokenizer, loader_dict['test']['class_names'], template, device=device
        )
        torch.save(clip_encodings, os.path.join(classification_heads_dir, f'{dataset_name}_head.pt'))
