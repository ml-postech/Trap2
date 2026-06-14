import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from dataset.templates import get_templates
from utils import get_config_from_name, prepare_experiment_config

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - runtime check for optional dependency
    raise ImportError(
        "open_clip is required for OpenCLIP heads. Install with `pip install open_clip_torch`."
    ) from exc


def build_classification_head(model, tokenizer, classnames, template, device):
    logit_scale = model.logit_scale
    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            embeddings = []
            for t in template:
                tokens = tokenizer([t(classname)]).to(device)
                embedding = model.encode_text(tokens)
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
    parser = argparse.ArgumentParser(description="Generate OpenCLIP classification heads.")
    parser.add_argument("--model", default="convnext_base_w", help="OpenCLIP model name.")
    parser.add_argument("--pretrained", default="laion2b_s13b_b82k", help="OpenCLIP pretrained tag.")
    parser.add_argument("--cache_dir", default="", help="OpenCLIP cache directory.")
    parser.add_argument("--heads_dir", default="data/heads/ConvNeXt-Base-OpenCLIP", help="Directory to save head tensors.")
    parser.add_argument("--config", default="8vision_train_openclip_convnext", help="Config name to load datasets from.")
    parser.add_argument("--precision", default="fp32", help="OpenCLIP precision (fp32/fp16/bf16).")
    args = parser.parse_args()

    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.model,
        pretrained=args.pretrained,
        precision=args.precision,
        device="cpu",
        cache_dir=args.cache_dir,
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(args.config, device=device)
    # Avoid LoRA injection when generating heads.
    raw_config.setdefault('model', {})
    raw_config['model']['ft_config'] = {'type': None}
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = preprocess_train
        dataset_config['eval_preprocess'] = preprocess_val
    raw_config['task_merge_config'] = None
    config = prepare_experiment_config(raw_config)

    dataset_names = np.array([i['name'] for i in raw_config['dataset']])

    os.makedirs(args.heads_dir, exist_ok=True)

    model = model.eval().to(device)
    for dataset_name, loader_dict in tqdm(zip(dataset_names, config['data'])):
        print(f'On {dataset_name}')
        template = get_templates(dataset_name)
        clip_encodings = build_classification_head(
            model, tokenizer, loader_dict['test']['class_names'], template, device=device
        )
        torch.save(clip_encodings, os.path.join(args.heads_dir, f'{dataset_name}_head.pt'))
