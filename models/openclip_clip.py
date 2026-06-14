from collections import defaultdict
import inspect

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

try:
    import open_clip
except ImportError as exc:  # pragma: no cover - runtime check for optional dependency
    raise ImportError(
        "open_clip is required for OpenCLIP models. Install with `pip install open_clip_torch`."
    ) from exc


def _filter_peft_args(config, config_cls):
    valid_args = set(inspect.signature(config_cls.__init__).parameters.keys())
    valid_args.discard("self")
    return {k: v for k, v in config.items() if k in valid_args}


class OpenCLIPVisionModel(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        cache_dir="",
        device="cpu",
        precision="fp32",
        force_quick_gelu=False,
    ):
        super().__init__()
        self.device = device
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
            force_quick_gelu=force_quick_gelu,
        )
        self.model = model
        self.train_preprocess = preprocess_train
        self.val_preprocess = preprocess_val

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("pixel_values", x.get("image", x))
        x = x.to(self.device)
        return self.model.encode_image(x)

    def get_base_model(self):
        return self


class OpenCLIPLoRAVisionModel(nn.Module):
    def __init__(
        self,
        model_name,
        pretrained,
        cache_dir="",
        lora_config=None,
        device="cpu",
        precision="fp32",
        force_quick_gelu=False,
    ):
        super().__init__()
        self.device = device
        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            precision=precision,
            device=device,
            cache_dir=cache_dir,
            force_quick_gelu=force_quick_gelu,
        )
        lora_cfg = LoraConfig(**_filter_peft_args(lora_config or {}, LoraConfig))
        model.visual = get_peft_model(model.visual, lora_cfg)
        self.model = model
        self.train_preprocess = preprocess_train
        self.val_preprocess = preprocess_val
        self.disable_adapters = False

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get("pixel_values", x.get("image", x))
        x = x.to(self.device)
        if self.disable_adapters:
            with self.model.visual.disable_adapter():
                return self.model.encode_image(x)
        return self.model.encode_image(x)

    def get_base_model(self):
        self.model.visual = self.model.visual.get_base_model()
        return self.model


def get_openclip_model_from_config(config, device):
    model_cfg = config.get("model", defaultdict(lambda: None))
    ft_cfg = model_cfg.get("ft_config", defaultdict(lambda: None))
    model_name = model_cfg.get("openclip_model")
    pretrained = model_cfg.get("openclip_pretrained")
    cache_dir = model_cfg.get("cachedir", "")
    precision = model_cfg.get("openclip_precision", "fp32")

    if ft_cfg.get("type") == "lora":
        return OpenCLIPLoRAVisionModel(
            model_name=model_name,
            pretrained=pretrained,
            cache_dir=cache_dir,
            lora_config=ft_cfg,
            device=device,
            precision=precision,
            force_quick_gelu=model_cfg.get("force_quick_gelu", False),
        )
    return OpenCLIPVisionModel(
        model_name=model_name,
        pretrained=pretrained,
        cache_dir=cache_dir,
        device=device,
        precision=precision,
        force_quick_gelu=model_cfg.get("force_quick_gelu", False),
    )
