"""QLoRA (4-bit quantized) CLIP Vision Model wrapper.

Loads the CLIP vision backbone in 4-bit NF4 quantization via bitsandbytes,
then applies standard LoRA adapters on top.  The LoRA adapter parameters
remain in fp16/fp32 and are saved/loaded via the PEFT _adapter/ directory.

NOTE: The quantized base weights are NOT suitable for direct .pt merging.
Use the _adapter/ save path for merge experiments.
"""

import torch
import torch.nn as nn
import inspect
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, CLIPModel, CLIPProcessor


def _filter_lora_args(config_dict):
    valid_args = set(inspect.signature(LoraConfig.__init__).parameters.keys())
    valid_args.discard("self")
    return {k: v for k, v in (config_dict or {}).items() if k in valid_args}


class HFQLoRACLIPVisionModel(nn.Module):
    """CLIP Vision encoder with 4-bit quantized backbone + LoRA.

    The model is placed on-device at load time via ``device_map``.
    Do NOT call ``.to(device)`` or ``deepcopy()`` after construction —
    quantized modules do not survive either operation.
    """

    def __init__(
        self,
        model_name,
        cache_dir="data",
        lora_config=None,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load with device_map so quantized layers land on GPU directly.
        # Do NOT .to(device) afterwards.
        model = CLIPModel.from_pretrained(
            model_name, cache_dir=cache_dir,
            quantization_config=bnb_config,
            device_map={"": device},
        )

        # prepare_model_for_kbit_training() calls get_input_embeddings()
        # which is not implemented for CLIPModel (vision-only usage).
        # Do the equivalent manually: freeze base, cast 1D params to fp32.
        #
        # Note: Do NOT enable gradient checkpointing here. With all backbone
        # params frozen, checkpointed blocks see no input requiring grad and
        # autograd drops the graph before LoRA weights, causing:
        # "element 0 of tensors does not require grad".
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1 and param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

        lora_cfg = LoraConfig(**_filter_lora_args(lora_config))

        self.vision_model = get_peft_model(model.vision_model, lora_cfg)
        self.vision_head = model.visual_projection
        self.vision_head.weight.requires_grad = False

        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.train_preprocess = lambda x: processor.image_processor(x, return_tensors="pt")
        self.val_preprocess = lambda x: processor.image_processor(x, return_tensors="pt")

        self.disable_adapters = False

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = {"pixel_values": x}
        if len(x["pixel_values"].shape) == 5:
            x["pixel_values"] = x["pixel_values"].squeeze(1)

        if self.disable_adapters:
            with self.vision_model.disable_adapter():
                vision_encodings = self.vision_model(**x)
        else:
            vision_encodings = self.vision_model(**x)
        text_encoding = self.vision_head(vision_encodings[1])
        return text_encoding

    def replace_sd_keys(self, sd, original, new):
        new_sd = {}
        for key, val in sd.items():
            new_key = key.replace(original, new)
            new_sd[new_key] = val
        return new_sd

    def get_base_model(self):
        return self.vision_model.get_base_model()
