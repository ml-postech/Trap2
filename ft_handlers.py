import torch.nn as nn
from collections import defaultdict, OrderedDict

"""
True base_model.model.classifier.original_module.dense.weight
True base_model.model.classifier.original_module.dense.bias
True base_model.model.classifier.original_module.out_proj.weight
True base_model.model.classifier.original_module.out_proj.bias

"""


class LoRAHandler():
    def __init__(self, state_dict):
        self.state_dict = state_dict

    def _normalize_base_name(self, base_name):
        # Align LoRA base names with base model parameter keys.
        base_name = base_name.replace('base_model.model.', 'model.')
        base_name = base_name.replace('base_model.', '')
        base_name = base_name.replace('model.model.', 'model.')
        if not (base_name.endswith('.weight') or base_name.endswith('_weight')):
            base_name = base_name + '.weight'
        return base_name

    def get_ft_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        if not hasattr(sd, 'items'):
            # peft > 0.3.0 fix
            sd = sd.state_dict()

        for key, val in sd.items():
            if '.lora_A.default' in key:
                base_name = self._normalize_base_name(key.replace('.lora_A.default', ''))
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_A' in key:
                base_name = self._normalize_base_name(key.replace('.lora_A', ''))
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_B.default' in key:
                base_name = self._normalize_base_name(key.replace('.lora_B.default', ''))
                layer2lora_parameters[base_name]['B'] = val
            elif '.lora_B' in key:
                base_name = self._normalize_base_name(key.replace('.lora_B', ''))
                layer2lora_parameters[base_name]['B'] = val

        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            # A: [r, I]. B: [O, r]. BxA: [O,r]x[r,I]:[O,I].
            task_parameters[name] = (key2val['B'] @ key2val['A'])
        return OrderedDict(sorted(task_parameters.items()))

    def get_ft_ab_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())
        sd = self.state_dict
        for key, val in sd.items():
            if '.lora_A.default' in key:
                base_name = self._normalize_base_name(key.replace('.lora_A.default', ''))
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_A' in key:
                base_name = self._normalize_base_name(key.replace('.lora_A', ''))
                layer2lora_parameters[base_name]['A'] = val
            elif '.lora_B.default' in key:
                base_name = self._normalize_base_name(key.replace('.lora_B.default', ''))
                layer2lora_parameters[base_name]['B'] = val
            elif '.lora_B' in key:
                base_name = self._normalize_base_name(key.replace('.lora_B', ''))
                layer2lora_parameters[base_name]['B'] = val

        task_parameters = {}
        for name, key2val in layer2lora_parameters.items():
            task_parameters[name] = (key2val['A'], key2val['B'])
        return OrderedDict(sorted(task_parameters.items()))


class HFCLIPLoRAHandler(LoRAHandler):
    def _normalize_base_name(self, base_name):
        # HF CLIP PEFT keys should map onto merge keys like
        # vision_model.base_model.model.encoder.layers.X....weight
        # TaskMerger internally strips ".base_layer" from base-model keys before lookup.
        if base_name.startswith('vision_model.base_model.model.'):
            if not base_name.endswith('.weight'):
                base_name = base_name + '.weight'
        elif base_name.startswith('base_model.model.'):
            base_name = 'vision_model.' + base_name
            if not base_name.endswith('.weight'):
                base_name = base_name + '.weight'
        elif base_name.startswith('vision_model.model.'):
            base_name = base_name.replace('vision_model.model.', 'vision_model.base_model.model.')
            if not base_name.endswith('.weight'):
                base_name = base_name + '.weight'
        else:
            base_name = super()._normalize_base_name(base_name)
        return base_name


class FFTHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))

    def get_final_model(self, **kwargs):
        return self.base_model


class GeneralHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))

    def get_final_model(self, **kwargs):
        return self.base_model
