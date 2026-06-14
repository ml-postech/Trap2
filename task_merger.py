import math
import time
from collections import OrderedDict, defaultdict
from copy import deepcopy
from functools import partial
import re
import importlib.util
import os
import sys

import einops
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from masking_ops import masked_merge
from merging_functions import ties_merging, tv_merging
from utils import get_mask_fn


_CHAIN_MERGERS = None


def _load_chain_of_merges_mergers():
    global _CHAIN_MERGERS
    if _CHAIN_MERGERS is not None:
        return _CHAIN_MERGERS
    # Compatibility shim for older torch.amp without GradScaler.
    try:
        import torch.amp as _torch_amp  # type: ignore
        if not hasattr(_torch_amp, "GradScaler"):
            try:
                from torch.cuda.amp import GradScaler as _GradScaler
                _torch_amp.GradScaler = _GradScaler
            except Exception:
                pass
    except Exception:
        pass
    # Compatibility shim for older transformers without transformers.masking_utils.
    try:
        import transformers.masking_utils  # noqa: F401
    except Exception:
        try:
            import types
            create_causal_mask = None
            try:
                from transformers.models.llama.modeling_llama import create_causal_mask as _ccm
                create_causal_mask = _ccm
            except Exception:
                try:
                    from transformers.modeling_utils import create_causal_mask as _ccm
                    create_causal_mask = _ccm
                except Exception:
                    create_causal_mask = None
            if create_causal_mask is None:
                def create_causal_mask(*args, **kwargs):
                    return None
            mod = types.ModuleType("transformers.masking_utils")
            mod.create_causal_mask = create_causal_mask
            sys.modules["transformers.masking_utils"] = mod
        except Exception:
            pass
    # Compatibility shim for older transformers without integrations.sdpa_attention.
    try:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: F401
    except Exception:
        try:
            import types
            def sdpa_attention_forward(*args, **kwargs):
                raise NotImplementedError("sdpa_attention_forward is unavailable in this transformers version.")
            mod = types.ModuleType("transformers.integrations.sdpa_attention")
            mod.sdpa_attention_forward = sdpa_attention_forward
            sys.modules["transformers.integrations.sdpa_attention"] = mod
        except Exception:
            pass
    chain_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chain-of-merges"))
    module_path = os.path.join(chain_root, "task_merger.py")
    if not os.path.exists(module_path):
        raise NotImplementedError(
            "RegMean / CoM (data-dependent merging, Table 5) require the external "
            "Chain of Merges implementation, which is not bundled here. "
            "See the README for setup instructions."
        )
    spec = importlib.util.spec_from_file_location("chain_of_merges_task_merger", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, chain_root)
    prev_utils = sys.modules.pop("utils", None)
    try:
        spec.loader.exec_module(module)
    finally:
        if prev_utils is not None:
            sys.modules["utils"] = prev_utils
        sys.path.pop(0)
    _CHAIN_MERGERS = (module.RegMeanMerger, module.CoMMerger)
    return _CHAIN_MERGERS


def directions_to_reps(directions):
    if isinstance(directions, list):
        return [directions_to_reps(direction) for direction in directions]
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for value in directions.values()])


class VectorOps(nn.Module):
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )

    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        if isinstance(vector, list) or len(vector.shape) == 2:
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict

    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)

    def forward(self, directions, merging_fn, merge_config):
        vectors = self.directions_to_reps(directions)
        merged_vector, rows_to_keep, topk_mask = merging_fn(vectors)

        ties_mask = [dict() for _ in range(len(rows_to_keep))]
        for idx in range(len(rows_to_keep)):
            ties_mask[idx] = self.rep_to_state_dict(rows_to_keep[idx], directions[0])
        sd = self.rep_to_state_dict(merged_vector, directions[0])

        return sd, ties_mask


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()

        self.device = device
        self.merge_device = device if merge_config and merge_config.get('merge_on_gpu') and torch.cuda.is_available() else 'cpu'
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models))
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        self.ftms_params = [param_handler(ft_model) for ft_model in finetuned_models]
        self.pretrained_model = pretrained_model.cpu()
        self.pt_params = self.pretrained_model.state_dict()
        self.merge_config = merge_config

    def randbin(self, M, N, P):
        P = 1 - P
        return torch.randint(2, size=(M, N), dtype=torch.float32, device=self.merge_device).bernoulli(P)

    def apply_dare(self, ftms_params, p, dare_seed=0):
        print("DARE seed: ", dare_seed)
        torch.manual_seed(dare_seed)
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}
            for key, finetuned_val in ftm_params.items():
                direction_sd[key] = finetuned_val * self.randbin(finetuned_val.shape[0], finetuned_val.shape[1], p) * (1 / (1 - p))
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def get_task_directions(self, ptm_params, ftms_params):
        finetuned_directions = []
        for ftm_params in ftms_params:
            direction_sd = {}

            for key, finetuned_val in ftm_params.items():
                if key not in ptm_params:
                    ptm_val = torch.zeros_like(finetuned_val)
                else:
                    ptm_val = ptm_params[key]
                direction_sd[key] = finetuned_val - ptm_val
            finetuned_directions += [OrderedDict(sorted(direction_sd.items()))]
        return finetuned_directions

    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, float) or len(scaling_coeffs) == 1:
            self.scaling_coeffs = torch.tensor([scaling_coeffs] * len(self.ftms_params))
        else:
            self.scaling_coeffs = torch.tensor(scaling_coeffs)

    def get_layer_names(self, state_dict):
        layer_names = defaultdict(lambda: dict())
        for key in state_dict:
            if ('.weight' in key) or ('_weight' in key):
                strip_key = key.replace('.weight', '').replace('_weight', '')
                layer_names[strip_key]['weight'] = key
            elif ('.bias' in key) or ('_bias' in key):
                strip_key = key.replace('.bias', '').replace('_bias', '')
                layer_names[strip_key]['bias'] = key
            else:
                layer_names[key]['other'] = key + ':other'
        return layer_names

    def add_task_parameters(self, base_model, parameters, concat_across_output=True, scaling_coeffs=1.):
        if isinstance(parameters, list):
            return [self.add_task_parameters(
                deepcopy(base_model),
                parameter,
                concat_across_output=concat_across_output,
                scaling_coeffs=scaling_coeffs
            ) for parameter in parameters]
        sd = base_model.state_dict()
        for key, val in parameters.items():
            if any('base_layer' in k for k in sd.keys()):
                key = '.'.join(key.split('.')[:-1] + ['base_layer'] + key.split('.')[-1:])
            if concat_across_output:
                sd[key].add_(val.cpu() * scaling_coeffs)
            else:
                sd[key].add_(val.T.cpu() * scaling_coeffs)
        return base_model

    def directions_to_matrices(self, directions, reference_layer_names=None):
        if isinstance(directions, list):
            return [self.directions_to_matrices(direction, reference_layer_names) for direction in directions]

        if reference_layer_names is None:
            layer_names = self.get_layer_names(directions)
        else:
            layer_names = reference_layer_names

        matrices = {}
        for layer_name, parameter_names in layer_names.items():
            if 'other' in parameter_names:
                other_parameter = directions[parameter_names['other'].replace(':other', '')].to(torch.float32)
                # Ensure parameters are always two dimensional
                if len(other_parameter.shape) == 1:  # e.g., class token, positional embeddings
                    other_parameter = other_parameter[None, :]
                elif len(other_parameter.shape) > 2:  # e.g., patch embeddings
                    other_parameter = other_parameter.flatten(1)
                matrices[layer_name + ':other'] = other_parameter
            elif 'weight' in parameter_names:
                weight_name = parameter_names['weight']
                weight = directions[weight_name]
                if 'norm' in layer_name or 'ln' in layer_name:
                    weight = torch.diag(weight)
                matrices[layer_name] = weight.flatten(1)
                if 'bias' in parameter_names:
                    bias = directions[parameter_names['bias']]
                    matrices[layer_name] = torch.concat((matrices[layer_name], bias.reshape(-1, 1)), dim=1)
        return matrices

    def matrix_to_state_dict(self, matrix, state_dict, remove_keys=[]):
        if isinstance(matrix, list):
            return [self.matrix_to_state_dict(m, state_dict) for m in matrix]

        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]

        layer_names = self.get_layer_names(reference_dict)
        merged_state_dict = {}
        for layer_name, value in matrix.items():

            parameter_types = layer_names[layer_name.replace(':other', '')]
            if 'other' in parameter_types:
                name = parameter_types['other'].replace(':other', '')
                merged_state_dict[name] = value.reshape(reference_dict[name].shape)
            else:
                if 'bias' in parameter_types:
                    bias_index = value.shape[1] - 1
                    value, bias = value[:, :bias_index], value[:, -1].flatten()
                    merged_state_dict[parameter_types['bias']] = bias
                if 'norm' in layer_name or 'ln' in layer_name:
                    value = torch.diagonal(value)
                name = parameter_types['weight']
                merged_state_dict[name] = value.reshape(*(reference_dict[name].shape))

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in merged_state_dict:
            for key in remove_keys:
                merged_state_dict[key] = merged_state_dict[
                    "transformer.shared.weight"
                ]
        return merged_state_dict

    def transform(self, *args, **kwargs):
        return


class MatrixPerLayerMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models,
            pretrained_model=pretrained_model,
            param_handler=param_handler,
            device=device,
            merge_config=merge_config
        )

        self.layer_names = self.get_layer_names(self.ftms_params[0].get_ft_parameters())
        self.ingredients = None
        self.cache = {}
        self.scalar_scaling = torch.tensor(1.0)
        self.per_task_scaling = None
        self.lmc = False

    def set_scaling_coeffs(self, scaling_coeffs):
        if isinstance(scaling_coeffs, (float, int, np.floating)):
            scaling_coeffs = [float(scaling_coeffs)]
        coeffs_tensor = torch.tensor(scaling_coeffs, dtype=torch.float32)
        # If all coeffs are identical, treat as a single global scale to preserve previous behavior
        if coeffs_tensor.numel() == 1 or torch.allclose(coeffs_tensor, coeffs_tensor[0]):
            self.scalar_scaling = torch.tensor(float(coeffs_tensor[0]))
            self.per_task_scaling = None
        else:
            if coeffs_tensor.numel() != len(self.ftms_params):
                raise ValueError(f"Expected {len(self.ftms_params)} scaling coeffs, got {coeffs_tensor.numel()}")
            self.scalar_scaling = torch.tensor(1.0)
            self.per_task_scaling = coeffs_tensor

    def _apply_delta(self, new_sd, key, delta_w):
        for name, param in new_sd.named_parameters():
            if name == key:
                param.data += self.scalar_scaling * delta_w.type_as(param.data)

    def _process_ties(self, tensor_list, topK=10):
        original_shape = tensor_list[0].shape
        tensor_list = list(map(torch.flatten, tensor_list))
        merged_tv, rows_to_keep, mask = ties_merging(tensor_list, topK=topK)
        return merged_tv.reshape(original_shape)

    def get_iso_matrix(self, ftms_task_dirs):
        summed_vectors = sum([ftms_task_dirs[i] for i in range(len(ftms_task_dirs))])
        return isotropize_matrix(summed_vectors)

    def get_tsv_delta_w(self, ftms_task_dirs):
        if any(vec.dim() != 2 for vec in ftms_task_dirs):
            # TSV requires 2D matrices; fallback to mean if any tensor is non-matrix.
            return torch.stack(ftms_task_dirs).mean(dim=0)
        sv_reduction = 1 / len(ftms_task_dirs)
        for i, vec in enumerate(ftms_task_dirs):
            u, s, v = torch.linalg.svd(vec.to(torch.float64), full_matrices=False)
            if i == 0:
                sum_u = torch.zeros_like(u)
                sum_s = torch.zeros_like(s)
                sum_v = torch.zeros_like(v)
            reduced_index_s = int(s.shape[0] * sv_reduction)
            # select only the first reduced_index_s columns of u and place them
            sum_u[:, i * reduced_index_s: (i + 1) * reduced_index_s] = u[
                :, :reduced_index_s
            ]
            sum_s[i * reduced_index_s: (i + 1) * reduced_index_s] = s[
                :reduced_index_s
            ]
            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s: (i + 1) * reduced_index_s, :] = v[
                :reduced_index_s, :
            ]
        u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
        u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)

        return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v)).type_as(ftms_task_dirs[0])

    def get_core_matrices(self, ftms_params_ab, key, merge_config):
        if key in self.cache:
            return self.cache[key]
        # Extract A and B matrices from all tasks
        A_list, B_list = zip(*[ftm[key] for ftm in ftms_params_ab])
        A_list = [A.to(self.merge_device) for A in A_list]
        B_list = [B.to(self.merge_device) for B in B_list]
        r, n = A_list[0].shape
        m, _ = B_list[0].shape

        A_stack = torch.cat(A_list, dim=0)  # shape: (T*r, n)
        B_stack = torch.cat(B_list, dim=1)  # shape: (m, T*r)

        Vh_A_ref = torch.linalg.svd(A_stack.to(torch.float64), full_matrices=False)[2]  # shape: (T*r, n)
        U_B_ref = torch.linalg.svd(B_stack.to(torch.float64), full_matrices=False)[0]  # shape: (m, T*r)

        M_list = []
        for i, (A, B) in enumerate(zip(A_list, B_list)):
            U_A, S_A, Vh_A = torch.linalg.svd(A.to(torch.float64), full_matrices=False)  # shape: (r, r), (r,), (r, n)
            U_B, S_B, Vh_B = torch.linalg.svd(B.to(torch.float64), full_matrices=False)  # shape: (m, r), (r,), (r, r)

            Q_A = Vh_A @ Vh_A_ref.T
            R_B = U_B_ref.T @ U_B

            # Middle core matrix M = Σ_B * V_B^T * U_A * Σ_A
            M = torch.diag(S_B) @ (Vh_B @ U_A) @ torch.diag(S_A)  # shape: (r, r)
            # Apply alignment to M
            M_aligned = R_B @ M @ Q_A  # shape: (T*r, T*r)

            # The optimized version is the following
            # M_aligned = U_B_ref.T @ B @ A @ Vh_A_ref.T

            M_list.append(M_aligned)

        self.cache[key] = (M_list, U_B_ref, Vh_A_ref)
        return M_list, U_B_ref, Vh_A_ref

    def get_dare_delta_w(self, M_list, dare_coeff=0.3, merge=True):
        # Ensure dare_coeff is a scalar float.
        if isinstance(dare_coeff, (list, tuple)):
            dare_coeff = float(dare_coeff[0])
        elif torch.is_tensor(dare_coeff):
            if dare_coeff.numel() == 1:
                dare_coeff = float(dare_coeff.item())
            else:
                dare_coeff = float(dare_coeff.flatten()[0].item())

        out = []
        for t in M_list:
            if t.numel() == 0:
                out.append(t)
                continue
            # Per-element Bernoulli mask, works for any dimensionality.
            m = (torch.rand_like(t, dtype=torch.float32) > dare_coeff).to(t.dtype)
            out.append((t * m) / (1.0 - dare_coeff))
        stacked = torch.stack(out)
        return stacked.sum(dim=0) if merge else stacked

    def get_cart_delta_w(self, ftms_task_dirs, pruning_rank=0.04, scaling_coeffs=1.):
        theta_avg = torch.stack(ftms_task_dirs).mean(dim=0)
        sum = torch.zeros_like(theta_avg)
        for i in range(len(ftms_task_dirs)):
            tau = ftms_task_dirs[i] - theta_avg
            if tau.dim() != 2:
                # Skip SVD for non-matrix tensors; just accumulate directly.
                sum += tau
                continue
            U, S, Vh = torch.linalg.svd(tau.to(torch.float64), full_matrices=False)
            pruning_rank_k = math.ceil(pruning_rank * S.shape[0])
            sum += U[:, :pruning_rank_k] @ torch.diag(S[:pruning_rank_k]) @ Vh[:pruning_rank_k, :]
        return theta_avg + scaling_coeffs * sum

    def get_knots_components(self, ftms_task_dirs):
        stack = torch.cat(ftms_task_dirs, dim=1)  # shape: (n, T*r*n)
        U, S, Vh = torch.linalg.svd(stack.to(torch.float64), full_matrices=False)
        # Keep only supported basis components
        U = U[:, S > 1e-5].type(torch.float32)
        Vh = Vh[S > 1e-5].type(torch.float32)
        S = S[S > 1e-5].type(torch.float32)

        S[S <= 1e-5] = 0
        Vs = einops.rearrange(Vh, 'Tr (b c) -> b Tr c', b=len(ftms_task_dirs))
        return U, S, list(Vs)

    def _merge_tensors(self, tensors, merge_config):
        # Ensure tensors are on the configured merge device
        tensors = [t.to(self.merge_device) for t in tensors]
        if self.per_task_scaling is not None:
            scaling = self.per_task_scaling.to(self.merge_device)
            tensors = [t * scaling[i] for i, t in enumerate(tensors)]
        if merge_config.get('merge_method') == 'mean':
            return torch.stack(tensors).mean(dim=0)
        elif merge_config.get('merge_method') in ('sum', 'tv'):
            return torch.stack(tensors).sum(dim=0)
        elif merge_config.get('merge_method') == 'ties':
            return self._process_ties(tensors, merge_config.get('topK', 10))
        elif merge_config.get('merge_method') == 'dare':
            if self.lmc:
                tensors_dare = self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=False)
                tensors_dare = [t * self.scaling_coeffs[i] for i, t in enumerate(tensors_dare)]
                return torch.stack(tensors_dare).sum(dim=0)

            return self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=True)

        elif merge_config.get('merge_method') == 'dare-ties':
            tensors_dare = self.get_dare_delta_w(tensors, merge_config.get('dare_pruning_coeffs', 0.3), merge=False)
            return self._process_ties(tensors_dare, merge_config.get('topK', 10))
        elif merge_config.get('merge_method') == 'tsv':
            result = self.get_tsv_delta_w(tensors)
            return result.type_as(tensors[0]) if hasattr(result, 'type_as') else result
        elif merge_config.get('merge_method') == 'cart':
            return self.get_cart_delta_w(
                tensors,
                merge_config.get('cart_pruning_rank', 0.04),
                merge_config.get('cart_scaling_coeffs', 0.1)
            )
        else:
            raise ValueError(f"Unknown merge_method: {merge_config.get('merge_method')}")

    def merge(self, merge_config):
        print(f"Merging using {merge_config.get('merge_space')} - {merge_config.get('merge_method')}")
        print(f"Isotropizing = {merge_config.get('isotropize', False)}")

        # Determine merge space and prepare parameters
        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()
        if merge_config.get('merge_space') in ('core', 'separate_a_b', 'core-vector'):
            ftms_params_ab = [ftm.get_ft_ab_parameters() for ftm in self.ftms_params]
            relevant_ab_keys = self.ftms_params[0].get_ft_ab_parameters().keys()
        else:
            ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
            ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)

        # with newer peft versions, the keys may not have '.base_layer' in them
        # so we handle it by replacing it with an empty string (a cleaner solution would be nice)
        all_keys = self.pretrained_model.state_dict().keys()
        new_sd = deepcopy(self.pretrained_model)

        avg_ranks = []
        if merge_config.get('merge_space') == 'full':
            base_sd = self.pretrained_model.state_dict()
            for key in tqdm(all_keys, desc="Merging full space"):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_task_dirs[0]:
                    tensor_list = [
                        deepcopy(ft_dir[key_base]).to(self.merge_device)
                        for ft_dir in ftms_task_dirs
                    ]
                    delta_w = self._merge_tensors(tensor_list, merge_config)
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                    if delta_w.ndim == 2:
                        rank = torch.linalg.matrix_rank(delta_w).item()
                        avg_ranks.append(rank)
                    self._apply_delta(new_sd, key, delta_w)

        elif merge_config.get('merge_space') == 'knots':
            for key in tqdm(all_keys, desc="Merging knots space"):
                key_base = key.replace('.base_layer', '')
                if key_base in ftms_task_dirs[0]:
                    if key_base in self.cache:
                        U, S, Vs = self.cache[key_base]
                    else:
                        tensor_list = [deepcopy(ft_dir[key_base]).to(self.merge_device) for ft_dir in ftms_task_dirs]
                        U, S, Vs = self.get_knots_components(tensor_list)
                        self.cache[key_base] = (U, S, Vs)

                    Vs_merged = self._merge_tensors(Vs, merge_config)

                    delta_w = U @ torch.diag(S) @ Vs_merged
                    if merge_config.get('isotropize', False):
                        delta_w = isotropize_matrix(delta_w)
                    if delta_w.ndim == 2:
                        rank = torch.linalg.matrix_rank(delta_w).item()
                        avg_ranks.append(rank)
                    self._apply_delta(new_sd, key, delta_w)

        elif merge_config.get('merge_space') == 'core':
            for key in tqdm(all_keys, desc="Merging core space"):
                key_base = key.replace('.base_layer', '')
                if key_base in relevant_ab_keys:
                    M_list, U_B_ref, Vh_A_ref = self.get_core_matrices(ftms_params_ab, key_base, merge_config)

                    M_merged = self._merge_tensors(M_list, merge_config)

                    if merge_config.get('isotropize', False):
                        M_merged = isotropize_matrix(M_merged)

                    delta_W = U_B_ref @ M_merged @ Vh_A_ref

                    rank = torch.linalg.matrix_rank(delta_W).item()
                    avg_ranks.append(rank)
                    self._apply_delta(new_sd, key, delta_W)

        else:
            raise ValueError(f"Unknown merge_space: {merge_config.get('merge_space')}")

        if avg_ranks:
            print(f"Average rank of delta_W across all layers: {sum(avg_ranks)/len(avg_ranks):.2f}")

        return new_sd


def isotropize_matrix(matrix):
    if matrix.ndim != 2:
        return matrix
    U, S, V = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    S_iso = S.mean() * torch.ones_like(S)
    return U @ torch.diag(S_iso) @ V


def get_merge_handler(rep_type):
    if rep_type == 'matrix_per_layer':
        return MatrixPerLayerMerger
    elif rep_type == 'regmean-vector':
        RegMeanMerger, _ = _load_chain_of_merges_mergers()
        return RegMeanMerger
    elif rep_type == 'com-vector':
        _, CoMMerger = _load_chain_of_merges_mergers()
        return CoMMerger
