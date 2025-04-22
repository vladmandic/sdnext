from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import torch
from diffusers.quantizers.base import DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationConfigMixin
from diffusers.utils import get_module_from_name
from accelerate import init_empty_weights

from modules import devices


class NNCF_T5DenseGatedActDense(torch.nn.Module): # forward can't find what self is without creating a class
    def __init__(self, T5DenseGatedActDense, dtype):
        super().__init__()
        self.wi_0 = T5DenseGatedActDense.wi_0
        self.wi_1 = T5DenseGatedActDense.wi_1
        self.wo = T5DenseGatedActDense.wo
        self.dropout = T5DenseGatedActDense.dropout
        self.act = T5DenseGatedActDense.act
        self.torch_dtype = dtype

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.to(self.torch_dtype) # this line needs to be forced
        hidden_states = self.wo(hidden_states)
        return hidden_states


class QuantizationMethod(str, Enum):
    NNCF = "nncf"


class NNCFQuantizer(DiffusersQuantizer):
    r"""
    Diffusers Quantizer for NNCF
    """

    requires_parameters_quantization = True
    use_keep_in_fp32_modules = True
    requires_calibration = False
    required_packages = ["nncf"]

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def check_if_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: Dict[str, Any],
        **kwargs,
    ):
        module, tensor_name = get_module_from_name(model, param_name)
        return module.__class__.__name__.startswith("NNCF") and param_name.endswith(".weight")

    def check_quantized_param(self, *args, **kwargs) -> bool:
        """
        needed for transformers compatibilty, returns self.check_if_quantized_param
        """
        return self.check_if_quantized_param(*args, **kwargs)

    def create_quantized_param(
        self,
        model,
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: Dict[str, Any],
        unexpected_keys: List[str],
        **kwargs,
    ):
        # load the model params to target_device first
        layer, tensor_name = get_module_from_name(model, param_name)
        layer._parameters[tensor_name] = torch.nn.Parameter(param_value).to(device=target_device)

        split_param_name = param_name.split(".")
        if param_name not in self.modules_to_not_convert and not any(param in split_param_name for param in self.modules_to_not_convert):
            from nncf.torch.quantization.quantize_functions import get_scale_zp_from_input_low_input_high
            from nncf.torch.quantization.weights_compression import WeightsDecompressor
            from nncf.torch.layers import NNCFEmbedding

            if not isinstance(layer, torch.nn.Embedding) and not isinstance(layer, NNCFEmbedding):
                target_dim = layer.target_weight_dim_for_compression
                stat_dim = (target_dim + 1) % 2
                input_low = torch.min(layer.weight, dim=stat_dim).values.detach()
                input_high = torch.max(layer.weight, dim=stat_dim).values.detach()
                scale, zero_point = get_scale_zp_from_input_low_input_high(0, 255, input_low, input_high)

                scale = scale.unsqueeze(stat_dim)
                zero_point = zero_point.unsqueeze(stat_dim)
                layer.register_pre_forward_operation(WeightsDecompressor(zero_point, scale))

                compressed_weight = layer.weight.data / scale + zero_point
                compressed_weight = torch.clamp(torch.round(compressed_weight), 0, 255)

                layer.weight.requires_grad = False
                layer.weight.data = compressed_weight.type(dtype=torch.uint8)

    def adjust_max_memory(self, max_memory: Dict[str, Union[int, str]]) -> Dict[str, Union[int, str]]:
        max_memory = {key: val * 0.80 for key, val in max_memory.items()}
        return max_memory

    def adjust_target_dtype(self, target_dtype: "torch.dtype") -> "torch.dtype":
        return torch.uint8

    def _process_model_before_weight_loading(
        self,
        model,
        device_map,
        keep_in_fp32_modules: List[str] = [],
        **kwargs,
    ):
        from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules

        self.modules_to_not_convert = self.quantization_config.modules_to_not_convert

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)
        model.config.quantization_config = self.quantization_config

        if model.__class__.__name__ in {"T5EncoderModel", "UMT5EncoderModel"}:
            for i in range(len(model.encoder.block)):
                model.encoder.block[i].layer[1].DenseReluDense = NNCF_T5DenseGatedActDense(
                    model.encoder.block[i].layer[1].DenseReluDense,
                    dtype=torch.float32 if devices.dtype != torch.bfloat16 else torch.bfloat16
                )

        with init_empty_weights():
            model, _ = replace_modules_by_nncf_modules(model)

    def _process_model_after_weight_loading(self, model, **kwargs):
        from modules.model_quant import nncf_send_to_device
        nncf_send_to_device(model, devices.device)
        return model

    def update_tp_plan(self, config):
        """
        needed for transformers compatibilty, no-op function
        """
        return config

    def update_unexpected_keys(self, model, unexpected_keys: List[str], prefix: str) -> List[str]:
        """
        needed for transformers compatibilty, no-op function
        """
        return unexpected_keys

    def update_missing_keys_after_loading(self, model, missing_keys: List[str], prefix: str) -> List[str]:
        """
        needed for transformers compatibilty, no-op function
        """
        return missing_keys

    def update_expected_keys(self, model, expected_keys: List[str], loaded_keys: List[str]) -> List[str]:
        """
        needed for transformers compatibilty, no-op function
        """
        return expected_keys

    @property
    def is_trainable(self):
        return False

    @property
    def is_serializable(self):
        return False


@dataclass
class NNCFConfig(QuantizationConfigMixin):
    """
    This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `nncf`.

    Args:
        weights_dtype (`str`, *optional*, defaults to `"int8"`):
            The target dtype for the weights after quantization. Supported values are ("int8")
       modules_to_not_convert (`list`, *optional*, default to `None`):
            The list of modules to not quantize, useful for quantizing models that explicitly require to have some
            modules left in their original precision (e.g. Whisper encoder, Llava encoder, Mixtral gate layers).
    """

    def __init__(
        self,
        weights_dtype: str = "int8",
        modules_to_not_convert: Optional[List[str]] = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.NNCF
        self.weights_dtype = weights_dtype
        self.modules_to_not_convert = modules_to_not_convert

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8", "uint8"]
        if self.weights_dtype not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights_dtype}")

