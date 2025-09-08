"""
Copied from: https://github.com/huggingface/diffusers/issues/9165
"""

import os
import torch
import torch.nn as nn
from transformers.quantizers.quantizers_utils import get_module_from_name
from huggingface_hub import hf_hub_download
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers
import safetensors.torch
from modules import shared, devices, model_quant


debug = os.environ.get('SD_LOAD_DEBUG', None) is not None


def _replace_with_bnb_linear(
    model,
    method="nf4",
    has_been_replaced=False,
):
    """
    Private method that wraps the recursion for module replacement.
    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    bnb = model_quant.load_bnb('Load model: type=FLUX')
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            with init_empty_weights():
                in_features = module.in_features
                out_features = module.out_features

                if method == "llm_int8":
                    model._modules[name] = bnb.nn.Linear8bitLt( # pylint: disable=protected-access
                        in_features,
                        out_features,
                        module.bias is not None,
                        has_fp16_weights=False,
                        threshold=6.0,
                    )
                    has_been_replaced = True
                else:
                    model._modules[name] = bnb.nn.Linear4bit( # pylint: disable=protected-access
                        in_features,
                        out_features,
                        module.bias is not None,
                        compute_dtype=devices.dtype,
                        compress_statistics=False,
                        quant_type="nf4",
                    )
                    has_been_replaced = True
                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module) # pylint: disable=protected-access
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False) # pylint: disable=protected-access

        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
    return model, has_been_replaced


def check_quantized_param(
    model,
    param_name: str,
) -> bool:
    bnb = model_quant.load_bnb('Load model: type=FLUX')
    module, tensor_name = get_module_from_name(model, param_name)
    if isinstance(module._parameters.get(tensor_name, None), bnb.nn.Params4bit): # pylint: disable=protected-access
        # Add here check for loaded components' dtypes once serialization is implemented
        return True
    elif isinstance(module, bnb.nn.Linear4bit) and tensor_name == "bias":
        # bias could be loaded by regular set_module_tensor_to_device() from accelerate,
        # but it would wrongly use uninitialized weight there.
        return True
    else:
        return False


def create_quantized_param(
    model,
    param_value: "torch.Tensor",
    param_name: str,
    target_device: "torch.device",
    state_dict=None,
    unexpected_keys=None,
    pre_quantized=False
):
    bnb = model_quant.load_bnb('Load model: type=FLUX')
    module, tensor_name = get_module_from_name(model, param_name)

    if tensor_name not in module._parameters: # pylint: disable=protected-access
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")

    old_value = getattr(module, tensor_name)

    if tensor_name == "bias":
        if param_value is None:
            new_value = old_value.to(target_device)
        else:
            new_value = param_value.to(target_device)
        new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad)
        module._parameters[tensor_name] = new_value # pylint: disable=protected-access
        return

    if not isinstance(module._parameters[tensor_name], bnb.nn.Params4bit): # pylint: disable=protected-access
        raise ValueError("this function only loads `Linear4bit components`")
    if (
        old_value.device == torch.device("meta")
        and target_device not in ["meta", torch.device("meta")]
        and param_value is None
    ):
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {target_device}.")

    if pre_quantized:
        if (param_name + ".quant_state.bitsandbytes__fp4" not in state_dict) and (param_name + ".quant_state.bitsandbytes__nf4" not in state_dict):
            raise ValueError(f"Supplied state dict for {param_name} does not contain `bitsandbytes__*` and possibly other `quantized_stats` components.")
        quantized_stats = {}
        for k, v in state_dict.items():
            # `startswith` to counter for edge cases where `param_name`
            # substring can be present in multiple places in the `state_dict`
            if param_name + "." in k and k.startswith(param_name):
                quantized_stats[k] = v
                if unexpected_keys is not None and k in unexpected_keys:
                    unexpected_keys.remove(k)
        new_value = bnb.nn.Params4bit.from_prequantized(
            data=param_value,
            quantized_stats=quantized_stats,
            requires_grad=False,
            device=target_device,
        )
    else:
        new_value = param_value.to("cpu")
        kwargs = old_value.__dict__
        new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)
    module._parameters[tensor_name] = new_value # pylint: disable=protected-access


def load_flux_nf4(checkpoint_info, prequantized: bool = True):
    transformer = None
    text_encoder_2 = None
    if isinstance(checkpoint_info, str):
        repo_path = checkpoint_info
    else:
        repo_path = checkpoint_info.path
    if os.path.exists(repo_path) and os.path.isfile(repo_path):
        ckpt_path = repo_path
    elif os.path.exists(repo_path) and os.path.isdir(repo_path) and os.path.exists(os.path.join(repo_path, "diffusion_pytorch_model.safetensors")):
        ckpt_path = os.path.join(repo_path, "diffusion_pytorch_model.safetensors")
    else:
        ckpt_path = hf_hub_download(repo_path, filename="diffusion_pytorch_model.safetensors", cache_dir=shared.opts.diffusers_dir)
    original_state_dict = safetensors.torch.load_file(ckpt_path)

    if 'sayakpaul' in repo_path:
        converted_state_dict = original_state_dict # already converted
    else:
        try:
            converted_state_dict = convert_flux_transformer_checkpoint_to_diffusers(original_state_dict)
        except Exception as e:
            shared.log.error(f"Load model: type=FLUX Failed to convert UNET: {e}")
            if debug:
                from modules import errors
                errors.display(e, 'FLUX convert:')
            converted_state_dict = original_state_dict

    with init_empty_weights():
        from diffusers import FluxTransformer2DModel
        config = FluxTransformer2DModel.load_config(os.path.join('configs', 'flux'), subfolder="transformer")
        transformer = FluxTransformer2DModel.from_config(config).to(devices.dtype)
        expected_state_dict_keys = list(transformer.state_dict().keys())

    _replace_with_bnb_linear(transformer, "nf4")

    try:
        for param_name, param in converted_state_dict.items():
            if param_name not in expected_state_dict_keys:
                continue
            is_param_float8_e4m3fn = hasattr(torch, "float8_e4m3fn") and param.dtype == torch.float8_e4m3fn
            if torch.is_floating_point(param) and not is_param_float8_e4m3fn:
                param = param.to(devices.dtype)
            if not check_quantized_param(transformer, param_name):
                set_module_tensor_to_device(transformer, param_name, device=0, value=param)
            else:
                create_quantized_param(transformer, param, param_name, target_device=0, state_dict=original_state_dict, pre_quantized=prequantized)
    except Exception as e:
        transformer, text_encoder_2 = None, None
        shared.log.error(f"Load model: type=FLUX failed to load UNET: {e}")
        if debug:
            from modules import errors
            errors.display(e, 'FLUX:')

    del original_state_dict
    devices.torch_gc(force=True, reason='load')
    return transformer, text_encoder_2
