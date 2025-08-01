from typing import Optional, Union
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import accelerate.utils.modeling
from modules import devices


tensor_to_timer = 0
orig_set_module = accelerate.utils.set_module_tensor_to_device
orig_torch_conv = torch.nn.modules.conv.Conv2d._conv_forward # pylint: disable=protected-access


# called for every item in state_dict by diffusers during model load
def hijack_set_module_tensor(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None, # pylint: disable=unused-argument
    fp16_statistics: Optional[torch.HalfTensor] = None, # pylint: disable=unused-argument
):
    global tensor_to_timer # pylint: disable=global-statement
    if device == 'cpu': # override to load directly to gpu
        device = devices.device
    t0 = time.time()
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            module = getattr(module, split)
        tensor_name = splits[-1]
    old_value = getattr(module, tensor_name)
    with devices.inference_context():
        # note: majority of time is spent on .to(old_value.dtype)
        if tensor_name in module._buffers: # pylint: disable=protected-access
            module._buffers[tensor_name] = value.to(device, old_value.dtype)  # pylint: disable=protected-access
        elif value is not None or not devices.same_device(device, module._parameters[tensor_name].device):  # pylint: disable=protected-access
            param_cls = type(module._parameters[tensor_name]) # pylint: disable=protected-access
            module._parameters[tensor_name] = param_cls(value, requires_grad=old_value.requires_grad).to(device, old_value.dtype) # pylint: disable=protected-access
    t1 = time.time()
    tensor_to_timer += (t1 - t0)


def hijack_set_module_tensor_simple(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None, # pylint: disable=unused-argument
    fp16_statistics: Optional[torch.HalfTensor] = None, # pylint: disable=unused-argument
):
    global tensor_to_timer # pylint: disable=global-statement
    if device == 'cpu': # override to load directly to gpu
        device = devices.device
    t0 = time.time()
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            module = getattr(module, split)
        tensor_name = splits[-1]
    old_value = getattr(module, tensor_name)
    with devices.inference_context():
        if tensor_name in module._buffers: # pylint: disable=protected-access
            module._buffers[tensor_name] = value.to(device, non_blocking=shared.opts.diffusers_offload_nonblocking)  # pylint: disable=protected-access
        elif value is not None or not devices.same_device(device, module._parameters[tensor_name].device):  # pylint: disable=protected-access
            param_cls = type(module._parameters[tensor_name]) # pylint: disable=protected-access
            module._parameters[tensor_name] = param_cls(value, requires_grad=old_value.requires_grad).to(device, non_blocking=shared.opts.diffusers_offload_nonblocking) # pylint: disable=protected-access
    t1 = time.time()
    tensor_to_timer += (t1 - t0)


def hijack_accelerate():
    accelerate.utils.set_module_tensor_to_device = hijack_set_module_tensor
    global tensor_to_timer # pylint: disable=global-statement
    tensor_to_timer = 0


def restore_accelerate():
    accelerate.utils.set_module_tensor_to_device = orig_set_module


def hijack_hfhub():
    import contextlib
    import huggingface_hub.file_download
    huggingface_hub.file_download.FileLock = contextlib.nullcontext


def torch_conv_forward(self, input, weight, bias): # pylint: disable=redefined-builtin
    if self.padding_mode != 'zeros':
        return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _pair(0), self.dilation, self.groups) # pylint: disable=protected-access
    if weight.dtype != bias.dtype:
        bias.to(weight.dtype)
    return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

def hijack_torch_conv():
    torch.nn.modules.conv.Conv2d._conv_forward = torch_conv_forward # pylint: disable=protected-access

def restore_torch_conv():
    torch.nn.modules.conv.Conv2d._conv_forward = orig_torch_conv # pylint: disable=protected-access
