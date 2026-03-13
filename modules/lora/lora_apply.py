import re
import time
import torch
import diffusers.models.lora
from modules.errorlimiter import ErrorLimiter
from modules.lora import lora_common as l
from modules import shared, devices, errors, model_quant
from modules.logger import log


bnb = None
re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")


def network_backup_weights(self: torch.nn.Conv2d | torch.nn.Linear | torch.nn.GroupNorm | torch.nn.LayerNorm | diffusers.models.lora.LoRACompatibleLinear | diffusers.models.lora.LoRACompatibleConv, network_layer_name: str, wanted_names: tuple):
    global bnb # pylint: disable=W0603
    backup_size = 0
    if len(l.loaded_networks) > 0 and network_layer_name is not None and any([net.modules.get(network_layer_name, None) for net in l.loaded_networks]): # noqa: C419 # pylint: disable=R1729
        t0 = time.time()

        weights_backup = getattr(self, "network_weights_backup", None)
        bias_backup = getattr(self, "network_bias_backup", None)
        if weights_backup is not None or bias_backup is not None:
            if (shared.opts.lora_fuse_native and not isinstance(weights_backup, bool)) or (not shared.opts.lora_fuse_native and isinstance(weights_backup, bool)): # invalidate so we can change direct/backup on-the-fly
                weights_backup = None
                bias_backup = None
                self.network_weights_backup = weights_backup
                self.network_bias_backup = bias_backup

        if weights_backup is None and wanted_names != (): # pylint: disable=C1803
            weight = getattr(self, 'weight', None)
            self.network_weights_backup = None
            if getattr(weight, "quant_type", None) in ['nf4', 'fp4']:
                if bnb is None:
                    bnb = model_quant.load_bnb('Network load: type=LoRA', silent=True)
                if bnb is not None:
                    if shared.opts.lora_fuse_native:
                        self.network_weights_backup = True
                    else:
                        self.network_weights_backup = bnb.functional.dequantize_4bit(weight, quant_state=weight.quant_state, quant_type=weight.quant_type, blocksize=weight.blocksize,)
                    self.quant_state, self.quant_type, self.blocksize = weight.quant_state, weight.quant_type, weight.blocksize
                else:
                    self.network_weights_backup = weight.clone().to(devices.cpu) if not shared.opts.lora_fuse_native else True
            else:
                if shared.opts.lora_fuse_native:
                    self.network_weights_backup = True
                else:
                    self.network_weights_backup = weight.clone().to(devices.cpu)
                    if hasattr(self, "sdnq_dequantizer"):
                        self.sdnq_dequantizer_backup = self.sdnq_dequantizer
                        self.sdnq_scale_backup = self.scale.clone().to(devices.cpu)
                        if self.zero_point is not None:
                            self.sdnq_zero_point_backup = self.zero_point.clone().to(devices.cpu)
                        else:
                            self.sdnq_zero_point_backup = None
                        if self.svd_up is not None:
                            self.sdnq_svd_up_backup = self.svd_up.clone().to(devices.cpu)
                            self.sdnq_svd_down_backup = self.svd_down.clone().to(devices.cpu)
                        else:
                            self.sdnq_svd_up_backup = None
                            self.sdnq_svd_down_backup = None

        if bias_backup is None:
            if getattr(self, 'bias', None) is not None:
                if shared.opts.lora_fuse_native:
                    self.network_bias_backup = True
                else:
                    bias_backup = self.bias.clone()
                    bias_backup = bias_backup.to(devices.cpu)

        if getattr(self, 'network_weights_backup', None) is not None:
            backup_size += self.network_weights_backup.numel() * self.network_weights_backup.element_size() if isinstance(self.network_weights_backup, torch.Tensor) else 0
        if getattr(self, 'network_bias_backup', None) is not None:
            backup_size += self.network_bias_backup.numel() * self.network_bias_backup.element_size() if isinstance(self.network_bias_backup, torch.Tensor) else 0
        l.timer.backup += time.time() - t0
    return backup_size


def network_calc_weights(self: torch.nn.Conv2d | torch.nn.Linear | torch.nn.GroupNorm | torch.nn.LayerNorm | diffusers.models.lora.LoRACompatibleLinear | diffusers.models.lora.LoRACompatibleConv, network_layer_name: str, use_previous: bool = False):
    if shared.opts.diffusers_offload_mode == "none":
        try:
            self.to(devices.device)
        except Exception:
            pass
    batch_updown = None
    batch_ex_bias = None
    loaded = l.loaded_networks if not use_previous else l.previously_loaded_networks
    for net in loaded:
        module = net.modules.get(network_layer_name, None)
        if module is None:
            continue
        try:
            t0 = time.time()
            if hasattr(self, "sdnq_dequantizer_backup"):
                weight = self.sdnq_dequantizer_backup(
                    self.weight.to(devices.device),
                    self.sdnq_scale_backup.to(devices.device),
                    self.sdnq_zero_point_backup.to(devices.device) if self.sdnq_zero_point_backup is not None else None,
                    self.sdnq_svd_up_backup.to(devices.device) if self.sdnq_svd_up_backup is not None else None,
                    self.sdnq_svd_down_backup.to(devices.device) if self.sdnq_svd_down_backup is not None else None,
                    skip_quantized_matmul=self.sdnq_dequantizer_backup.use_quantized_matmul
                )
            elif hasattr(self, "sdnq_dequantizer"):
                weight = self.sdnq_dequantizer(
                    self.weight.to(devices.device),
                    self.scale.to(devices.device),
                    self.zero_point.to(devices.device) if self.zero_point is not None else None,
                    self.svd_up.to(devices.device) if self.svd_up is not None else None,
                    self.svd_down.to(devices.device) if self.svd_down is not None else None,
                    skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul
                )
            else:
                weight = self.weight.to(devices.device) # must perform calc on gpu due to performance
            updown, ex_bias = module.calc_updown(weight)
            weight = None
            del weight

            if updown is not None:
                if batch_updown is not None:
                    batch_updown += updown.to(batch_updown.device)
                else:
                    batch_updown = updown.to(devices.device)
            if ex_bias is not None:
                if batch_ex_bias:
                    batch_ex_bias += ex_bias.to(batch_ex_bias.device)
                else:
                    batch_ex_bias = ex_bias.to(devices.device)
            l.timer.calc += time.time() - t0

            if shared.opts.diffusers_offload_mode == "sequential":
                t0 = time.time()
                if batch_updown is not None:
                    batch_updown = batch_updown.to(devices.cpu)
                if batch_ex_bias is not None:
                    batch_ex_bias = batch_ex_bias.to(devices.cpu)
                t1 = time.time()
                l.timer.move += t1 - t0
        except RuntimeError as e:
            l.extra_network_lora.errors[net.name] = l.extra_network_lora.errors.get(net.name, 0) + 1
            module_name = net.modules.get(network_layer_name, None)
            log.error(f'Network: type=LoRA name="{net.name}" module="{module_name}" layer="{network_layer_name}" apply weight: {e}')
            if l.debug:
                errors.display(e, 'LoRA')
                raise RuntimeError('LoRA apply weight') from e
            ErrorLimiter.notify(("network_activate", "network_deactivate"))
        continue
    return batch_updown, batch_ex_bias


def network_add_weights(self: torch.nn.Conv2d | torch.nn.Linear | torch.nn.GroupNorm | torch.nn.LayerNorm | diffusers.models.lora.LoRACompatibleLinear | diffusers.models.lora.LoRACompatibleConv, model_weights: None | torch.Tensor = None, lora_weights: torch.Tensor = None, deactivate: bool = False, device: torch.device = None, bias: bool = False):
    if lora_weights is None:
        return
    if deactivate:
        lora_weights *= -1
    if model_weights is None: # weights are used if provided-from-backup else use self.weight
        model_weights = self.weight
    weight, new_weight = None, None
    # TODO lora: add other quantization types
    if self.__class__.__name__ == 'Linear4bit' and bnb is not None:
        try:
            dequant_weight = bnb.functional.dequantize_4bit(model_weights.to(devices.device), quant_state=self.quant_state, quant_type=self.quant_type, blocksize=self.blocksize)
            new_weight = dequant_weight.to(devices.device) + lora_weights.to(devices.device)
            weight = bnb.nn.Params4bit(new_weight.to(device), quant_state=self.quant_state, quant_type=self.quant_type, blocksize=self.blocksize, requires_grad=False)
            # TODO lora: maybe force imediate quantization
            # weight._quantize(devices.device) / weight.to(device=device)
        except Exception as e:
            log.error(f'Network load: type=LoRA quant=bnb cls={self.__class__.__name__} type={self.quant_type} blocksize={self.blocksize} state={vars(self.quant_state)} weight={self.weight} bias={lora_weights} {e}')
    elif not bias and hasattr(self, "sdnq_dequantizer"):
        try:
            from modules.sdnq import sdnq_quantize_layer
            if hasattr(self, "sdnq_dequantizer_backup"):
                use_svd = bool(self.sdnq_svd_up_backup is not None)
                dequantize_fp32 = bool(self.sdnq_scale_backup.dtype == torch.float32)
                sdnq_dequantizer = self.sdnq_dequantizer_backup
                dequant_weight = self.sdnq_dequantizer_backup(
                    model_weights.to(devices.device),
                    self.sdnq_scale_backup.to(devices.device),
                    self.sdnq_zero_point_backup.to(devices.device) if self.sdnq_zero_point_backup is not None else None,
                    self.sdnq_svd_up_backup.to(devices.device) if use_svd else None,
                    self.sdnq_svd_down_backup.to(devices.device) if use_svd else None,
                    skip_quantized_matmul=self.sdnq_dequantizer_backup.use_quantized_matmul,
                    dtype=torch.float32,
                )
            else:
                use_svd = bool(self.svd_up is not None)
                dequantize_fp32 = bool(self.scale.dtype == torch.float32)
                sdnq_dequantizer = self.sdnq_dequantizer
                dequant_weight = self.sdnq_dequantizer(
                    model_weights.to(devices.device),
                    self.scale.to(devices.device),
                    self.zero_point.to(devices.device) if self.zero_point is not None else None,
                    self.svd_up.to(devices.device) if use_svd else None,
                    self.svd_down.to(devices.device) if use_svd else None,
                    skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul,
                    dtype=torch.float32,
                )

            new_weight = dequant_weight.to(devices.device, dtype=torch.float32) + lora_weights.to(devices.device, dtype=torch.float32)
            self.weight = torch.nn.Parameter(new_weight, requires_grad=False)
            del self.sdnq_dequantizer, self.scale, self.zero_point, self.svd_up, self.svd_down
            self = sdnq_quantize_layer(
                self,
                weights_dtype=sdnq_dequantizer.weights_dtype,
                quantized_matmul_dtype=sdnq_dequantizer.quantized_matmul_dtype,
                torch_dtype=sdnq_dequantizer.result_dtype,
                group_size=sdnq_dequantizer.group_size,
                svd_rank=sdnq_dequantizer.svd_rank,
                use_quantized_matmul=sdnq_dequantizer.use_quantized_matmul,
                use_quantized_matmul_conv=sdnq_dequantizer.use_quantized_matmul,
                use_svd=use_svd,
                dequantize_fp32=dequantize_fp32,
                svd_steps=shared.opts.sdnq_svd_steps,
                quant_conv=True, # quant_conv is True if conv layers ends up here
                non_blocking=False,
                quantization_device=devices.device,
                return_device=device,
                param_name=getattr(self, 'network_layer_name', None),
            )[0].to(device)
            weight = None
            del dequant_weight
        except Exception as e:
            log.error(f'Network load: type=LoRA quant=sdnq cls={self.__class__.__name__} weight={self.weight} lora_weights={lora_weights} {e}')
    else:
        try:
            new_weight = model_weights.to(devices.device) + lora_weights.to(devices.device)
        except Exception as e:
            log.warning(f'Network load: {e}')
            if 'The size of tensor' in str(e):
                log.error(f'Network load: type=LoRA model={shared.sd_model.__class__.__name__} incompatible lora shape')
                new_weight = model_weights
            else:
                new_weight = model_weights + lora_weights # try without device cast
        weight = torch.nn.Parameter(new_weight.to(device), requires_grad=False)
    if weight is not None:
        if not bias:
            self.weight = weight
        else:
            self.bias = weight
    del model_weights, lora_weights, new_weight, weight # required to avoid memory leak


def network_apply_direct(self: torch.nn.Conv2d | torch.nn.Linear | torch.nn.GroupNorm | torch.nn.LayerNorm | diffusers.models.lora.LoRACompatibleLinear | diffusers.models.lora.LoRACompatibleConv, updown: torch.Tensor, ex_bias: torch.Tensor, deactivate: bool = False, device: torch.device = devices.device):
    weights_backup = getattr(self, "network_weights_backup", False)
    bias_backup = getattr(self, "network_bias_backup", False)
    if not isinstance(weights_backup, bool): # remove previous backup if we switched settings
        weights_backup = True
    if not isinstance(bias_backup, bool):
        bias_backup = True
    if not weights_backup and not bias_backup:
        return
    t0 = time.time()

    if weights_backup:
        if updown is not None and len(self.weight.shape) == 4 and self.weight.shape[1] == 9: # inpainting model so zero pad updown to make channel 4 to 9
            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5)) # pylint: disable=not-callable
        if updown is not None:
            network_add_weights(self, lora_weights=updown, deactivate=deactivate, device=device, bias=False)

    if bias_backup:
        if ex_bias is not None:
            network_add_weights(self, lora_weights=ex_bias, deactivate=deactivate, device=device, bias=True)

    if hasattr(self, "qweight") and hasattr(self, "freeze"):
        self.freeze()

    l.timer.apply += time.time() - t0


def network_apply_weights(self: torch.nn.Conv2d | torch.nn.Linear | torch.nn.GroupNorm | torch.nn.LayerNorm | diffusers.models.lora.LoRACompatibleLinear | diffusers.models.lora.LoRACompatibleConv, updown: torch.Tensor, ex_bias: torch.Tensor, device: torch.device, deactivate: bool = False):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)
    if weights_backup is None and bias_backup is None:
        return
    t0 = time.time()

    if weights_backup is not None:
        self.weight = None
        if updown is not None and len(weights_backup.shape) == 4 and weights_backup.shape[1] == 9: # inpainting model. zero pad updown to make channel[1]  4 to 9
            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5)) # pylint: disable=not-callable
        if updown is not None:
            network_add_weights(self, model_weights=weights_backup, lora_weights=updown, deactivate=deactivate, device=device, bias=False)
        else:
            self.weight = torch.nn.Parameter(weights_backup.to(device), requires_grad=False)
            if hasattr(self, "sdnq_dequantizer_backup"):
                self.sdnq_dequantizer = self.sdnq_dequantizer_backup
                self.scale = torch.nn.Parameter(self.sdnq_scale_backup.to(device), requires_grad=False)
                if self.sdnq_zero_point_backup is not None:
                    self.zero_point = torch.nn.Parameter(self.sdnq_zero_point_backup.to(device), requires_grad=False)
                else:
                    self.zero_point = None
                if self.sdnq_svd_up_backup is not None:
                    self.svd_up = torch.nn.Parameter(self.sdnq_svd_up_backup.to(device), requires_grad=False)
                    self.svd_down = torch.nn.Parameter(self.sdnq_svd_down_backup.to(device), requires_grad=False)
                else:
                    self.svd_up, self.svd_down = None, None
                del self.sdnq_dequantizer_backup, self.sdnq_scale_backup, self.sdnq_zero_point_backup, self.sdnq_svd_up_backup, self.sdnq_svd_down_backup

    if bias_backup is not None:
        self.bias = None
        if ex_bias is not None:
            network_add_weights(self, model_weights=bias_backup, lora_weights=ex_bias, deactivate=deactivate, device=device, bias=True)
        else:
            self.bias = torch.nn.Parameter(bias_backup.to(device), requires_grad=False)

    if hasattr(self, "qweight") and hasattr(self, "freeze"):
        self.freeze()

    l.timer.apply += time.time() - t0
