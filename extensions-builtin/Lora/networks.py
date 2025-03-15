from typing import Union, List
import os
import re
import time
import concurrent
import lora_patches
import network
import network_lora
import network_hada
import network_ia3
import network_oft
import network_lokr
import network_full
import network_norm
import network_glora
import network_overrides
import lora_convert
import torch
import diffusers.models.lora
from modules import shared, devices, sd_models, sd_models_compile, errors, scripts, files_cache, model_quant


debug = os.environ.get('SD_LORA_DEBUG', None) is not None
originals: lora_patches.LoraPatches = None
extra_network_lora = None
available_networks = {}
available_network_aliases = {}
loaded_networks: List[network.Network] = []
timer = { 'load': 0, 'apply': 0, 'restore': 0, 'deactivate': 0 }
# networks_in_memory = {}
lora_cache = {}
diffuser_loaded = []
diffuser_scales = []
available_network_hash_lookup = {}
forbidden_network_aliases = {}
re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")
module_types = [
    network_lora.ModuleTypeLora(),
    network_hada.ModuleTypeHada(),
    network_ia3.ModuleTypeIa3(),
    network_oft.ModuleTypeOFT(),
    network_lokr.ModuleTypeLokr(),
    network_full.ModuleTypeFull(),
    network_norm.ModuleTypeNorm(),
    network_glora.ModuleTypeGLora(),
]
convert_diffusers_name_to_compvis = lora_convert.convert_diffusers_name_to_compvis # supermerger compatibility item


def assign_network_names_to_compvis_modules(sd_model):
    if sd_model is None:
        return
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatiblility
    network_layer_mapping = {}
    if shared.native:
        if hasattr(sd_model, 'text_encoder') and sd_model.text_encoder is not None:
            for name, module in sd_model.text_encoder.named_modules():
                prefix = "lora_te1_" if hasattr(sd_model, 'text_encoder_2') else "lora_te_"
                network_name = prefix + name.replace(".", "_")
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name
        if hasattr(sd_model, 'text_encoder_2'):
            for name, module in sd_model.text_encoder_2.named_modules():
                network_name = "lora_te2_" + name.replace(".", "_")
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name
        if hasattr(sd_model, 'unet'):
            for name, module in sd_model.unet.named_modules():
                network_name = "lora_unet_" + name.replace(".", "_")
                network_layer_mapping[network_name] = module
                module.network_layer_name = network_name
        if hasattr(sd_model, 'transformer'):
            for name, module in sd_model.transformer.named_modules():
                network_name = "lora_transformer_" + name.replace(".", "_")
                network_layer_mapping[network_name] = module
                if "norm" in network_name and "linear" not in network_name:
                    continue
                module.network_layer_name = network_name
    else:
        if not hasattr(sd_model, 'cond_stage_model'):
            sd_model.network_layer_mapping = {}
            return
        for name, module in sd_model.cond_stage_model.wrapped.named_modules():
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
        for name, module in sd_model.model.named_modules():
            network_name = name.replace(".", "_")
            network_layer_mapping[network_name] = module
            module.network_layer_name = network_name
    sd_model.network_layer_mapping = network_layer_mapping


def load_diffusers(name, network_on_disk, lora_scale=shared.opts.extra_networks_default_multiplier) -> network.Network:
    t0 = time.time()
    name = name.replace(".", "_")
    #cached = lora_cache.get(name, None)
    shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" detected={network_on_disk.sd_version} method=diffusers scale={lora_scale} fuse={shared.opts.lora_fuse_diffusers}')
    # if cached is not None:
    #    return cached
    if not shared.native:
        return None
    if not hasattr(shared.sd_model, 'load_lora_weights'):
        shared.log.error(f'Network load: type=LoRA class={shared.sd_model.__class__} does not implement load lora')
        return None
    try:
        shared.sd_model.load_lora_weights(network_on_disk.filename, adapter_name=name)
    except Exception as e:
        if 'already in use' in str(e):
            pass
        else:
            if 'The following keys have not been correctly renamed' in str(e):
                shared.log.error(f'Network load: type=LoRA name="{name}" diffusers unsupported format')
            else:
                shared.log.error(f'Network load: type=LoRA name="{name}" {e}')
            if debug:
                errors.display(e, "LoRA")
            return None
    if name not in diffuser_loaded:
        diffuser_loaded.append(name)
        diffuser_scales.append(lora_scale)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    # lora_cache[name] = net
    t1 = time.time()
    timer['load'] += t1 - t0
    return net


def load_network(name, network_on_disk) -> network.Network:
    if not shared.sd_loaded:
        return None
    t0 = time.time()
    cached = lora_cache.get(name, None)
    if debug:
        shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" type=lora {"cached" if cached else ""}')
    if cached is not None:
        return cached
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    sd = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if shared.sd_model_type == 'f1':  # if kohya flux lora, convert state_dict
        sd = lora_convert._convert_kohya_flux_lora_to_diffusers(sd) or sd  # pylint: disable=protected-access
    assign_network_names_to_compvis_modules(shared.sd_model)
    keys_failed_to_match = {}
    matched_networks = {}
    bundle_embeddings = {}
    convert = lora_convert.KeyConvert()
    for key_network, weight in sd.items():
        parts = key_network.split('.')
        if parts[0] == "bundle_emb":
            emb_name, vec_name = parts[1], key_network.split(".", 2)[-1]
            emb_dict = bundle_embeddings.get(emb_name, {})
            emb_dict[vec_name] = weight
            bundle_embeddings[emb_name] = emb_dict
        if len(parts) > 5: # messy handler for diffusers peft lora
            key_network_without_network_parts = '_'.join(parts[:-2])
            if not key_network_without_network_parts.startswith('lora_'):
                key_network_without_network_parts = 'lora_' + key_network_without_network_parts
            network_part = '.'.join(parts[-2:]).replace('lora_A', 'lora_down').replace('lora_B', 'lora_up')
        else:
            key_network_without_network_parts, network_part = key_network.split(".", 1)
        key, sd_module = convert(key_network_without_network_parts)  # Now returns lists
        if sd_module[0] is None:
            if "bundle_emb" not in key_network:
                keys_failed_to_match[key_network] = key
            continue
        for k, module in zip(key, sd_module):
            if k not in matched_networks:
                matched_networks[k] = network.NetworkWeights(network_key=key_network, sd_key=k, w={}, sd_module=module)
            matched_networks[k].w[network_part] = weight
    network_types = []
    for key, weights in matched_networks.items():
        net_module = None
        for nettype in module_types:
            net_module = nettype.create_module(net, weights)
            if net_module is not None:
                network_types.append(nettype.__class__.__name__)
                break
        if net_module is None:
            shared.log.error(f'LoRA unhandled: name={name} key={key} weights={weights.w.keys()}')
        else:
            net.modules[key] = net_module
    if len(keys_failed_to_match) > 0:
        shared.log.warning(f'Network load: type=LoRA name="{name}" type={set(network_types)} unmatched={len(keys_failed_to_match)} matched={len(matched_networks)}')
        if debug:
            shared.log.debug(f'Network load: type=LoRA name="{name}" unmatched={keys_failed_to_match}')
    else:
        shared.log.debug(f'Network load: type=LoRA name="{name}" type={set(network_types)} keys={len(matched_networks)}')
    if len(matched_networks) == 0:
        return None
    lora_cache[name] = net
    t1 = time.time()
    net.bundle_embeddings = bundle_embeddings
    timer['load'] += t1 - t0
    return net


def load_networks(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    networks_on_disk: list[network.NetworkOnDisk] = [available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk: list[network.NetworkOnDisk] = [available_network_aliases.get(name, None) for name in names]
    failed_to_load_networks = []
    recompile_model = False
    if shared.compiled_model_state is not None and shared.compiled_model_state.is_compiled:
        if len(names) == len(shared.compiled_model_state.lora_model):
            for i, name in enumerate(names):
                if shared.compiled_model_state.lora_model[i] != f"{name}:{te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier}":
                    recompile_model = True
                    shared.compiled_model_state.lora_model = []
                    break
            if not recompile_model:
                if len(loaded_networks) > 0 and debug:
                    shared.log.debug('Model Compile: Skipping LoRa loading')
                return
        else:
            recompile_model = True
            shared.compiled_model_state.lora_model = []
    if recompile_model:
        backup_cuda_compile = shared.opts.cuda_compile
        sd_models.unload_model_weights(op='model')
        shared.opts.cuda_compile = []
        sd_models.reload_model_weights(op='model')
        shared.opts.cuda_compile = backup_cuda_compile

    loaded_networks.clear()
    diffuser_loaded.clear()
    diffuser_scales.clear()

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = None
        if network_on_disk is not None:
            shorthash = getattr(network_on_disk, 'shorthash', '').lower()
            if debug:
                shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" hash="{shorthash}"')
            try:
                if recompile_model:
                    shared.compiled_model_state.lora_model.append(f"{name}:{te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier}")
                if shared.native and (shared.opts.lora_force_diffusers or network_overrides.check_override(shorthash)): # OpenVINO only works with Diffusers LoRa loading
                    net = load_diffusers(name, network_on_disk, lora_scale=te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier)
                else:
                    net = load_network(name, network_on_disk)
                if net is not None:
                    net.mentioned_name = name
                    network_on_disk.read_hash()
            except Exception as e:
                shared.log.error(f'Network load: type=LoRA file="{network_on_disk.filename}" {e}')
                if debug:
                    errors.display(e, 'LoRA')
                continue
        if net is None:
            failed_to_load_networks.append(name)
            shared.log.error(f'Network load: type=LoRA name="{name}" detected={network_on_disk.sd_version if network_on_disk is not None else None} failed')
            continue
        if shared.native:
            shared.sd_model.embedding_db.load_diffusers_embedding(None, net.bundle_embeddings)
        net.te_multiplier = te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else shared.opts.extra_networks_default_multiplier
        net.dyn_dim = dyn_dims[i] if dyn_dims else shared.opts.extra_networks_default_multiplier
        loaded_networks.append(net)

    while len(lora_cache) > shared.opts.lora_in_memory_limit:
        name = next(iter(lora_cache))
        lora_cache.pop(name, None)

    if len(diffuser_loaded) > 0:
        shared.log.debug(f'Network load: type=LoRA loaded={diffuser_loaded} available={shared.sd_model.get_list_adapters()} active={shared.sd_model.get_active_adapters()} scales={diffuser_scales}')
        try:
            shared.sd_model.set_adapters(adapter_names=diffuser_loaded, adapter_weights=diffuser_scales)
            if shared.opts.lora_fuse_diffusers:
                shared.sd_model.fuse_lora(adapter_names=diffuser_loaded, lora_scale=1.0, fuse_unet=True, fuse_text_encoder=True) # fuse uses fixed scale since later apply does the scaling
                shared.sd_model.unload_lora_weights()
        except Exception as e:
            shared.log.error(f'Network load: type=LoRA {e}')
            if debug:
                errors.display(e, 'LoRA')

    if len(loaded_networks) > 0 and debug:
        shared.log.debug(f'Network load: type=LoRA loaded={len(loaded_networks)} cache={list(lora_cache)}')

    devices.torch_gc()

    if recompile_model:
        shared.log.info("Network load: type=LoRA recompiling model")
        backup_lora_model = shared.compiled_model_state.lora_model
        if 'Model' in shared.opts.cuda_compile:
            shared.sd_model = sd_models_compile.compile_diffusers(shared.sd_model)

        shared.compiled_model_state.lora_model = backup_lora_model
    if shared.opts.diffusers_offload_mode == "balanced":
        sd_models.apply_balanced_offload(shared.sd_model)


def network_restore_weights_from_backup(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv]):
    t0 = time.time()
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)
    if weights_backup is None and bias_backup is None:
        t1 = time.time()
        timer['restore'] += t1 - t0
        return
    # if debug:
    #     shared.log.debug('LoRA restore weights')
    if weights_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.in_proj_weight.copy_(weights_backup[0])
            self.out_proj.weight.copy_(weights_backup[1])
        elif hasattr(self, "qweight") and hasattr(self, "freeze"):
            self.weight = torch.nn.Parameter(weights_backup.to(self.weight.device, copy=True))
            self.freeze()
        elif getattr(self, "quant_type", None) in ['nf4', 'fp4']:
            bnb = model_quant.load_bnb('Network load: type=LoRA', silent=True)
            if bnb is not None:
                device = self.weight.device
                self.weight = bnb.nn.Params4bit(weights_backup, quant_state=self.quant_state, quant_type=self.quant_type, blocksize=self.blocksize)
                self.weight.to(device)
            else:
                self.weight.copy_(weights_backup)
        else:
            self.weight.copy_(weights_backup)
    if bias_backup is not None:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias.copy_(bias_backup)
        else:
            self.bias.copy_(bias_backup)
    else:
        if isinstance(self, torch.nn.MultiheadAttention):
            self.out_proj.bias = None
        else:
            self.bias = None
    t1 = time.time()
    timer['restore'] += t1 - t0


def maybe_backup_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv], wanted_names, current_names): # pylint: disable=W0613
    weights_backup = getattr(self, "network_weights_backup", None)
    if weights_backup is None and wanted_names != (): # pylint: disable=C1803
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.clone().to(devices.cpu), self.out_proj.weight.clone().to(devices.cpu))
        elif getattr(self.weight, "quant_type", None) in ['nf4', 'fp4']:
            bnb = model_quant.load_bnb('Network load: type=LoRA', silent=True)
            if bnb is not None:
                with devices.inference_context():
                    weights_backup = bnb.functional.dequantize_4bit(self.weight, quant_state=self.weight.quant_state, quant_type=self.weight.quant_type, blocksize=self.weight.blocksize,).to(devices.cpu)
                    self.quant_state = self.weight.quant_state
                    self.quant_type = self.weight.quant_type
                    self.blocksize = self.weight.blocksize
            else:
                weights_backup = self.weight.clone().to(devices.cpu)
        else:
            weights_backup = self.weight.clone().to(devices.cpu)
        self.network_weights_backup = weights_backup
    bias_backup = getattr(self, "network_bias_backup", None)
    if bias_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention) and self.out_proj.bias is not None:
            bias_backup = self.out_proj.bias.clone().to(devices.cpu)
        elif getattr(self, 'bias', None) is not None:
            bias_backup = self.bias.clone().to(devices.cpu)
        else:
            bias_backup = None
        self.network_bias_backup = bias_backup


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, torch.nn.MultiheadAttention, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv]):
    """
    Applies the currently selected set of networks to the weights of torch layer self.
    If weights already have this particular set of networks applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to networks.
    """
    network_layer_name = getattr(self, 'network_layer_name', None)
    if network_layer_name is None:
        return
    t0 = time.time()
    current_names = getattr(self, "network_current_names", ())
    wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks)
    if any([net.modules.get(network_layer_name, None) for net in loaded_networks]): # noqa: C419 # pylint: disable=R1729
        maybe_backup_weights(self, wanted_names, current_names)
    if current_names != wanted_names:
        network_restore_weights_from_backup(self)
        for net in loaded_networks:
            # default workflow where module is known and has weights
            module = net.modules.get(network_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                try:
                    with devices.inference_context():
                        weight = self.weight # calculate quant weights once
                        updown, ex_bias = module.calc_updown(weight)
                        if len(weight.shape) == 4 and weight.shape[1] == 9:
                            # inpainting model. zero pad updown to make channel[1]  4 to 9
                            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5)) # pylint: disable=not-callable
                        if getattr(self.weight, "quant_type", None) in ['nf4', 'fp4']: # or self.weight.numel() != updown.numel():
                            bnb = model_quant.load_bnb('Network load: type=LoRA', silent=True)
                            if bnb is not None:
                                device = self.weight.device
                                weight = bnb.functional.dequantize_4bit(self.weight, quant_state=self.weight.quant_state, quant_type=self.weight.quant_type, blocksize=self.weight.blocksize)
                                self.weight = bnb.nn.Params4bit(weight + updown, quant_state=self.quant_state, quant_type=shared.opts.lora_quant.lower(), blocksize=self.blocksize)
                                self.weight.to(device)
                            else:
                                self.weight = torch.nn.Parameter(weight + updown)
                        else:
                            self.weight = torch.nn.Parameter(weight + updown)
                        if hasattr(self, "qweight") and hasattr(self, "freeze"):
                            self.freeze()
                        if ex_bias is not None and hasattr(self, 'bias'):
                            if self.bias is None:
                                self.bias = torch.nn.Parameter(ex_bias)
                            else:
                                self.bias += ex_bias
                except RuntimeError as e:
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                    if debug:
                        module_name = net.modules.get(network_layer_name, None)
                        shared.log.error(f'LoRA apply weight name="{net.name}" module="{module_name}" layer="{network_layer_name}" {e}')
                        errors.display(e, 'LoRA')
                        raise RuntimeError('LoRA apply weight') from e
                continue
            # alternative workflow looking at _*_proj layers
            module_q = net.modules.get(network_layer_name + "_q_proj", None)
            module_k = net.modules.get(network_layer_name + "_k_proj", None)
            module_v = net.modules.get(network_layer_name + "_v_proj", None)
            module_out = net.modules.get(network_layer_name + "_out_proj", None)
            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                try:
                    with devices.inference_context():
                        updown_q, _ = module_q.calc_updown(self.in_proj_weight)
                        updown_k, _ = module_k.calc_updown(self.in_proj_weight)
                        updown_v, _ = module_v.calc_updown(self.in_proj_weight)
                        updown_qkv = torch.vstack([updown_q, updown_k, updown_v])
                        updown_out, ex_bias = module_out.calc_updown(self.out_proj.weight)
                        self.in_proj_weight += updown_qkv
                        self.out_proj.weight += updown_out
                    if ex_bias is not None:
                        if self.out_proj.bias is None:
                            self.out_proj.bias = torch.nn.Parameter(ex_bias)
                        else:
                            self.out_proj.bias += ex_bias
                except RuntimeError as e:
                    if debug:
                        shared.log.debug(f'LoRA network="{net.name}" layer="{network_layer_name}" {e}')
                    extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
                continue
            if module is None:
                continue
            shared.log.warning(f'LoRA network="{net.name}" layer="{network_layer_name}" unsupported operation')
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
        self.network_current_names = wanted_names
    t1 = time.time()
    timer['apply'] += t1 - t0


def network_forward(module, input, original_forward): # pylint: disable=W0622
    """
    Old way of applying Lora by executing operations during layer's forward.
    Stacking many loras this way results in big performance degradation.
    """
    if len(loaded_networks) == 0:
        return original_forward(module, input)
    input = devices.cond_cast_unet(input)
    network_restore_weights_from_backup(module)
    network_reset_cached_weight(module)
    y = original_forward(module, input)
    network_layer_name = getattr(module, 'network_layer_name', None)
    for lora in loaded_networks:
        module = lora.modules.get(network_layer_name, None)
        if module is None:
            continue
        y = module.forward(input, y)
    return y


def network_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    self.network_current_names = ()
    self.network_weights_backup = None


def network_Linear_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return originals.Linear_forward(self, input)


def network_QLinear_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return torch.nn.functional.linear(input, self.qweight, bias=self.bias)


def network_Linear_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.Linear_load_state_dict(self, *args, **kwargs)


def network_Linear4bit_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return originals.Linear4bit_forward(self, input)
#
# def network_Linear4bit_load_state_dict(self, *args, **kwargs):
#     network_reset_cached_weight(self)
#     return originals.Linear4bit_load_state_dict(self, *args, **kwargs)

def network_Conv2d_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return originals.Conv2d_forward(self, input)


def network_QConv2d_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return self._conv_forward(input, self.qweight, self.bias) # pylint: disable=protected-access


def network_Conv2d_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.Conv2d_load_state_dict(self, *args, **kwargs)


def network_GroupNorm_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return originals.GroupNorm_forward(self, input)


def network_GroupNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.GroupNorm_load_state_dict(self, *args, **kwargs)


def network_LayerNorm_forward(self, input): # pylint: disable=W0622
    network_apply_weights(self)
    return originals.LayerNorm_forward(self, input)


def network_LayerNorm_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.LayerNorm_load_state_dict(self, *args, **kwargs)


def network_MultiheadAttention_forward(self, *args, **kwargs):
    network_apply_weights(self)
    return originals.MultiheadAttention_forward(self, *args, **kwargs)


def network_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    network_reset_cached_weight(self)
    return originals.MultiheadAttention_load_state_dict(self, *args, **kwargs)


def list_available_networks():
    t0 = time.time()
    available_networks.clear()
    available_network_aliases.clear()
    forbidden_network_aliases.clear()
    available_network_hash_lookup.clear()
    forbidden_network_aliases.update({"none": 1, "Addams": 1})
    if not os.path.exists(shared.cmd_opts.lora_dir):
        shared.log.warning(f'LoRA directory not found: path="{shared.cmd_opts.lora_dir}"')

    def add_network(filename):
        if not os.path.isfile(filename):
            return
        name = os.path.splitext(os.path.basename(filename))[0]
        name = name.replace('.', '_')
        try:
            entry = network.NetworkOnDisk(name, filename)
            available_networks[entry.name] = entry
            if entry.alias in available_network_aliases:
                forbidden_network_aliases[entry.alias.lower()] = 1
            if shared.opts.lora_preferred_name == 'filename':
                available_network_aliases[entry.name] = entry
            else:
                available_network_aliases[entry.alias] = entry
            if entry.shorthash:
                available_network_hash_lookup[entry.shorthash] = entry
        except OSError as e:  # should catch FileNotFoundError and PermissionError etc.
            shared.log.error(f'LoRA: filename="{filename}" {e}')

    candidates = list(files_cache.list_files(shared.cmd_opts.lora_dir, ext_filter=[".pt", ".ckpt", ".safetensors"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
        for fn in candidates:
            executor.submit(add_network, fn)
    t1 = time.time()
    shared.log.info(f'Available LoRAs: path="{shared.cmd_opts.lora_dir}" items={len(available_networks)} folders={len(forbidden_network_aliases)} time={t1 - t0:.2f}')


def infotext_pasted(infotext, params): # pylint: disable=W0613
    if "AddNet Module 1" in [x[1] for x in scripts.scripts_txt2img.infotext_fields]:
        return  # if the other extension is active, it will handle those fields, no need to do anything
    added = []
    for k in params:
        if not k.startswith("AddNet Model "):
            continue
        num = k[13:]
        if params.get("AddNet Module " + num) != "LoRA":
            continue
        name = params.get("AddNet Model " + num)
        if name is None:
            continue
        m = re_network_name.match(name)
        if m:
            name = m.group(1)
        multiplier = params.get("AddNet Weight A " + num, "1.0")
        added.append(f"<lora:{name}:{multiplier}>")
    if added:
        params["Prompt"] += "\n" + "".join(added)


list_available_networks()
