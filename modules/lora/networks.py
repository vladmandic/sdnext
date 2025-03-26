from typing import Union, List
from contextlib import nullcontext
import os
import re
import time
import concurrent
import torch
import diffusers.models.lora
import rich.progress as rp

from modules.lora import lora_timers, network, lora_convert, network_overrides
from modules.lora import network_lora, network_hada, network_ia3, network_oft, network_lokr, network_full, network_norm, network_glora
from modules.lora.extra_networks_lora import ExtraNetworkLora
from modules import shared, devices, sd_models, sd_models_compile, errors, files_cache, model_quant


debug = os.environ.get('SD_LORA_DEBUG', None) is not None
extra_network_lora = ExtraNetworkLora()
available_networks = {}
available_network_aliases = {}
loaded_networks: List[network.Network] = []
previously_loaded_networks: List[network.Network] = []
applied_layers: list[str] = []
bnb = None
lora_cache = {}
diffuser_loaded = []
diffuser_scales = []
available_network_hash_lookup = {}
forbidden_network_aliases = {}
re_network_name = re.compile(r"(.*)\s*\([0-9a-fA-F]+\)")
timer = lora_timers.Timer()
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

# section: load networks from disk

def load_diffusers(name, network_on_disk, lora_scale=shared.opts.extra_networks_default_multiplier) -> Union[network.Network, None]:
    t0 = time.time()
    name = name.replace(".", "_")
    shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" detected={network_on_disk.sd_version} method=diffusers scale={lora_scale} fuse={shared.opts.lora_fuse_diffusers}')
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
    timer.activate += time.time() - t0
    return net


def load_safetensors(name, network_on_disk) -> Union[network.Network, None]:
    if not shared.sd_loaded:
        return None

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
    if shared.sd_model_type == 'sd3':  # if kohya flux lora, convert state_dict
        try:
            sd = lora_convert._convert_kohya_sd3_lora_to_diffusers(sd) or sd  # pylint: disable=protected-access
        except ValueError:  # EAFP for diffusers PEFT keys
            pass
    lora_convert.assign_network_names_to_compvis_modules(shared.sd_model)
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
            continue
        if len(parts) > 5: # messy handler for diffusers peft lora
            key_network_without_network_parts = '_'.join(parts[:-2])
            if not key_network_without_network_parts.startswith('lora_'):
                key_network_without_network_parts = 'lora_' + key_network_without_network_parts
            network_part = '.'.join(parts[-2:]).replace('lora_A', 'lora_down').replace('lora_B', 'lora_up')
        else:
            key_network_without_network_parts, network_part = key_network.split(".", 1)
        key, sd_module = convert(key_network_without_network_parts)
        if sd_module is None:
            keys_failed_to_match[key_network] = key
            continue
        if key not in matched_networks:
            matched_networks[key] = network.NetworkWeights(network_key=key_network, sd_key=key, w={}, sd_module=sd_module)
        matched_networks[key].w[network_part] = weight
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
        shared.log.debug(f'Network load: type=LoRA name="{name}" type={set(network_types)} keys={len(matched_networks)} direct={shared.opts.lora_fuse_diffusers}')
    if len(matched_networks) == 0:
        return None
    lora_cache[name] = net
    net.bundle_embeddings = bundle_embeddings
    return net


def maybe_recompile_model(names, te_multipliers):
    recompile_model = False
    skip_lora_load = False
    if shared.compiled_model_state is not None and shared.compiled_model_state.is_compiled:
        if len(names) == len(shared.compiled_model_state.lora_model):
            for i, name in enumerate(names):
                if shared.compiled_model_state.lora_model[
                    i] != f"{name}:{te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier}":
                    recompile_model = True
                    shared.compiled_model_state.lora_model = []
                    break
            if not recompile_model:
                skip_lora_load = True
                if len(loaded_networks) > 0 and debug:
                    shared.log.debug('Model Compile: Skipping LoRa loading')
                return recompile_model, skip_lora_load
        else:
            recompile_model = True
            shared.compiled_model_state.lora_model = []
    if recompile_model:
        backup_cuda_compile = shared.opts.cuda_compile
        backup_scheduler = getattr(shared.sd_model, "scheduler", None)
        sd_models.unload_model_weights(op='model')
        shared.opts.cuda_compile = []
        sd_models.reload_model_weights(op='model')
        shared.opts.cuda_compile = backup_cuda_compile
        if backup_scheduler is not None:
            shared.sd_model.scheduler = backup_scheduler
    return recompile_model, skip_lora_load


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

    candidates = sorted(files_cache.list_files(shared.cmd_opts.lora_dir, ext_filter=[".pt", ".ckpt", ".safetensors"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
        for fn in candidates:
            executor.submit(add_network, fn)
    t1 = time.time()
    timer.list = t1 - t0
    shared.log.info(f'Available LoRAs: path="{shared.cmd_opts.lora_dir}" items={len(available_networks)} folders={len(forbidden_network_aliases)} time={t1 - t0:.2f}')


def network_download(name):
    from huggingface_hub import hf_hub_download
    if os.path.exists(name):
        return network.NetworkOnDisk(name, name)
    parts = name.split('/')
    if len(parts) >= 5 and parts[1] == 'huggingface.co':
        repo_id = f'{parts[2]}/{parts[3]}'
        filename = '/'.join(parts[4:])
        fn = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=shared.opts.hfcache_dir)
        return network.NetworkOnDisk(name, fn)
    return None


def network_load(names, te_multipliers=None, unet_multipliers=None, dyn_dims=None):
    networks_on_disk: list[network.NetworkOnDisk] = [available_network_aliases.get(name, None) for name in names]
    if any(x is None for x in networks_on_disk):
        list_available_networks()
        networks_on_disk: list[network.NetworkOnDisk] = [available_network_aliases.get(name, None) for name in names]
    for i in range(len(names)):
        if names[i].startswith('/'):
            networks_on_disk[i] = network_download(names[i])
    failed_to_load_networks = []
    recompile_model, skip_lora_load = maybe_recompile_model(names, te_multipliers)

    loaded_networks.clear()
    diffuser_loaded.clear()
    diffuser_scales.clear()
    t0 = time.time()

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = None
        if network_on_disk is not None:
            shorthash = getattr(network_on_disk, 'shorthash', '').lower()
            if debug:
                shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" hash="{shorthash}"')
            try:
                if recompile_model:
                    shared.compiled_model_state.lora_model.append(f"{name}:{te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier}")
                if shared.opts.lora_force_diffusers or network_overrides.check_override(shorthash): # OpenVINO only works with Diffusers LoRa loading
                    net = load_diffusers(name, network_on_disk, lora_scale=te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier)
                else:
                    net = load_safetensors(name, network_on_disk)
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
        if hasattr(shared.sd_model, 'embedding_db'):
            shared.sd_model.embedding_db.load_diffusers_embedding(None, net.bundle_embeddings)
        net.te_multiplier = te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier
        net.unet_multiplier = unet_multipliers[i] if unet_multipliers else shared.opts.extra_networks_default_multiplier
        net.dyn_dim = dyn_dims[i] if dyn_dims else shared.opts.extra_networks_default_multiplier
        loaded_networks.append(net)

    while len(lora_cache) > shared.opts.lora_in_memory_limit:
        name = next(iter(lora_cache))
        lora_cache.pop(name, None)

    if not skip_lora_load and len(diffuser_loaded) > 0:
        shared.log.debug(f'Network load: type=LoRA loaded={diffuser_loaded} available={shared.sd_model.get_list_adapters()} active={shared.sd_model.get_active_adapters()} scales={diffuser_scales}')
        try:
            t0 = time.time()
            shared.sd_model.set_adapters(adapter_names=diffuser_loaded, adapter_weights=diffuser_scales)
            if shared.opts.lora_fuse_diffusers and not network_overrides.check_fuse():
                shared.sd_model.fuse_lora(adapter_names=diffuser_loaded, lora_scale=1.0, fuse_unet=True, fuse_text_encoder=True) # fuse uses fixed scale since later apply does the scaling
                shared.sd_model.unload_lora_weights()
            timer.activate += time.time() - t0
        except Exception as e:
            shared.log.error(f'Network load: type=LoRA {e}')
            if debug:
                errors.display(e, 'LoRA')

    if len(loaded_networks) > 0 and debug:
        shared.log.debug(f'Network load: type=LoRA loaded={[n.name for n in loaded_networks]} cache={list(lora_cache)}')

    if recompile_model:
        shared.log.info("Network load: type=LoRA recompiling model")
        backup_lora_model = shared.compiled_model_state.lora_model
        if 'Model' in shared.opts.cuda_compile:
            shared.sd_model = sd_models_compile.compile_diffusers(shared.sd_model)
        shared.compiled_model_state.lora_model = backup_lora_model

    if len(loaded_networks) > 0:
        devices.torch_gc()

    timer.load = time.time() - t0


# section: process loaded networks

def network_backup_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv], network_layer_name: str, wanted_names: tuple):
    global bnb # pylint: disable=W0603
    backup_size = 0
    if len(loaded_networks) > 0 and network_layer_name is not None and any([net.modules.get(network_layer_name, None) for net in loaded_networks]): # noqa: C419 # pylint: disable=R1729
        t0 = time.time()

        weights_backup = getattr(self, "network_weights_backup", None)
        bias_backup = getattr(self, "network_bias_backup", None)
        if weights_backup is not None or bias_backup is not None:
            if (shared.opts.lora_fuse_diffusers and not isinstance(weights_backup, bool)) or (not shared.opts.lora_fuse_diffusers and isinstance(weights_backup, bool)): # invalidate so we can change direct/backup on-the-fly
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
                    with devices.inference_context():
                        if shared.opts.lora_fuse_diffusers:
                            self.network_weights_backup = True
                        else:
                            self.network_weights_backup = bnb.functional.dequantize_4bit(weight, quant_state=weight.quant_state, quant_type=weight.quant_type, blocksize=weight.blocksize,)
                        self.quant_state = weight.quant_state
                        self.quant_type = weight.quant_type
                        self.blocksize = weight.blocksize
                else:
                    if shared.opts.lora_fuse_diffusers:
                        self.network_weights_backup = True
                    else:
                        weights_backup = weight.clone()
                        self.network_weights_backup = weights_backup.to(devices.cpu)
            else:
                if shared.opts.lora_fuse_diffusers:
                    self.network_weights_backup = True
                else:
                    self.network_weights_backup = weight.clone().to(devices.cpu)

        if bias_backup is None:
            if getattr(self, 'bias', None) is not None:
                if shared.opts.lora_fuse_diffusers:
                    self.network_bias_backup = True
                else:
                    bias_backup = self.bias.clone()
                    bias_backup = bias_backup.to(devices.cpu)

        if getattr(self, 'network_weights_backup', None) is not None:
            backup_size += self.network_weights_backup.numel() * self.network_weights_backup.element_size() if isinstance(self.network_weights_backup, torch.Tensor) else 0
        if getattr(self, 'network_bias_backup', None) is not None:
            backup_size += self.network_bias_backup.numel() * self.network_bias_backup.element_size() if isinstance(self.network_bias_backup, torch.Tensor) else 0
        timer.backup += time.time() - t0
    return backup_size


def network_calc_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv], network_layer_name: str, use_previous: bool = False):
    if shared.opts.diffusers_offload_mode == "none":
        try:
            self.to(devices.device)
        except Exception:
            pass
    batch_updown = None
    batch_ex_bias = None
    loaded = loaded_networks if not use_previous else previously_loaded_networks
    for net in loaded:
        module = net.modules.get(network_layer_name, None)
        if module is None:
            continue
        try:
            t0 = time.time()
            try:
                weight = self.weight.to(devices.device)
            except Exception:
                weight = self.weight

            updown, ex_bias = module.calc_updown(weight)
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
            timer.calc += time.time() - t0

            if shared.opts.diffusers_offload_mode == "sequential":
                t0 = time.time()
                if batch_updown is not None:
                    batch_updown = batch_updown.to(devices.cpu)
                if batch_ex_bias is not None:
                    batch_ex_bias = batch_ex_bias.to(devices.cpu)
                t1 = time.time()
                timer.move += t1 - t0
        except RuntimeError as e:
            extra_network_lora.errors[net.name] = extra_network_lora.errors.get(net.name, 0) + 1
            if debug:
                module_name = net.modules.get(network_layer_name, None)
                shared.log.error(f'LoRA apply weight name="{net.name}" module="{module_name}" layer="{network_layer_name}" {e}')
                errors.display(e, 'LoRA')
                raise RuntimeError('LoRA apply weight') from e
        continue
    return batch_updown, batch_ex_bias


def network_add_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv], model_weights: Union[None, torch.Tensor] = None, lora_weights: torch.Tensor = None, deactivate: bool = False):
    if lora_weights is None:
        return None
    if deactivate:
        lora_weights *= -1
    if model_weights is None: # weights are used if provided-from-backup else use self.weight
        model_weights = self.weight
    # TODO lora: add other quantization types
    weight = None
    if self.__class__.__name__ == 'Linear4bit' and bnb is not None:
        try:
            dequant_weight = bnb.functional.dequantize_4bit(model_weights.to(devices.device), quant_state=self.quant_state, quant_type=self.quant_type, blocksize=self.blocksize)
            new_weight = dequant_weight.to(devices.device) + lora_weights.to(devices.device)
            weight = bnb.nn.Params4bit(new_weight, quant_state=self.quant_state, quant_type=self.quant_type, blocksize=self.blocksize)
        except Exception as e:
            shared.log.error(f'Network load: type=LoRA quant=bnb cls={self.__class__.__name__} type={self.quant_type} blocksize={self.blocksize} state={vars(self.quant_state)} weight={self.weight} bias={lora_weights} {e}')
    else:
        try:
            new_weight = model_weights.to(devices.device) + lora_weights.to(devices.device)
        except Exception:
            new_weight = model_weights + lora_weights # try without device cast
        weight = torch.nn.Parameter(new_weight, requires_grad=False)
    try:
        # weight = weight.to(device=devices.device) # required since quantization happens only during .to call, not during params creation
        pass
    except Exception:
        pass # may fail if weights is meta tensor
    return weight


def network_apply_direct(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv], updown: torch.Tensor, ex_bias: torch.Tensor, deactivate: bool = False):
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
            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))  # pylint: disable=not-callable
        if updown is not None:
            weight = network_add_weights(self, lora_weights=updown, deactivate=deactivate)
            if weight is not None:
                self.weight = weight

    if bias_backup:
        if ex_bias is not None:
            bias = network_add_weights(self, lora_weights=ex_bias, deactivate=deactivate)
            if bias is not None:
                self.bias = bias

    if hasattr(self, "qweight") and hasattr(self, "freeze"):
        self.freeze()

    timer.apply += time.time() - t0


def network_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm, torch.nn.LayerNorm, diffusers.models.lora.LoRACompatibleLinear, diffusers.models.lora.LoRACompatibleConv], updown: torch.Tensor, ex_bias: torch.Tensor, orig_device: torch.device, deactivate: bool = False):
    weights_backup = getattr(self, "network_weights_backup", None)
    bias_backup = getattr(self, "network_bias_backup", None)
    if weights_backup is None and bias_backup is None:
        return
    t0 = time.time()

    if weights_backup is not None:
        self.weight = None
        if updown is not None and len(weights_backup.shape) == 4 and weights_backup.shape[1] == 9: # inpainting model. zero pad updown to make channel[1]  4 to 9
            updown = torch.nn.functional.pad(updown, (0, 0, 0, 0, 0, 5))  # pylint: disable=not-callable
        if updown is not None:
            weight = network_add_weights(self, model_weights=weights_backup, lora_weights=updown, deactivate=deactivate)
            if weight is not None:
                self.weight = weight
        else:
            self.weight = torch.nn.Parameter(weights_backup.to(device=orig_device), requires_grad=False)

    if bias_backup is not None:
        self.bias = None
        if ex_bias is not None:
            bias = network_add_weights(self, model_weights=weights_backup, lora_weights=ex_bias, deactivate=deactivate)
            if bias:
                self.weight = bias
        else:
            self.bias = torch.nn.Parameter(bias_backup.to(device=orig_device), requires_grad=False)

    if hasattr(self, "qweight") and hasattr(self, "freeze"):
        self.freeze()

    timer.apply += time.time() - t0


def network_deactivate(include=[], exclude=[]):
    if not shared.opts.lora_fuse_diffusers or shared.opts.lora_force_diffusers:
        return
    t0 = time.time()
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatiblility
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.disable_offload(sd_model)
        sd_models.move_model(sd_model, device=devices.cpu)
    modules = {}

    components = include if len(include) > 0 else ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'unet', 'transformer']
    components = [x for x in components if x not in exclude]
    active_components = []
    for name in components:
        component = getattr(sd_model, name, None)
        if component is not None and hasattr(component, 'named_modules'):
            modules[name] = list(component.named_modules())
            active_components.append(name)
    total = sum(len(x) for x in modules.values())
    if len(previously_loaded_networks) > 0 and debug:
        pbar = rp.Progress(rp.TextColumn('[cyan]Network: type=LoRA action=deactivate'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
        task = pbar.add_task(description='', total=total)
    else:
        task = None
        pbar = nullcontext()
    with devices.inference_context(), pbar:
        applied_layers.clear()
        for component in modules.keys():
            orig_device = getattr(sd_model, component, None).device
            for _, module in modules[component]:
                network_layer_name = getattr(module, 'network_layer_name', None)
                if shared.state.interrupted or network_layer_name is None:
                    if task is not None:
                        pbar.update(task, advance=1)
                    continue
                batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name, use_previous=True)
                if shared.opts.lora_fuse_diffusers:
                    network_apply_direct(module, batch_updown, batch_ex_bias, deactivate=True)
                else:
                    network_apply_weights(module, batch_updown, batch_ex_bias, orig_device, deactivate=True)
                if batch_updown is not None or batch_ex_bias is not None:
                    applied_layers.append(network_layer_name)
                del batch_updown, batch_ex_bias
                module.network_current_names = ()
                if task is not None:
                    pbar.update(task, advance=1, description=f'networks={len(previously_loaded_networks)} modules={active_components} layers={total} unapply={len(applied_layers)}')

    timer.deactivate = time.time() - t0
    if debug and len(previously_loaded_networks) > 0:
        shared.log.debug(f'Network deactivate: type=LoRA networks={[n.name for n in previously_loaded_networks]} modules={active_components} layers={total} apply={len(applied_layers)} fuse={shared.opts.lora_fuse_diffusers} time={timer.summary}')
    modules.clear()
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.set_diffuser_offload(sd_model, op="model")


def network_activate(include=[], exclude=[]):
    t0 = time.time()
    sd_model = getattr(shared.sd_model, "pipe", shared.sd_model)  # wrapped model compatiblility
    if shared.opts.diffusers_offload_mode == "sequential":
        sd_models.disable_offload(sd_model)
        sd_models.move_model(sd_model, device=devices.cpu)
    modules = {}
    components = include if len(include) > 0 else ['text_encoder', 'text_encoder_2', 'text_encoder_3', 'unet', 'transformer']
    components = [x for x in components if x not in exclude]
    active_components = []
    for name in components:
        component = getattr(sd_model, name, None)
        if component is not None and hasattr(component, 'named_modules'):
            active_components.append(name)
            modules[name] = list(component.named_modules())
    total = sum(len(x) for x in modules.values())
    if len(loaded_networks) > 0:
        pbar = rp.Progress(rp.TextColumn('[cyan]Network: type=LoRA action=activate'), rp.BarColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
        task = pbar.add_task(description='' , total=total)
    else:
        task = None
        pbar = nullcontext()
    applied_weight = 0
    applied_bias = 0
    with devices.inference_context(), pbar:
        wanted_names = tuple((x.name, x.te_multiplier, x.unet_multiplier, x.dyn_dim) for x in loaded_networks) if len(loaded_networks) > 0 else ()
        applied_layers.clear()
        backup_size = 0
        for component in modules.keys():
            orig_device = getattr(sd_model, component, None).device
            for _, module in modules[component]:
                network_layer_name = getattr(module, 'network_layer_name', None)
                current_names = getattr(module, "network_current_names", ())
                if getattr(module, 'weight', None) is None or shared.state.interrupted or network_layer_name is None or current_names == wanted_names:
                    if task is not None:
                        pbar.update(task, advance=1)
                    continue
                backup_size += network_backup_weights(module, network_layer_name, wanted_names)
                batch_updown, batch_ex_bias = network_calc_weights(module, network_layer_name)
                if shared.opts.lora_fuse_diffusers:
                    network_apply_direct(module, batch_updown, batch_ex_bias)
                else:
                    network_apply_weights(module, batch_updown, batch_ex_bias, orig_device)
                if batch_updown is not None or batch_ex_bias is not None:
                    applied_layers.append(network_layer_name)
                    if batch_updown is not None:
                        applied_weight += 1
                    if batch_ex_bias is not None:
                        applied_bias += 1
                del batch_updown, batch_ex_bias
                module.network_current_names = wanted_names
                if task is not None:
                    pbar.update(task, advance=1, description=f'networks={len(loaded_networks)} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={backup_size}')

        if task is not None and len(applied_layers) == 0:
            pbar.remove_task(task) # hide progress bar for no action
    timer.activate += time.time() - t0
    if debug and len(loaded_networks) > 0:
        shared.log.debug(f'Network load: type=LoRA networks={[n.name for n in loaded_networks]} modules={active_components} layers={total} weights={applied_weight} bias={applied_bias} backup={backup_size} fuse={shared.opts.lora_fuse_diffusers} time={timer.summary}')
    modules.clear()
    if len(loaded_networks) > 0 and (applied_weight > 0 or applied_bias > 0):
        if shared.opts.diffusers_offload_mode == "sequential":
            sd_models.set_diffuser_offload(sd_model, op="model")
