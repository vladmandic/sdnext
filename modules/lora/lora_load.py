from typing import Union
import os
import time
import concurrent
from modules import shared, errors, devices, sd_models, sd_models_compile, files_cache
from modules.lora import network, lora_overrides, lora_convert
from modules.lora import lora_common as l


diffuser_loaded = []
diffuser_scales = []
lora_cache = {}
available_networks = {}
available_network_aliases = {}
forbidden_network_aliases = {}
available_network_hash_lookup = {}


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
            if l.debug:
                errors.display(e, "LoRA")
            return None
    if name not in diffuser_loaded:
        diffuser_loaded.append(name)
        diffuser_scales.append(lora_scale)
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    l.timer.activate += time.time() - t0
    return net


def load_safetensors(name, network_on_disk) -> Union[network.Network, None]:
    if not shared.sd_loaded:
        return None

    cached = lora_cache.get(name, None)
    if l.debug:
        shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" type=lora {"cached" if cached else ""}')
    if cached is not None:
        return cached
    net = network.Network(name, network_on_disk)
    net.mtime = os.path.getmtime(network_on_disk.filename)
    state_dict = sd_models.read_state_dict(network_on_disk.filename, what='network')
    if shared.sd_model_type == 'f1': # if kohya flux lora, convert state_dict
        state_dict = lora_convert._convert_kohya_flux_lora_to_diffusers(state_dict) or state_dict # pylint: disable=protected-access
    if shared.sd_model_type == 'sd3': # if kohya flux lora, convert state_dict
        try:
            state_dict = lora_convert._convert_kohya_sd3_lora_to_diffusers(state_dict) or state_dict # pylint: disable=protected-access
        except ValueError: # EAFP for diffusers PEFT keys
            pass
    lora_convert.assign_network_names_to_compvis_modules(shared.sd_model)
    keys_failed_to_match = {}
    matched_networks = {}
    bundle_embeddings = {}
    dtypes = []
    convert = lora_convert.KeyConvert()
    device = devices.device if shared.opts.lora_apply_gpu else devices.cpu
    for key_network, weight in state_dict.items():
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
        if weight.dtype not in dtypes:
            dtypes.append(weight.dtype)
    network_types = []
    state_dict = None
    del state_dict
    for key, weights in matched_networks.items():
        net_module = None
        for nettype in l.module_types:
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
        if l.debug:
            shared.log.debug(f'Network load: type=LoRA name="{name}" unmatched={keys_failed_to_match}')
    else:
        shared.log.debug(f'Network load: type=LoRA name="{name}" type={set(network_types)} keys={len(matched_networks)} device={device} dtypes={dtypes} direct={shared.opts.lora_fuse_diffusers}')
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
                if len(l.loaded_networks) > 0 and l.debug:
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
        except OSError as e: # should catch FileNotFoundError and PermissionError etc.
            shared.log.error(f'LoRA: filename="{filename}" {e}')

    candidates = sorted(files_cache.list_files(shared.cmd_opts.lora_dir, ext_filter=[".pt", ".ckpt", ".safetensors"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=shared.max_workers) as executor:
        for fn in candidates:
            executor.submit(add_network, fn)
    t1 = time.time()
    l.timer.list = t1 - t0
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

    l.loaded_networks.clear()
    diffuser_loaded.clear()
    diffuser_scales.clear()
    t0 = time.time()

    for i, (network_on_disk, name) in enumerate(zip(networks_on_disk, names)):
        net = None
        if network_on_disk is not None:
            shorthash = getattr(network_on_disk, 'shorthash', '').lower()
            if l.debug:
                shared.log.debug(f'Network load: type=LoRA name="{name}" file="{network_on_disk.filename}" hash="{shorthash}"')
            try:
                if recompile_model:
                    shared.compiled_model_state.lora_model.append(f"{name}:{te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier}")
                if shared.opts.lora_force_diffusers or lora_overrides.check_override(shorthash): # OpenVINO only works with Diffusers LoRa loading
                    net = load_diffusers(name, network_on_disk, lora_scale=te_multipliers[i] if te_multipliers else shared.opts.extra_networks_default_multiplier)
                else:
                    net = load_safetensors(name, network_on_disk)
                if net is not None:
                    net.mentioned_name = name
                    network_on_disk.read_hash()
            except Exception as e:
                shared.log.error(f'Network load: type=LoRA file="{network_on_disk.filename}" {e}')
                if l.debug:
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
        l.loaded_networks.append(net)

    while len(lora_cache) > shared.opts.lora_in_memory_limit:
        name = next(iter(lora_cache))
        lora_cache.pop(name, None)

    if not skip_lora_load and len(diffuser_loaded) > 0:
        shared.log.debug(f'Network load: type=LoRA loaded={diffuser_loaded} available={shared.sd_model.get_list_adapters()} active={shared.sd_model.get_active_adapters()} scales={diffuser_scales}')
        try:
            t0 = time.time()
            shared.sd_model.set_adapters(adapter_names=diffuser_loaded, adapter_weights=diffuser_scales)
            if shared.opts.lora_fuse_diffusers and not lora_overrides.check_fuse():
                shared.sd_model.fuse_lora(adapter_names=diffuser_loaded, lora_scale=1.0, fuse_unet=True, fuse_text_encoder=True) # diffusers with fuse uses fixed scale since later apply does the scaling
                shared.sd_model.unload_lora_weights()
            l.timer.activate += time.time() - t0
        except Exception as e:
            shared.log.error(f'Network load: type=LoRA {e}')
            if l.debug:
                errors.display(e, 'LoRA')

    if len(l.loaded_networks) > 0 and l.debug:
        shared.log.debug(f'Network load: type=LoRA loaded={[n.name for n in l.loaded_networks]} cache={list(lora_cache)}')

    if recompile_model:
        shared.log.info("Network load: type=LoRA recompiling model")
        backup_lora_model = shared.compiled_model_state.lora_model
        if 'Model' in shared.opts.cuda_compile:
            shared.sd_model = sd_models_compile.compile_diffusers(shared.sd_model)
        shared.compiled_model_state.lora_model = backup_lora_model

    l.timer.load = time.time() - t0
