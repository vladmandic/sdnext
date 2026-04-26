from modules import shared
from modules.api import models, helpers


def _format_tags(raw_tags):
    if isinstance(raw_tags, dict):
        return '|'.join(raw_tags.keys()) if raw_tags else None
    if isinstance(raw_tags, str) and raw_tags:
        return raw_tags
    return None


def get_samplers():
    from modules import sd_samplers_diffusers
    all_samplers = []
    for k, v in sd_samplers_diffusers.config.items():
        if k in ['All', 'Default', 'Res4Lyf']:
            continue
        all_samplers.append({'name': k, 'options': v})
    return all_samplers

def get_sampler():
    """Return the active scheduler's class name and configuration for the currently loaded model."""
    if not shared.sd_loaded or shared.sd_model is None:
        return {}
    if hasattr(shared.sd_model, 'scheduler'):
        scheduler = shared.sd_model.scheduler
        config = {k: v for k, v in scheduler.config.items() if not k.startswith('_')}
        return {
            'name': scheduler.__class__.__name__,
            'options': config
        }
    return {}

def get_sd_vaes():
    """List available VAE models with their filenames."""
    from modules.sd_vae import vae_dict
    return [{"model_name": x, "filename": vae_dict[x]} for x in vae_dict.keys()]

def get_upscalers():
    """List available upscaler models with their names, paths, and scale factors."""
    return [{"name": upscaler.name, "model_name": upscaler.scaler.model_name, "model_path": upscaler.data_path, "model_url": None, "scale": upscaler.scale} for upscaler in shared.sd_upscalers]

def get_sd_models():
    """List all registered checkpoint models with title, filename, type, and hash."""
    from modules import sd_checkpoint
    checkpoints = []
    for v in sd_checkpoint.checkpoints_list.values():
        model = models.ItemModel(title=v.title, model_name=v.name, filename=v.filename, type=v.type, hash=v.shorthash, sha256=v.sha256, config=None)
        checkpoints.append(model)
    return checkpoints

def get_controlnets(model_type: str | None = None):
    """List available ControlNet models. Optionally filter by model type."""
    from modules.control.units.controlnet import api_list_models
    return api_list_models(model_type)

def get_detailers():
    """List available detailer (YOLO) models for face/object detection and inpainting."""
    shared.yolo.enumerate()
    return [{"name": k, "path": v} for k, v in shared.yolo.list.items()]

get_restorers = get_detailers  # legacy alias for /sdapi/v1/face-restorers

def get_ip_adapters():
    """
    List available IP-Adapter models.
    Returns adapter names that can be used for image-prompt conditioning during generation.
    """
    from modules import ipadapter
    return ipadapter.get_adapters()

def get_prompt_styles():
    """List all saved prompt styles with their prompt, negative prompt, and preview."""
    return [{ 'name': v.name, 'prompt': v.prompt, 'negative_prompt': v.negative_prompt, 'extra': v.extra, 'filename': v.filename, 'preview': v.preview} for v in shared.prompt_styles.styles.values()]

def get_unets():
    """List available UNet models with their names and filenames."""
    from modules.sd_unet import unet_dict
    return [{"name": k, "filename": v} for k, v in unet_dict.items()]

def get_embeddings():
    """List loaded and skipped textual-inversion embeddings for the current model."""
    db = getattr(shared.sd_model, 'embedding_db', None) if shared.sd_loaded else None
    if db is None:
        return models.ResEmbeddings(loaded=[], skipped=[])
    return models.ResEmbeddings(loaded=list(db.word_embeddings.keys()), skipped=list(db.skipped_embeddings.keys()))

def get_wildcards():
    """List wildcard basenames (relative path with `.txt` stripped) from the configured wildcards directory."""
    from modules import ui_extra_networks_wildcards
    return [{"name": n} for n in ui_extra_networks_wildcards.list_wildcard_names()]

def get_extra_networks(page: str | None = None, name: str | None = None, filename: str | None = None, title: str | None = None, fullname: str | None = None, hash: str | None = None): # pylint: disable=redefined-builtin
    """List extra networks (LoRA, checkpoints, embeddings, etc.) with optional filtering by page, name, filename, title, fullname, or hash."""
    res = []
    for pg in shared.extra_networks:
        if page is not None and pg.name != page.lower():
            continue
        for item in pg.items:
            if name is not None and item.get('name', '') != name:
                continue
            if title is not None and item.get('title', '') != title:
                continue
            if filename is not None and item.get('filename', '') != filename:
                continue
            if fullname is not None and item.get('fullname', '') != fullname:
                continue
            if hash is not None and (item.get('shorthash', None) or item.get('hash')) != hash:
                continue
            res.append({
                'name': item.get('name', ''),
                'type': pg.name,
                'title': item.get('title', None),
                'fullname': item.get('fullname', None),
                'filename': item.get('filename', None),
                'hash': item.get('shorthash', None) or item.get('hash'),
                "preview": item.get('preview', None),
                "version": item.get('version', None),
                "tags": _format_tags(item.get('tags', None)),
            })
    return res

def get_extra_network_detail(page: str, name: str):
    """
    Get detailed metadata for a single extra network item.

    Returns name, filename, hash, alias, file size, modification time, version, tags,
    description, and embedded info dict for the item matching ``page`` and ``name``.
    """
    from datetime import datetime
    for pg in shared.extra_networks:
        if pg.name.lower() != page.lower():
            continue
        for item in pg.items:
            if item.get('name', '').lower() != name.lower():
                continue
            mtime = item.get('mtime', None)
            if isinstance(mtime, datetime):
                mtime = mtime.isoformat()
            elif mtime is not None:
                mtime = str(mtime)
            return {
                'name': item.get('name', ''),
                'type': pg.name,
                'title': item.get('title', None),
                'filename': item.get('filename', None),
                'hash': item.get('shorthash', None) or item.get('hash'),
                'alias': item.get('alias', None),
                'size': item.get('size', None),
                'mtime': mtime,
                'version': item.get('version', None),
                'tags': _format_tags(item.get('tags', None)),
                'description': item.get('description', None),
                'info': item.get('info', None) if isinstance(item.get('info'), dict) else None,
            }
    return {}

def get_extra_network_details(page: str | None = None, name: str | None = None, filename: str | None = None, title: str | None = None, fullname: str | None = None, hash: str | None = None, offset: int = 0, limit: int = 50): # pylint: disable=redefined-builtin
    """Batch-fetch full detail for extra network items with optional filtering and pagination."""
    from datetime import datetime
    matched = []
    for pg in shared.extra_networks:
        if page is not None and pg.name != page.lower():
            continue
        for item in pg.items:
            if name is not None and item.get('name', '') != name:
                continue
            if title is not None and item.get('title', '') != title:
                continue
            if filename is not None and item.get('filename', '') != filename:
                continue
            if fullname is not None and item.get('fullname', '') != fullname:
                continue
            if hash is not None and (item.get('shorthash', None) or item.get('hash')) != hash:
                continue
            mtime = item.get('mtime', None)
            if isinstance(mtime, datetime):
                mtime = mtime.isoformat()
            elif mtime is not None:
                mtime = str(mtime)
            matched.append({
                'name': item.get('name', ''),
                'type': pg.name,
                'title': item.get('title', None),
                'fullname': item.get('fullname', None),
                'filename': item.get('filename', None),
                'hash': item.get('shorthash', None) or item.get('hash'),
                'preview': item.get('preview', None),
                'alias': item.get('alias', None),
                'size': item.get('size', None),
                'mtime': mtime,
                'version': item.get('version', None),
                'tags': _format_tags(item.get('tags', None)),
                'description': item.get('description', None),
                'info': item.get('info', None) if isinstance(item.get('info'), dict) else None,
            })
    total = len(matched)
    return {
        'items': matched[offset:offset + limit],
        'total': total,
        'offset': offset,
        'limit': limit,
    }

def get_schedulers():
    """List all available schedulers with their class names and options."""
    from modules.sd_samplers import list_samplers
    all_schedulers = list_samplers()
    return all_schedulers

def post_unload_checkpoint():
    """Unload the current model and refiner from memory to free VRAM."""
    from modules import sd_models
    sd_models.unload_model_weights(op='model')
    sd_models.unload_model_weights(op='refiner')
    return {}

def post_reload_checkpoint(force:bool=False):
    """Reload the selected checkpoint. Set ``force=True`` to unload first and do a clean reload."""
    from modules import sd_models
    if force:
        sd_models.unload_model_weights(op='model')
    sd_models.reload_model_weights()
    return {}

def post_lock_checkpoint(lock:bool=False):
    """Lock or unlock the current model to prevent automatic model swaps."""
    from modules import modeldata
    modeldata.model_data.locked = lock
    return {}

def post_refresh_unets():
    """Rescan UNet directories and update the available UNet list."""
    import modules.sd_unet
    return modules.sd_unet.refresh_unet_list()

def get_checkpoint():
    """Return information about the currently loaded checkpoint including type, class, title, and hash."""
    if not shared.sd_loaded or shared.sd_model is None:
        checkpoint = {
            'type': None,
            'class': None,
        }
    else:
        checkpoint = {
            'type': shared.sd_model_type,
            'class': shared.sd_model.__class__.__name__,
        }
        if hasattr(shared.sd_model, 'sd_model_checkpoint'):
            checkpoint['checkpoint'] = shared.sd_model.sd_model_checkpoint
        if hasattr(shared.sd_model, 'sd_checkpoint_info'):
            checkpoint['title'] = shared.sd_model.sd_checkpoint_info.title
            checkpoint['name'] = shared.sd_model.sd_checkpoint_info.name
            checkpoint['filename'] = shared.sd_model.sd_checkpoint_info.filename
            checkpoint['hash'] = shared.sd_model.sd_checkpoint_info.shorthash
    return checkpoint

def set_checkpoint(sd_model_checkpoint: str, dtype: str | None = None, force: bool = False):
    """Load a checkpoint by name. Optionally set dtype and force a clean reload."""
    from modules import sd_models, devices
    if force:
        sd_models.unload_model_weights(op='model')
    if dtype is not None:
        shared.opts.cuda_dtype = dtype
        devices.set_dtype()
    shared.opts.sd_model_checkpoint = sd_model_checkpoint
    model = sd_models.reload_model_weights()
    return { 'ok': model is not None }

def post_refresh_checkpoints():
    """Rescan checkpoint directories and update the available models list."""
    shared.refresh_checkpoints()
    return {}

def post_refresh_vae():
    """Rescan VAE directories and update the available VAE list."""
    shared.refresh_vaes()
    return {}

def get_modules():
    """Analyze the loaded model and return its sub-module breakdown with device, dtype, and parameter info."""
    from modules import modelstats
    model = modelstats.analyze()
    if model is None:
        return {}
    model_obj = {
        'model': model.name,
        'type': model.type,
        'class': model.cls,
        'size': model.size,
        'mtime': str(model.mtime),
        'modules': []
    }
    for m in model.modules:
        model_obj['modules'].append({
            'class': m.cls,
            'params': m.params,
            'modules': m.modules,
            'quant': m.quant,
            'device': str(m.device),
            'dtype': str(m.dtype)
        })
    return model_obj

def get_extensions_list():
    """List installed extensions with their remote URLs, branches, versions, and enabled status."""
    from modules import extensions
    extensions.list_extensions()
    ext_list = []
    for ext in extensions.extensions:
        ext: extensions.Extension
        ext.read_info()
        if ext.remote is not None:
            ext_list.append({
                "name": ext.name,
                "remote": ext.remote,
                "branch": ext.branch,
                "commit_hash":ext.commit_hash,
                "commit_date":ext.commit_date,
                "version":ext.version,
                "enabled":ext.enabled
            })
    return ext_list

def post_pnginfo(req: models.ReqImageInfo):
    """Extract generation parameters from a PNG image's metadata. Returns raw info string and parsed parameters dict."""
    from modules import images, script_callbacks, infotext
    if not req.image.strip():
        return models.ResImageInfo(info="")
    image = helpers.decode_base64_to_image(req.image.strip())
    if image is None:
        return models.ResImageInfo(info="")
    geninfo, items = images.read_info_from_image(image)
    if geninfo is None:
        geninfo = ""
    params = infotext.parse(geninfo)
    script_callbacks.infotext_pasted_callback(geninfo, params)
    return models.ResImageInfo(info=geninfo, items=items, parameters=params)

def get_control_models(unit_type: str = "controlnet"):
    """
    List available models for a control unit type.

    Returns model names for the specified ``unit_type``: ``controlnet`` (default),
    ``t2i`` / ``t2i adapter``, ``xs``, ``lite``, or ``reference``.
    """
    if unit_type == "controlnet":
        from modules.control.units.controlnet import api_list_models
        return api_list_models()
    if unit_type in ("t2i", "t2i adapter"):
        from modules.control.units.t2iadapter import list_models
        result = list_models()
        return list(result) if isinstance(result, dict) else result
    if unit_type == "xs":
        from modules.control.units.xs import list_models
        result = list_models()
        return list(result) if isinstance(result, dict) else result
    if unit_type == "lite":
        from modules.control.units.lite import list_models
        result = list_models()
        return list(result) if isinstance(result, dict) else result
    if unit_type == "reference":
        from modules.control.units.reference import list_models
        return list_models()
    return []

def get_control_modes():
    """
    List mode choices for control models that support modes.

    Returns a mapping of model name to available mode choices (e.g., canny, depth, pose).
    Only models with non-default mode options are included.
    """
    from modules.control.unit import Unit
    u = Unit.__new__(Unit)
    u.model_name = None
    u.choices = ['default']
    from modules.control.units import controlnet
    result = {}
    for models_dict in [controlnet.predefined_sd15, controlnet.predefined_sdxl, controlnet.predefined_f1, controlnet.predefined_sd3, getattr(controlnet, 'predefined_qwen', {})]:
        for name in models_dict:
            u.update_choices(name)
            if u.choices != ['default']:
                result[name] = list(u.choices)
    return result

def get_latent_history():
    """List available latent history entries by name."""
    return shared.history.list

def post_latent_history(req: models.ReqLatentHistory):
    """Select a latent history entry by name. Returns the index of the selected entry."""
    shared.history.index = shared.history.find(req.name)
    return shared.history.index
