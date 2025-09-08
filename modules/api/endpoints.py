from typing import Optional
from fastapi.exceptions import HTTPException
from modules import shared
from modules.api import models, helpers



def get_samplers():
    from modules import sd_samplers
    return [{"name": sampler[0], "aliases":sampler[2], "options":sampler[3]} for sampler in sd_samplers.all_samplers]

def get_sd_vaes():
    from modules.sd_vae import vae_dict
    return [{"model_name": x, "filename": vae_dict[x]} for x in vae_dict.keys()]

def get_upscalers():
    return [{"name": upscaler.name, "model_name": upscaler.scaler.model_name, "model_path": upscaler.data_path, "model_url": None, "scale": upscaler.scale} for upscaler in shared.sd_upscalers]

def get_sd_models():
    from modules import sd_checkpoint
    checkpoints = []
    for v in sd_checkpoint.checkpoints_list.values():
        checkpoints.append({"title": v.title, "model_name": v.name, "filename": v.filename, "type": v.type, "hash": v.shorthash, "sha256": v.sha256})
    return checkpoints

def get_controlnets(model_type: Optional[str] = None):
    from modules.control.units.controlnet import api_list_models
    return api_list_models(model_type)

def get_detailers():
    return [{"name":x.name(), "cmd_dir": getattr(x, "cmd_dir", None)} for x in shared.detailers]

def get_prompt_styles():
    return [{ 'name': v.name, 'prompt': v.prompt, 'negative_prompt': v.negative_prompt, 'extra': v.extra, 'filename': v.filename, 'preview': v.preview} for v in shared.prompt_styles.styles.values()]

def get_embeddings():
    db = getattr(shared.sd_model, 'embedding_db', None) if shared.sd_loaded else None
    if db is None:
        return models.ResEmbeddings(loaded=[], skipped=[])
    return models.ResEmbeddings(loaded=list(db.word_embeddings.keys()), skipped=list(db.skipped_embeddings.keys()))

def get_extra_networks(page: Optional[str] = None, name: Optional[str] = None, filename: Optional[str] = None, title: Optional[str] = None, fullname: Optional[str] = None, hash: Optional[str] = None): # pylint: disable=redefined-builtin
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
            })
    return res

def get_interrogate():
    from modules.interrogate.openclip import refresh_clip_models
    return ['deepdanbooru'] + refresh_clip_models()

def post_interrogate(req: models.ReqInterrogate):
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')
    if req.model == "deepdanbooru" or req.model == 'deepbooru':
        from modules.interrogate import deepbooru
        caption = deepbooru.model.tag(image)
        return models.ResInterrogate(caption=caption)
    else:
        from modules.interrogate.openclip import interrogate_image, analyze_image, refresh_clip_models
        if req.model not in refresh_clip_models():
            raise HTTPException(status_code=404, detail="Model not found")
        try:
            caption = interrogate_image(image, clip_model=req.clip_model, blip_model=req.blip_model, mode=req.mode)
        except Exception as e:
            caption = str(e)
        if not req.analyze:
            return models.ResInterrogate(caption=caption)
        else:
            medium, artist, movement, trending, flavor = analyze_image(image, clip_model=req.clip_model, blip_model=req.blip_model)
            return models.ResInterrogate(caption=caption, medium=medium, artist=artist, movement=movement, trending=trending, flavor=flavor)

def post_vqa(req: models.ReqVQA):
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')
    from modules.interrogate import vqa
    answer = vqa.interrogate(req.question, req.system, '', image, req.model)
    return models.ResVQA(answer=answer)

def post_unload_checkpoint():
    from modules import sd_models
    sd_models.unload_model_weights(op='model')
    sd_models.unload_model_weights(op='refiner')
    return {}

def post_reload_checkpoint(force:bool=False):
    from modules import sd_models
    if force:
        sd_models.unload_model_weights(op='model')
    sd_models.reload_model_weights()
    return {}

def post_lock_checkpoint(lock:bool=False):
    from modules import modeldata
    modeldata.model_data.locked = lock
    return {}

def get_checkpoint():
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

def set_checkpoint(sd_model_checkpoint: str, dtype:str=None, force:bool=False):
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
    shared.refresh_checkpoints()
    return {}

def post_refresh_vae():
    shared.refresh_vaes()
    return {}

def get_modules():
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

def get_latent_history():
    return shared.history.list

def post_latent_history(req: models.ReqLatentHistory):
    shared.history.index = shared.history.find(req.name)
    return shared.history.index
