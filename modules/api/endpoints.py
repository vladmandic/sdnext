from typing import Optional
from fastapi.exceptions import HTTPException
from modules import shared
from modules.api import models, helpers



def get_samplers():
    from modules import sd_samplers_diffusers
    all_samplers = []
    for k, v in sd_samplers_diffusers.config.items():
        if k in ['All', 'Default', 'Res4Lyf']:
            continue
        all_samplers.append({
            'name': k,
            'options': v,
        })
    return all_samplers

def get_sampler():
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
    from modules.sd_vae import vae_dict
    return [{"model_name": x, "filename": vae_dict[x]} for x in vae_dict.keys()]

def get_upscalers():
    return [{"name": upscaler.name, "model_name": upscaler.scaler.model_name, "model_path": upscaler.data_path, "model_url": None, "scale": upscaler.scale} for upscaler in shared.sd_upscalers]

def get_sd_models():
    from modules import sd_checkpoint
    checkpoints = []
    for v in sd_checkpoint.checkpoints_list.values():
        model = models.ItemModel(title=v.title, model_name=v.name, filename=v.filename, type=v.type, hash=v.shorthash, sha256=v.sha256, config=None)
        checkpoints.append(model)
    return checkpoints

def get_controlnets(model_type: Optional[str] = None):
    from modules.control.units.controlnet import api_list_models
    return api_list_models(model_type)

def get_restorers():
    return [{"name":x.name(), "path": getattr(x, "cmd_dir", None)} for x in shared.face_restorers]

def get_detailers():
    shared.yolo.enumerate()
    return [{"name": k, "path": v} for k, v in shared.yolo.list.items()]

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
    """
    List available interrogation models.

    Returns model identifiers for use with POST /sdapi/v1/interrogate.

    **Model Types:**
    - OpenCLIP models: Format `architecture/pretrained_dataset` (e.g., `ViT-L-14/openai`)

    For anime-style tagging (WaifuDiffusion, DeepBooru), use `/sdapi/v1/tagger` instead.

    **Example Response:**
    ```json
    ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]
    ```
    """
    from modules.interrogate.openclip import refresh_clip_models
    return refresh_clip_models()

def get_schedulers():
    from modules.sd_samplers import list_samplers
    all_schedulers = list_samplers()
    for s in all_schedulers:
        shared.log.critical(s)
    return all_schedulers

def post_interrogate(req: models.ReqInterrogate):
    """
    Interrogate an image using OpenCLIP/BLIP.

    Analyze image using CLIP model via OpenCLIP to generate Stable Diffusion prompts.

    - Uses CLIP for image-text matching, BLIP for captioning
    - **Modes:**
      - `best`: Highest quality, combines multiple techniques
      - `fast`: Quick results with fewer iterations
      - `classic`: Traditional CLIP interrogator style
      - `caption`: BLIP caption only
      - `negative`: Generate negative prompt suggestions
    - Set `analyze=True` for detailed breakdown (medium, artist, movement, trending, flavor)

    For anime/illustration tagging, use `/sdapi/v1/tagger` with WaifuDiffusion or DeepBooru models.

    **Error Codes:**
    - 404: Image not provided or model not found
    """
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')
    from modules.interrogate.openclip import interrogate_image, analyze_image, refresh_clip_models
    if req.model not in refresh_clip_models():
        raise HTTPException(status_code=404, detail="Model not found")
    # Build clip overrides from request (only include non-None values)
    clip_overrides = {}
    if req.min_length is not None:
        clip_overrides['min_length'] = req.min_length
    if req.max_length is not None:
        clip_overrides['max_length'] = req.max_length
    if req.chunk_size is not None:
        clip_overrides['chunk_size'] = req.chunk_size
    if req.min_flavors is not None:
        clip_overrides['min_flavors'] = req.min_flavors
    if req.max_flavors is not None:
        clip_overrides['max_flavors'] = req.max_flavors
    if req.flavor_count is not None:
        clip_overrides['flavor_count'] = req.flavor_count
    if req.num_beams is not None:
        clip_overrides['num_beams'] = req.num_beams
    try:
        caption = interrogate_image(image, clip_model=req.clip_model, blip_model=req.blip_model, mode=req.mode, overrides=clip_overrides if clip_overrides else None)
    except Exception as e:
        caption = str(e)
    if not req.analyze:
        return models.ResInterrogate(caption=caption)
    analyze_results = analyze_image(image, clip_model=req.clip_model, blip_model=req.blip_model)
    # Extract top-ranked item from each Gradio update dict
    def get_top_item(result):
        if isinstance(result, dict) and 'value' in result:
            value = result['value']
            if isinstance(value, dict) and value:
                return next(iter(value.keys()))  # First key = top ranked
            if isinstance(value, str):
                return value
        return None
    medium = get_top_item(analyze_results[0])
    artist = get_top_item(analyze_results[1])
    movement = get_top_item(analyze_results[2])
    trending = get_top_item(analyze_results[3])
    flavor = get_top_item(analyze_results[4])
    return models.ResInterrogate(caption=caption, medium=medium, artist=artist, movement=movement, trending=trending, flavor=flavor)

def post_vqa(req: models.ReqVQA):
    """
    Caption an image using Vision-Language Models (VLM).

    Analyze image using vision language model for flexible image understanding.
    Supports 60+ models including Google Gemma, Alibaba Qwen, Microsoft Florence, Moondream.

    **Common Tasks:**

    1. **General Captioning**
       - `question="Short Caption"` or `"Normal Caption"` or `"Long Caption"`

    2. **Object Detection** (Florence-2):
       - `question="<OD>"` - Returns bounding boxes
       - `question="<DENSE_REGION_CAPTION>"` - Captions for regions

    3. **OCR/Text Recognition** (Florence-2):
       - `question="<OCR>"` - Extract text from image

    4. **Point/Detect** (Moondream):
       - `question="Point at the cat"` - Returns coordinates
       - `question="Detect all faces"` - Returns all instances

    **Annotated Images:**
    Set `include_annotated=True` to receive an annotated image with detection results.
    Returns Base64 PNG with bounding boxes and points drawn for:
    - Florence-2: Object detection, phrase grounding, region proposals
    - Moondream 2/3: Point detection, object detection, gaze detection

    **Model Selection:**
    - Small/Fast: Florence 2 Base, SmolVLM 0.5B, FastVLM 0.5B
    - Balanced: Qwen 2.5 VL 3B, Florence 2 Large, Gemma 3 4B
    - High Quality: Qwen 3 VL 8B, JoyCaption Beta, Moondream 3

    Use GET /sdapi/v1/vqa/models for complete model list.
    """
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')

    # Build generation kwargs from request parameters (None values are ignored)
    generation_kwargs = {}
    if req.max_tokens is not None:
        generation_kwargs['max_tokens'] = req.max_tokens
    if req.temperature is not None:
        generation_kwargs['temperature'] = req.temperature
    if req.top_k is not None:
        generation_kwargs['top_k'] = req.top_k
    if req.top_p is not None:
        generation_kwargs['top_p'] = req.top_p
    if req.num_beams is not None:
        generation_kwargs['num_beams'] = req.num_beams
    if req.do_sample is not None:
        generation_kwargs['do_sample'] = req.do_sample
    if req.keep_thinking is not None:
        generation_kwargs['keep_thinking'] = req.keep_thinking
    if req.keep_prefill is not None:
        generation_kwargs['keep_prefill'] = req.keep_prefill

    from modules.interrogate import vqa
    answer = vqa.interrogate(
        question=req.question,
        system_prompt=req.system,
        prompt=req.prompt or '',
        image=image,
        model_name=req.model,
        prefill=req.prefill,
        thinking_mode=req.thinking_mode,
        generation_kwargs=generation_kwargs if generation_kwargs else None
    )
    # Return annotated image if requested and available
    annotated_b64 = None
    if req.include_annotated:
        annotated_img = vqa.get_last_annotated_image()
        if annotated_img is not None:
            annotated_b64 = helpers.encode_pil_to_base64(annotated_img)
    return models.ResVQA(answer=answer, annotated_image=annotated_b64)

def get_vqa_models():
    """
    List available VLM models for captioning.

    Returns all Vision-Language Models available for POST /sdapi/v1/vqa.

    **Response includes:**
    - `name`: Display name
    - `repo`: HuggingFace repository ID
    - `prompts`: Available prompts/tasks
    - `capabilities`: Model features (caption, vqa, detection, ocr, thinking)
    """
    from modules.interrogate import vqa
    models_list = []
    for name, repo in vqa.vlm_models.items():
        prompts = vqa.get_prompts_for_model(name)
        capabilities = ["caption", "vqa"]
        # Detect additional capabilities based on model name
        name_lower = name.lower()
        if 'florence' in name_lower or 'promptgen' in name_lower:
            capabilities.extend(["detection", "ocr"])
        if 'moondream' in name_lower:
            capabilities.append("detection")
        if vqa.is_thinking_model(name):
            capabilities.append("thinking")
        models_list.append({
            "name": name,
            "repo": repo,
            "prompts": prompts,
            "capabilities": list(set(capabilities))
        })
    return models_list

def get_vqa_prompts(model: Optional[str] = None):
    """
    List available prompts/tasks for VLM models.

    **Query Parameters:**
    - `model` (optional): Filter prompts for a specific model

    **Prompt Categories:**
    - Common: Use Prompt, Short/Normal/Long Caption
    - Florence: Phrase Grounding, Object Detection, OCR, Dense Region Caption
    - Moondream: Point at..., Detect all..., Detect Gaze
    """
    from modules.interrogate import vqa
    if model:
        prompts = vqa.get_prompts_for_model(model)
        return {"available": prompts}
    return {
        "common": vqa.vlm_prompts_common,
        "florence": vqa.vlm_prompts_florence,
        "moondream": vqa.vlm_prompts_moondream,
        "moondream2_only": vqa.vlm_prompts_moondream2
    }

def get_tagger_models():
    """
    List available tagger models.

    Returns WaifuDiffusion and DeepBooru models for image tagging.

    **WaifuDiffusion Models:**
    - `wd-eva02-large-tagger-v3` (recommended)
    - `wd-vit-tagger-v3`, `wd-convnext-tagger-v3`, `wd-swinv2-tagger-v3`

    **DeepBooru:**
    - Legacy tagger for anime images
    """
    from modules.interrogate import waifudiffusion
    models_list = []
    # Add WaifuDiffusion models
    for name in waifudiffusion.get_models():
        models_list.append({"name": name, "type": "waifudiffusion"})
    # Add DeepBooru
    models_list.append({"name": "deepbooru", "type": "deepbooru"})
    return models_list

def post_tagger(req: models.ReqTagger):
    """
    Tag an image using WaifuDiffusion or DeepBooru.

    Generate anime/illustration tags for images.

    **WaifuDiffusion Models:**
    - `wd-eva02-large-tagger-v3` (recommended)
    - `wd-vit-tagger-v3`, `wd-convnext-tagger-v3`, `wd-swinv2-tagger-v3`

    **DeepBooru:**
    - Legacy tagger, use `deepbooru` or `deepdanbooru`

    **Thresholds:**
    - `threshold`: General tag confidence (default: 0.5)
    - `character_threshold`: Character identification (default: 0.85, WaifuDiffusion only)
    """
    if req.image is None or len(req.image) < 64:
        raise HTTPException(status_code=404, detail="Image not found")
    image = helpers.decode_base64_to_image(req.image)
    image = image.convert('RGB')
    from modules.interrogate import tagger
    # Determine if using DeepBooru
    is_deepbooru = req.model.lower() in ('deepbooru', 'deepdanbooru')
    # Store original settings and apply request settings
    original_opts = {
        'tagger_threshold': shared.opts.tagger_threshold,
        'tagger_max_tags': shared.opts.tagger_max_tags,
        'tagger_include_rating': shared.opts.tagger_include_rating,
        'tagger_sort_alpha': shared.opts.tagger_sort_alpha,
        'tagger_use_spaces': shared.opts.tagger_use_spaces,
        'tagger_escape_brackets': shared.opts.tagger_escape_brackets,
        'tagger_exclude_tags': shared.opts.tagger_exclude_tags,
        'tagger_show_scores': shared.opts.tagger_show_scores,
    }
    # WaifuDiffusion-specific settings (not applicable to DeepBooru)
    if not is_deepbooru:
        original_opts['waifudiffusion_character_threshold'] = shared.opts.waifudiffusion_character_threshold
        original_opts['waifudiffusion_model'] = shared.opts.waifudiffusion_model
    try:
        shared.opts.tagger_threshold = req.threshold
        shared.opts.tagger_max_tags = req.max_tags
        shared.opts.tagger_include_rating = req.include_rating
        shared.opts.tagger_sort_alpha = req.sort_alpha
        shared.opts.tagger_use_spaces = req.use_spaces
        shared.opts.tagger_escape_brackets = req.escape_brackets
        shared.opts.tagger_exclude_tags = req.exclude_tags
        shared.opts.tagger_show_scores = req.show_scores
        # WaifuDiffusion-specific settings (not applicable to DeepBooru)
        if not is_deepbooru:
            shared.opts.waifudiffusion_character_threshold = req.character_threshold
            shared.opts.waifudiffusion_model = req.model
        tags = tagger.tag(image, model_name='DeepBooru' if is_deepbooru else None)
        # Parse scores if requested - format is "(tag:0.95)" from WaifuDiffusion/DeepBooru
        scores = None
        if req.show_scores:
            scores = {}
            for item in tags.split(', '):
                item = item.strip()
                # Format: "(tag:0.95)" - tag and score wrapped in parentheses
                if item.startswith('(') and item.endswith(')') and ':' in item:
                    inner = item[1:-1]  # Remove outer parentheses
                    tag, score_str = inner.rsplit(':', 1)
                    try:
                        scores[tag.strip()] = float(score_str.strip())
                    except ValueError:
                        pass
                elif ':' in item:
                    # Fallback format: "tag:0.95" without parentheses
                    tag, score_str = item.rsplit(':', 1)
                    try:
                        scores[tag.strip()] = float(score_str.strip())
                    except ValueError:
                        pass
            if not scores:
                scores = None
        return models.ResTagger(tags=tags, scores=scores)
    finally:
        # Restore original settings
        for key, value in original_opts.items():
            setattr(shared.opts, key, value)

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
