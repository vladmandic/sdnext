import os
import inspect
from modules.logger import log


def execute_generate(params: dict, job_id: str) -> dict:
    from modules import shared, processing_helpers
    from modules.api import helpers
    from modules.control import run as control_run_module
    from modules.control.unit import Unit

    # Decode base64 images
    inputs = [helpers.decode_base64_to_image(x) for x in params.get('inputs', [])] if params.get('inputs') else None
    inits = [helpers.decode_base64_to_image(x) for x in params.get('inits', [])] if params.get('inits') else None
    mask = helpers.decode_base64_to_image(params['mask']) if params.get('mask') else None

    # Merge asset unit images (init_control) into inits
    init_control = params.get('init_control')
    if init_control:
        extra_inits = [helpers.decode_base64_to_image(x) for x in init_control]
        inits = (inits or []) + extra_inits

    # Build units from control dicts
    units = []
    control_dicts = params.get('control') or []
    for u in control_dicts:
        if not isinstance(u, dict):
            continue
        unit = Unit(
            enabled=True,
            unit_type=u.get('unit_type', 'controlnet'),
            model_id=u.get('model', ''),
            process_id=u.get('process', ''),
            strength=u.get('strength', 1.0),
            start=u.get('start', 0.0),
            end=u.get('end', 1.0),
        )
        unit.guess = u.get('guess', False)
        unit.factor = u.get('factor', 1.0)
        unit.attention = u.get('attention', 'Attention')
        unit.fidelity = u.get('fidelity', 0.5)
        unit.query_weight = u.get('query_weight', 1.0)
        unit.adain_weight = u.get('adain_weight', 1.0)
        unit.process_params = u.get('process_params') or {}
        unit.update_choices(u.get('model', ''))
        mode = u.get('mode', 'default')
        if mode != 'default' and unit.choices and mode in unit.choices:
            unit.mode = mode
        elif unit.choices:
            unit.mode = unit.choices[0]
        if unit.process is not None:
            unit.process.override = None
        override_b64 = u.get('override') or u.get('image')
        if override_b64:
            unit.override = helpers.decode_base64_to_image(override_b64)
        units.append(unit)

    # Build IP adapter args
    ip_adapter_args = {}
    ip_adapter_list = params.get('ip_adapter') or []
    if ip_adapter_list:
        ip_adapter_args = {'ip_adapter_names': [], 'ip_adapter_scales': [], 'ip_adapter_crops': [], 'ip_adapter_starts': [], 'ip_adapter_ends': [], 'ip_adapter_images': [], 'ip_adapter_masks': []}
        for ipa in ip_adapter_list:
            if not isinstance(ipa, dict) or not ipa.get('images'):
                continue
            ip_adapter_args['ip_adapter_names'].append(ipa.get('adapter', ''))
            ip_adapter_args['ip_adapter_scales'].append(ipa.get('scale', 1.0))
            ip_adapter_args['ip_adapter_starts'].append(ipa.get('start', 0.0))
            ip_adapter_args['ip_adapter_ends'].append(ipa.get('end', 1.0))
            ip_adapter_args['ip_adapter_crops'].append(ipa.get('crop', False))
            ip_adapter_args['ip_adapter_images'].append([helpers.decode_base64_to_image(x) for x in ipa['images']])
            if ipa.get('masks'):
                ip_adapter_args['ip_adapter_masks'].append([helpers.decode_base64_to_image(x) for x in ipa['masks']])

    save_images = params.get('save_images', True)
    sampler_name = params.get('sampler_name', 'Default')
    sampler_index = processing_helpers.get_sampler_index(sampler_name)

    # Build args dict for control_run, only passing params it accepts
    _valid_params = set(inspect.signature(control_run_module.control_run).parameters.keys())
    skip_keys = {'type', 'inputs', 'inits', 'mask', 'control', 'init_control', 'ip_adapter', 'save_images', 'sampler_name', 'script_name', 'script_args', 'alwayson_scripts', 'extra', 'priority'}
    run_args = {k: v for k, v in params.items() if k in _valid_params and k not in skip_keys}
    run_args['sampler_index'] = sampler_index
    run_args['is_generator'] = True
    run_args['inputs'] = inputs
    run_args['inits'] = inits
    run_args['mask'] = mask
    run_args['units'] = units
    if units:
        run_args['unit_type'] = units[0].type

    extra = params.get('extra', {}) or {}
    run_args['extra'] = extra

    extra_p_args = {
        'do_not_save_grid': not save_images,
        'do_not_save_samples': not save_images,
        **ip_adapter_args,
    }

    override_script_name = params.get('script_name')
    override_script_args = params.get('script_args', [])
    if override_script_name:
        run_args['override_script_name'] = override_script_name
        run_args['override_script_args'] = override_script_args

    # Run generation
    jobid = shared.state.begin('API-V2', api=True)
    try:
        control_run_module.control_set(extra_p_args)
        res = control_run_module.control_run(**run_args)

        output_images = []
        output_processed = []
        for item in res:
            if len(item) > 0 and (isinstance(item[0], list) or item[0] is None):
                output_images += item[0] if item[0] is not None else []
            if len(item) > 1 and item[1] is not None:
                output_processed.append(item[1])

        # Capture saved file paths BEFORE end() clears state.results
        saved_paths = list(shared.state.results) if hasattr(shared.state, 'results') and shared.state.results else []
    finally:
        shared.state.end(jobid)

    # Collect saved file paths
    image_refs = []

    for i, img in enumerate(output_images):
        path = saved_paths[i] if i < len(saved_paths) else None
        if path and os.path.isfile(str(path)):
            path = str(path)
            ext = os.path.splitext(path)[1].lstrip('.').lower()
            image_refs.append({
                'index': i,
                'path': path,
                'url': f'/sdapi/v2/jobs/{job_id}/images/{i}',
                'width': img.width if hasattr(img, 'width') else 0,
                'height': img.height if hasattr(img, 'height') else 0,
                'format': ext if ext else 'png',
                'size': os.path.getsize(path),
            })
        elif save_images and img is not None:
            # Fallback: save image manually if not saved by the pipeline
            from modules import images as img_module
            try:
                output_dir = shared.opts.outdir_txt2img_samples if not inits else shared.opts.outdir_img2img_samples
                path_info = img_module.save_image(img, output_dir, "", seed=params.get('seed', -1), prompt=params.get('prompt', ''))
                if path_info and len(path_info) > 0:
                    fpath = path_info[0] if isinstance(path_info, (list, tuple)) else str(path_info)
                    if os.path.isfile(str(fpath)):
                        ext = os.path.splitext(str(fpath))[1].lstrip('.').lower()
                        image_refs.append({
                            'index': i,
                            'path': str(fpath),
                            'url': f'/sdapi/v2/jobs/{job_id}/images/{i}',
                            'width': img.width if hasattr(img, 'width') else 0,
                            'height': img.height if hasattr(img, 'height') else 0,
                            'format': ext if ext else 'png',
                            'size': os.path.getsize(str(fpath)),
                        })
            except Exception as e:
                log.warning(f'Job {job_id}: failed to save fallback image {i}: {e}')

    # Save processed control images to disk and build refs
    processed_refs = []
    if output_processed:
        from modules import images as img_module
        output_dir = shared.opts.outdir_extras_samples if hasattr(shared.opts, 'outdir_extras_samples') else shared.opts.outdir_txt2img_samples
        for pi, proc_img in enumerate(output_processed):
            try:
                path_info = img_module.save_image(proc_img, output_dir, "control-", prompt="processed")
                fpath = path_info[0] if isinstance(path_info, (list, tuple)) else str(path_info) if path_info else None
                if fpath and os.path.isfile(str(fpath)):
                    fpath = str(fpath)
                    ext = os.path.splitext(fpath)[1].lstrip('.').lower()
                    processed_refs.append({
                        'index': pi,
                        'path': fpath,
                        'url': f'/sdapi/v2/jobs/{job_id}/processed/{pi}',
                        'width': proc_img.width if hasattr(proc_img, 'width') else 0,
                        'height': proc_img.height if hasattr(proc_img, 'height') else 0,
                        'format': ext or 'png',
                        'size': os.path.getsize(fpath),
                    })
            except Exception as e:
                log.warning(f'Job {job_id}: failed to save processed image {pi}: {e}')

    return {'images': image_refs, 'processed': processed_refs, 'info': {}, 'params': {k: v for k, v in params.items() if k != 'type'}}


def execute_upscale(params: dict, job_id: str) -> dict:
    from modules import shared, postprocessing
    from modules.api import helpers

    image = helpers.decode_base64_to_image(params.get('image', ''))
    upscaler = params.get('upscaler', 'None')
    scale = params.get('scale', 2.0)

    jobid = shared.state.begin('API-V2-UP', api=True)
    try:
        result = postprocessing.run_extras(
            extras_mode=0, image=image, image_folder="", input_dir="", output_dir="",
            save_output=True, extras_upscaler_1=upscaler, upscaling_resize=scale,
        )
        saved_paths = list(shared.state.results) if hasattr(shared.state, 'results') and shared.state.results else []
    finally:
        shared.state.end(jobid)

    output_image = result[0][0] if result and result[0] else None
    image_refs = []
    if output_image is not None:
        if saved_paths and os.path.isfile(str(saved_paths[0])):
            path = str(saved_paths[0])
            ext = os.path.splitext(path)[1].lstrip('.').lower()
            image_refs.append({'index': 0, 'path': path, 'url': f'/sdapi/v2/jobs/{job_id}/images/0', 'width': output_image.width, 'height': output_image.height, 'format': ext or 'png', 'size': os.path.getsize(path)})

    return {'images': image_refs, 'info': {}, 'params': {k: v for k, v in params.items() if k not in ('type', 'image')}}


def execute_caption(params: dict, job_id: str) -> dict:
    from modules import shared
    from modules.api import helpers

    backend = params.get('backend', 'vlm')
    image = helpers.decode_base64_to_image(params.get('image', ''))
    model = params.get('model')

    jobid = shared.state.begin('API-V2-CAP', api=True)
    try:
        if backend == 'vlm':
            from modules.api.caption import do_vqa, ReqVQA
            req = ReqVQA(image='', model=model, prompt=params.get('prompt'))
            answer, _annotated = do_vqa(image, req)
            caption_text = answer
        elif backend == 'openclip':
            from modules.api.caption import do_openclip, ReqCaptionOpenCLIP
            req = ReqCaptionOpenCLIP(image='', model=model)
            caption_text, *_ = do_openclip(image, req)
        elif backend == 'tagger':
            from modules.api.caption import do_tagger, ReqTagger
            req = ReqTagger(image='', model=model)
            tags, _scores = do_tagger(image, req)
            caption_text = tags
        else:
            raise ValueError(f"Unknown caption backend: {backend}")
    finally:
        shared.state.end(jobid)

    return {'images': [], 'info': {'caption': caption_text}, 'params': {k: v for k, v in params.items() if k not in ('type', 'image')}}


def execute_enhance(params: dict, job_id: str) -> dict:
    from modules import shared, processing_helpers
    from modules.api import helpers

    prompt = params.get('prompt', '')
    model = params.get('model')
    enhance_type = params.get('enhance_type', 'text')
    seed = processing_helpers.get_fixed_seed(params.get('seed', -1))
    image = helpers.decode_base64_to_image(params['image']) if params.get('image') else None

    jobid = shared.state.begin('API-V2-ENH', api=True)
    try:
        if enhance_type == 'video':
            from modules.ui_video_vlm import enhance_prompt
            default_model = 'Google Gemma 3 4B' if model is None or len(model) < 4 else model
            result_prompt = enhance_prompt(enable=True, image=image, prompt=prompt, model=default_model, system_prompt=params.get('system_prompt', ''), nsfw=params.get('nsfw', False))
        else:
            from modules.scripts_manager import scripts_txt2img
            default_model = 'google/gemma-3-4b-it' if enhance_type == 'image' else 'google/gemma-3-1b-it'
            use_model = default_model if model is None or len(model) < 4 else model
            instance = [s for s in scripts_txt2img.scripts if 'prompt_enhance.py' in s.filename][0]
            result_prompt = instance.enhance(
                model=use_model, prompt=prompt,
                system=params.get('system_prompt', ''), prefix=params.get('prefix', ''), suffix=params.get('suffix', ''),
                sample=params.get('do_sample', True), tokens=params.get('max_tokens', 256),
                temperature=params.get('temperature', 0.7), penalty=params.get('repetition_penalty', 1.2),
                top_k=params.get('top_k', 50), top_p=params.get('top_p', 0.9),
                thinking=params.get('thinking', False), keep_thinking=params.get('keep_thinking', False),
                use_vision=params.get('use_vision', False), prefill=params.get('prefill', ''),
                keep_prefill=params.get('keep_prefill', False), image=image, seed=seed,
                nsfw=params.get('nsfw', False),
            )
    finally:
        shared.state.end(jobid)

    return {'images': [], 'info': {'prompt': result_prompt, 'seed': seed}, 'params': {k: v for k, v in params.items() if k not in ('type', 'image')}}


def execute_detect(params: dict, job_id: str) -> dict:
    from modules import shared
    from modules.api import helpers
    from modules.shared import yolo

    image = helpers.decode_base64_to_image(params.get('image', ''))
    model = params.get('model')

    jobid = shared.state.begin('API-V2-DET', api=True)
    try:
        items = yolo.predict(model, image)
        detections = []
        for item in items:
            detections.append({
                'label': item.label,
                'score': item.score,
                'cls': item.cls,
                'box': item.box,
            })
    finally:
        shared.state.end(jobid)

    return {'images': [], 'info': {'detections': detections}, 'params': {k: v for k, v in params.items() if k not in ('type', 'image')}}


def execute_preprocess(params: dict, job_id: str) -> dict:
    from modules import shared
    from modules.api import helpers
    from modules.control import processors

    image = helpers.decode_base64_to_image(params.get('image', ''))
    model = params.get('model', '')
    proc_params = params.get('params', {}) or {}

    processors_list = list(processors.config)
    if model not in processors_list:
        raise ValueError(f"Processor model not found: {model}")

    jobid = shared.state.begin('API-V2-PRE', api=True)
    try:
        proc = processors.Processor(model)
        processed = proc(image, local_config=proc_params)
        # Save processed image to disk
        from modules import images as img_module
        output_dir = shared.opts.outdir_extras_samples if hasattr(shared.opts, 'outdir_extras_samples') else shared.opts.outdir_txt2img_samples
        path_info = img_module.save_image(processed, output_dir, "", prompt=f"preprocess-{model}")
    finally:
        shared.state.end(jobid)

    image_refs = []
    if path_info:
        fpath = path_info[0] if isinstance(path_info, (list, tuple)) else str(path_info)
        if os.path.isfile(str(fpath)):
            ext = os.path.splitext(str(fpath))[1].lstrip('.').lower()
            image_refs.append({'index': 0, 'path': str(fpath), 'url': f'/sdapi/v2/jobs/{job_id}/images/0', 'width': processed.width, 'height': processed.height, 'format': ext or 'png', 'size': os.path.getsize(str(fpath))})

    return {'images': image_refs, 'info': {'model': model}, 'params': {k: v for k, v in params.items() if k not in ('type', 'image')}}


def execute_video(params: dict, job_id: str) -> dict:
    from modules import shared
    from modules.api import helpers
    from modules.video_models import video_run, video_ui

    engine = params.get('engine', '')
    model = params.get('model', '')
    prompt = params.get('prompt', '')
    negative = params.get('negative', '')
    width = params.get('width', 848)
    height = params.get('height', 480)
    frames = params.get('frames', 25)
    steps = params.get('steps', 30)
    sampler_index = params.get('sampler', 0)
    sampler_shift = params.get('sampler_shift', -1)
    dynamic_shift = params.get('dynamic_shift', False)
    seed = params.get('seed', -1)
    guidance_scale = params.get('guidance_scale', 6.0)
    guidance_true = params.get('guidance_true', -1)
    init_strength = params.get('init_strength', 0.5)
    vae_type = params.get('vae_type', 'Default')
    vae_tile_frames = params.get('vae_tile_frames', 0)
    mp4_fps = params.get('fps', 24)
    mp4_interpolate = params.get('interpolate', 0)
    mp4_codec = params.get('codec', 'libx264')
    mp4_ext = params.get('format', 'mp4')
    mp4_opt = params.get('codec_options', 'crf:16')
    mp4_video = params.get('save_video', True)
    mp4_frames = params.get('save_frames', False)
    mp4_sf = params.get('save_safetensors', False)

    # Decode optional images
    init_image = helpers.decode_base64_to_image(params['init_image']) if params.get('init_image') else None
    last_image = helpers.decode_base64_to_image(params['last_image']) if params.get('last_image') else None

    # Ensure model is loaded
    for _msg in video_ui.model_load(engine, model):
        pass

    jobid = shared.state.begin('API-V2-VID', api=True)
    try:
        result = video_run.generate(
            '', '',  # task_id, ui_state
            engine, model,
            prompt, negative, [],  # styles
            width, height, frames,
            steps, sampler_index, sampler_shift, dynamic_shift,
            seed, guidance_scale, guidance_true,
            init_image, init_strength, last_image,
            vae_type, vae_tile_frames,
            mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf,
            False, '', '',  # vlm_enhance, vlm_model, vlm_system_prompt
            {},  # override_settings
        )
    finally:
        shared.state.end(jobid)

    # result = (images, video_file, gen_info_js, info, html_log)
    video_file = result[1] if result and len(result) > 1 else None

    image_refs = []
    # Video file as first "image" ref
    if video_file and os.path.isfile(str(video_file)):
        path = str(video_file)
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        image_refs.append({
            'index': 0,
            'path': path,
            'url': f'/sdapi/v2/jobs/{job_id}/images/0',
            'width': width,
            'height': height,
            'format': ext or 'mp4',
            'size': os.path.getsize(path),
        })

    return {'images': image_refs, 'info': {}, 'params': {k: v for k, v in params.items() if k not in ('type', 'init_image', 'last_image')}}


def execute_framepack(params: dict, job_id: str) -> dict:
    from modules import shared
    from modules.api import helpers
    from modules.framepack import framepack_wrappers

    init_image = helpers.decode_base64_to_image(params['init_image']) if params.get('init_image') else None
    end_image = helpers.decode_base64_to_image(params['end_image']) if params.get('end_image') else None

    prompt = params.get('prompt', '')
    negative = params.get('negative', '')
    styles = params.get('styles', [])
    seed = params.get('seed', -1)
    resolution = params.get('resolution', 640)
    duration = params.get('duration', 4)
    variant = params.get('variant', 'bi-directional')
    attention = params.get('attention', 'Default')

    jobid = shared.state.begin('API-V2-FP', api=True)
    try:
        gen = framepack_wrappers.run_framepack(
            '', '',  # task_id, ui_state
            init_image, end_image,
            params.get('start_weight', 1.0), params.get('end_weight', 1.0), params.get('vision_weight', 1.0),
            prompt, params.get('system_prompt', ''), params.get('optimized_prompt', True), params.get('section_prompt', ''),
            negative, styles,
            seed, resolution, duration,
            params.get('latent_ws', 9), params.get('steps', 25),
            params.get('cfg_scale', 1.0), params.get('cfg_distilled', 10.0), params.get('cfg_rescale', 0.0),
            params.get('shift', 3.0),
            params.get('use_teacache', True), params.get('use_cfgzero', False), params.get('use_preview', True),
            params.get('fps', 30), params.get('codec', 'libx264'), params.get('save_safetensors', False),
            params.get('save_video', True), params.get('save_frames', False),
            params.get('codec_options', 'crf:16'), params.get('format', 'mp4'),
            params.get('interpolate', 0),
            attention, params.get('vae_type', 'Full'), variant,
            params.get('vlm_enhance', False), params.get('vlm_model', ''), params.get('vlm_system_prompt', ''),
        )
        video_file = None
        for item in gen:
            if item and len(item) > 0 and isinstance(item[0], str) and item[0] and not item[0].startswith('<'):
                video_file = item[0]
    finally:
        shared.state.end(jobid)

    image_refs = []
    if video_file and os.path.isfile(str(video_file)):
        path = str(video_file)
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        image_refs.append({
            'index': 0,
            'path': path,
            'url': f'/sdapi/v2/jobs/{job_id}/images/0',
            'width': resolution,
            'height': resolution,
            'format': ext or 'mp4',
            'size': os.path.getsize(path),
        })

    return {'images': image_refs, 'info': {}, 'params': {k: v for k, v in params.items() if k not in ('type', 'init_image', 'end_image')}}


def execute_ltx(params: dict, job_id: str) -> dict:
    from modules import shared
    from modules.api import helpers
    from modules.ltx import ltx_process

    model = params.get('model', '')
    prompt = params.get('prompt', '')
    negative = params.get('negative', '')
    styles = params.get('styles', [])
    width = params.get('width', 768)
    height = params.get('height', 512)
    frames = params.get('frames', 97)
    steps = params.get('steps', 50)
    sampler_index = params.get('sampler', 0)
    seed = params.get('seed', -1)

    condition_image = helpers.decode_base64_to_image(params['condition_image']) if params.get('condition_image') else None
    condition_last = helpers.decode_base64_to_image(params['condition_last']) if params.get('condition_last') else None

    jobid = shared.state.begin('API-V2-LTX', api=True)
    try:
        gen = ltx_process.run_ltx(
            '', '',  # task_id, ui_state
            model, prompt, negative, styles,
            width, height, frames, steps, sampler_index, seed,
            params.get('upsample_enable', False), params.get('upsample_ratio', 2.0),
            params.get('refine_enable', False), params.get('refine_strength', 0.4),
            params.get('condition_strength', 0.8),
            condition_image, condition_last,
            None, None,  # condition_files, condition_video
            params.get('condition_video_frames', 0), params.get('condition_video_skip', 0),
            params.get('decode_timestep', 0.05), params.get('image_cond_noise_scale', 0.025),
            params.get('fps', 24), params.get('interpolate', 0),
            params.get('codec', 'libx264'), params.get('format', 'mp4'),
            params.get('codec_options', 'crf:16'),
            params.get('save_video', True), params.get('save_frames', False), params.get('save_safetensors', False),
            params.get('audio_enable', False),
            {},  # overrides
        )
        video_file = None
        for item in gen:
            if item and len(item) > 0 and isinstance(item[0], str) and item[0]:
                video_file = item[0]
    finally:
        shared.state.end(jobid)

    image_refs = []
    if video_file and os.path.isfile(str(video_file)):
        path = str(video_file)
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        image_refs.append({
            'index': 0,
            'path': path,
            'url': f'/sdapi/v2/jobs/{job_id}/images/0',
            'width': width,
            'height': height,
            'format': ext or 'mp4',
            'size': os.path.getsize(path),
        })

    return {'images': image_refs, 'info': {}, 'params': {k: v for k, v in params.items() if k not in ('type', 'condition_image', 'condition_last')}}


EXECUTORS = {
    'generate': execute_generate,
    'upscale': execute_upscale,
    'caption': execute_caption,
    'enhance': execute_enhance,
    'detect': execute_detect,
    'preprocess': execute_preprocess,
    'video': execute_video,
    'framepack': execute_framepack,
    'ltx': execute_ltx,
}
