import os
import time
from typing import List, Union
import cv2
import numpy as np
from PIL import Image
from modules.control import util # helper functions
from modules.control import unit # control units
from modules.control import processors # image preprocessors
from modules.control import tile # tiling module
from modules.control.units import controlnet # lllyasviel ControlNet
from modules.control.units import xs # VisLearn ControlNet-XS
from modules.control.units import lite # Kohya ControlLLLite
from modules.control.units import t2iadapter # TencentARC T2I-Adapter
from modules.control.units import reference # ControlNet-Reference
from modules import devices, shared, errors, processing, images, sd_models, scripts_manager, masking
from modules.processing_class import StableDiffusionProcessingControl
from modules.ui_common import infotext_to_html
from modules.api import script
from modules.timer import process as process_timer


debug = os.environ.get('SD_CONTROL_DEBUG', None) is not None
debug_log = shared.log.trace if debug else lambda *args, **kwargs: None
pipe = None
instance = None
original_pipeline = None
p_extra_args = {}
unified_models = ['Flex2Pipeline'] # models that have controlnet builtin


def restore_pipeline():
    global pipe, instance # pylint: disable=global-statement
    if instance is not None and hasattr(instance, 'restore'):
        instance.restore()
    if original_pipeline is not None and (original_pipeline.__class__.__name__ != shared.sd_model.__class__.__name__):
        debug_log(f'Control restored pipeline: class={shared.sd_model.__class__.__name__} to={original_pipeline.__class__.__name__}')
        shared.sd_model = original_pipeline
    pipe = None
    instance = None
    devices.torch_gc()


def terminate(msg):
    restore_pipeline()
    shared.log.error(f'Control terminated: {msg}')
    return msg


def is_unified_model():
    return shared.sd_model.__class__.__name__ in unified_models


def set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, control_conditioning, control_guidance_start, control_guidance_end, inits):
    global pipe, instance # pylint: disable=global-statement
    pipe = None
    if has_models:
        # Track control operation
        from modules import pipeline_viz
        p.ops.append('control')
        pipeline_viz.track_operation_start('control', p)
        
        p.extra_generation_params["Control type"] = unit_type # overriden later with pretty-print
        p.extra_generation_params["Control model"] = ';'.join([(m.model_id or '') for m in active_model if m.model is not None])
        p.extra_generation_params["Control conditioning"] = control_conditioning if isinstance(control_conditioning, list) else [control_conditioning]
        p.extra_generation_params['Control start'] = control_guidance_start if isinstance(control_guidance_start, list) else [control_guidance_start]
        p.extra_generation_params['Control end'] = control_guidance_end if isinstance(control_guidance_end, list) else [control_guidance_end]
        p.extra_generation_params["Control conditioning"] = ';'.join([str(c) for c in p.extra_generation_params["Control conditioning"]])
        p.extra_generation_params['Control start'] = ';'.join([str(c) for c in p.extra_generation_params['Control start']])
        p.extra_generation_params['Control end'] = ';'.join([str(c) for c in p.extra_generation_params['Control end']])
    if unit_type == 't2i adapter' and has_models:
        p.extra_generation_params["Control type"] = 'T2I-Adapter'
        p.task_args['adapter_conditioning_scale'] = control_conditioning
        instance = t2iadapter.AdapterPipeline(selected_models, shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: T2I-Adapter does not support separate init image')
    elif unit_type == 'controlnet' and has_models:
        p.extra_generation_params["Control type"] = 'ControlNet'
        if shared.sd_model_type == 'f1':
            p.task_args['controlnet_conditioning_scale'] = control_conditioning if isinstance(control_conditioning, list) else [control_conditioning]
        else:
            p.task_args['controlnet_conditioning_scale'] = control_conditioning
        p.task_args['control_guidance_start'] = control_guidance_start
        p.task_args['control_guidance_end'] = control_guidance_end
        p.task_args['guess_mode'] = p.guess_mode
        if not is_unified_model():
            instance = controlnet.ControlNetPipeline(selected_models, shared.sd_model, p=p)
            pipe = instance.pipeline
        else:
            pipe = shared.sd_model
    elif unit_type == 'xs' and has_models:
        p.extra_generation_params["Control type"] = 'ControlNet-XS'
        p.controlnet_conditioning_scale = control_conditioning
        p.control_guidance_start = control_guidance_start
        p.control_guidance_end = control_guidance_end
        instance = xs.ControlNetXSPipeline(selected_models, shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: ControlNet-XS does not support separate init image')
    elif unit_type == 'lite' and has_models:
        p.extra_generation_params["Control type"] = 'ControlLLLite'
        p.controlnet_conditioning_scale = control_conditioning
        instance = lite.ControlLLitePipeline(shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: ControlLLLite does not support separate init image')
    elif unit_type == 'reference' and has_models:
        p.extra_generation_params["Control type"] = 'Reference'
        p.extra_generation_params["Control attention"] = p.attention
        p.task_args['reference_attn'] = 'Attention' in p.attention
        p.task_args['reference_adain'] = 'Adain' in p.attention
        p.task_args['attention_auto_machine_weight'] = p.query_weight
        p.task_args['gn_auto_machine_weight'] = p.adain_weight
        p.task_args['style_fidelity'] = p.fidelity
        instance = reference.ReferencePipeline(shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            shared.log.warning('Control: ControlNet-XS does not support separate init image')
    else: # run in txt2img/img2img mode
        if len(active_strength) > 0:
            p.strength = active_strength[0]
        pipe = shared.sd_model
        instance = None
    debug_log(f'Control: run type={unit_type} models={has_models} pipe={pipe.__class__.__name__ if pipe is not None else None}')
    return pipe


def check_active(p, unit_type, units):
    active_process: List[processors.Processor] = [] # all active preprocessors
    active_model: List[Union[controlnet.ControlNet, xs.ControlNetXS, t2iadapter.Adapter]] = [] # all active models
    active_strength: List[float] = [] # strength factors for all active models
    active_start: List[float] = [] # start step for all active models
    active_end: List[float] = [] # end step for all active models
    num_units = 0
    for u in units:
        if u.type != unit_type:
            continue
        num_units += 1
        debug_log(f'Control unit: i={num_units} type={u.type} enabled={u.enabled} cn={u.controlnet} proc={u.process}')
        if not u.enabled:
            if u.controlnet is not None and u.controlnet.model is not None:
                debug_log(f'Control unit offload: model="{u.controlnet.model_id}" device={devices.cpu}')
                sd_models.move_model(u.controlnet.model, devices.cpu)
            continue
        if u.controlnet is not None and u.controlnet.model is not None:
            debug_log(f'Control unit offload: model="{u.controlnet.model_id}" device={devices.device}')
            sd_models.move_model(u.controlnet.model, devices.device)
        if unit_type == 't2i adapter' and u.adapter.model is not None:
            active_process.append(u.process)
            active_model.append(u.adapter)
            active_strength.append(float(u.strength))
            p.adapter_conditioning_factor = u.factor
            shared.log.debug(f'Control T2I-Adapter unit: i={num_units} process="{u.process.processor_id}" model="{u.adapter.model_id}" strength={u.strength} factor={u.factor}')
        elif unit_type == 'controlnet' and (u.controlnet.model is not None or is_unified_model()):
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_start.append(float(u.start))
            active_end.append(float(u.end))
            p.guess_mode = u.guess
            if isinstance(u.mode, str):
                if not hasattr(p, 'control_mode'):
                    p.control_mode = []
                p.control_mode.append(u.choices.index(u.mode) if u.mode in u.choices else 0)
                p.is_tile = p.is_tile or 'tile' in u.mode.lower()
                p.control_tile = u.tile
                p.extra_generation_params["Control mode"] = u.mode
            shared.log.debug(f'Control ControlNet unit: i={num_units} process="{u.process.processor_id}" model="{u.controlnet.model_id}" strength={u.strength} guess={u.guess} start={u.start} end={u.end} mode={u.mode}')
        elif unit_type == 'xs' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_start.append(float(u.start))
            active_end.append(float(u.end))
            shared.log.debug(f'Control ControlNet-XS unit: i={num_units} process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'lite' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            shared.log.debug(f'Control ControlLLite unit: i={num_units} process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'reference':
            p.override = u.override
            p.attention = u.attention
            p.query_weight = float(u.query_weight)
            p.adain_weight = float(u.adain_weight)
            p.fidelity = u.fidelity
            shared.log.debug('Control Reference unit')
        else:
            if u.process.processor_id is not None:
                active_process.append(u.process)
                shared.log.debug(f'Control process unit: i={num_units} process={u.process.processor_id}')
            active_strength.append(float(u.strength))
    debug_log(f'Control active: process={len(active_process)} model={len(active_model)}')
    return active_process, active_model, active_strength, active_start, active_end


def check_enabled(p, unit_type, units, active_model, active_strength, active_start, active_end):
    has_models = False
    selected_models: List[Union[controlnet.ControlNetModel, xs.ControlNetXSModel, t2iadapter.AdapterModel]] = None
    control_conditioning = None
    control_guidance_start = None
    control_guidance_end = None
    if unit_type == 't2i adapter' or unit_type == 'controlnet' or unit_type == 'xs' or unit_type == 'lite':
        if len(active_model) == 0:
            selected_models = None
        elif len(active_model) == 1:
            selected_models = active_model[0].model if active_model[0].model is not None else None
            p.is_tile = p.is_tile or 'tile' in (active_model[0].model_id or '').lower()
            has_models = (selected_models is not None) or is_unified_model()
            control_conditioning = active_strength[0] if len(active_strength) > 0 else 1 # strength or list[strength]
            control_guidance_start = active_start[0] if len(active_start) > 0 else 0
            control_guidance_end = active_end[0] if len(active_end) > 0 else 1
        else:
            selected_models = [m.model for m in active_model if m.model is not None]
            has_models = len(selected_models) > 0
            control_conditioning = active_strength[0] if len(active_strength) == 1 else list(active_strength) # strength or list[strength]
            control_guidance_start = active_start[0] if len(active_start) == 1 else list(active_start)
            control_guidance_end = active_end[0] if len(active_end) == 1 else list(active_end)
    elif unit_type == 'reference':
        has_models = any(u.enabled for u in units if u.type == 'reference')
    else:
        pass
    return has_models, selected_models, control_conditioning, control_guidance_start, control_guidance_end


def control_set(kwargs):
    if kwargs:
        global p_extra_args # pylint: disable=global-statement
        p_extra_args = {}
        debug_log(f'Control extra args: {kwargs}')
    for k, v in kwargs.items():
        p_extra_args[k] = v


def control_run(state: str = '', # pylint: disable=keyword-arg-before-vararg
                units: List[unit.Unit] = [], inputs: List[Image.Image] = [], inits: List[Image.Image] = [], mask: Image.Image = None, unit_type: str = None, is_generator: bool = True,
                input_type: int = 0,
                prompt: str = '', negative_prompt: str = '', styles: List[str] = [],
                steps: int = 20, sampler_index: int = None,
                seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1,
                cfg_scale: float = 6.0, clip_skip: float = 1.0, image_cfg_scale: float = 6.0, diffusers_guidance_rescale: float = 0.7, pag_scale: float = 0.0, pag_adaptive: float = 0.5, cfg_end: float = 1.0,
                vae_type: str = 'Full', tiling: bool = False, hidiffusion: bool = False,
                detailer_enabled: bool = True, detailer_prompt: str = '', detailer_negative: str = '', detailer_steps: int = 10, detailer_strength: float = 0.3,
                hdr_mode: int = 0, hdr_brightness: float = 0, hdr_color: float = 0, hdr_sharpen: float = 0, hdr_clamp: bool = False, hdr_boundary: float = 4.0, hdr_threshold: float = 0.95,
                hdr_maximize: bool = False, hdr_max_center: float = 0.6, hdr_max_boundary: float = 1.0, hdr_color_picker: str = None, hdr_tint_ratio: float = 0,
                resize_mode_before: int = 0, resize_name_before: str = 'None', resize_context_before: str = 'None', width_before: int = 512, height_before: int = 512, scale_by_before: float = 1.0, selected_scale_tab_before: int = 0,
                resize_mode_after: int = 0, resize_name_after: str = 'None', resize_context_after: str = 'None', width_after: int = 0, height_after: int = 0, scale_by_after: float = 1.0, selected_scale_tab_after: int = 0,
                resize_mode_mask: int = 0, resize_name_mask: str = 'None', resize_context_mask: str = 'None', width_mask: int = 0, height_mask: int = 0, scale_by_mask: float = 1.0, selected_scale_tab_mask: int = 0,
                denoising_strength: float = 0.3, batch_count: int = 1, batch_size: int = 1,
                enable_hr: bool = False, hr_sampler_index: int = None, hr_denoising_strength: float = 0.0, hr_resize_mode: int = 0, hr_resize_context: str = 'None', hr_upscaler: str = None, hr_force: bool = False, hr_second_pass_steps: int = 20,
                hr_scale: float = 1.0, hr_resize_x: int = 0, hr_resize_y: int = 0, refiner_steps: int = 5, refiner_start: float = 0.0, refiner_prompt: str = '', refiner_negative: str = '',
                video_skip_frames: int = 0, video_type: str = 'None', video_duration: float = 2.0, video_loop: bool = False, video_pad: int = 0, video_interpolate: int = 0,
                *input_script_args,
        ):
    # handle optional initialization via ui
    for u in units:
        if not u.enabled:
            continue
        if u.process_name is not None and u.process_name != '' and u.process_name != 'None':
            u.process.load(u.process_name, force=False)
        if u.model_name is not None and u.model_name != '' and u.model_name != 'None':
            if u.type == 't2i adapter':
                u.adapter.load(u.model_name, force=False)
            else:
                u.controlnet.load(u.model_name, force=False)
                u.update_choices(u.model_name)
        if u.process is not None and u.process.override is None and u.override is not None:
            u.process.override = u.override

    global pipe, original_pipeline # pylint: disable=global-statement
    debug_log(f'Control: type={unit_type} input={inputs} init={inits} type={input_type}')
    if inputs is None or (type(inputs) is list and len(inputs) == 0):
        inputs = [None]
    output_images: List[Image.Image] = [] # output images
    processed_image: Image.Image = None # last processed image
    if mask is not None and input_type == 0:
        input_type = 1 # inpaint always requires control_image

    if sampler_index is None:
        shared.log.warning('Sampler: invalid')
        sampler_index = 0
    if hr_sampler_index is None:
        hr_sampler_index = sampler_index

    p = StableDiffusionProcessingControl(
        prompt = prompt,
        negative_prompt = negative_prompt,
        styles = styles,
        steps = steps,
        n_iter = batch_count,
        batch_size = batch_size,
        sampler_name = processing.get_sampler_name(sampler_index),
        seed = seed,
        subseed = subseed,
        subseed_strength = subseed_strength,
        seed_resize_from_h = seed_resize_from_h,
        seed_resize_from_w = seed_resize_from_w,
        # advanced
        cfg_scale = cfg_scale,
        cfg_end = cfg_end,
        clip_skip = clip_skip,
        image_cfg_scale = image_cfg_scale,
        diffusers_guidance_rescale = diffusers_guidance_rescale,
        pag_scale = pag_scale,
        pag_adaptive = pag_adaptive,
        vae_type = vae_type,
        tiling = tiling,
        hidiffusion = hidiffusion,
        # detailer
        detailer_enabled = detailer_enabled,
        detailer_prompt = detailer_prompt,
        detailer_negative = detailer_negative,
        detailer_steps = detailer_steps,
        detailer_strength = detailer_strength,
        # resize
        resize_mode = resize_mode_before if resize_name_before != 'None' else 0,
        resize_name = resize_name_before,
        scale_by = scale_by_before,
        selected_scale_tab = selected_scale_tab_before,
        denoising_strength = denoising_strength,
        # inpaint
        inpaint_full_res = masking.opts.mask_only,
        inpainting_mask_invert = 1 if masking.opts.invert else 0,
        inpainting_fill = 1,
        # hdr
        hdr_mode=hdr_mode, hdr_brightness=hdr_brightness, hdr_color=hdr_color, hdr_sharpen=hdr_sharpen, hdr_clamp=hdr_clamp,
        hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold, hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundary=hdr_max_boundary, hdr_color_picker=hdr_color_picker, hdr_tint_ratio=hdr_tint_ratio,
        # path
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_control_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_control_grids,
    )
    p.state = state
    p.is_tile = False
    # processing.process_init(p)
    resize_mode_before = resize_mode_before if resize_name_before != 'None' and inputs is not None and len(inputs) > 0 else 0

    # TODO modernui: monkey-patch for missing tabs.select event
    if selected_scale_tab_before == 0 and resize_name_before != 'None' and scale_by_before != 1 and inputs is not None and len(inputs) > 0:
        shared.log.debug('Control: override resize mode=before')
        selected_scale_tab_before = 1
    if selected_scale_tab_after == 0 and resize_name_after != 'None' and scale_by_after != 1:
        shared.log.debug('Control: override resize mode=after')
        selected_scale_tab_after = 1
    if selected_scale_tab_mask == 0 and resize_name_mask != 'None' and scale_by_mask != 1:
        shared.log.debug('Control: override resize mode=mask')
        selected_scale_tab_mask = 1

    # set control sizing
    if resize_mode_before != 0 or inputs is None or inputs == [None]:
        p.width, p.height = width_before, height_before # pylint: disable=attribute-defined-outside-init
        p.width_before = width_before
        p.height_before = height_before
        if resize_name_before != 'None':
            p.resize_mode_before = resize_mode_before
            p.resize_name_before = resize_name_before
        p.scale_by_before = scale_by_before
        p.selected_scale_tab_before = selected_scale_tab_before
    else:
        del p.width
        del p.height
    if resize_name_after != 'None':
        p.resize_mode_after = resize_mode_after
        p.resize_name_after = resize_name_after
        p.width_after = width_after
        p.height_after = height_after
        p.scale_by_after = scale_by_after
        p.selected_scale_tab_after = selected_scale_tab_after
    if resize_name_mask != 'None':
        p.resize_mode_mask = resize_mode_mask
        p.resize_name_mask = resize_name_mask
        p.width_mask = width_mask
        p.height_mask = height_mask
        p.scale_by_mask = scale_by_mask
        p.selected_scale_tab_mask = selected_scale_tab_mask

    # hires/refine defined outside of main init
    p.enable_hr = enable_hr
    p.hr_sampler_name = processing.get_sampler_name(hr_sampler_index)
    p.hr_denoising_strength = hr_denoising_strength
    p.hr_resize_mode = hr_resize_mode
    p.hr_resize_context = hr_resize_context
    p.hr_upscaler = hr_upscaler
    p.hr_force = hr_force
    p.hr_second_pass_steps = hr_second_pass_steps
    p.hr_scale = hr_scale
    p.hr_resize_x = hr_resize_x
    p.hr_resize_y = hr_resize_y
    p.refiner_steps = refiner_steps
    p.refiner_start = refiner_start
    p.refiner_prompt = refiner_prompt
    p.refiner_negative = refiner_negative
    if p.enable_hr and (p.hr_resize_x == 0 or p.hr_resize_y == 0):
        p.hr_upscale_to_x, p.hr_upscale_to_y = 8 * int(width_before * p.hr_scale / 8), 8 * int(height_before * p.hr_scale / 8)
    elif p.enable_hr and (p.hr_upscale_to_x == 0 or p.hr_upscale_to_y == 0):
        p.hr_upscale_to_x, p.hr_upscale_to_y = 8 * int(p.hr_resize_x / 8), 8 * int(hr_resize_y / 8)

    if is_unified_model():
        p.init_images = inputs

    global p_extra_args # pylint: disable=global-statement
    for k, v in p_extra_args.items():
        setattr(p, k, v)
    p_extra_args = {}

    if shared.sd_model is None:
        shared.log.warning('Aborted: op=control model not loaded')
        return [], '', '', 'Error: model not loaded'

    unit_type = unit_type.strip().lower() if unit_type is not None else ''
    t0 = time.time()

    active_process, active_model, active_strength, active_start, active_end = check_active(p, unit_type, units)
    has_models, selected_models, control_conditioning, control_guidance_start, control_guidance_end = check_enabled(p, unit_type, units, active_model, active_strength, active_start, active_end)

    processed: processing.Processed = None
    image_txt = ''
    info_txt = []

    p.is_tile = p.is_tile and has_models

    pipe = set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, control_conditioning, control_guidance_start, control_guidance_end, inits)
    debug_log(f'Control pipeline: class={pipe.__class__.__name__} args={vars(p)}')
    t1, t2, t3 = time.time(), 0, 0
    status = True
    frame = None
    video = None
    output_filename = None
    index = 0
    frames = 0
    blended_image = None

    # set pipeline
    if pipe is None:
        return [], '', '', 'Pipeline not set'
    elif pipe.__class__.__name__ != shared.sd_model.__class__.__name__:
        original_pipeline = shared.sd_model
        shared.sd_model = pipe
        sd_models.move_model(shared.sd_model, shared.device)
        debug_log(f'Control device={devices.device} dtype={devices.dtype}')
        sd_models.copy_diffuser_options(shared.sd_model, original_pipeline) # copy options from original pipeline
        sd_models.set_diffuser_options(shared.sd_model)
    else:
        original_pipeline = None

    try:
        with devices.inference_context():
            if isinstance(inputs, str): # only video, the rest is a list
                if input_type == 2: # separate init image
                    if isinstance(inits, str) and inits != inputs:
                        shared.log.warning('Control: separate init video not support for video input')
                        input_type = 1
                try:
                    video = cv2.VideoCapture(inputs)
                    if not video.isOpened():
                        if is_generator:
                            yield terminate(f'Video open failed: path={inputs}')
                        return [], '', '', 'Error: video open failed'
                    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(video.get(cv2.CAP_PROP_FPS))
                    w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    codec = util.decode_fourcc(video.get(cv2.CAP_PROP_FOURCC))
                    status, frame = video.read()
                    if status:
                        shared.state.frame_count = 1 + frames // (video_skip_frames + 1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    shared.log.debug(f'Control: input video: path={inputs} frames={frames} fps={fps} size={w}x{h} codec={codec}')
                except Exception as e:
                    if is_generator:
                        yield terminate(f'Video open failed: path={inputs} {e}')
                    return [], '', '', 'Error: video open failed'

            while status:
                if pipe is None: # pipe may have been reset externally
                    pipe = set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, control_conditioning, control_guidance_start, control_guidance_end, inits)
                    debug_log(f'Control pipeline reinit: class={pipe.__class__.__name__}')
                possible = sd_models.get_call(pipe).keys()
                processed_image = None
                if frame is not None:
                    inputs = [Image.fromarray(frame)] # cv2 to pil
                for i, input_image in enumerate(inputs):
                    debug_log(f'Control Control image: {i + 1} of {len(inputs)}')
                    if shared.state.skipped:
                        shared.state.skipped = False
                        continue
                    if shared.state.interrupted:
                        shared.state.interrupted = False
                        if is_generator:
                            yield terminate('Interrupted')
                        return [], '', '', 'Interrupted'
                    # get input
                    if isinstance(input_image, str):
                        try:
                            input_image = Image.open(inputs[i])
                        except Exception as e:
                            shared.log.error(f'Control: image open failed: path={inputs[i]} type=control error={e}')
                            continue
                    # match init input
                    if input_type == 1:
                        debug_log('Control Init image: same as control')
                        init_image = input_image
                    elif inits is None:
                        debug_log('Control Init image: none')
                        init_image = None
                    elif isinstance(inits[i], str):
                        debug_log(f'Control: init image: {inits[i]}')
                        try:
                            init_image = Image.open(inits[i])
                        except Exception as e:
                            shared.log.error(f'Control: image open failed: path={inits[i]} type=init error={e}')
                            continue
                    else:
                        debug_log(f'Control Init image: {i % len(inits) + 1} of {len(inits)}')
                        init_image = inits[i % len(inits)]
                    if video is not None and index % (video_skip_frames + 1) != 0:
                        index += 1
                        continue
                    index += 1

                    # resize before
                    if resize_mode_before != 0 and resize_name_before != 'None':
                        if selected_scale_tab_before == 1 and input_image is not None:
                            width_before, height_before = int(input_image.width * scale_by_before), int(input_image.height * scale_by_before)
                        if input_image is not None:
                            p.extra_generation_params["Control resize"] = f'{resize_name_before}'
                            debug_log(f'Control resize: op=before image={input_image} width={width_before} height={height_before} mode={resize_mode_before} name={resize_name_before} context="{resize_context_before}"')
                            input_image = images.resize_image(resize_mode_before, input_image, width_before, height_before, resize_name_before, context=resize_context_before)
                    if input_image is not None and init_image is not None and init_image.size != input_image.size:
                        debug_log(f'Control resize init: image={init_image} target={input_image}')
                        init_image = images.resize_image(resize_mode=1, im=init_image, width=input_image.width, height=input_image.height)
                    if input_image is not None and p.override is not None and p.override.size != input_image.size:
                        debug_log(f'Control resize override: image={p.override} target={input_image}')
                        p.override = images.resize_image(resize_mode=1, im=p.override, width=input_image.width, height=input_image.height)
                    if input_image is not None:
                        p.width = input_image.width
                        p.height = input_image.height
                        debug_log(f'Control: input image={input_image}')

                    processed_images = []
                    if mask is not None:
                        p.extra_generation_params["Mask only"] = masking.opts.mask_only if masking.opts.mask_only else None
                        p.extra_generation_params["Mask auto"] = masking.opts.auto_mask if masking.opts.auto_mask != 'None' else None
                        p.extra_generation_params["Mask invert"] = masking.opts.invert if masking.opts.invert else None
                        p.extra_generation_params["Mask blur"] = masking.opts.mask_blur if masking.opts.mask_blur > 0 else None
                        p.extra_generation_params["Mask erode"] = masking.opts.mask_erode if masking.opts.mask_erode > 0 else None
                        p.extra_generation_params["Mask dilate"] = masking.opts.mask_dilate if masking.opts.mask_dilate > 0 else None
                        p.extra_generation_params["Mask model"] = masking.opts.model if masking.opts.model is not None else None
                        masked_image = masking.run_mask(input_image=input_image, input_mask=mask, return_type='Masked', invert=p.inpainting_mask_invert==1) if mask is not None else input_image
                    else:
                        masked_image = input_image
                    for i, process in enumerate(active_process): # list[image]
                        debug_log(f'Control: i={i+1} process="{process.processor_id}" input={masked_image} override={process.override}')
                        processed_image = process(
                            image_input=masked_image,
                            mode='RGB',
                            resize_mode=resize_mode_before,
                            resize_name=resize_name_before,
                            scale_tab=selected_scale_tab_before,
                            scale_by=scale_by_before,
                        )
                        if processed_image is not None:
                            processed_images.append(processed_image)
                        if shared.opts.control_unload_processor and process.processor_id is not None:
                            processors.config[process.processor_id]['dirty'] = True # to force reload
                            process.model = None

                    debug_log(f'Control processed: {len(processed_images)}')
                    if len(processed_images) > 0:
                        try:
                            if len(p.extra_generation_params["Control process"]) == 0:
                                p.extra_generation_params["Control process"] = None
                            else:
                                p.extra_generation_params["Control process"] = ';'.join([p.processor_id for p in active_process if p.processor_id is not None])
                        except Exception:
                            pass
                        if any(img is None for img in processed_images):
                            if is_generator:
                                yield terminate('Attempting process but output is none')
                            return [], '', '', 'Error: output is none'
                        if len(processed_images) > 1 and len(active_process) != len(active_model):
                            processed_image = [np.array(i) for i in processed_images]
                            processed_image = util.blend(processed_image) # blend all processed images into one
                            processed_image = Image.fromarray(processed_image)
                            blended_image = processed_image
                        elif len(processed_images) == 1:
                            processed_image = processed_images
                            blended_image = processed_image[0]
                        else:
                            blended_image = [np.array(i) for i in processed_images]
                            blended_image = util.blend(blended_image) # blend all processed images into one
                            blended_image = Image.fromarray(blended_image)
                        if isinstance(selected_models, list) and len(processed_images) == len(selected_models):
                            debug_log(f'Control: inputs match: input={len(processed_images)} models={len(selected_models)}')
                            p.init_images = processed_images
                        elif isinstance(selected_models, list) and len(processed_images) != len(selected_models):
                            if is_generator:
                                yield terminate(f'Number of inputs does not match: input={len(processed_images)} models={len(selected_models)}')
                            return [], '', '', 'Error: number of inputs does not match'
                        elif selected_models is not None:
                            p.init_images = processed_image
                    else:
                        debug_log('Control processed: using input direct')
                        processed_image = input_image

                    if unit_type == 'reference' and has_models:
                        p.ref_image = p.override or input_image
                        p.task_args.pop('image', None)
                        p.task_args['ref_image'] = p.ref_image
                        debug_log(f'Control: process=None image={p.ref_image}')
                        if p.ref_image is None:
                            if is_generator:
                                yield terminate('Attempting reference mode but image is none')
                            return [], '', '', 'Reference mode without image'
                    elif unit_type == 'controlnet' and has_models:
                        if input_type == 0: # Control only
                            if 'control_image' in possible:
                                p.task_args['control_image'] = [p.init_images] if isinstance(p.init_images, Image.Image) else p.init_images
                            elif 'image' in possible:
                                p.task_args['image'] = [p.init_images] if isinstance(p.init_images, Image.Image) else p.init_images
                            if 'control_mode' in possible:
                                p.task_args['control_mode'] = getattr(p, 'control_mode', None)
                            if 'strength' in possible:
                                p.task_args['strength'] = p.denoising_strength
                            p.init_images = None
                        elif input_type == 1: # Init image same as control
                            p.init_images = [p.override or input_image] * max(1, len(active_model))
                            if 'inpaint_image' in possible: # flex
                                p.task_args['inpaint_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                                p.task_args['inpaint_mask'] = Image.new('L', p.task_args['inpaint_image'].size, int(p.denoising_strength * 255))
                                p.task_args['control_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                                p.task_args['width'] = p.width
                                p.task_args['height'] = p.height
                            elif 'control_image' in possible:
                                p.task_args['control_image'] = p.init_images # switch image and control_image
                            if 'control_mode' in possible:
                                p.task_args['control_mode'] = getattr(p, 'control_mode', None)
                            if 'strength' in possible:
                                p.task_args['strength'] = p.denoising_strength
                        elif input_type == 2: # Separate init image
                            if init_image is None:
                                shared.log.warning('Control: separate init image not provided')
                                init_image = input_image
                            if 'inpaint_image' in possible: # flex
                                p.task_args['inpaint_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                                p.task_args['inpaint_mask'] = Image.new('L', p.task_args['inpaint_image'].size, int(p.denoising_strength * 255))
                                p.task_args['control_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                                p.task_args['width'] = p.width
                                p.task_args['height'] = p.height
                            elif 'control_image' in possible:
                                p.task_args['control_image'] = p.init_images # switch image and control_image
                            if 'control_mode' in possible:
                                p.task_args['control_mode'] = getattr(p, 'control_mode', None)
                            if 'strength' in possible:
                                p.task_args['strength'] = p.denoising_strength
                            p.init_images = [init_image] * len(active_model)
                        if hasattr(shared.sd_model, 'controlnet') and hasattr(p.task_args, 'control_image') and len(p.task_args['control_image']) > 1 and (shared.sd_model.__class__.__name__ == 'StableDiffusionXLControlNetUnionPipeline'): # special case for controlnet-union
                            p.task_args['control_image'] = [[x] for x in p.task_args['control_image']]
                            p.task_args['control_mode'] = [[x] for x in p.task_args['control_mode']]

                    if is_generator:
                        image_txt = f'{blended_image.width}x{blended_image.height}' if blended_image is not None else 'None'
                        msg = f'process | {index} of {frames if video is not None else len(inputs)} | {"Image" if video is None else "Frame"} {image_txt}'
                        debug_log(f'Control yield: {msg}')
                        if is_generator:
                            yield (None, blended_image, f'Control {msg}')
                    t2 += time.time() - t2

                    # determine txt2img, img2img, inpaint pipeline
                    if unit_type == 'reference' and has_models: # special case
                        p.is_control = True
                        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                    elif not has_models: # run in txt2img/img2img/inpaint mode
                        if mask is not None:
                            p.task_args['strength'] = p.denoising_strength
                            p.image_mask = mask
                            p.init_images = input_image if isinstance(input_image, list) else [input_image]
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING)
                        elif processed_image is not None:
                            p.init_images = processed_image if isinstance(processed_image, list) else [processed_image]
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
                        else:
                            p.init_hr(p.scale_by, p.resize_name, force=True)
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                    elif has_models: # actual control
                        p.is_control = True
                        if mask is not None:
                            p.task_args['strength'] = denoising_strength
                            p.image_mask = mask
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # only controlnet supports inpaint
                        if hasattr(p, 'init_images') and p.init_images is not None:
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE) # only controlnet supports img2img
                        else:
                            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                            if hasattr(p, 'init_images') and p.init_images is not None and 'image' in possible:
                                p.task_args['image'] = p.init_images # need to set explicitly for txt2img
                                del p.init_images
                        if unit_type == 'lite':
                            if input_type == 0:
                                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                                shared.sd_model.no_task_switch = True
                            elif input_type == 1:
                                p.init_images = [input_image]
                            elif input_type == 2:
                                if init_image is None:
                                    shared.log.warning('Control: separate init image not provided')
                                    init_image = input_image
                                p.init_images = [init_image]
                            instance.apply(selected_models, processed_image, control_conditioning)
                    if hasattr(p, 'init_images') and p.init_images is None: # delete empty
                        del p.init_images

                    # final check
                    if has_models and shared.sd_model.__class__.__name__ not in unified_models:
                        if unit_type in ['controlnet', 't2i adapter', 'lite', 'xs'] \
                            and p.task_args.get('image', None) is None \
                            and p.task_args.get('control_image', None) is None \
                            and getattr(p, 'init_images', None) is None \
                            and getattr(p, 'image', None) is None:
                            if is_generator:
                                shared.log.debug(f'Control args: {p.task_args}')
                                yield terminate(f'Mode={p.extra_generation_params.get("Control type", None)} input image is none')
                            return [], '', '', 'Error: Input image is none'

                    # resize mask
                    if mask is not None and resize_mode_mask != 0 and resize_name_mask != 'None':
                        if selected_scale_tab_mask == 1:
                            width_mask, height_mask = int(input_image.width * scale_by_mask), int(input_image.height * scale_by_mask)
                        p.width, p.height = width_mask, height_mask
                        debug_log(f'Control resize: op=mask image={mask} width={width_mask} height={height_mask} mode={resize_mode_mask} name={resize_name_mask} context="{resize_context_mask}"')

                    # pipeline
                    output = None
                    script_run = False
                    if pipe is not None: # run new pipeline
                        if not hasattr(pipe, 'restore_pipeline') and video is None:
                            pipe.restore_pipeline = restore_pipeline
                            shared.sd_model.restore_pipeline = restore_pipeline
                        debug_log(f'Control exec pipeline: task={sd_models.get_diffusers_task(pipe)} class={pipe.__class__}')
                        # debug_log(f'Control exec pipeline: p={vars(p)}')
                        # debug_log(f'Control exec pipeline: args={p.task_args} image={p.task_args.get("image", None)} control={p.task_args.get("control_image", None)} mask={p.task_args.get("mask_image", None) or p.image_mask} ref={p.task_args.get("ref_image", None)}')
                        if sd_models.get_diffusers_task(pipe) != sd_models.DiffusersTaskType.TEXT_2_IMAGE: # force vae back to gpu if not in txt2img mode
                            sd_models.move_model(pipe.vae, devices.device)

                        p.scripts = scripts_manager.scripts_control
                        p.script_args = input_script_args or []
                        if len(p.script_args) == 0:
                            script_runner = scripts_manager.scripts_control
                            if not script_runner.scripts:
                                script_runner.initialize_scripts(False)
                            p.script_args = script.init_default_script_args(script_runner)

                        # actual processing
                        if p.is_tile:
                            processed: processing.Processed = tile.run_tiling(p, input_image)
                        if processed is None and p.scripts is not None:
                            processed = p.scripts.run(p, *p.script_args)
                        if processed is None:
                            processed: processing.Processed = processing.process_images(p) # run actual pipeline
                        else:
                            script_run = True

                        # postprocessing
                        if p.scripts is not None:
                            processed = p.scripts.after(p, processed, *p.script_args)
                        output = None
                        if processed is not None:
                            output = processed.images
                            info_txt = [processed.infotext(p, i) for i in range(len(output))]

                        # output = pipe(**vars(p)).images # alternative direct pipe exec call
                    else: # blend all processed images and return
                        output = [processed_image]
                    t3 += time.time() - t3

                    # outputs
                    output = output or []
                    for i, output_image in enumerate(output):
                        if output_image is not None:

                            # resize after
                            is_grid = len(output) == p.batch_size * p.n_iter + 1 and i == 0
                            if selected_scale_tab_after == 1:
                                width_after = int(output_image.width * scale_by_after)
                                height_after = int(output_image.height * scale_by_after)
                            if resize_mode_after != 0 and resize_name_after != 'None' and not is_grid:
                                debug_log(f'Control resize: op=after image={output_image} width={width_after} height={height_after} mode={resize_mode_after} name={resize_name_after} context="{resize_context_after}"')
                                output_image = images.resize_image(resize_mode_after, output_image, width_after, height_after, resize_name_after, context=resize_context_after)

                            output_images.append(output_image)
                            if shared.opts.include_mask and not script_run:
                                if processed_image is not None and isinstance(processed_image, Image.Image):
                                    output_images.append(processed_image)

                            if is_generator:
                                image_txt = f'{output_image.width}x{output_image.height}' if output_image is not None else 'None'
                                if video is not None:
                                    msg = f'Control output | {index} of {frames} skip {video_skip_frames} | Frame {image_txt}'
                                else:
                                    msg = f'Control output | {index} of {len(inputs)} | Image {image_txt}'
                                yield (output_image, blended_image, msg) # result is control_output, proces_output

                if video is not None and frame is not None:
                    status, frame = video.read()
                    if status:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    debug_log(f'Control: video frame={index} frames={frames} status={status} skip={index % (video_skip_frames + 1)} progress={index/frames:.2f}')
                else:
                    status = False

            if video is not None:
                video.release()

            debug_log(f'Control: pipeline units={len(active_model)} process={len(active_process)} time={t3-t0:.2f} init={t1-t0:.2f} proc={t2-t1:.2f} ctrl={t3-t2:.2f} outputs={len(output_images)}')
            process_timer.add('init', t1-t0)
            process_timer.add('proc', t2-t1)
    except Exception as e:
        shared.log.error(f'Control pipeline failed: type={unit_type} units={len(active_model)} error={e}')
        errors.display(e, 'Control')

    if len(output_images) == 0:
        output_images = None
        image_txt = '| Images None'
    else:
        image_txt = ''
        p.init_images = output_images # may be used for hires

    if video_type != 'None' and isinstance(output_images, list):
        p.do_not_save_grid = True # pylint: disable=attribute-defined-outside-init
        output_filename = images.save_video(p, filename=None, images=output_images, video_type=video_type, duration=video_duration, loop=video_loop, pad=video_pad, interpolate=video_interpolate, sync=True)
        if shared.opts.gradio_skip_video:
            output_filename = ''
        image_txt = f'| Frames {len(output_images)} | Size {output_images[0].width}x{output_images[0].height}'

    p.close()
    restore_pipeline()
    debug_log(f'Ready: {image_txt}')

    # Track control operation completion
    if has_models and 'control' in p.ops:
        from modules import pipeline_viz
        pipeline_viz.track_operation_complete('control', p)

    html_txt = f'<p>Ready {image_txt}</p>' if image_txt != '' else ''
    if len(info_txt) > 0:
        html_txt = html_txt + infotext_to_html(info_txt[0])
    if is_generator:
        yield (output_images, blended_image, html_txt, output_filename)
    return (output_images, blended_image, html_txt, output_filename)
