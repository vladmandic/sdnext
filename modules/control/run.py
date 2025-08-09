import os
import sys
from typing import List, Union
import cv2
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
from modules.control.processor import preprocess_image
from modules import devices, shared, errors, processing, images, sd_models, sd_vae, scripts_manager, masking
from modules.processing_class import StableDiffusionProcessingControl
from modules.ui_common import infotext_to_html
from modules.api import script


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
    if (original_pipeline is not None) and (original_pipeline.__class__.__name__ != shared.sd_model.__class__.__name__):
        if debug:
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            shared.log.trace(f'Control restored pipeline: class={shared.sd_model.__class__.__name__} to={original_pipeline.__class__.__name__} fn={fn}')
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


def set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, control_conditioning, control_guidance_start, control_guidance_end, inits=None):
    global pipe, instance # pylint: disable=global-statement
    pipe = None
    if has_models:
        p.ops.append('control')
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
    if (pipe is not None) and (pipe.__class__.__name__ != shared.sd_model.__class__.__name__):
        sd_models.copy_diffuser_options(pipe, shared.sd_model) # copy options from original pipeline
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


def init_units(units: List[unit.Unit]):
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
    global pipe, original_pipeline # pylint: disable=global-statement

    unit.current = units
    debug_log(f'Control: type={unit_type} input={inputs} init={inits} type={input_type}')
    init_units(units)
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
        denoising_strength = denoising_strength,
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
        # resize
        width = width_before,
        height = height_before,
        width_before = width_before,
        width_after = width_after,
        width_mask = width_mask,
        height_before = height_before,
        height_after = height_after,
        height_mask = height_mask,
        resize_name_before = resize_name_before,
        resize_name_after = resize_name_after,
        resize_name_mask = resize_name_mask,
        resize_mode_before = resize_mode_before if resize_name_before != 'None' and inputs is not None and len(inputs) > 0 else 0,
        resize_mode_after = resize_mode_after if resize_name_after != 'None' else 0,
        resize_mode_mask = resize_mode_mask if resize_name_mask != 'None' else 0,
        resize_context_before = resize_context_before,
        resize_context_after = resize_context_after,
        resize_context_mask = resize_context_mask,
        selected_scale_tab_before = selected_scale_tab_before,
        selected_scale_tab_after = selected_scale_tab_after,
        selected_scale_tab_mask = selected_scale_tab_mask,
        scale_by_before = scale_by_before,
        scale_by_after = scale_by_after,
        scale_by_mask = scale_by_mask,
        # hires
        enable_hr = enable_hr,
        hr_sampler_name = processing.get_sampler_name(hr_sampler_index),
        hr_denoising_strength = hr_denoising_strength,
        hr_resize_mode = hr_resize_mode if enable_hr else 0,
        hr_resize_context = hr_resize_context if enable_hr else 'None',
        hr_upscaler = hr_upscaler if enable_hr else None,
        hr_force = hr_force,
        hr_second_pass_steps = hr_second_pass_steps if enable_hr else 0,
        hr_scale = hr_scale if enable_hr else 1.0,
        hr_resize_x = hr_resize_x if enable_hr else 0,
        hr_resize_y = hr_resize_y if enable_hr else 0,
        # refiner
        refiner_steps = refiner_steps,
        refiner_start = refiner_start,
        refiner_prompt = refiner_prompt,
        refiner_negative = refiner_negative,
        # detailer
        detailer_enabled = detailer_enabled,
        detailer_prompt = detailer_prompt,
        detailer_negative = detailer_negative,
        detailer_steps = detailer_steps,
        detailer_strength = detailer_strength,
        # inpaint
        inpaint_full_res = masking.opts.mask_only,
        inpainting_mask_invert = 1 if masking.opts.invert else 0,
        # hdr
        hdr_mode=hdr_mode, hdr_brightness=hdr_brightness, hdr_color=hdr_color, hdr_sharpen=hdr_sharpen, hdr_clamp=hdr_clamp,
        hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold, hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundary=hdr_max_boundary, hdr_color_picker=hdr_color_picker, hdr_tint_ratio=hdr_tint_ratio,
        # path
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_control_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_control_grids,
    )
    p.state = state
    p.is_tile = False
    p.orig_init_images = inputs

    # TODO modernui: monkey-patch for missing tabs.select event
    if p.selected_scale_tab_before == 0 and p.resize_name_before != 'None' and p.scale_by_before != 1 and inputs is not None and len(inputs) > 0:
        shared.log.debug('Control: override resize mode=before')
        p.selected_scale_tab_before = 1
    if p.selected_scale_tab_after == 0 and p.resize_name_after != 'None' and p.scale_by_after != 1:
        shared.log.debug('Control: override resize mode=after')
        p.selected_scale_tab_after = 1
    if p.selected_scale_tab_mask == 0 and p.resize_name_mask != 'None' and p.scale_by_mask != 1:
        shared.log.debug('Control: override resize mode=mask')
        p.selected_scale_tab_mask = 1

    # hires/refine defined outside of main init
    vae_scale_factor = sd_vae.get_vae_scale_factor()
    if p.enable_hr and (p.hr_resize_x == 0 or p.hr_resize_y == 0):
        p.hr_upscale_to_x, p.hr_upscale_to_y = vae_scale_factor * int(p.width_before * p.hr_scale / vae_scale_factor), vae_scale_factor * int(p.height_before * p.hr_scale / vae_scale_factor)
    elif p.enable_hr and (p.hr_upscale_to_x == 0 or p.hr_upscale_to_y == 0):
        p.hr_upscale_to_x, p.hr_upscale_to_y = 8 * int(p.hr_resize_x / vae_scale_factor), vae_scale_factor * int(p.hr_resize_y / vae_scale_factor)

    global p_extra_args # pylint: disable=global-statement
    for k, v in p_extra_args.items():
        setattr(p, k, v)
    p_extra_args = {}

    if shared.sd_model is None:
        shared.log.warning('Aborted: op=control model not loaded')
        return [], '', '', 'Error: model not loaded'

    unit_type = unit_type.strip().lower() if unit_type is not None else ''
    active_process, active_model, active_strength, active_start, active_end = check_active(p, unit_type, units)
    has_models, selected_models, control_conditioning, control_guidance_start, control_guidance_end = check_enabled(p, unit_type, units, active_model, active_strength, active_start, active_end)

    image_txt = ''
    info_txt = []

    p.is_tile = p.is_tile and has_models
    if is_unified_model():
        p.init_images = inputs

    pipe = set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, control_conditioning, control_guidance_start, control_guidance_end, inits)
    debug_log(f'Control pipeline: class={pipe.__class__.__name__} args={vars(p)}')
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
                processed_image = None
                if frame is not None:
                    inputs = [Image.fromarray(frame)] # cv2 to pil
                for i, input_image in enumerate(inputs):
                    if input_image is not None:
                        p.ops.append('img2img')
                    if pipe is None: # pipe may have been reset externally
                        pipe = set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, control_conditioning, control_guidance_start, control_guidance_end, inits)
                        debug_log(f'Control pipeline reinit: class={pipe.__class__.__name__}')
                    pipe.restore_pipeline = restore_pipeline
                    shared.sd_model.restore_pipeline = restore_pipeline
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
                            input_image = Image.open(input_image)
                        except Exception as e:
                            shared.log.error(f'Control: image open failed: path={input_image} type=control error={e}')
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

                    processed_image, blended_image = preprocess_image(p, pipe, input_image, init_image, mask, input_type, unit_type, active_process, active_model, selected_models, has_models)
                    if is_generator:
                        yield (None, blended_image, '') # result is control_output, proces_output

                    # final check
                    if has_models:
                        if shared.sd_model.__class__.__name__ not in unified_models:
                            if unit_type in ['controlnet', 't2i adapter', 'lite', 'xs'] \
                                and p.task_args.get('image', None) is None \
                                and p.task_args.get('control_image', None) is None \
                                and getattr(p, 'init_images', None) is None \
                                and getattr(p, 'image', None) is None:
                                if is_generator:
                                    shared.log.debug(f'Control args: {p.task_args}')
                                    yield terminate(f'Mode={p.extra_generation_params.get("Control type", None)} input image is none')
                                return [], '', '', 'Error: Input image is none'
                        if unit_type == 'lite':
                            instance.apply(selected_models, processed_image, control_conditioning)

                    # what are we doing?
                    if 'control' in p.ops:
                        p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_control_samples
                    elif 'img2img' in p.ops:
                        p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_img2img_samples
                    elif 'txt2img' in p.ops:
                        p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples
                    else: # fallback to txt2img
                        p.outpath_samples = shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples

                    # pipeline
                    output = None
                    script_run = False
                    if pipe is not None: # run new pipeline
                        debug_log(f'Control exec pipeline: task={sd_models.get_diffusers_task(pipe)} class={pipe.__class__}')
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
                        processed: processing.Processed = None
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
                        output = processed_image

                    # outputs
                    output = output or []
                    for _i, output_image in enumerate(output):
                        if output_image is not None:
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

            debug_log(f'Control: pipeline units={len(active_model)} process={len(active_process)} outputs={len(output_images)}')
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

    html_txt = f'<p>Ready {image_txt}</p>' if image_txt != '' else ''
    if len(info_txt) > 0:
        html_txt = html_txt + infotext_to_html(info_txt[0])
    if is_generator:
        yield (output_images, blended_image, html_txt, output_filename)
    return (output_images, blended_image, html_txt, output_filename)
