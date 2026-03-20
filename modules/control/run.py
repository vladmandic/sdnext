import os
import sys
import cv2
from PIL import Image
from modules.logger import log
from modules import devices, shared, errors, processing, sd_models, sd_vae, scripts_manager, masking
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
from modules.processing_class import StableDiffusionProcessingControl
from modules.ui_common import infotext_to_html
from modules.api import script
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.paths import resolve_output_path


debug = os.environ.get('SD_CONTROL_DEBUG', None) is not None
debug_log = log.trace if debug else lambda *args, **kwargs: None
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
            log.trace(f'Control restored pipeline: class={shared.sd_model.__class__.__name__} to={original_pipeline.__class__.__name__} fn={fn}')
        shared.sd_model = original_pipeline
    pipe = None
    instance = None
    devices.torch_gc()


def terminate(msg):
    restore_pipeline()
    log.error(f'Control terminated: {msg}')
    return msg


def is_unified_model():
    return shared.sd_model.__class__.__name__ in unified_models


def has_inputs(inputs):
    current = inputs or []
    current = current if isinstance(current, list) else [current]
    current = [input for input in current if input is not None]
    if current is None or len(current) == 0:
        return False
    return True


def set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, active_units, control_conditioning, control_guidance_start, control_guidance_end, inits=None, inputs=None):
    global pipe, instance # pylint: disable=global-statement
    pipe = None
    if has_models and not has_inputs(inits) and not has_inputs(inputs):
        if not any(has_inputs(u.override) for u in active_units if u.enabled): # check overrides
            log.error('Control: no input images')
            return pipe
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
            log.warning('Control: T2I-Adapter does not support separate init image')
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
            log.warning('Control: ControlNet-XS does not support separate init image')
    elif unit_type == 'lite' and has_models:
        p.extra_generation_params["Control type"] = 'ControlLLLite'
        p.controlnet_conditioning_scale = control_conditioning
        instance = lite.ControlLLitePipeline(shared.sd_model)
        pipe = instance.pipeline
        if inits is not None:
            log.warning('Control: ControlLLLite does not support separate init image')
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
            log.warning('Control: ControlNet-XS does not support separate init image')
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
    active_process: list[processors.Processor] = [] # all active preprocessors
    active_model: list[controlnet.ControlNet | xs.ControlNetXS | t2iadapter.Adapter] = [] # all active models
    active_strength: list[float] = [] # strength factors for all active models
    active_start: list[float] = [] # start step for all active models
    active_end: list[float] = [] # end step for all active models
    active_units: list[unit.Unit] = [] # all active units
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
            active_units.append(u)
            log.debug(f'Control T2I-Adapter unit: i={num_units} process="{u.process.processor_id}" model="{u.adapter.model_id}" strength={u.strength} factor={u.factor}')
        elif unit_type == 'controlnet' and (u.controlnet.model is not None or is_unified_model()):
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_start.append(float(u.start))
            active_end.append(float(u.end))
            p.guess_mode = u.guess
            active_units.append(u)
            if isinstance(u.mode, str):
                if not hasattr(p, 'control_mode'):
                    p.control_mode = []
                p.control_mode.append(u.choices.index(u.mode) if u.mode in u.choices else 0)
                p.is_tile = p.is_tile or 'tile' in u.mode.lower()
                p.control_tile = u.tile
                p.extra_generation_params["Control mode"] = u.mode
            log.debug(f'Control unit: i={num_units} type=ControlNet process="{u.process.processor_id}" model="{u.controlnet.model_id}" strength={u.strength} guess={u.guess} start={u.start} end={u.end} mode={u.mode}')
        elif unit_type == 'xs' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_start.append(float(u.start))
            active_end.append(float(u.end))
            active_units.append(u)
            log.debug(f'Control unit: i={num_units} type=ControlNetXS process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'lite' and u.controlnet.model is not None:
            active_process.append(u.process)
            active_model.append(u.controlnet)
            active_strength.append(float(u.strength))
            active_units.append(u)
            log.debug(f'Control unit: i={num_units} type=ControlLLite process={u.process.processor_id} model={u.controlnet.model_id} strength={u.strength} guess={u.guess} start={u.start} end={u.end}')
        elif unit_type == 'reference':
            p.override = u.override
            p.attention = u.attention
            p.query_weight = float(u.query_weight)
            p.adain_weight = float(u.adain_weight)
            p.fidelity = u.fidelity
            active_units.append(u)
            log.debug('Control Reference unit')
        else:
            if u.process.processor_id is not None:
                active_process.append(u.process)
                active_units.append(u)
                log.debug(f'Control unit: i={num_units} type=Process process={u.process.processor_id}')
            active_strength.append(float(u.strength))
    debug_log(f'Control active: process={len(active_process)} model={len(active_model)}')
    return active_process, active_model, active_strength, active_start, active_end, active_units


def check_enabled(p, unit_type, units, active_model, active_strength, active_start, active_end):
    has_models = False
    selected_models: list[controlnet.ControlNetModel | xs.ControlNetXSModel | t2iadapter.AdapterModel] = None
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


def init_units(units: list[unit.Unit]):
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
                units: list[unit.Unit] = None, inputs: list[Image.Image] = None, inits: list[Image.Image] = None, mask: Image.Image = None, unit_type: str = None, is_generator: bool = True,
                input_type: int = 0,
                prompt: str = '', negative_prompt: str = '', styles: list[str] = None,
                steps: int = 20, sampler_index: int = None,
                seed: int = -1, subseed: int = -1, subseed_strength: float = 0, seed_resize_from_h: int = -1, seed_resize_from_w: int = -1,
                guidance_name: str = 'Default', guidance_scale: float = 6.0, guidance_rescale: float = 0.0, guidance_start: float = 0.0, guidance_stop: float = 1.0,
                cfg_scale: float = 6.0, clip_skip: float = 1.0, image_cfg_scale: float = 6.0, diffusers_guidance_rescale: float = 0.7, pag_scale: float = 0.0, pag_adaptive: float = 0.5, cfg_end: float = 1.0,
                vae_type: str = 'Full', tiling: bool = False, hidiffusion: bool = False,
                detailer_enabled: bool = False, detailer_prompt: str = '', detailer_negative: str = '', detailer_steps: int = 10, detailer_strength: float = 0.3, detailer_resolution: int = 1024,
                hdr_mode: int = 0, hdr_brightness: float = 0, hdr_color: float = 0, hdr_sharpen: float = 0, hdr_clamp: bool = False, hdr_boundary: float = 4.0, hdr_threshold: float = 0.95,
                hdr_maximize: bool = False, hdr_max_center: float = 0.6, hdr_max_boundary: float = 1.0, hdr_color_picker: str = None, hdr_tint_ratio: float = 0, hdr_apply_hires: bool = True,
                grading_brightness: float = 0.0, grading_contrast: float = 0.0, grading_saturation: float = 0.0, grading_hue: float = 0.0,
                grading_gamma: float = 1.0, grading_sharpness: float = 0.0, grading_color_temp: float = 6500,
                grading_shadows: float = 0.0, grading_midtones: float = 0.0, grading_highlights: float = 0.0,
                grading_clahe_clip: float = 0.0, grading_clahe_grid: int = 8,
                grading_shadows_tint: str = "#000000", grading_highlights_tint: str = "#ffffff", grading_split_tone_balance: float = 0.5,
                grading_vignette: float = 0.0, grading_grain: float = 0.0,
                grading_lut_file: str = "", grading_lut_strength: float = 1.0,
                resize_mode_before: int = 0, resize_name_before: str = 'None', resize_context_before: str = 'None', width_before: int = 512, height_before: int = 512, scale_by_before: float = 1.0, selected_scale_tab_before: int = 0,
                resize_mode_after: int = 0, resize_name_after: str = 'None', resize_context_after: str = 'None', width_after: int = 0, height_after: int = 0, scale_by_after: float = 1.0, selected_scale_tab_after: int = 0,
                resize_mode_mask: int = 0, resize_name_mask: str = 'None', resize_context_mask: str = 'None', width_mask: int = 0, height_mask: int = 0, scale_by_mask: float = 1.0, selected_scale_tab_mask: int = 0,
                denoising_strength: float = 0.3, batch_count: int = 1, batch_size: int = 1,
                enable_hr: bool = False, hr_sampler_index: int = None, hr_denoising_strength: float = 0.0, hr_resize_mode: int = 0, hr_resize_context: str = 'None', hr_upscaler: str = None, hr_force: bool = False, hr_second_pass_steps: int = 20,
                hr_scale: float = 1.0, hr_resize_x: int = 0, hr_resize_y: int = 0, refiner_steps: int = 5, refiner_start: float = 0.0, refiner_prompt: str = '', refiner_negative: str = '',
                video_skip_frames: int = 0, video_type: str = 'None', video_duration: float = 2.0, video_loop: bool = False, video_pad: int = 0, video_interpolate: int = 0,
                override_script_name: str = None, override_script_args = None, extra: dict = None,
                *input_script_args,
                # API-only params (keyword-only, not wired to Gradio)
                detailer_segmentation: bool = None, detailer_include_detections: bool = None, detailer_merge: bool = None, detailer_sort: bool = None, detailer_classes: str = None,
                detailer_conf: float = None, detailer_iou: float = None, detailer_max: int = None,
                detailer_min_size: float = None, detailer_max_size: float = None,
                detailer_blur: int = None, detailer_padding: int = None,
                detailer_sigma_adjust: float = None, detailer_sigma_adjust_max: float = None,
                detailer_models: list = None, detailer_augment: bool = None,
                img2img_color_correction: bool = None, color_correction_method: str = None, img2img_background_color: str = None,
                img2img_fix_steps: bool = None, mask_apply_overlay: bool = None,
                include_mask: bool = None, inpainting_mask_weight: float = None,
                # output and saving
                samples_save: bool = None, samples_format: str = None,
                save_images_before_highres_fix: bool = None, save_images_before_refiner: bool = None,
                save_images_before_detailer: bool = None, save_images_before_color_correction: bool = None,
                grid_save: bool = None, grid_format: str = None, return_grid: bool = None,
                save_mask: bool = None, save_mask_composite: bool = None,
                return_mask: bool = None, return_mask_composite: bool = None,
                keep_incomplete: bool = None, image_metadata: bool = None, jpeg_quality: int = None,
                # scheduler/noise overrides
                schedulers_prediction_type: str = None, schedulers_beta_schedule: str = None, schedulers_timesteps: str = None,
                schedulers_sigma: str = None, schedulers_use_thresholding: bool = None, schedulers_use_loworder: bool = None,
                schedulers_solver_order: int = None, uni_pc_variant: str = None, schedulers_beta_start: float = None,
                schedulers_beta_end: float = None, schedulers_shift: float = None, schedulers_dynamic_shift: bool = None,
                schedulers_base_shift: float = None, schedulers_max_shift: float = None, schedulers_rescale_betas: bool = None,
                schedulers_timestep_spacing: str = None, schedulers_timesteps_range: int = None,
                schedulers_sigma_adjust: float = None, schedulers_sigma_adjust_min: float = None, schedulers_sigma_adjust_max: float = None,
                scheduler_eta: float = None, eta_noise_seed_delta: int = None, enable_batch_seeds: bool = None,
                diffusers_generator_device: str = None, nan_skip: bool = None,
                sequential_seed: bool = None,
                # prompt/attention overrides
                prompt_attention: str = None, prompt_mean_norm: bool = None, diffusers_zeros_prompt_pad: bool = None,
                te_pooled_embeds: bool = None, lora_apply_te: bool = None, te_complex_human_instruction: str = None, te_use_mask: bool = None,
                # generation modifier overrides (hijack)
                freeu_enabled: bool = None, freeu_b1: float = None, freeu_b2: float = None, freeu_s1: float = None, freeu_s2: float = None,
                hypertile_unet_enabled: bool = None, hypertile_hires_only: bool = None, hypertile_unet_tile: int = None, hypertile_unet_min_tile: int = None,
                hypertile_unet_swap_size: int = None, hypertile_unet_depth: int = None,
                hypertile_vae_enabled: bool = None, hypertile_vae_tile: int = None, hypertile_vae_swap_size: int = None,
                teacache_enabled: bool = None, teacache_thresh: float = None,
                token_merging_method: str = None, tome_ratio: float = None, todo_ratio: float = None,
                # lora behavior
                lora_fuse_native: bool = None, lora_fuse_diffusers: bool = None,
                lora_force_reload: bool = None, extra_networks_default_multiplier: float = None,
                lora_apply_tags: int = None,
        ):
    if override_script_args is None:
        override_script_args = []
    if extra is None:
        extra = {}
    if styles is None:
        styles = []
    if inits is None:
        inits = []
    if inputs is None:
        inputs = []
    if units is None:
        units = []
    global pipe, original_pipeline # pylint: disable=global-statement
    if 'refine' in state:
        enable_hr = True

    unit.current = units
    debug_log(f'Control: type={unit_type} input={inputs} init={inits} type={input_type}')
    init_units(units)
    if inputs is None or (type(inputs) is list and len(inputs) == 0):
        inputs = [None]
    output_images: list[Image.Image] = [] # output images
    processed_image: Image.Image = None # last processed image
    if mask is not None and input_type == 0:
        input_type = 1 # inpaint always requires control_image

    if sampler_index is None:
        log.warning('Sampler: invalid')
        sampler_index = 0
    if hr_sampler_index is None:
        hr_sampler_index = sampler_index
    if isinstance(extra, list):
        extra = create_override_settings_dict(extra)

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
        # modular guidance
        guidance_name = guidance_name,
        guidance_scale = guidance_scale,
        guidance_rescale = guidance_rescale,
        guidance_start = guidance_start,
        guidance_stop = guidance_stop,
        # legacy guidance
        cfg_scale = cfg_scale,
        cfg_end = cfg_end,
        clip_skip = clip_skip,
        image_cfg_scale = image_cfg_scale,
        diffusers_guidance_rescale = diffusers_guidance_rescale,
        pag_scale = pag_scale,
        pag_adaptive = pag_adaptive,
        # advanced
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
        detailer_resolution = detailer_resolution,
        detailer_segmentation = detailer_segmentation,
        detailer_include_detections = detailer_include_detections,
        detailer_merge = detailer_merge,
        detailer_sort = detailer_sort,
        detailer_classes = detailer_classes,
        detailer_conf=detailer_conf, detailer_iou=detailer_iou, detailer_max=detailer_max,
        detailer_min_size=detailer_min_size, detailer_max_size=detailer_max_size,
        detailer_blur=detailer_blur, detailer_padding=detailer_padding,
        detailer_sigma_adjust=detailer_sigma_adjust, detailer_sigma_adjust_max=detailer_sigma_adjust_max,
        detailer_models=detailer_models, detailer_augment=detailer_augment,
        # img2img and mask
        img2img_color_correction=img2img_color_correction, color_correction_method=color_correction_method, img2img_background_color=img2img_background_color,
        img2img_fix_steps=img2img_fix_steps, mask_apply_overlay=mask_apply_overlay,
        include_mask=include_mask, inpainting_mask_weight=inpainting_mask_weight,
        # output and saving
        samples_save=samples_save, samples_format=samples_format,
        save_images_before_highres_fix=save_images_before_highres_fix, save_images_before_refiner=save_images_before_refiner,
        save_images_before_detailer=save_images_before_detailer, save_images_before_color_correction=save_images_before_color_correction,
        grid_save=grid_save, grid_format=grid_format, return_grid=return_grid,
        save_mask=save_mask, save_mask_composite=save_mask_composite,
        return_mask=return_mask, return_mask_composite=return_mask_composite,
        keep_incomplete=keep_incomplete, image_metadata=image_metadata, jpeg_quality=jpeg_quality,
        # inpaint
        inpaint_full_res = masking.opts.mask_only,
        inpainting_mask_invert = 1 if masking.opts.invert else 0,
        # hdr
        hdr_mode=hdr_mode, hdr_brightness=hdr_brightness, hdr_color=hdr_color, hdr_sharpen=hdr_sharpen, hdr_clamp=hdr_clamp,
        hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold, hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundary=hdr_max_boundary, hdr_color_picker=hdr_color_picker, hdr_tint_ratio=hdr_tint_ratio, hdr_apply_hires=hdr_apply_hires,
        # grading
        grading_brightness=grading_brightness, grading_contrast=grading_contrast, grading_saturation=grading_saturation, grading_hue=grading_hue,
        grading_gamma=grading_gamma, grading_sharpness=grading_sharpness, grading_color_temp=grading_color_temp,
        grading_shadows=grading_shadows, grading_midtones=grading_midtones, grading_highlights=grading_highlights,
        grading_clahe_clip=grading_clahe_clip, grading_clahe_grid=grading_clahe_grid,
        grading_shadows_tint=grading_shadows_tint, grading_highlights_tint=grading_highlights_tint, grading_split_tone_balance=grading_split_tone_balance,
        grading_vignette=grading_vignette, grading_grain=grading_grain,
        grading_lut_file=grading_lut_file.name if hasattr(grading_lut_file, 'name') else (grading_lut_file or ''), grading_lut_strength=grading_lut_strength,
        # path
        outpath_samples=resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_control_samples),
        outpath_grids=resolve_output_path(shared.opts.outdir_grids, shared.opts.outdir_control_grids),
        # scheduler/noise overrides
        schedulers_prediction_type=schedulers_prediction_type, schedulers_beta_schedule=schedulers_beta_schedule,
        schedulers_timesteps=schedulers_timesteps, schedulers_sigma=schedulers_sigma,
        schedulers_use_thresholding=schedulers_use_thresholding, schedulers_use_loworder=schedulers_use_loworder,
        schedulers_solver_order=schedulers_solver_order, uni_pc_variant=uni_pc_variant,
        schedulers_beta_start=schedulers_beta_start, schedulers_beta_end=schedulers_beta_end,
        schedulers_shift=schedulers_shift, schedulers_dynamic_shift=schedulers_dynamic_shift,
        schedulers_base_shift=schedulers_base_shift, schedulers_max_shift=schedulers_max_shift,
        schedulers_rescale_betas=schedulers_rescale_betas, schedulers_timestep_spacing=schedulers_timestep_spacing,
        schedulers_timesteps_range=schedulers_timesteps_range,
        schedulers_sigma_adjust=schedulers_sigma_adjust, schedulers_sigma_adjust_min=schedulers_sigma_adjust_min,
        schedulers_sigma_adjust_max=schedulers_sigma_adjust_max,
        scheduler_eta=scheduler_eta, eta_noise_seed_delta=eta_noise_seed_delta,
        enable_batch_seeds=enable_batch_seeds, diffusers_generator_device=diffusers_generator_device, nan_skip=nan_skip,
        sequential_seed=sequential_seed,
        # prompt/attention overrides
        prompt_attention=prompt_attention, prompt_mean_norm=prompt_mean_norm,
        diffusers_zeros_prompt_pad=diffusers_zeros_prompt_pad, te_pooled_embeds=te_pooled_embeds,
        lora_apply_te=lora_apply_te, te_complex_human_instruction=te_complex_human_instruction, te_use_mask=te_use_mask,
        # generation modifier overrides (hijack)
        freeu_enabled=freeu_enabled, freeu_b1=freeu_b1, freeu_b2=freeu_b2, freeu_s1=freeu_s1, freeu_s2=freeu_s2,
        hypertile_unet_enabled=hypertile_unet_enabled, hypertile_hires_only=hypertile_hires_only,
        hypertile_unet_tile=hypertile_unet_tile, hypertile_unet_min_tile=hypertile_unet_min_tile,
        hypertile_unet_swap_size=hypertile_unet_swap_size, hypertile_unet_depth=hypertile_unet_depth,
        hypertile_vae_enabled=hypertile_vae_enabled, hypertile_vae_tile=hypertile_vae_tile, hypertile_vae_swap_size=hypertile_vae_swap_size,
        teacache_enabled=teacache_enabled, teacache_thresh=teacache_thresh,
        token_merging_method=token_merging_method, tome_ratio=tome_ratio, todo_ratio=todo_ratio,
        # lora behavior
        lora_fuse_native=lora_fuse_native, lora_fuse_diffusers=lora_fuse_diffusers,
        lora_force_reload=lora_force_reload, extra_networks_default_multiplier=extra_networks_default_multiplier,
        lora_apply_tags=lora_apply_tags,
        # overrides
        override_settings=extra
    )

    p.state = state
    p.is_tile = False
    p.init_control = inits or []
    p.orig_init_images = inputs

    # TODO modernui: monkey-patch for missing tabs.select event
    if p.selected_scale_tab_before == 0 and p.resize_name_before != 'None' and p.scale_by_before != 1 and inputs is not None and len(inputs) > 0:
        log.debug('Control: override resize mode=before')
        p.selected_scale_tab_before = 1
    if p.selected_scale_tab_after == 0 and p.resize_name_after != 'None' and p.scale_by_after != 1:
        log.debug('Control: override resize mode=after')
        p.selected_scale_tab_after = 1
    if p.selected_scale_tab_mask == 0 and p.resize_name_mask != 'None' and p.scale_by_mask != 1:
        log.debug('Control: override resize mode=mask')
        p.selected_scale_tab_mask = 1

    # hires/refine defined outside of main init
    vae_scale_factor = sd_vae.get_vae_scale_factor()
    if p.enable_hr and (p.hr_resize_x == 0 or p.hr_resize_y == 0):
        p.hr_upscale_to_x, p.hr_upscale_to_y = int(vae_scale_factor * int(p.width_before * p.hr_scale / vae_scale_factor)), int(vae_scale_factor * int(p.height_before * p.hr_scale / vae_scale_factor))
    elif p.enable_hr and (p.hr_upscale_to_x == 0 or p.hr_upscale_to_y == 0):
        p.hr_upscale_to_x, p.hr_upscale_to_y = 8 * int(p.hr_resize_x / vae_scale_factor), int(vae_scale_factor * int(p.hr_resize_y / vae_scale_factor))

    global p_extra_args # pylint: disable=global-statement
    for k, v in p_extra_args.items():
        setattr(p, k, v)
    p_extra_args = {}

    if shared.sd_model is None: # triggers load if not loaded
        log.warning('Aborted: op=generate model not loaded')
        return [], '', '', 'Error: model not loaded'

    unit_type = unit_type.strip().lower() if unit_type is not None else ''
    active_process, active_model, active_strength, active_start, active_end, active_units = check_active(p, unit_type, units)
    has_models, selected_models, control_conditioning, control_guidance_start, control_guidance_end = check_enabled(p, unit_type, units, active_model, active_strength, active_start, active_end)

    image_txt = ''
    info_txt = []

    p.is_tile = p.is_tile and has_models
    if is_unified_model():
        p.init_images = inputs

    pipe = set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, active_units, control_conditioning, control_guidance_start, control_guidance_end, inits, inputs)
    debug_log(f'Control pipeline: class={pipe.__class__.__name__} args={vars(p)}')
    status = True
    frame = None
    video = None
    output_filename = None
    index = 0
    frames = 0
    blended_image = None
    cap = None

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
            if isinstance(inputs, str) and os.path.exists(inputs): # only video, the rest is a list
                if input_type == 2: # separate init image
                    if isinstance(inits, str) and inits != inputs:
                        log.warning('Control: separate init video not support for video input')
                        input_type = 1
                try:
                    cap = cv2.VideoCapture(inputs)
                    if not cap.isOpened():
                        if is_generator:
                            yield terminate(f'Video open failed: path={inputs}')
                        return terminate(f'Video open failed: path={inputs}')
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    codec = util.decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
                    status, frame = cap.read()
                    if status:
                        shared.state.frame_count = 1 + frames // (video_skip_frames + 1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    log.debug(f'Control: input video: path={inputs} frames={frames} fps={fps} size={w}x{h} codec={codec}')
                except Exception as e:
                    if is_generator:
                        yield terminate(f'Video open failed: path={inputs} {e}')
                    return terminate(f'Video open failed: path={inputs} {e}')

            while status:
                processed_image = None
                if frame is not None:
                    inputs = [Image.fromarray(frame)] # cv2 to pil
                for i, input_image in enumerate(inputs):
                    if input_image is not None:
                        p.ops.append('img2img')
                    if pipe is None: # pipe may have been reset externally
                        if video is None:
                            break # non-video: pipeline was consumed, no need to re-process remaining inputs
                        pipe = set_pipe(p, has_models, unit_type, selected_models, active_model, active_strength, active_units, control_conditioning, control_guidance_start, control_guidance_end, inits)
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
                        return terminate('Interrupted')
                    # get input
                    if isinstance(input_image, str) and os.path.exists(input_image):
                        try:
                            input_image = Image.open(input_image)
                        except Exception as e:
                            log.error(f'Control: image open failed: path={input_image} type=control error={e}')
                            continue
                    # match init input
                    if input_type == 1:
                        debug_log('Control Init image: same as control')
                        init_image = input_image
                    elif inits is None or len(inits) == 0:
                        debug_log('Control Init image: none')
                        init_image = None
                    elif len(inits) > i and isinstance(inits[i], str):
                        debug_log(f'Control: init image: {inits[i]}')
                        try:
                            init_image = Image.open(inits[i])
                        except Exception as e:
                            log.error(f'Control: image open failed: path={inits[i]} type=init error={e}')
                            continue
                    elif len(inits) > i:
                        debug_log(f'Control Init image: {i % len(inits) + 1} of {len(inits)}')
                        init_image = inits[i % len(inits)]
                    else:
                        init_image = None
                    if video is not None and index % (video_skip_frames + 1) != 0:
                        index += 1
                        continue
                    index += 1

                    processed_image, blended_image = preprocess_image(p, pipe, input_image, init_image, mask, input_type, unit_type, active_process, active_model, selected_models, has_models, active_units)
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
                                    yield terminate(f'Mode={p.extra_generation_params.get("Control type", None)} input image is none')
                                return terminate(f'Mode={p.extra_generation_params.get("Control type", None)} input image is none')
                        if unit_type == 'lite':
                            instance.apply(selected_models, processed_image, control_conditioning)

                    # what are we doing?
                    if 'control' in p.ops:
                        p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_control_samples)
                    elif 'img2img' in p.ops:
                        p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_img2img_samples)
                    elif 'txt2img' in p.ops:
                        p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_txt2img_samples)
                    else: # fallback to txt2img
                        p.outpath_samples = resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_txt2img_samples)

                    # pipeline
                    output = None
                    script_run = False
                    if pipe is not None: # run new pipeline
                        debug_log(f'Control exec pipeline: task={sd_models.get_diffusers_task(pipe)} class={pipe.__class__}')
                        if sd_models.get_diffusers_task(pipe) != sd_models.DiffusersTaskType.TEXT_2_IMAGE: # force vae back to gpu if not in txt2img mode
                            sd_models.move_model(pipe.vae, devices.device)

                        # init scripts
                        p.scripts = scripts_manager.scripts_control
                        p.script_args = input_script_args or []
                        if len(p.script_args) == 0:
                            if not p.scripts:
                                p.scripts.initialize_scripts(False)
                            p.script_args = script.init_default_script_args(p.scripts)

                        # init override scripts
                        if override_script_name and override_script_args and len(override_script_name) > 0:
                            selectable_scripts, selectable_script_idx = script.get_selectable_script(override_script_name, p.scripts)
                            if selectable_scripts:
                                for idx in range(len(override_script_args)):
                                    p.script_args[selectable_scripts.args_from + idx] = override_script_args[idx]
                                p.script_args[0] = selectable_script_idx + 1

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
                        if processed is not None and processed.images is not None:
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
                            _inc_mask = getattr(p, 'include_mask', None)
                            if (_inc_mask if _inc_mask is not None else shared.opts.include_mask) and not script_run:
                                if processed_image is not None and isinstance(processed_image, Image.Image):
                                    output_images.append(processed_image)

                            if is_generator and frame is not None and video is not None:
                                image_txt = f'{output_image.width}x{output_image.height}' if output_image is not None else 'None'
                                msg = f'Control output | {index} of {frames} skip {video_skip_frames} | Frame {image_txt}'
                                yield (output_image, blended_image, msg) # result is control_output, proces_output

                if video is not None and frame is not None:
                    status, frame = video.read()
                    if status:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    debug_log(f'Control: video frame={index} frames={frames} status={status} skip={index % (video_skip_frames + 1)} progress={index/frames:.2f}')
                else:
                    status = False

            if cap is not None:
                cap.release()

            debug_log(f'Control: pipeline units={len(active_model)} process={len(active_process)} outputs={len(output_images)}')
    except Exception as e:
        log.error(f'Control: type={unit_type} units={len(active_model)} {e}')
        errors.display(e, 'Control')

    if len(output_images) == 0:
        output_images = None
        image_txt = '| Images None'
    else:
        image_txt = ''
        p.init_images = output_images # may be used for hires

    if video_type != 'None' and isinstance(output_images, list) and 'video' in p.ops:
        p.do_not_save_grid = True # pylint: disable=attribute-defined-outside-init
        output_filename = video.save_video(p, filename=None, images=output_images, video_type=video_type, duration=video_duration, loop=video_loop, pad=video_pad, interpolate=video_interpolate, sync=True)
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
    else:
        return (output_images, blended_image, html_txt, output_filename)
