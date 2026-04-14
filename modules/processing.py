import os
import json
import time
import numpy as np
from PIL import Image, ImageOps
from modules import shared, devices, errors, images, scripts_manager, memstats, script_callbacks, extra_networks, detailer, sd_models, sd_checkpoint, sd_vae, processing_helpers, processing_grading, timer
from modules.logger import log
from modules.sd_hijack_hypertile import context_hypertile_vae, context_hypertile_unet
from modules.processing_class import ( # pylint: disable=unused-import
    StableDiffusionProcessing,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingVideo,
    StableDiffusionProcessingControl,
)
from modules.processing_info import create_infotext


opt_C = 4
opt_f = 8
debug = log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')
create_binary_mask = processing_helpers.create_binary_mask
apply_overlay = processing_helpers.apply_overlay
apply_color_correction = processing_helpers.apply_color_correction
setup_color_correction = processing_helpers.setup_color_correction
fix_seed = processing_helpers.fix_seed
get_fixed_seed = processing_helpers.get_fixed_seed
create_random_tensors = processing_helpers.create_random_tensors
old_hires_fix_first_pass_dimensions = processing_helpers.old_hires_fix_first_pass_dimensions
get_sampler_name = processing_helpers.get_sampler_name
get_sampler_index = processing_helpers.get_sampler_index
validate_sample = processing_helpers.validate_sample
decode_first_stage = processing_helpers.decode_first_stage
images_tensor_to_samples = processing_helpers.images_tensor_to_samples
processed = None # last known processed results


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info=None, subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments="", binary=None, audio=None):
        self.sd_model_hash = getattr(shared.sd_model, 'sd_model_hash', '') if shared.sd_loaded is not None else ''

        self.prompt = p.prompt or ''
        self.negative_prompt = p.negative_prompt or ''
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.styles = p.styles

        self.bytes = binary
        self.images = images_list
        self.width = p.width if hasattr(p, 'width') else (self.images[0].width if len(self.images) > 0 else 0)
        self.height = p.height if hasattr(p, 'height') else (self.images[0].height if len(self.images) > 0 else 0)

        self.sampler_name = p.sampler_name or ''
        self.cfg_scale = p.cfg_scale if p.cfg_scale > 1 else None
        self.cfg_end = p.cfg_end if p.cfg_end < 1 else None
        self.image_cfg_scale = p.image_cfg_scale or 0
        self.steps = p.steps or 0
        self.batch_size = max(1, p.batch_size)
        self.denoising_strength = p.denoising_strength

        self.audio = audio

        self.detailer = p.detailer_enabled or False
        self.detailer_model = shared.opts.detailer_model if p.detailer_enabled else None
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.job_timestamp = shared.state.job_timestamp
        self.clip_skip = p.clip_skip
        self.eta = p.eta

        self.seed = seed if seed != -1 else p.seed
        self.subseed = subseed
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.subseed_strength = p.subseed_strength

        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning

        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]

        self.info = info or create_infotext(p)
        self.infotexts = infotexts or [self.info]
        self.comments = comments or ''
        memstats.reset_stats()

    def js(self):
        obj = {
            "prompt": self.all_prompts[0],
            "all_prompts": self.all_prompts,
            "negative_prompt": self.all_negative_prompts[0],
            "all_negative_prompts": self.all_negative_prompts,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_name": self.sampler_name,
            "cfg_scale": self.cfg_scale,
            "cfg_end": self.cfg_end,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "detailer": self.detailer,
            "detailer_model": self.detailer_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
        }
        return json.dumps(obj)

    def infotext(self, p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

    def __str___(self):
        return f'{self.__class__.__name__}: {self.__dict__}'


def get_processed(*args, **kwargs):
    global processed # pylint: disable=global-statement
    processed = Processed(*args, **kwargs)
    return processed


def process_images(p: StableDiffusionProcessing) -> Processed:
    timer.process.reset()
    debug(f'Process images: class={p.__class__.__name__} {vars(p)}')
    if shared.sd_model is None:
        log.warning('Aborted: op=process model not loaded')
        return None
    if not hasattr(shared.sd_model, 'sd_checkpoint_info'):
        log.error('Aborted: op=process incomplete model')
        return None
    if p.abort:
        log.debug('Processing: aborted')
        return None
    if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
        p.scripts.before_process(p)
    stored_opts = {}
    for k, v in p.override_settings.copy().items():
        if shared.opts.data.get(k, None) is None and shared.opts.data_labels.get(k, None) is None:
            continue
        orig = shared.opts.data.get(k, None) or shared.opts.data_labels[k].default
        if orig == v or (type(orig) == str and os.path.splitext(orig)[0] == v):
            p.override_settings.pop(k, None)
    for k in p.override_settings.keys():
        stored_opts[k] = shared.opts.data.get(k, None) or shared.opts.data_labels[k].default
    results = None
    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        if p.override_settings.get('sd_model_checkpoint', None) is not None and sd_checkpoint.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            log.warning(f"Override not found: checkpoint={p.override_settings.get('sd_model_checkpoint', None)}")
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()
        if p.override_settings.get('sd_model_refiner', None) is not None and sd_checkpoint.checkpoint_aliases.get(p.override_settings.get('sd_model_refiner')) is None:
            log.warning(f"Override not found: refiner={p.override_settings.get('sd_model_refiner', None)}")
            p.override_settings.pop('sd_model_refiner', None)
            sd_models.reload_model_weights()
        if p.override_settings.get('sd_vae', None) is not None:
            if p.override_settings.get('sd_vae', None) == 'TAESD':
                p.vae_type = 'Tiny'
                p.override_settings.pop('sd_vae', None)
            if p.override_settings.get('sd_vae', None) == 'REPA-E':
                p.vae_type = 'Repa'
                p.override_settings.pop('sd_vae', None)
        if p.override_settings.get('Hires upscaler', None) is not None:
            p.enable_hr = True
        if len(p.override_settings.keys()) > 0:
            log.debug(f'Override: {p.override_settings}')
        for k, v in p.override_settings.items():
            setattr(shared.opts, k, v)
            if k == 'sd_model_checkpoint':
                sd_models.reload_model_weights()
            if k == 'sd_vae':
                sd_vae.reload_vae_weights()

        shared.prompt_styles.apply_styles_to_extra(p)
        shared.prompt_styles.extract_comments(p)
        vae_scale_factor = sd_vae.get_vae_scale_factor()

        if p.width is not None:
            p.width = vae_scale_factor * int(p.width / vae_scale_factor)
        if p.height is not None:
            p.height = vae_scale_factor * int(p.height / vae_scale_factor)

        script_callbacks.before_process_callback(p)
        timer.process.record('pre')

        if shared.cmd_opts.profile:
            timer.startup.profile = True
            timer.process.profile = True
            with context_hypertile_vae(p), context_hypertile_unet(p):
                import torch.profiler # pylint: disable=redefined-outer-name
                activities=[torch.profiler.ProfilerActivity.CPU]
                if torch.cuda.is_available():
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                if devices.has_xpu() and hasattr(torch.profiler.ProfilerActivity, "XPU"):
                    activities.append(torch.profiler.ProfilerActivity.XPU)
                log.debug(f'Torch profile: activities={activities}')
                if shared.profiler is None:
                    profile_args = {
                        'activities': activities,
                        'profile_memory': True,
                        'with_modules': True,
                        'with_stack': os.environ.get('SD_PROFILE_STACK', None) is not None,
                        'experimental_config': torch._C._profiler._ExperimentalConfig(verbose=True) if os.environ.get('SD_PROFILE_STACK', None) is not None else None, # pylint: disable=protected-access
                        'with_flops': os.environ.get('SD_PROFILE_FLOPS', None) is not None,
                        'record_shapes': os.environ.get('SD_PROFILE_SHAPES', None) is not None,
                        'on_trace_ready': torch.profiler.tensorboard_trace_handler(os.environ.get('SD_PROFILE_FOLDER', None)) if os.environ.get('SD_PROFILE_FOLDER', None) is not None else None,
                    }
                    log.debug(f'Torch profile: {profile_args}')
                    shared.profiler = torch.profiler.profile(**profile_args)
                shared.profiler.start()
                results = process_images_inner(p)
                errors.profile_torch(shared.profiler, 'Process')
        else:
            with context_hypertile_vae(p), context_hypertile_unet(p):
                results = process_images_inner(p)

    finally:
        script_callbacks.after_process_callback(p)

        if p.override_settings_restore_afterwards: # restore opts to original state
            for k, v in stored_opts.items():
                setattr(shared.opts, k, v)
                if k == 'sd_model_checkpoint':
                    sd_models.reload_model_weights()
                if k == 'sd_model_refiner':
                    sd_models.reload_model_weights()
                if k == 'sd_vae':
                    sd_vae.reload_vae_weights()
        timer.process.record('post')
    return results


def process_init(p: StableDiffusionProcessing):
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
    reset_prompts = False
    if not p.all_prompts:
        p.all_prompts = p.prompt if isinstance(p.prompt, list) else p.batch_size * p.n_iter * [p.prompt]
        reset_prompts = True
    if not p.all_negative_prompts:
        p.all_negative_prompts = p.negative_prompt if isinstance(p.negative_prompt, list) else p.batch_size * p.n_iter * [p.negative_prompt]
        reset_prompts = True
    if not p.all_seeds:
        reset_prompts = True
        if type(seed) == list:
            p.all_seeds = [int(s) for s in seed]
        else:
            _sequential_seed = getattr(p, 'sequential_seed', None)
            if _sequential_seed is None:
                _sequential_seed = shared.opts.sequential_seed
            if _sequential_seed:
                p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
            else:
                p.all_seeds = []
                if p.all_prompts is not None:
                    for i in range(len(p.all_prompts)):
                        seed = get_fixed_seed(p.seed)
                        p.all_seeds.append(int(seed) + (i if p.subseed_strength == 0 else 0))
    if not p.all_subseeds:
        if type(subseed) == list:
            p.all_subseeds = [int(s) for s in subseed]
        else:
            p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]
    if reset_prompts:
        if not hasattr(p, 'keep_prompts'):
            p.all_prompts, p.all_negative_prompts = shared.prompt_styles.apply_styles_to_prompts(p.all_prompts, p.all_negative_prompts, p.styles, p.all_seeds)
            p.prompts = p.all_prompts[(p.iteration * p.batch_size):((p.iteration+1) * p.batch_size)]
            p.negative_prompts = p.all_negative_prompts[(p.iteration * p.batch_size):((p.iteration+1) * p.batch_size)]


def get_opt(p, key):
    val = getattr(p, key, None)
    return val if val is not None else getattr(shared.opts, key)


def process_samples(p: StableDiffusionProcessing, samples):
    out_images = []
    out_infotexts = []
    if not isinstance(samples, list):
        return samples, []
    for i, sample in enumerate(samples):
        debug(f'Processing result: index={i+1}/{len(samples)}')
        p.batch_index = i
        if isinstance(sample, Image.Image) or (isinstance(sample, list) and isinstance(sample[0], Image.Image)):
            image = sample
            sample = np.array(sample)
        else:
            sample = validate_sample(sample)
            image = Image.fromarray(sample)

        if isinstance(image, list):
            if len(image) > 1:
                log.warning(f'Processing: images={image} contains multiple images using first one only')
            image = image[0]

        if not shared.state.interrupted and not shared.state.skipped:

            if p.detailer_enabled:
                p.ops.append('detailer')
                if not p.do_not_save_samples and get_opt(p, 'save_images_before_detailer'):
                    info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                    images.save_image(image, path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=get_opt(p, 'samples_format'), info=info, p=p, suffix="-before-detailer")
                sample = detailer.detail(sample, p)
                if isinstance(sample, list):
                    if len(sample) > 0:
                        image = Image.fromarray(sample[0])
                    if len(sample) > 1:
                        annotated = sample[1] if isinstance(sample[1], Image.Image) else Image.fromarray(sample[1])
                        out_images.append(annotated)
                        out_infotexts.append("Detailer annotations")
                elif sample is not None:
                    image = Image.fromarray(sample)

            if p.color_corrections is not None and i < len(p.color_corrections):
                p.ops.append('color')
                if not p.do_not_save_samples and get_opt(p, 'save_images_before_color_correction'):
                    image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                    info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                    images.save_image(image_without_cc, path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=get_opt(p, 'samples_format'), info=info, p=p, suffix="-before-color-correct")
                method = p.color_correction_method if p.color_correction_method is not None else getattr(shared.opts, 'color_correction_method', 'histogram')
                image = apply_color_correction(p.color_corrections[i], image, method=method)

            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                pp = scripts_manager.PostprocessImageArgs(image)
                p.scripts.postprocess_image(p, pp)
                if pp.image is not None:
                    image = pp.image

            grading_params = processing_grading.GradingParams(
                brightness=getattr(p, 'grading_brightness', 0.0),
                contrast=getattr(p, 'grading_contrast', 0.0),
                saturation=getattr(p, 'grading_saturation', 0.0),
                hue=getattr(p, 'grading_hue', 0.0),
                gamma=getattr(p, 'grading_gamma', 1.0),
                sharpness=getattr(p, 'grading_sharpness', 0.0),
                color_temp=getattr(p, 'grading_color_temp', 6500),
                shadows=getattr(p, 'grading_shadows', 0.0),
                midtones=getattr(p, 'grading_midtones', 0.0),
                highlights=getattr(p, 'grading_highlights', 0.0),
                clahe_clip=getattr(p, 'grading_clahe_clip', 0.0),
                clahe_grid=getattr(p, 'grading_clahe_grid', 8),
                shadows_tint=getattr(p, 'grading_shadows_tint', '#000000'),
                highlights_tint=getattr(p, 'grading_highlights_tint', '#ffffff'),
                split_tone_balance=getattr(p, 'grading_split_tone_balance', 0.5),
                vignette=getattr(p, 'grading_vignette', 0.0),
                grain=getattr(p, 'grading_grain', 0.0),
                lut_cube_file=getattr(p, 'grading_lut_file', ''),
                lut_strength=getattr(p, 'grading_lut_strength', 1.0),
            )
            if processing_grading.is_active(grading_params):
                p.ops.append('grading')
                image = processing_grading.grade_image(image, grading_params)

            _overlay = getattr(p, 'mask_apply_overlay', None)
            if _overlay is None:
                _overlay = shared.opts.mask_apply_overlay
            if _overlay:
                image = apply_overlay(image, p.paste_to, i, p.overlay_images)

            _save_mask = get_opt(p, 'save_mask')
            _save_mask_composite = get_opt(p, 'save_mask_composite')
            _return_mask = get_opt(p, 'return_mask')
            _return_mask_composite = get_opt(p, 'return_mask_composite')
            if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([_save_mask, _save_mask_composite, _return_mask, _return_mask_composite]):
                image_mask = p.mask_for_overlay.convert('RGB')
                image1 = image.convert('RGBA').convert('RGBa')
                image2 = Image.new('RGBa', image.size)
                mask = images.resize_image(3, p.mask_for_overlay, image.width, image.height).convert('L')
                image_mask_composite = Image.composite(image1, image2, mask).convert('RGBA')
                info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                _fmt = get_opt(p, 'samples_format')
                if _save_mask:
                    images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], _fmt, info=info, p=p, suffix="-mask")
                if _save_mask_composite:
                    images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], _fmt, info=info, p=p, suffix="-mask-composite")
                if _return_mask:
                    out_infotexts.append(info)
                    out_images.append(image_mask)
                if _return_mask_composite:
                    out_infotexts.append(info)
                    out_images.append(image_mask_composite)

            _inc_mask = getattr(p, 'include_mask', None)
            if _inc_mask is None:
                _inc_mask = shared.opts.include_mask
            if _inc_mask:
                info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
                if _overlay and p.overlay_images is not None and len(p.overlay_images) > 0:
                    p.image_mask = create_binary_mask(p.overlay_images[0])
                    p.image_mask = ImageOps.invert(p.image_mask)
                    out_infotexts.append(info)
                    out_images.append(p.image_mask)
                elif getattr(p, 'image_mask', None) is not None and isinstance(p.image_mask, Image.Image):
                    if getattr(p, 'mask_for_detailer', None) is not None:
                        out_infotexts.append(info)
                        out_images.append(p.mask_for_detailer)
                    else:
                        out_infotexts.append(info)
                        out_images.append(p.image_mask)

            if p.selected_scale_tab_after == 1:
                p.width_after, p.height_after = int(image.width * p.scale_by_after), int(image.height * p.scale_by_after)
            if p.resize_mode_after != 0 and p.resize_name_after != 'None':
                image = images.resize_image(p.resize_mode_after, image, p.width_after, p.height_after, p.resize_name_after, context=p.resize_context_after)

        info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
        if get_opt(p, 'samples_save') and not p.do_not_save_samples and p.outpath_samples is not None:
            images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], get_opt(p, 'samples_format'), info=info, p=p) # main save image

        image.info["parameters"] = info
        out_infotexts.append(info)
        out_images.append(image)
    shared.history.add(None, info=out_infotexts, ops=p.ops, images=out_images)
    return out_images, out_infotexts


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    comments = {}
    infotexts = []
    output_images = []
    output_binary = None
    audio = None

    process_init(p)
    if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
        p.scripts.process(p)

    jobid = shared.state.begin('Process')
    shared.state.batch_count = p.n_iter
    with devices.inference_context():
        t0 = time.time()
        if not hasattr(p, 'skip_init'):
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
        debug(f'Processing inner: args={vars(p)}')
        p.iter_init_images = p.init_images # required so we use same starting non-processed images for each batch sequence
        for n in range(p.n_iter):
            p.init_images = p.iter_init_images
            if p.n_iter > 1:
                log.debug(f'Processing: batch={n+1} total={p.n_iter} progress={(n+1)/p.n_iter:.2f}')
            shared.state.batch_no = n + 1
            debug(f'Processing inner: iteration={n+1}/{p.n_iter}')
            p.iteration = n
            if shared.state.interrupted:
                log.debug(f'Process interrupted: {n+1}/{p.n_iter}')
                break
            if shared.state.skipped:
                log.debug(f'Process skipped: {n+1}/{p.n_iter}')
                shared.state.skipped = False
                continue

            if not hasattr(p, 'keep_prompts'):
                p.prompts = p.all_prompts[(n * p.batch_size):((n+1) * p.batch_size)]
                p.negative_prompts = p.all_negative_prompts[(n * p.batch_size):((n+1) * p.batch_size)]
            p.seeds = p.all_seeds[(n * p.batch_size):((n+1) * p.batch_size)]
            p.subseeds = p.all_subseeds[(n * p.batch_size):((n+1) * p.batch_size)]
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)
            if not p.prompts:
                break
            p.prompts, p.network_data = extra_networks.parse_prompts(p.prompts)
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            samples = None
            timer.process.record('init')
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                results = p.scripts.process_images(p)
                if results is not None:
                    samples = results.images
                    for script_image, script_infotext in zip(results.images, results.infotexts, strict=False):
                        output_images.append(script_image)
                        infotexts.append(script_infotext)

            # main processing
            if samples is None:
                from modules.processing_diffusers import process_diffusers
                samples = process_diffusers(p)
            timer.process.record('process')

            if shared.state.interrupted:
                log.debug(f'Process: batch={n+1}/{p.n_iter} interrupted')
                _keep = get_opt(p, 'keep_incomplete')
                p.do_not_save_samples = not _keep
                if shared.state.current_image is not None and isinstance(shared.state.current_image, Image.Image):
                    samples = [shared.state.current_image]
                    infotexts = [create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0)]
                else:
                    samples = []
                if not _keep:
                    break

            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.scripts.postprocess_batch(p, samples, batch_number=n)
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.prompts = p.all_prompts[(n * p.batch_size):((n+1) * p.batch_size)]
                p.negative_prompts = p.all_negative_prompts[(n * p.batch_size):((n+1) * p.batch_size)]
                batch_params = scripts_manager.PostprocessBatchListArgs(list(samples))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                samples = batch_params.images

            if hasattr(samples, 'bytes') and samples.bytes is not None:
                output_binary = samples.bytes
            else:
                batch_images, batch_infotexts = process_samples(p, samples)
                for batch_image, batch_infotext in zip(batch_images, batch_infotexts, strict=False):
                    if batch_image is not None and batch_image not in output_images:
                        output_images.append(batch_image)
                        infotexts.append(batch_infotext)

            audio = getattr(samples, 'audio', None)

            if shared.cmd_opts.lowvram:
                devices.torch_gc(force=True, reason='lowvram')
            timer.process.record('post')
            if shared.state.interrupted:
                break

        if not p.xyz:
            if hasattr(shared.sd_model, 'restore_pipeline') and (shared.sd_model.restore_pipeline is not None):
                shared.sd_model.restore_pipeline()
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)

        t1 = time.time()

        p.color_corrections = None
        index_of_first_image = 0
        _return_grid = get_opt(p, 'return_grid')
        _grid_save = get_opt(p, 'grid_save')
        if (_return_grid or _grid_save) and (not p.do_not_save_grid) and (len(output_images) > 1):
            if images.check_grid_size(output_images):
                r, c = images.get_grid_size(output_images, p.batch_size)
                grid = images.image_grid(output_images, p.batch_size)
                grid_text = f'{r}x{c}'
                grid_info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0, grid=grid_text)
                if _return_grid:
                    infotexts.insert(0, grid_info)
                    output_images.insert(0, grid)
                    index_of_first_image = 1
                if _grid_save:
                    images.save_image(grid, p.outpath_grids, "", p.all_seeds[0], p.all_prompts[0], get_opt(p, 'grid_format'), info=grid_info, p=p, grid=True) # main save grid

    results = get_processed(
        p,
        images_list=output_images,
        binary=output_binary,
        seed=p.all_seeds[0],
        info=infotexts[0] if len(infotexts) > 0 else '',
        comments="\n".join(comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
        audio=audio,
    )
    if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner) and not (shared.state.interrupted or shared.state.skipped):
        p.scripts.postprocess(p, results)
    timer.process.record('post')
    p.ops = list(set(p.ops))
    if not p.disable_extra_networks:
        log.info(f'Processed: images={len(output_images)} its={(p.steps * len(output_images)) / (t1 - t0):.2f} ops={p.ops}')
        log.debug(f'Processed: timers={timer.process.dct()}')
        log.debug(f'Processed: memory={memstats.memory_stats()}')

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        devices.torch_gc(force=True, reason='final')
    shared.state.end(jobid)
    return results
