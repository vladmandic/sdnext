import os
import json
import time
import numpy as np
from PIL import Image, ImageOps
from modules import shared, devices, errors, images, scripts_manager, memstats, script_callbacks, extra_networks, detailer, sd_models, sd_checkpoint, sd_vae, processing_helpers, timer, face_restoration
from modules.sd_hijack_hypertile import context_hypertile_vae, context_hypertile_unet
from modules.processing_class import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img, StableDiffusionProcessingControl, StableDiffusionProcessingVideo # pylint: disable=unused-import
from modules.processing_info import create_infotext
from modules.modeldata import model_data


opt_C = 4
opt_f = 8
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
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


class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info=None, subseed=None, all_prompts=None, all_negative_prompts=None, all_seeds=None, all_subseeds=None, index_of_first_image=0, infotexts=None, comments=""):
        self.images = images_list
        self.prompt = p.prompt or ''
        self.negative_prompt = p.negative_prompt or ''
        self.seed = seed if seed != -1 else p.seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info or create_infotext(p)
        self.comments = comments or ''
        self.width = p.width if hasattr(p, 'width') else (self.images[0].width if len(self.images) > 0 else 0)
        self.height = p.height if hasattr(p, 'height') else (self.images[0].height if len(self.images) > 0 else 0)
        self.sampler_name = p.sampler_name or ''
        self.cfg_scale = p.cfg_scale if p.cfg_scale > 1 else None
        self.cfg_end = p.cfg_end if p.cfg_end < 0 else None
        self.image_cfg_scale = p.image_cfg_scale or 0
        self.steps = p.steps or 0
        self.batch_size = max(1, p.batch_size)
        self.restore_faces = p.restore_faces or False
        self.face_restoration_model = shared.opts.face_restoration_model if p.restore_faces else None
        self.detailer = p.detailer_enabled or False
        self.detailer_model = shared.opts.detailer_model if p.detailer_enabled else None
        self.sd_model_hash = getattr(shared.sd_model, 'sd_model_hash', '') if model_data.sd_model is not None else ''
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = p.denoising_strength
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = shared.state.job_timestamp
        self.clip_skip = p.clip_skip
        self.eta = p.eta
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0]) if self.seed is not None else -1
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1
        self.is_using_inpainting_conditioning = p.is_using_inpainting_conditioning
        self.all_prompts = all_prompts or p.all_prompts or [self.prompt]
        self.all_negative_prompts = all_negative_prompts or p.all_negative_prompts or [self.negative_prompt]
        self.all_seeds = all_seeds or p.all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or p.all_subseeds or [self.subseed]
        self.infotexts = infotexts or [self.info]
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


def process_images(p: StableDiffusionProcessing) -> Processed:
    timer.process.reset()
    debug(f'Process images: {vars(p)}')
    if not hasattr(p.sd_model, 'sd_checkpoint_info'):
        shared.log.error('Processing: incomplete model')
        return None
    if p.abort:
        shared.log.debug('Processing: aborted')
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
    processed = None
    try:
        # if no checkpoint override or the override checkpoint can't be found, remove override entry and load opts checkpoint
        if p.override_settings.get('sd_model_checkpoint', None) is not None and sd_checkpoint.checkpoint_aliases.get(p.override_settings.get('sd_model_checkpoint')) is None:
            shared.log.warning(f"Override not found: checkpoint={p.override_settings.get('sd_model_checkpoint', None)}")
            p.override_settings.pop('sd_model_checkpoint', None)
            sd_models.reload_model_weights()
        if p.override_settings.get('sd_model_refiner', None) is not None and sd_checkpoint.checkpoint_aliases.get(p.override_settings.get('sd_model_refiner')) is None:
            shared.log.warning(f"Override not found: refiner={p.override_settings.get('sd_model_refiner', None)}")
            p.override_settings.pop('sd_model_refiner', None)
            sd_models.reload_model_weights()
        if p.override_settings.get('sd_vae', None) is not None:
            if p.override_settings.get('sd_vae', None) == 'TAESD':
                p.vae_type = 'Tiny'
                p.override_settings.pop('sd_vae', None)
        if p.override_settings.get('Hires upscaler', None) is not None:
            p.enable_hr = True
        if len(p.override_settings.keys()) > 0:
            shared.log.debug(f'Override: {p.override_settings}')
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
                shared.log.debug(f'Torch profile: activities={activities}')
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
                    shared.log.debug(f'Torch profile: {profile_args}')
                    shared.profiler = torch.profiler.profile(**profile_args)
                shared.profiler.start()
                processed = process_images_inner(p)
                errors.profile_torch(shared.profiler, 'Process')
        else:
            with context_hypertile_vae(p), context_hypertile_unet(p):
                processed = process_images_inner(p)

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
    return processed


def process_init(p: StableDiffusionProcessing):
    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)
    reset_prompts = False
    if p.all_prompts is None:
        p.all_prompts = p.prompt if isinstance(p.prompt, list) else p.batch_size * p.n_iter * [p.prompt]
        reset_prompts = True
    if p.all_negative_prompts is None:
        p.all_negative_prompts = p.negative_prompt if isinstance(p.negative_prompt, list) else p.batch_size * p.n_iter * [p.negative_prompt]
        reset_prompts = True
    if p.all_seeds is None:
        reset_prompts = True
        if type(seed) == list:
            p.all_seeds = [int(s) for s in seed]
        else:
            if shared.opts.sequential_seed:
                p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]
            else:
                p.all_seeds = []
                if p.all_prompts is not None:
                    for i in range(len(p.all_prompts)):
                        seed = get_fixed_seed(p.seed)
                        p.all_seeds.append(int(seed) + (i if p.subseed_strength == 0 else 0))
    if p.all_subseeds is None:
        if type(subseed) == list:
            p.all_subseeds = [int(s) for s in subseed]
        else:
            p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]
    if reset_prompts:
        if not hasattr(p, 'keep_prompts'):
            p.all_prompts, p.all_negative_prompts = shared.prompt_styles.apply_styles_to_prompts(p.all_prompts, p.all_negative_prompts, p.styles, p.all_seeds)
            p.prompts = p.all_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]
            p.negative_prompts = p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]
        p.prompts, _ = extra_networks.parse_prompts(p.prompts)


def process_samples(p: StableDiffusionProcessing, samples):
    out_images = []
    out_infotexts = []
    for i, sample in enumerate(samples):
        debug(f'Processing result: index={i+1}/{len(samples)}')
        p.batch_index = i
        info = create_infotext(p, p.prompts, p.seeds, p.subseeds, index=i)
        if isinstance(sample, Image.Image) or (isinstance(sample, list) and isinstance(sample[0], Image.Image)):
            image = sample
            sample = np.array(sample)
        else:
            sample = validate_sample(sample)
            image = Image.fromarray(sample)

        if p.restore_faces:
            p.ops.append('restore')
            if not p.do_not_save_samples and shared.opts.save_images_before_detailer:
                images.save_image(Image.fromarray(sample), path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-restore")
            sample = face_restoration.restore_faces(sample, p)
            if sample is not None:
                image = Image.fromarray(sample)

        if p.detailer_enabled:
            p.ops.append('detailer')
            if not p.do_not_save_samples and shared.opts.save_images_before_detailer:
                images.save_image(Image.fromarray(sample), path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-detailer")
            sample = detailer.detail(sample, p)
            if sample is not None:
                image = Image.fromarray(sample)

        if p.color_corrections is not None and i < len(p.color_corrections):
            p.ops.append('color')
            if not p.do_not_save_samples and shared.opts.save_images_before_color_correction:
                orig = p.color_corrections
                p.color_corrections = None
                p.color_corrections = orig
                image_without_cc = apply_overlay(image, p.paste_to, i, p.overlay_images)
                images.save_image(image_without_cc, path=p.outpath_samples, basename="", seed=p.seeds[i], prompt=p.prompts[i], extension=shared.opts.samples_format, info=info, p=p, suffix="-before-color-correct")
            image = apply_color_correction(p.color_corrections[i], image)

        if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
            pp = scripts_manager.PostprocessImageArgs(image)
            p.scripts.postprocess_image(p, pp)
            if pp.image is not None:
                image = pp.image

        if shared.opts.mask_apply_overlay:
            image = apply_overlay(image, p.paste_to, i, p.overlay_images)

        if hasattr(p, 'mask_for_overlay') and p.mask_for_overlay and any([shared.opts.save_mask, shared.opts.save_mask_composite, shared.opts.return_mask, shared.opts.return_mask_composite]):
            image_mask = p.mask_for_overlay.convert('RGB')
            image1 = image.convert('RGBA').convert('RGBa')
            image2 = Image.new('RGBa', image.size)
            mask = images.resize_image(3, p.mask_for_overlay, image.width, image.height).convert('L')
            image_mask_composite = Image.composite(image1, image2, mask).convert('RGBA')
            if shared.opts.save_mask:
                images.save_image(image_mask, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=info, p=p, suffix="-mask")
            if shared.opts.save_mask_composite:
                images.save_image(image_mask_composite, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=info, p=p, suffix="-mask-composite")
            if shared.opts.return_mask:
                out_infotexts.append(info)
                out_images.append(image_mask)
            if shared.opts.return_mask_composite:
                out_infotexts.append(info)
                out_images.append(image_mask_composite)

        if shared.opts.include_mask:
            if shared.opts.mask_apply_overlay and p.overlay_images is not None and len(p.overlay_images) > 0:
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

        if shared.opts.samples_save and not p.do_not_save_samples and p.outpath_samples is not None:
            images.save_image(image, p.outpath_samples, "", p.seeds[i], p.prompts[i], shared.opts.samples_format, info=info, p=p) # main save image

        image.info["parameters"] = info
        out_infotexts.append(info)
        out_images.append(image)
    return out_images, out_infotexts


def process_images_inner(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""
    if type(p.prompt) == list:
        assert len(p.prompt) > 0
    else:
        assert p.prompt is not None

    comments = {}
    infotexts = []
    output_images = []

    process_init(p)
    if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
        p.scripts.process(p)

    shared.state.batch_count = p.n_iter
    with devices.inference_context():
        t0 = time.time()
        if not hasattr(p, 'skip_init'):
            p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
        debug(f'Processing inner: args={vars(p)}')
        for n in range(p.n_iter):
            shared.state.batch_no = n + 1
            debug(f'Processing inner: iteration={n+1}/{p.n_iter}')
            p.iteration = n
            if shared.state.skipped:
                shared.log.debug(f'Process skipped: {n+1}/{p.n_iter}')
                shared.state.skipped = False
                continue
            if shared.state.interrupted:
                shared.log.debug(f'Process interrupted: {n+1}/{p.n_iter}')
                break

            if not hasattr(p, 'keep_prompts'):
                p.prompts = p.all_prompts[n * p.batch_size:(n+1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n+1) * p.batch_size]
            p.seeds = p.all_seeds[n * p.batch_size:(n+1) * p.batch_size]
            p.subseeds = p.all_subseeds[n * p.batch_size:(n+1) * p.batch_size]
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.scripts.before_process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)
            if len(p.prompts) == 0:
                break
            p.prompts, p.network_data = extra_networks.parse_prompts(p.prompts)
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.scripts.process_batch(p, batch_number=n, prompts=p.prompts, seeds=p.seeds, subseeds=p.subseeds)

            samples = None
            timer.process.record('init')
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                processed = p.scripts.process_images(p)
                if processed is not None:
                    samples = processed.images
                    for script_image, script_infotext in zip(processed.images, processed.infotexts):
                        output_images.append(script_image)
                        infotexts.append(script_infotext)
            if samples is None:
                from modules.processing_diffusers import process_diffusers
                samples = process_diffusers(p)
            timer.process.record('process')

            if not shared.opts.keep_incomplete and shared.state.interrupted:
                samples = []

            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.scripts.postprocess_batch(p, samples, batch_number=n)
            if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner):
                p.prompts = p.all_prompts[n * p.batch_size:(n+1) * p.batch_size]
                p.negative_prompts = p.all_negative_prompts[n * p.batch_size:(n+1) * p.batch_size]
                batch_params = scripts_manager.PostprocessBatchListArgs(list(samples))
                p.scripts.postprocess_batch_list(p, batch_params, batch_number=n)
                samples = batch_params.images

            batch_images, batch_infotexts = process_samples(p, samples)
            for batch_image, batch_infotext in zip(batch_images, batch_infotexts):
                output_images.append(batch_image)
                infotexts.append(batch_infotext)

            if shared.cmd_opts.lowvram:
                devices.torch_gc(force=True, reason='lowvram')
            timer.process.record('post')

        if not p.xyz:
            if hasattr(shared.sd_model, 'restore_pipeline') and (shared.sd_model.restore_pipeline is not None):
                shared.sd_model.restore_pipeline()
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)

        t1 = time.time()

        p.color_corrections = None
        index_of_first_image = 0
        if (shared.opts.return_grid or shared.opts.grid_save) and (not p.do_not_save_grid) and (len(output_images) > 1):
            if images.check_grid_size(output_images):
                r, c = images.get_grid_size(output_images, p.batch_size)
                grid = images.image_grid(output_images, p.batch_size)
                grid_text = f'{r}x{c}'
                grid_info = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds, index=0, grid=grid_text)
                if shared.opts.return_grid:
                    infotexts.insert(0, grid_info)
                    output_images.insert(0, grid)
                    index_of_first_image = 1
                if shared.opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "", p.all_seeds[0], p.all_prompts[0], shared.opts.grid_format, info=grid_info, p=p, grid=True) # main save grid

    processed = Processed(
        p,
        images_list=output_images,
        seed=p.all_seeds[0],
        info=infotexts[0] if len(infotexts) > 0 else '',
        comments="\n".join(comments),
        subseed=p.all_subseeds[0],
        index_of_first_image=index_of_first_image,
        infotexts=infotexts,
    )
    if p.scripts is not None and isinstance(p.scripts, scripts_manager.ScriptRunner) and not (shared.state.interrupted or shared.state.skipped):
        p.scripts.postprocess(p, processed)
    timer.process.record('post')
    p.ops = list(set(p.ops))
    if not p.disable_extra_networks:
        shared.log.info(f'Processed: images={len(output_images)} its={(p.steps * len(output_images)) / (t1 - t0):.2f} ops={p.ops}')
        shared.log.debug(f'Processed: timers={timer.process.dct()}')
        shared.log.debug(f'Processed: memory={memstats.memory_stats()}')

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        devices.torch_gc(force=True, reason='final')
    return processed
