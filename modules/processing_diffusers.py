from types import SimpleNamespace
import os
import time
import numpy as np
import torch
from PIL import Image
from modules import shared, devices, processing, sd_models, errors, sd_hijack_hypertile, processing_vae, sd_models_compile, timer, modelstats, extra_networks, attention, images_sharpfin
from modules.processing_helpers import resize_hires, calculate_base_steps, calculate_hires_steps, calculate_refiner_steps, save_intermediate, update_sampler, is_txt2img, is_refiner_enabled, get_job_name
from modules.processing_args import set_pipeline_args
from modules.onnx_impl import preprocess_pipeline as preprocess_onnx_pipeline, check_parameters_changed as olive_check_parameters_changed
from modules.lora import lora_common


debug = os.environ.get('SD_DIFFUSERS_DEBUG', None) is not None
output_type = 'np' if os.environ.get('SD_VAE_DEFAULT', None) is not None else 'latent'
last_p = None
orig_pipeline = shared.sd_model


def restore_state(p: processing.StableDiffusionProcessing):
    if p.state in ['reprocess_refine', 'reprocess_detail']:
        # validate
        if last_p is None:
            shared.log.warning(f'Restore state: op={p.state} last state missing')
            return p
        if p.__class__ != last_p.__class__:
            shared.log.warning(f'Restore state: op={p.state} last state is different type')
            return p
        if shared.history.count == 0:
            shared.log.warning(f'Restore state: op={p.state} last latents missing')
            return p
        state = p.state

        # set ops
        if state == 'reprocess_refine':
            width, width_before, width_after, width_mask = p.width, p.width_before, p.width_after, p.width_mask
            height, height_before, height_after, height_mask = p.height, p.height_before, p.height_after, p.height_mask
            scale_by, scale_by_before, scale_by_after, scale_by_mask = p.scale_by, p.scale_by_before, p.scale_by_after, p.scale_by_mask
            resize_name, resize_name_before, resize_name_after, resize_name_mask = p.resize_name, p.resize_name_before, p.resize_name_after, p.resize_name_mask
            resize_mode, resize_mode_before, resize_mode_after, resize_mode_mask = p.resize_mode, p.resize_mode_before, p.resize_mode_after, p.resize_mode_mask
            resize_context, resize_context_before, resize_context_after, resize_context_mask = p.resize_context, p.resize_context_before, p.resize_context_after, p.resize_context_mask
            selected_scale_tab, selected_scale_tab_before, selected_scale_tab_after, selected_scale_tab_mask = p.selected_scale_tab, p.selected_scale_tab_before, p.selected_scale_tab_after, p.selected_scale_tab_mask
            hr_scale, hr_resize_mode, hr_resize_context, hr_upscaler, hr_second_pass_steps = p.hr_scale, p.hr_resize_mode, p.hr_resize_context, p.hr_upscaler, p.hr_second_pass_steps
            hr_resize_x, hr_resize_y, hr_upscale_to_x, hr_upscale_to_y, hr_denoising_strength = p.hr_resize_x, p.hr_resize_y, p.hr_upscale_to_x, p.hr_upscale_to_y, p.hr_denoising_strength

            p = last_p
            p.skip = ['encode', 'base']
            p.state = state
            p.enable_hr = True
            p.hr_force = True
            p.init_images = None

            p.width, p.width_before, p.width_after, p.width_mask = width, width_before, width_after, width_mask
            p.height, p.height_before, p.height_after, p.height_mask = height, height_before, height_after, height_mask
            p.resize_name, p.resize_name_before, p.resize_name_after, p.resize_name_mask = resize_name, resize_name_before, resize_name_after, resize_name_mask
            p.resize_mode, p.resize_mode_before, p.resize_mode_after, p.resize_mode_mask = resize_mode, resize_mode_before, resize_mode_after, resize_mode_mask
            p.resize_context, p.resize_context_before, p.resize_context_after, p.resize_context_mask = resize_context, resize_context_before, resize_context_after, resize_context_mask
            p.selected_scale_tab, p.selected_scale_tab_before, p.selected_scale_tab_after, p.selected_scale_tab_mask = selected_scale_tab, selected_scale_tab_before, selected_scale_tab_after, selected_scale_tab_mask
            p.scale_by, p.scale_by_before, p.scale_by_after, p.scale_by_mask = scale_by, scale_by_before, scale_by_after, scale_by_mask
            p.hr_scale, p.hr_resize_mode, p.hr_resize_context, p.hr_upscaler, p.hr_second_pass_steps = hr_scale, hr_resize_mode, hr_resize_context, hr_upscaler, hr_second_pass_steps
            p.hr_resize_x, p.hr_resize_y, p.hr_upscale_to_x, p.hr_upscale_to_y, p.hr_denoising_strength = hr_resize_x, hr_resize_y, hr_upscale_to_x, hr_upscale_to_y, hr_denoising_strength
        if state == 'reprocess_detail':
            p.skip = ['encode', 'base', 'hires']
            p.detailer_enabled = True
        shared.log.info(f'Restore state: op={p.state} skip={p.skip}')
    return p


def process_pre(p: processing.StableDiffusionProcessing):
    from modules import ipadapter, sd_hijack_freeu, para_attention, teacache, hidiffusion, ras, pag, cfgzero, transformer_cache, token_merge, linfusion, cachedit
    shared.log.info('Processing modifiers: apply')

    try:
        # apply-with-unapply
        sd_models_compile.check_deepcache(enable=True)
        ipadapter.apply(shared.sd_model, p)
        token_merge.apply_token_merging(p.sd_model)
        hidiffusion.apply(p, shared.sd_model_type)
        ras.apply(shared.sd_model, p)
        pag.apply(p)
        cfgzero.apply(p)
        linfusion.apply(shared.sd_model)
        cachedit.apply_cache_dit(shared.sd_model)

        # apply-only
        sd_hijack_freeu.apply_freeu(p)
        transformer_cache.set_cache()
        para_attention.apply_first_block_cache()
        teacache.apply_teacache(p)
    except Exception as e:
        shared.log.error(f'Processing apply: {e}')
        errors.display(e, 'apply')

    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    # if hasattr(shared.sd_model, 'unet'):
    #     sd_models.move_model(shared.sd_model.unet, devices.device)
    # if hasattr(shared.sd_model, 'transformer'):
    #     sd_models.move_model(shared.sd_model.transformer, devices.device)

    from modules import modular
    if modular.is_compatible(shared.sd_model):
        modular_pipe = modular.convert_to_modular(shared.sd_model)
        if modular_pipe is not None:
            shared.sd_model = modular_pipe
    if modular.is_guider(shared.sd_model):
        from modules import modular_guiders
        modular_guiders.set_guider(p)

    timer.process.record('pre')


def process_post(p: processing.StableDiffusionProcessing):
    from modules import ipadapter, hidiffusion, ras, pag, cfgzero, token_merge, linfusion, cachedit
    shared.log.info('Processing modifiers: unapply')

    try:
        sd_models_compile.check_deepcache(enable=False)
        ipadapter.unapply(shared.sd_model, unload=getattr(p, 'ip_adapter_unload', False))
        token_merge.remove_token_merging(p.sd_model)
        hidiffusion.unapply()
        ras.unapply(shared.sd_model)
        pag.unapply()
        cfgzero.unapply()
        linfusion.unapply(shared.sd_model)
        cachedit.unapply_cache_dir(shared.sd_model)
    except Exception as e:
        shared.log.error(f'Processing unapply: {e}')
        errors.display(e, 'unapply')
    timer.process.record('post')


def process_base(p: processing.StableDiffusionProcessing):
    jobid = shared.state.begin('Base')
    txt2img = is_txt2img()
    use_refiner_start = is_refiner_enabled(p) and (not p.is_hr_pass)
    use_denoise_start = not txt2img and p.refiner_start > 0 and p.refiner_start < 1

    shared.sd_model = update_pipeline(shared.sd_model, p)
    update_sampler(p, shared.sd_model)
    timer.process.record('prepare')
    process_pre(p)
    desc = 'Base'
    if 'detailer' in p.ops:
        desc = 'Detail'
    base_args = set_pipeline_args(
        p=p,
        model=shared.sd_model,
        prompts=p.prompts,
        negative_prompts=p.negative_prompts,
        prompts_2=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
        negative_prompts_2=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
        num_inference_steps=calculate_base_steps(p, use_refiner_start=use_refiner_start, use_denoise_start=use_denoise_start),
        eta=shared.opts.scheduler_eta,
        guidance_scale=p.cfg_scale,
        guidance_rescale=p.diffusers_guidance_rescale,
        true_cfg_scale=p.pag_scale,
        denoising_start=0 if use_refiner_start else p.refiner_start if use_denoise_start else None,
        denoising_end=p.refiner_start if use_refiner_start else 1 if use_denoise_start else None,
        num_frames=getattr(p, 'frames', 1),
        output_type=output_type,
        clip_skip=p.clip_skip,
        desc=desc,
    )
    base_steps = base_args.get('prior_num_inference_steps', None) or p.steps or base_args.get('num_inference_steps', None)
    shared.state.update(get_job_name(p, shared.sd_model), base_steps, 1)
    if shared.opts.scheduler_eta is not None and shared.opts.scheduler_eta > 0 and shared.opts.scheduler_eta < 1:
        p.extra_generation_params["Sampler Eta"] = shared.opts.scheduler_eta
    output = None
    if debug:
        modelstats.analyze()
    try:
        t0 = time.time()
        p.prompts, p.network_data = extra_networks.parse_prompts(p.prompts, p.network_data)
        extra_networks.activate(p, exclude=['text_encoder', 'text_encoder_2', 'text_encoder_3'])

        if hasattr(shared.sd_model, 'tgate') and getattr(p, 'gate_step', -1) > 0:
            base_args['gate_step'] = p.gate_step
            output = shared.sd_model.tgate(**base_args) # pylint: disable=not-callable
        else:
            taskid = shared.state.begin('Inference')
            output = shared.sd_model(**base_args)
            shared.state.end(taskid)
        if isinstance(output, dict):
            output = SimpleNamespace(**output)
        if isinstance(output, list):
            output = SimpleNamespace(images=output)
        if isinstance(output, Image.Image):
            output = SimpleNamespace(images=[output])
        if hasattr(output, 'image'):
            output.images = output.image
        if hasattr(output, 'images'):
            shared.history.add(output.images, info=processing.create_infotext(p), ops=p.ops)
        timer.process.record('pipeline')
        sd_models_compile.openvino_post_compile(op="base") # only executes on compiled vino models
        if shared.cmd_opts.profile:
            t1 = time.time()
            shared.log.debug(f'Profile: pipeline call: {t1-t0:.2f}')
        if not hasattr(output, 'images') and hasattr(output, 'frames'):
            if hasattr(output.frames[0], 'shape'):
                shared.log.debug(f'Generated: frames={output.frames[0].shape[1]}')
            else:
                shared.log.debug(f'Generated: frames={len(output.frames[0])}')
            output.images = output.frames[0]
        if hasattr(output, 'images') and isinstance(output.images, np.ndarray):
            output.images = torch.from_numpy(output.images)
    except AssertionError as e:
        shared.log.info(e)
    except ValueError as e:
        shared.state.interrupted = True
        err_args = base_args.copy()
        for k, v in base_args.items():
            if isinstance(v, torch.Tensor):
                err_args[k] = f'{v.device}:{v.dtype}:{v.shape}'
        shared.log.error(f'Processing: args={err_args} {e}')
        if shared.cmd_opts.debug:
            errors.display(e, 'Processing')
    except RuntimeError as e:
        shared.state.interrupted = True
        err_args = base_args.copy()
        for k, v in base_args.items():
            if isinstance(v, torch.Tensor):
                err_args[k] = f'{v.device}:{v.dtype}:{v.shape}'
        shared.log.error(f'Processing: step=base args={err_args} {e}')
        errors.display(e, 'Processing')
        modelstats.analyze()
    finally:
        process_post(p)

    if hasattr(shared.sd_model, 'postprocess') and callable(shared.sd_model.postprocess):
        output = shared.sd_model.postprocess(p, output)

    shared.state.end(jobid)
    shared.state.nextjob()
    return output


def process_hires(p: processing.StableDiffusionProcessing, output):
    # optional second pass
    if (output is None) or (output.images is None):
        return output
    if p.enable_hr:
        jobid = shared.state.begin('Hires')
        p.is_hr_pass = True
        if hasattr(p, 'init_hr'):
            p.init_hr(p.hr_scale, p.hr_upscaler, force=p.hr_force)
        else:
            if not p.is_hr_pass: # fake hires for img2img if not actual hr pass
                p.hr_scale = p.scale_by
                p.hr_upscaler = p.resize_name
                p.hr_resize_mode = p.resize_mode
                p.hr_resize_context = p.resize_context
            p.hr_upscale_to_x = int(p.width * p.hr_scale) if p.hr_resize_x == 0 else p.hr_resize_x
            p.hr_upscale_to_y = int(p.height * p.hr_scale) if p.hr_resize_y == 0 else p.hr_resize_y

        # hires runs on original pipeline
        if hasattr(shared.sd_model, 'restore_pipeline') and (shared.sd_model.restore_pipeline is not None) and (not shared.opts.control_hires):
            shared.sd_model.restore_pipeline()
        if (getattr(shared.sd_model, 'controlnet', None) is not None) and (((isinstance(shared.sd_model.controlnet, list) and len(shared.sd_model.controlnet) > 1)) or ('Multi' in type(shared.sd_model.controlnet).__name__)):
            shared.log.warning(f'Process: control={type(shared.sd_model.controlnet)} not supported in hires')
            return output

        # upscale
        if hasattr(p, 'height') and hasattr(p, 'width') and p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5):
            shared.log.info(f'Upscale: mode={p.hr_resize_mode} upscaler="{p.hr_upscaler}" context="{p.hr_resize_context}" resize={p.hr_resize_x}x{p.hr_resize_y} upscale={p.hr_upscale_to_x}x{p.hr_upscale_to_y}')
            p.ops.append('upscale')
            if shared.opts.samples_save and not p.do_not_save_samples and shared.opts.save_images_before_highres_fix and hasattr(shared.sd_model, 'vae'):
                save_intermediate(p, latents=output.images, suffix="-before-hires")
            output.images = resize_hires(p, latents=output.images)
            sd_hijack_hypertile.hypertile_set(p, hr=True)
        elif torch.is_tensor(output.images) and output.images.shape[-1] == 3: # nhwc
            if output.images.dim() == 3:
                output.images = images_sharpfin.to_pil(output.images)
            elif output.images.dim() == 4:
                output.images = [images_sharpfin.to_pil(output.images[i]) for i in range(output.images.shape[0])]

        strength = p.hr_denoising_strength if p.hr_denoising_strength > 0 else p.denoising_strength
        if (p.hr_upscaler is not None) and (p.hr_upscaler.lower().startswith('latent') or p.hr_force) and strength > 0:
            p.ops.append('hires')
            sd_models_compile.openvino_recompile_model(p, hires=True, refiner=False)
            if shared.sd_model.__class__.__name__ == "OnnxRawPipeline":
                shared.sd_model = preprocess_onnx_pipeline(p)
            p.hr_force = True

        # hires
        if p.hr_force and strength == 0:
            shared.log.warning('Hires skip: denoising=0')
            p.hr_force = False
        if p.hr_force:
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
            if 'Upscale' in shared.sd_model.__class__.__name__ or 'Flux' in shared.sd_model.__class__.__name__ or 'Kandinsky' in shared.sd_model.__class__.__name__:
                output.images = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, vae_type=p.vae_type, output_type='pil', width=p.width, height=p.height)
            if p.is_control and hasattr(p, 'task_args') and p.task_args.get('image', None) is not None:
                if hasattr(shared.sd_model, "vae") and output.images is not None and len(output.images) > 0:
                    output.images = processing_vae.vae_decode(latents=output.images, model=shared.sd_model, vae_type=p.vae_type, output_type='pil', width=p.hr_upscale_to_x, height=p.hr_upscale_to_y) # controlnet cannnot deal with latent input
            update_sampler(p, shared.sd_model, second_pass=True)
            orig_denoise = p.denoising_strength
            p.denoising_strength = strength
            orig_image = p.task_args.pop('image', None) # remove image override from hires
            process_pre(p)

            prompts = p.prompts
            reset_prompts = False
            if len(p.refiner_prompt) > 0:
                prompts = len(output.images)* [p.refiner_prompt]
                prompts, p.network_data = extra_networks.parse_prompts(prompts)
                reset_prompts = True
            if reset_prompts or ('base' in p.skip):
                extra_networks.activate(p)

            hires_args = set_pipeline_args(
                p=p,
                model=shared.sd_model,
                prompts=prompts,
                negative_prompts=len(output.images) * [p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
                prompts_2=len(output.images) * [p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts,
                negative_prompts_2=len(output.images) * [p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts,
                num_inference_steps=calculate_hires_steps(p),
                eta=shared.opts.scheduler_eta,
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                output_type=output_type,
                clip_skip=p.clip_skip,
                image=output.images,
                strength=strength,
                desc='Hires',
            )

            hires_steps = hires_args.get('prior_num_inference_steps', None) or p.hr_second_pass_steps or hires_args.get('num_inference_steps', None)
            shared.state.update(get_job_name(p, shared.sd_model), hires_steps, 1)
            try:
                taskid = shared.state.begin('Inference')
                output = shared.sd_model(**hires_args) # pylint: disable=not-callable
                shared.state.end(taskid)
                if isinstance(output, dict):
                    output = SimpleNamespace(**output)
                if hasattr(output, 'images'):
                    shared.history.add(output.images, info=processing.create_infotext(p), ops=p.ops)
                sd_models_compile.check_deepcache(enable=False)
                sd_models_compile.openvino_post_compile(op="base")
            except AssertionError as e:
                shared.log.info(e)
            except RuntimeError as e:
                shared.state.interrupted = True
                shared.log.error(f'Processing step=hires: args={hires_args} {e}')
                errors.display(e, 'Processing')
                modelstats.analyze()
            finally:
                process_post(p)
            if hasattr(shared.sd_model, 'postprocess') and callable(shared.sd_model.postprocess):
                output = shared.sd_model.postprocess(p, output)
            if orig_image is not None:
                p.task_args['image'] = orig_image
            p.denoising_strength = orig_denoise
        shared.state.end(jobid)
        shared.state.nextjob()
        p.is_hr_pass = False
        timer.process.record('hires')
    return output


def process_refine(p: processing.StableDiffusionProcessing, output):
    # optional refiner pass or decode
    if (output is None) or (output.images is None):
        return output
    if is_refiner_enabled(p):
        if shared.opts.samples_save and not p.do_not_save_samples and shared.opts.save_images_before_refiner and hasattr(shared.sd_model, 'vae'):
            save_intermediate(p, latents=output.images, suffix="-before-refiner")
        if shared.opts.diffusers_move_base:
            shared.log.debug('Moving to CPU: model=base')
            sd_models.move_model(shared.sd_model, devices.cpu)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return output
        jobid = shared.state.begin('Refine')
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        if shared.opts.diffusers_move_refiner:
            sd_models.move_model(shared.sd_refiner, devices.device)
            if hasattr(shared.sd_refiner, 'unet'):
                sd_models.move_model(shared.sd_model.unet, devices.device)
            if hasattr(shared.sd_refiner, 'transformer'):
                sd_models.move_model(shared.sd_model.transformer, devices.device)
        p.ops.append('refine')
        p.is_refiner_pass = True
        sd_models_compile.openvino_recompile_model(p, hires=False, refiner=True)
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
        shared.sd_refiner = sd_models.set_diffuser_pipe(shared.sd_refiner, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        for i in range(len(output.images)):
            image = output.images[i]
            noise_level = round(350 * p.denoising_strength)
            refiner_output_type = output_type
            if 'Upscale' in shared.sd_refiner.__class__.__name__ or 'Flux' in shared.sd_refiner.__class__.__name__ or 'Kandinsky' in shared.sd_refiner.__class__.__name__:
                image = processing_vae.vae_decode(latents=image, model=shared.sd_model, vae_type=p.vae_type, output_type='pil', width=p.width, height=p.height)
                p.extra_generation_params['Noise level'] = noise_level
                refiner_output_type = 'np'
            update_sampler(p, shared.sd_refiner, second_pass=True)
            shared.opts.prompt_attention = 'fixed'
            refiner_args = set_pipeline_args(
                p=p,
                model=shared.sd_refiner,
                prompts=[p.refiner_prompt] if len(p.refiner_prompt) > 0 else p.prompts[i],
                negative_prompts=[p.refiner_negative] if len(p.refiner_negative) > 0 else p.negative_prompts[i],
                num_inference_steps=calculate_refiner_steps(p),
                eta=shared.opts.scheduler_eta,
                # strength=p.denoising_strength,
                noise_level=noise_level, # StableDiffusionUpscalePipeline only
                guidance_scale=p.image_cfg_scale if p.image_cfg_scale is not None else p.cfg_scale,
                guidance_rescale=p.diffusers_guidance_rescale,
                denoising_start=p.refiner_start if p.refiner_start > 0 and p.refiner_start < 1 else None,
                denoising_end=1 if p.refiner_start > 0 and p.refiner_start < 1 else None,
                image=image,
                output_type=refiner_output_type,
                clip_skip=p.clip_skip,
                prompt_attention='fixed',
                desc='Refiner',
            )
            refiner_steps = refiner_args.get('prior_num_inference_steps', None) or p.steps or refiner_args.get('num_inference_steps', None)
            shared.state.update(get_job_name(p, shared.sd_refiner), refiner_steps, 1)
            try:
                if 'requires_aesthetics_score' in shared.sd_refiner.config: # sdxl-model needs false and sdxl-refiner needs true
                    shared.sd_refiner.register_to_config(requires_aesthetics_score = getattr(shared.sd_refiner, 'tokenizer', None) is None)
                output = shared.sd_refiner(**refiner_args) # pylint: disable=not-callable
                if isinstance(output, dict):
                    output = SimpleNamespace(**output)
                if hasattr(output, 'images'):
                    shared.history.add(output.images, info=processing.create_infotext(p), ops=p.ops)
                sd_models_compile.openvino_post_compile(op="refiner")
            except AssertionError as e:
                shared.log.info(e)
            except RuntimeError as e:
                shared.state.interrupted = True
                shared.log.error(f'Processing step=refine: args={refiner_args} {e}')
                errors.display(e, 'Processing')
                modelstats.analyze()

        if shared.opts.diffusers_offload_mode == "balanced":
            shared.sd_refiner = sd_models.apply_balanced_offload(shared.sd_refiner)
        elif shared.opts.diffusers_move_refiner:
            shared.log.debug('Moving to CPU: model=refiner')
            sd_models.move_model(shared.sd_refiner, devices.cpu)
        shared.state.end(jobid)
        shared.state.nextjob()
        p.is_refiner_pass = False
        timer.process.record('refine')
    return output


def process_decode(p: processing.StableDiffusionProcessing, output):
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    if output is not None:
        if hasattr(output, 'bytes') and output.bytes is not None:
            shared.log.debug(f'Generated: bytes={len(output.bytes)}')
            return output
        if not hasattr(output, 'images') and hasattr(output, 'frames'):
            shared.log.debug(f'Generated: frames={len(output.frames[0])}')
            output.images = output.frames[0]
        if output.images is not None and len(output.images) > 0 and isinstance(output.images[0], Image.Image):
            return output.images
        model = shared.sd_model if not is_refiner_enabled(p) else shared.sd_refiner
        if not hasattr(model, 'vae'):
            if hasattr(model, 'pipe') and hasattr(model.pipe, 'vae'):
                model = model.pipe
        if (hasattr(model, "vae") or hasattr(model, "vqgan")) and (output.images is not None) and (len(output.images) > 0):
            if p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5):
                width = max(getattr(p, 'width', 0), getattr(p, 'hr_upscale_to_x', 0))
                height = max(getattr(p, 'height', 0), getattr(p, 'hr_upscale_to_y', 0))
            else:
                width = getattr(p, 'width', 0)
                height = getattr(p, 'height', 0)
            frames = p.task_args.get('num_frames', None) or getattr(p, 'frames', None)
            if isinstance(output.images, list):
                results = []
                for i in range(len(output.images)):
                    result_batch = processing_vae.vae_decode(
                        latents = output.images[i],
                        model = model,
                        vae_type = p.vae_type,
                        width = width,
                        height = height,
                        frames = frames,
                    )
                    for result in list(result_batch):
                        results.append(result)
            else:
                results = processing_vae.vae_decode(
                    latents = output.images,
                    model = model,
                    vae_type = p.vae_type,
                    width = width,
                    height = height,
                    frames = frames,
                )
                if not isinstance(results, list):
                    results = list(results)
        elif hasattr(output, 'images'):
            results = output.images
        else:
            shared.log.warning('Processing: no results')
            results = []
    else:
        shared.log.warning('Processing: no results')
        results = []
    return results


def update_pipeline(sd_model, p: processing.StableDiffusionProcessing):
    if sd_models.get_diffusers_task(sd_model) == sd_models.DiffusersTaskType.INPAINTING and getattr(p, 'image_mask', None) is None and p.task_args.get('image_mask', None) is None and getattr(p, 'mask', None) is None:
        shared.log.warning('Processing: mode=inpaint mask=None')
        sd_model = sd_models.set_diffuser_pipe(sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
    if shared.opts.cuda_compile_backend == "olive-ai":
        sd_model = olive_check_parameters_changed(p, is_refiner_enabled(p))
    if sd_model.__class__.__name__ == "OnnxRawPipeline":
        sd_model = preprocess_onnx_pipeline(p)
        global orig_pipeline # pylint: disable=global-statement
        orig_pipeline = sd_model # processed ONNX pipeline should not be replaced with original pipeline.
    if getattr(sd_model, "current_attn_name", None) != shared.opts.cross_attention_optimization:
        shared.log.info(f"Setting attention optimization: {shared.opts.cross_attention_optimization}")
        attention.set_diffusers_attention(sd_model)
    return sd_model


def validate_pipeline(p: processing.StableDiffusionProcessing):
    from modules.video_models.models_def import models as video_models
    models_cls = []
    for family in video_models:
        for m in video_models[family]:
            if m.repo_cls is not None:
                models_cls.append(m.repo_cls.__name__)
            if m.custom is not None:
                models_cls.append(m.custom)
    is_video_model = shared.sd_model.__class__.__name__ in models_cls
    override_video_pipelines = ['WanPipeline', 'WanImageToVideoPipeline', 'WanVACEPipeline']
    is_video_pipeline = ('video' in p.__class__.__name__.lower()) or (shared.sd_model.__class__.__name__ in override_video_pipelines)
    if is_video_model and not is_video_pipeline:
        shared.log.error(f'Mismatch: type={shared.sd_model_type} cls={shared.sd_model.__class__.__name__} request={p.__class__.__name__} video model with non-video pipeline')
        return False
    elif not is_video_model and is_video_pipeline:
        shared.log.error(f'Mismatch: type={shared.sd_model_type} cls={shared.sd_model.__class__.__name__} request={p.__class__.__name__} non-video model with video pipeline')
        return False
    return True


def process_diffusers(p: processing.StableDiffusionProcessing):
    results = []
    if debug:
        shared.log.trace(f'Process diffusers args: {vars(p)}')
    if not validate_pipeline(p):
        return results

    p = restore_state(p)
    global orig_pipeline # pylint: disable=global-statement
    orig_pipeline = shared.sd_model

    if shared.state.interrupted or shared.state.skipped:
        shared.sd_model = orig_pipeline
        return results

    # sanitize init_images
    if hasattr(p, 'init_images') and not isinstance(getattr(p, 'init_images', []), list):
        p.init_images = [p.init_images]
    if hasattr(p, 'init_images') and isinstance(getattr(p, 'init_images', []), list):
        p.init_images = [i for i in p.init_images if i is not None]
    if len(getattr(p, 'init_images', [])) > 0:
        while len(p.init_images) < len(p.prompts):
            p.init_images.append(p.init_images[-1])

    # pipeline type is set earlier in processing, but check for sanity
    is_control = getattr(p, 'is_control', False) is True
    has_images = len(getattr(p, 'init_images', [])) > 0
    if (sd_models.get_diffusers_task(shared.sd_model) != sd_models.DiffusersTaskType.TEXT_2_IMAGE) and (not has_images) and (not is_control):
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE) # reset pipeline
    if hasattr(shared.sd_model, 'unet') and hasattr(shared.sd_model.unet, 'config') and hasattr(shared.sd_model.unet.config, 'in_channels') and shared.sd_model.unet.config.in_channels == 9 and not is_control:
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # force pipeline
        if len(getattr(p, 'init_images', [])) == 0:
            p.init_images = [images_sharpfin.to_pil(torch.rand((3, getattr(p, 'height', 512), getattr(p, 'width', 512))))]
    if not p.prompts:
        p.prompts = p.all_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]
    if not p.negative_prompts:
        p.negative_prompts = p.all_negative_prompts[p.iteration * p.batch_size:(p.iteration+1) * p.batch_size]

    sd_models_compile.openvino_recompile_model(p, hires=False, refiner=False) # recompile if a parameter changes

    if hasattr(p, 'dummy'):
        images = [Image.new(mode='RGB', size=(p.width, p.height))]
        return images
    if 'base' not in p.skip:
        output = process_base(p)
    else:
        images, _index=shared.history.selected
        output = SimpleNamespace(images=images)

    if (output is None or (hasattr(output, 'images') and len(output.images) == 0)) and has_images:
        if output is not None:
            shared.log.debug('Processing: using input as base output')
            output.images = p.init_images

    if shared.state.interrupted or shared.state.skipped:
        shared.sd_model = orig_pipeline
        return results

    if 'hires' not in p.skip:
        output = process_hires(p, output)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return results

    if 'refine' not in p.skip:
        output = process_refine(p, output)
        if shared.state.interrupted or shared.state.skipped:
            shared.sd_model = orig_pipeline
            return results

    extra_networks.deactivate(p)
    timer.process.add('lora', lora_common.timer.total)
    lora_common.timer.clear(complete=True)

    results = process_decode(p, output)
    timer.process.record('decode')

    shared.sd_model = orig_pipeline

    if p.state == '':
        global last_p # pylint: disable=global-statement
        last_p = p
    return results
