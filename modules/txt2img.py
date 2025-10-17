import os
from modules import shared, processing, scripts_manager
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui_common import plaintext_to_html


debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')


def txt2img(id_task, state,
            prompt, negative_prompt, prompt_styles,
            steps, sampler_index, hr_sampler_index,
            vae_type, tiling, hidiffusion,
            detailer_enabled, detailer_prompt, detailer_negative, detailer_steps, detailer_strength, detailer_resolution,
            n_iter, batch_size,
            guidance_name, guidance_scale, guidance_rescale, guidance_start, guidance_stop,
            cfg_scale, image_cfg_scale, diffusers_guidance_rescale, pag_scale, pag_adaptive, cfg_end,
            clip_skip,
            seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
            height, width,
            enable_hr, denoising_strength,
            hr_scale, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_resize_x, hr_resize_y,
            refiner_steps, refiner_start, refiner_prompt, refiner_negative,
            hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio,
            override_settings_texts,
            *args):

    debug(f'txt2img: {id_task}')

    if shared.sd_model is None:
        shared.log.warning('Aborted: op=txt model not loaded')
        return [], '', '', 'Error: model not loaded'

    override_settings = create_override_settings_dict(override_settings_texts)
    if sampler_index is None:
        shared.log.warning('Sampler: invalid')
        sampler_index = 0
    if hr_sampler_index is None:
        hr_sampler_index = sampler_index

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_txt2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=prompt_styles,
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name = processing.get_sampler_name(sampler_index),
        hr_sampler_name = processing.get_sampler_name(hr_sampler_index),
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        guidance_name=guidance_name,
        guidance_scale=guidance_scale,
        guidance_rescale=guidance_rescale,
        guidance_start=guidance_start,
        guidance_stop=guidance_stop,
        cfg_scale=cfg_scale,
        image_cfg_scale=image_cfg_scale,
        diffusers_guidance_rescale=diffusers_guidance_rescale,
        pag_scale=pag_scale,
        pag_adaptive=pag_adaptive,
        cfg_end=cfg_end,
        clip_skip=clip_skip,
        width=width,
        height=height,
        vae_type=vae_type,
        detailer_enabled=detailer_enabled,
        detailer_prompt=detailer_prompt,
        detailer_negative=detailer_negative,
        detailer_steps=detailer_steps,
        detailer_strength=detailer_strength,
        detailer_resolution=detailer_resolution,
        tiling=tiling,
        hidiffusion=hidiffusion,
        enable_hr=enable_hr,
        denoising_strength=denoising_strength,
        hr_scale=hr_scale,
        hr_resize_mode=hr_resize_mode,
        hr_resize_context=hr_resize_context,
        hr_upscaler=hr_upscaler,
        hr_force=hr_force,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        refiner_steps=refiner_steps,
        refiner_start=refiner_start,
        refiner_prompt=refiner_prompt,
        refiner_negative=refiner_negative,
        hdr_mode=hdr_mode, hdr_brightness=hdr_brightness, hdr_color=hdr_color, hdr_sharpen=hdr_sharpen, hdr_clamp=hdr_clamp,
        hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold, hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundary=hdr_max_boundary, hdr_color_picker=hdr_color_picker, hdr_tint_ratio=hdr_tint_ratio,
        override_settings=override_settings,
    )
    p.scripts = scripts_manager.scripts_txt2img
    p.script_args = args
    p.state = state
    processed: processing.Processed = scripts_manager.scripts_txt2img.run(p, *args)
    if processed is None:
        processed = processing.process_images(p)
    processed = scripts_manager.scripts_txt2img.after(p, processed, *args)
    p.close()
    if processed is None:
        return [], '', '', 'Error: processing failed'
    generation_info_js = processed.js() if processed is not None else ''
    if processed is None:
        return [], generation_info_js, '', 'Error: no images'
    return processed.images, generation_info_js, processed.info, plaintext_to_html(processed.comments)
