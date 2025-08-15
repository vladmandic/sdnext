import os
import itertools # SBM Batch frames
import numpy as np
import filetype
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageChops, UnidentifiedImageError
from modules import scripts_manager, shared, processing, images, errors
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.ui_common import plaintext_to_html
from modules.memstats import memory_stats


debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PROCESS')


def validate_inputs(inputs):
    outputs = []
    for image in inputs:
        if filetype.is_image(image):
            outputs.append(image)
        else:
            shared.log.warning(f'Input skip: file="{image}" filetype={filetype.guess(image)}')
    return outputs


def process_batch(p, input_files, input_dir, output_dir, inpaint_mask_dir, args):
    # shared.log.debug(f'batch: {input_files}|{input_dir}|{output_dir}|{inpaint_mask_dir}')
    processing.fix_seed(p)
    image_files = []
    if input_files is not None and len(input_files) > 0:
        image_files = [f.name for f in input_files]
        image_files = validate_inputs(image_files)
        shared.log.info(f'Process batch: input images={len(image_files)}')
    elif os.path.isdir(input_dir):
        image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)]
        image_files = validate_inputs(image_files)
        shared.log.info(f'Process batch: input folder="{input_dir}" images={len(image_files)}')
    is_inpaint_batch = False
    if inpaint_mask_dir and os.path.isdir(inpaint_mask_dir):
        inpaint_masks = [os.path.join(inpaint_mask_dir, f) for f in os.listdir(inpaint_mask_dir)]
        inpaint_masks = validate_inputs(inpaint_masks)
        is_inpaint_batch = len(inpaint_masks) > 0
        shared.log.info(f'Process batch: mask folder="{input_dir}" images={len(inpaint_masks)}')
    p.do_not_save_grid = True
    p.do_not_save_samples = True
    p.default_prompt = p.prompt
    if p.n_iter > 1:
        p.n_iter = 1
        shared.log.warning(f'Process batch: batch_count={p.n_iter} forced to 1')
    shared.state.job_count = len(image_files) * p.n_iter
    if shared.opts.batch_frame_mode: # SBM Frame mode is on, process each image in batch with same seed
        window_size = p.batch_size
        btcrept = 1
        p.seed = [p.seed] * window_size # SBM MONKEYPATCH: Need to change processing to support a fixed seed value.
        p.subseed = [p.subseed] * window_size # SBM MONKEYPATCH
        shared.log.info(f"Process batch: inputs={len(image_files)} outputs={p.n_iter}x{len(image_files)} parallel={window_size}")
    else: # SBM Frame mode is off, standard operation of repeating same images with sequential seed.
        window_size = 1
        btcrept = p.batch_size
        shared.log.info(f"Process batch: inputs={len(image_files)} outputs={p.n_iter*p.batch_size}x{len(image_files)}")
    for i in range(0, len(image_files), window_size):
        if shared.state.skipped:
            shared.state.skipped = False
        if shared.state.interrupted:
            break
        batch_image_files = image_files[i:i+window_size]
        batch_images = []
        for image_file in batch_image_files:
            try:
                img = Image.open(image_file)
                img = ImageOps.exif_transpose(img)
                batch_images.append(img)
                # p.init()
                p.width = int(img.width * p.scale_by)
                p.height = int(img.height * p.scale_by)
                caption_file = os.path.splitext(image_file)[0] + '.txt'
                prompt_type='default'
                if os.path.exists(caption_file):
                    with open(caption_file, 'r', encoding='utf8') as f:
                        p.prompt = f.read()
                        prompt_type='file'
                else:
                    p.prompt = p.default_prompt
                p.all_prompts = None
                p.all_negative_prompts = None
                p.all_seeds = None
                p.all_subseeds = None
                shared.log.debug(f'Process batch: image="{image_file}" prompt={prompt_type} i={i+1}/{len(image_files)}')
            except UnidentifiedImageError as e:
                shared.log.error(f'Process batch: image="{image_file}" {e}')
        if len(batch_images) == 0:
            shared.log.warning("Process batch: no images found in batch")
            continue
        batch_images = batch_images * btcrept # Standard mode sends the same image per batchsize.
        p.init_images = batch_images

        if is_inpaint_batch:
            # try to find corresponding mask for an image using simple filename matching
            batch_mask_images = []
            for image_file in batch_image_files:
                mask_image_path = os.path.join(inpaint_mask_dir, os.path.basename(image_file))
                # if not found use first one ("same mask for all images" use-case)
                if mask_image_path not in inpaint_masks:
                    mask_image_path = inpaint_masks[0]
                mask_image = Image.open(mask_image_path)
                batch_mask_images.append(mask_image)
            batch_mask_images = batch_mask_images * btcrept
            p.image_mask = batch_mask_images

        batch_image_files = batch_image_files * btcrept # List used for naming later.

        try:
            processed = scripts_manager.scripts_img2img.run(p, *args)
            if processed is None:
                processed = processing.process_images(p)
        except Exception as e:
            shared.log.error(f'Process batch: {e}')
            errors.display(e, 'batch')
            processed = None

        if processed is None or len(processed.images) == 0:
            shared.log.warning(f'Process batch: i={i+1}/{len(image_files)} no images processed')
            continue

        for n, (image, image_file) in enumerate(itertools.zip_longest(processed.images, batch_image_files)):
            if image is None:
                continue
            basename = ''
            if shared.opts.use_original_name_batch:
                forced_filename, ext = os.path.splitext(os.path.basename(image_file))
            else:
                forced_filename = None
                ext = shared.opts.samples_format
            if len(processed.images) > 1:
                basename = f'{n + i}' if shared.opts.batch_frame_mode else f'{n}'
            else:
                basename = ''
            if output_dir == '':
                output_dir = shared.opts.outdir_img2img_samples
            os.makedirs(output_dir, exist_ok=True)
            info, items = images.read_info_from_image(image)
            for k, v in items.items():
                image.info[k] = v
            images.save_image(image, path=output_dir, basename=basename, seed=None, prompt=None, extension=ext, info=info, grid=False, pnginfo_section_name="extras", existing_info=image.info, forced_filename=forced_filename)
        processed = scripts_manager.scripts_img2img.after(p, processed, *args)
        shared.log.debug(f'Processed: images={len(batch_image_files)} memory={memory_stats()} batch')


def img2img(id_task: str, state: str, mode: int,
            prompt, negative_prompt, prompt_styles,
            init_img,
            sketch,
            init_img_with_mask,
            inpaint_color_sketch,
            inpaint_color_sketch_orig,
            init_img_inpaint,
            init_mask_inpaint,
            steps,
            sampler_index,
            mask_blur, mask_alpha,
            vae_type, tiling, hidiffusion,
            detailer_enabled, detailer_prompt, detailer_negative, detailer_steps, detailer_strength,
            n_iter, batch_size,
            cfg_scale, image_cfg_scale,
            diffusers_guidance_rescale,
            pag_scale, pag_adaptive,
            cfg_end,
            refiner_start,
            clip_skip,
            denoising_strength,
            seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w,
            selected_scale_tab,
            height, width,
            scale_by,
            resize_mode, resize_name, resize_context,
            inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert,
            img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir,
            hdr_mode, hdr_brightness, hdr_color, hdr_sharpen, hdr_clamp, hdr_boundary, hdr_threshold, hdr_maximize, hdr_max_center, hdr_max_boundary, hdr_color_picker, hdr_tint_ratio,
            enable_hr, hr_sampler_index, hr_denoising_strength, hr_resize_mode, hr_resize_context, hr_upscaler, hr_force, hr_second_pass_steps, hr_scale, hr_resize_x, hr_resize_y, refiner_steps, hr_refiner_start, refiner_prompt, refiner_negative,
            override_settings_texts,
            *args):


    debug(f'img2img: {id_task}')

    if shared.sd_model is None:
        shared.log.warning('Aborted: op=img model not loaded')
        return [], '', '', 'Error: model not loaded'

    if sampler_index is None:
        shared.log.warning('Sampler: invalid')
        sampler_index = 0

    mode = int(mode)
    image = None
    mask = None
    override_settings = create_override_settings_dict(override_settings_texts)

    if mode == 0: # img2img
        if init_img is None:
            return [], '', '', 'Error: init image not provided'
        image = init_img.convert("RGB")
    elif mode == 1: # inpaint
        if init_img_with_mask is None:
            return [], '', '', 'Error: init image with mask not provided'
        image = init_img_with_mask["image"]
        mask = init_img_with_mask["mask"]
        alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
        mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
        image = image.convert("RGB")
    elif mode == 2:  # sketch
        if sketch is None:
            return [], '', '', 'Error: sketch image not provided'
        image = sketch.convert("RGB")
    elif mode == 3: # composite
        if inpaint_color_sketch is None:
            return [], '', '', 'Error: color sketch image not provided'
        image = inpaint_color_sketch
        orig = inpaint_color_sketch_orig or inpaint_color_sketch
        pred = np.any(np.array(image) != np.array(orig), axis=-1)
        mask = Image.fromarray((255.0 * pred).astype(np.uint8), "L")
        mask = ImageEnhance.Brightness(mask).enhance(mask_alpha)
        blur = ImageFilter.GaussianBlur(mask_blur)
        image = Image.composite(image.filter(blur), orig, mask.filter(blur))
        image = image.convert("RGB")
    elif mode == 4: # inpaint upload mask
        if init_img_inpaint is None:
            return [], '', '', 'Error: inpaint image not provided'
        image = init_img_inpaint
        mask = init_mask_inpaint
    elif mode == 5: # process batch
        pass # handled later
    else:
        shared.log.error(f'Image processing unknown mode: {mode}')

    if image is not None:
        image = ImageOps.exif_transpose(image)
        if selected_scale_tab == 1 and resize_mode != 0:
            width = int(image.width * scale_by)
            height = int(image.height * scale_by)

    p = processing.StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=shared.opts.outdir_samples or shared.opts.outdir_img2img_samples,
        outpath_grids=shared.opts.outdir_grids or shared.opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=prompt_styles,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        sampler_name = processing.get_sampler_name(sampler_index, img=True),
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        cfg_end=cfg_end,
        clip_skip=clip_skip,
        width=width,
        height=height,
        vae_type=vae_type,
        tiling=tiling,
        hidiffusion=hidiffusion,
        detailer_enabled=detailer_enabled,
        detailer_prompt=detailer_prompt,
        detailer_negative=detailer_negative,
        detailer_steps=detailer_steps,
        detailer_strength=detailer_strength,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        resize_mode=resize_mode,
        resize_name=resize_name,
        resize_context=resize_context,
        scale_by=scale_by,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        diffusers_guidance_rescale=diffusers_guidance_rescale,
        pag_scale=pag_scale,
        pag_adaptive=pag_adaptive,
        refiner_start=refiner_start,
        inpaint_full_res=inpaint_full_res != 0,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        hdr_mode=hdr_mode, hdr_brightness=hdr_brightness, hdr_color=hdr_color, hdr_sharpen=hdr_sharpen, hdr_clamp=hdr_clamp,
        hdr_boundary=hdr_boundary, hdr_threshold=hdr_threshold, hdr_maximize=hdr_maximize, hdr_max_center=hdr_max_center, hdr_max_boundary=hdr_max_boundary, hdr_color_picker=hdr_color_picker, hdr_tint_ratio=hdr_tint_ratio,
        # refiner
        enable_hr=enable_hr,
        hr_denoising_strength=hr_denoising_strength,
        hr_scale=hr_scale,
        hr_resize_mode=hr_resize_mode,
        hr_resize_context=hr_resize_context,
        hr_upscaler=hr_upscaler,
        hr_force=hr_force,
        hr_second_pass_steps=hr_second_pass_steps,
        hr_resize_x=hr_resize_x,
        hr_resize_y=hr_resize_y,
        hr_sampler_name = processing.get_sampler_name(hr_sampler_index),
        refiner_steps=refiner_steps,
        hr_refiner_start=hr_refiner_start,
        refiner_prompt=refiner_prompt,
        refiner_negative=refiner_negative,
        # override
        override_settings=override_settings,
    )
    p.scripts = scripts_manager.scripts_img2img
    p.script_args = args
    p.state = state
    if mask:
        p.extra_generation_params["Mask blur"] = mask_blur
        p.extra_generation_params["Mask alpha"] = mask_alpha
        p.extra_generation_params["Mask padding"] = inpaint_full_res_padding
        p.extra_generation_params["Mask invert"] = ['masked', 'invert'][inpainting_mask_invert]
        p.extra_generation_params["Mask area"] = ["full", "masked"][inpaint_full_res]
    p.is_batch = mode == 5
    if p.is_batch:
        process_batch(p, img2img_batch_files, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, args)
        processed = processing.Processed(p, [], p.seed, "")
    else:
        processed = scripts_manager.scripts_img2img.run(p, *args)
        if processed is None:
            processed = processing.process_images(p)
        processed = scripts_manager.scripts_img2img.after(p, processed, *args)
    p.close()
    generation_info_js = processed.js() if processed is not None else ''
    if processed is None:
        return [], generation_info_js, '', 'Error: no images'
    return processed.images, generation_info_js, processed.info, plaintext_to_html(processed.comments)
