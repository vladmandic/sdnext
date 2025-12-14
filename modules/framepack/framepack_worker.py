import time
import torch
import rich.progress as rp
from sdnext_core import MODELDATA
from modules import shared, errors ,devices, sd_models, timer, memstats
from modules.framepack import framepack_vae # pylint: disable=wrong-import-order
from modules.framepack import framepack_hijack # pylint: disable=wrong-import-order
from modules.video_models.video_save import save_video # pylint: disable=wrong-import-order


stream = None # AsyncStream


def get_latent_paddings(mp4_fps, mp4_interpolate, latent_window_size, total_second_length, variant):
    try:
        real_fps = mp4_fps / (mp4_interpolate + 1)
        is_f1 = variant == 'forward-only'
        if is_f1:
            total_latent_sections = (total_second_length * real_fps) / (latent_window_size * 4)
            total_latent_sections = int(max(round(total_latent_sections), 1))
            latent_paddings = list(range(total_latent_sections))
        else:
            total_latent_sections = int(max((total_second_length * real_fps) / (latent_window_size * 4), 1))
            latent_paddings = list(reversed(range(total_latent_sections)))
            if total_latent_sections > 4: # extra padding for better quality
                # latent_paddings = list(reversed(range(total_latent_sections)))
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
    except Exception:
        latent_paddings = [0]
    return latent_paddings


def worker(
        input_image, end_image,
        start_weight, end_weight, vision_weight,
        prompts, n_prompt, system_prompt, optimized_prompt, unmodified_prompt,
        seed,
        total_second_length,
        latent_window_size,
        steps,
        cfg_scale, cfg_distilled, cfg_rescale,
        shift,
        use_teacache, use_cfgzero, use_preview,
        mp4_fps, mp4_codec, mp4_sf, mp4_video, mp4_frames, mp4_opt, mp4_ext, mp4_interpolate,
        vae_type,
        variant,
        metadata:dict={},
    ):
    timer.process.reset()
    memstats.reset_stats()
    if stream is None or shared.state.interrupted or shared.state.skipped:
        shared.log.error('FramePack: stream is None')
        stream.output_queue.push(('end', None))
        return

    from modules.framepack.pipeline import hunyuan
    from modules.framepack.pipeline import utils
    from modules.framepack.pipeline import k_diffusion_hunyuan

    is_f1 = variant == 'forward-only'
    total_generated_frames = 0
    total_generated_latent_frames = 0
    latent_paddings = get_latent_paddings(mp4_fps, mp4_interpolate, latent_window_size, total_second_length, variant)
    num_frames = latent_window_size * 4 - 3 # number of frames to generate in each section

    metadata['title'] = 'sdnext framepack'
    metadata['description'] = f'variant:{variant} seed:{seed} steps:{steps} scale:{cfg_scale} distilled:{cfg_distilled} rescale:{cfg_rescale} shift:{shift} start:{start_weight} end:{end_weight} vision:{vision_weight}'

    videojob = shared.state.begin('Video')
    shared.state.job_count = 1

    text_encoder = MODELDATA.sd_model.text_encoder
    text_encoder_2 = MODELDATA.sd_model.text_encoder_2
    tokenizer = MODELDATA.sd_model.tokenizer
    tokenizer_2 = MODELDATA.sd_model.tokenizer_2
    feature_extractor = MODELDATA.sd_model.feature_extractor
    image_encoder = MODELDATA.sd_model.image_processor
    transformer = MODELDATA.sd_model.transformer
    sd_models.apply_balanced_offload(MODELDATA.sd_model)
    pbar = rp.Progress(rp.TextColumn('[cyan]Video'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
    task = pbar.add_task('starting', total=steps * len(latent_paddings))
    t_last = time.time()
    if not is_f1:
        prompts = list(reversed(prompts))

    def text_encode(prompt, i:int=None):
        jobid = shared.state.begin('TE Encode')
        pbar.update(task, description=f'text encode section={i}')
        t0 = time.time()
        torch.manual_seed(seed)
        # shared.log.debug(f'FramePack: section={i} prompt="{prompt}"')
        shared.state.textinfo = 'Text encode'
        stream.output_queue.push(('progress', (None, 'Text encoding...')))
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        framepack_hijack.set_prompt_template(prompt, system_prompt, optimized_prompt, unmodified_prompt)
        llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        metadata['comment'] = prompt
        if cfg_scale > 1 and n_prompt is not None and len(n_prompt) > 0:
            llama_vec_n, clip_l_pooler_n = hunyuan.encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        else:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        llama_vec, llama_attention_mask = utils.crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = utils.crop_or_pad_yield_mask(llama_vec_n, length=512)
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        timer.process.add('prompt', time.time()-t0)
        shared.state.end(jobid)
        return llama_vec, llama_vec_n, llama_attention_mask, llama_attention_mask_n, clip_l_pooler, clip_l_pooler_n

    def latents_encode(input_image, end_image):
        jobid = shared.state.begin('VAE Encode')
        pbar.update(task, description='image encode')
        # shared.log.debug(f'FramePack: image encode init={input_image.shape} end={end_image.shape if end_image is not None else None}')
        t0 = time.time()
        torch.manual_seed(seed)
        stream.output_queue.push(('progress', (None, 'VAE encoding...')))
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        if input_image is not None:
            input_image_pt = torch.from_numpy(input_image).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
            start_latent = framepack_vae.vae_encode(input_image_pt)
        if start_weight < 1:
            noise = torch.randn_like(start_latent)
            start_latent = start_latent * start_weight + noise * (1 - start_weight)
        if end_image is not None:
            end_image_pt = torch.from_numpy(end_image).float() / 127.5 - 1
            end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]
            end_latent = framepack_vae.vae_encode(end_image_pt)
        else:
            end_latent = None
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        timer.process.add('encode', time.time()-t0)
        shared.state.end(jobid)
        return start_latent, end_latent

    def vision_encode(input_image, end_image):
        pbar.update(task, description='vision encode')
        # shared.log.debug(f'FramePack: vision encode init={input_image.shape} end={end_image.shape if end_image is not None else None}')
        t0 = time.time()
        shared.state.textinfo = 'Vision encode'
        stream.output_queue.push(('progress', (None, 'Vision encoding...')))
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        # siglip doesn't work with offload
        sd_models.move_model(feature_extractor, devices.device, force=True)
        sd_models.move_model(image_encoder, devices.device, force=True)
        preprocessed = feature_extractor.preprocess(images=input_image, return_tensors="pt").to(device=image_encoder.device, dtype=image_encoder.dtype)
        image_encoder_output = image_encoder(**preprocessed)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        if end_image is not None:
            preprocessed = feature_extractor.preprocess(images=end_image, return_tensors="pt").to(device=image_encoder.device, dtype=image_encoder.dtype)
            end_image_encoder_output = image_encoder(**preprocessed)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state * start_weight) + (end_image_encoder_last_hidden_state * end_weight) / (start_weight + end_weight) # use weighted approach
        image_encoder_last_hidden_state = image_encoder_last_hidden_state * vision_weight
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        timer.process.add('vision', time.time()-t0)
        return image_encoder_last_hidden_state

    def step_callback(d):
        if use_cfgzero and is_first_section and d['i'] == 0:
            d['denoised'] = d['denoised'] * 0
        t_current = time.time()
        if stream.input_queue.top() == 'end' or shared.state.interrupted or shared.state.skipped:
            stream.output_queue.push(('progress', (None, 'Interrupted...')))
            stream.output_queue.push(('end', None))
            raise AssertionError('Interrupted...')
        if shared.state.paused:
            shared.log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)
        nonlocal total_generated_frames, t_last
        t_preview = time.time()
        current_step = d['i'] + 1
        shared.state.textinfo = ''
        shared.state.sampling_step = ((lattent_padding_loop-1) * steps) + current_step
        shared.state.sampling_steps = steps * len(latent_paddings)
        progress = shared.state.sampling_step / shared.state.sampling_steps
        total_generated_frames = int(max(0, total_generated_latent_frames * 4 - 3))
        pbar.update(task, advance=1, description=f'its={1/(t_current-t_last):.2f} sample={d["i"]+1}/{steps} section={lattent_padding_loop}/{len(latent_paddings)} frames={total_generated_frames}/{num_frames*len(latent_paddings)}')
        desc = f'Step {shared.state.sampling_step}/{shared.state.sampling_steps} | Current {current_step}/{steps} | Section {lattent_padding_loop}/{len(latent_paddings)} | Progress {progress:.2%}'
        if use_preview:
            preview = framepack_vae.vae_decode(d['denoised'], 'Preview')
            stream.output_queue.push(('progress', (preview, desc)))
        else:
            stream.output_queue.push(('progress', (None, desc)))
        timer.process.add('preview', time.time() - t_preview)
        t_last = t_current

    try:
        with devices.inference_context(), pbar:
            t0 = time.time()

            height, width, _C = input_image.shape
            start_latent, end_latent = latents_encode(input_image, end_image)
            image_encoder_last_hidden_state = vision_encode(input_image, end_image)

            # Sample loop
            stream.output_queue.push(('progress', (None, 'Start sampling...')))
            generator = torch.Generator("cpu").manual_seed(seed)
            if is_f1:
                history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
            else:
                history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=devices.dtype).cpu()
            history_pixels = None
            lattent_padding_loop = 0
            last_prompt = None

            for latent_padding in latent_paddings:
                current_prompt = prompts[lattent_padding_loop]
                if current_prompt != last_prompt:
                    llama_vec, llama_vec_n, llama_attention_mask, llama_attention_mask_n, clip_l_pooler, clip_l_pooler_n = text_encode(current_prompt, i=lattent_padding_loop+1)
                    last_prompt = current_prompt

                sammplejob = shared.state.begin('Sample')
                lattent_padding_loop += 1
                # shared.log.trace(f'FramePack: op=sample section={lattent_padding_loop}/{len(latent_paddings)} frames={total_generated_frames}/{num_frames*len(latent_paddings)} window={latent_window_size} size={num_frames}')
                if is_f1:
                    is_first_section, is_last_section = False, False
                else:
                    is_first_section, is_last_section = latent_padding == latent_paddings[0], latent_padding == 0
                if stream.input_queue.top() == 'end' or shared.state.interrupted or shared.state.skipped:
                    stream.output_queue.push(('end', None))
                    return
                if is_f1:
                    indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                    clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
                else:
                    latent_padding_size = latent_padding * latent_window_size
                    indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                    clean_latent_indices_pre, _blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                    clean_latents_pre = start_latent.to(history_latents)
                    clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                    if end_image is not None and is_first_section:
                        clean_latents_post = (clean_latents_post * start_weight / len(latent_paddings)) + (end_weight * end_latent.to(history_latents)) / (start_weight/len(latent_paddings) + end_weight) # pylint: disable=possibly-used-before-assignment
                        clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                sd_models.apply_balanced_offload(MODELDATA.sd_model)
                transformer.initialize_teacache(enable_teacache=use_teacache, num_steps=steps, rel_l1_thresh=shared.opts.teacache_thresh)

                t_sample = time.time()
                generated_latents = k_diffusion_hunyuan.sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    num_inference_steps=steps,
                    real_guidance_scale=cfg_scale,
                    distilled_guidance_scale=cfg_distilled,
                    guidance_rescale=cfg_rescale,
                    shift=shift if shift > 0 else None,
                    generator=generator,
                    prompt_embeds=llama_vec, # pylint: disable=possibly-used-before-assignment
                    prompt_embeds_mask=llama_attention_mask, # pylint: disable=possibly-used-before-assignment
                    prompt_poolers=clip_l_pooler, # pylint: disable=possibly-used-before-assignment
                    negative_prompt_embeds=llama_vec_n, # pylint: disable=possibly-used-before-assignment
                    negative_prompt_embeds_mask=llama_attention_mask_n, # pylint: disable=possibly-used-before-assignment
                    negative_prompt_poolers=clip_l_pooler_n, # pylint: disable=possibly-used-before-assignment
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    device=devices.device,
                    dtype=devices.dtype,
                    callback=step_callback,
                )

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                total_generated_latent_frames += int(generated_latents.shape[2])

                if is_f1:
                    history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
                    real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]
                else:
                    history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                    real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                sd_models.apply_balanced_offload(MODELDATA.sd_model)
                timer.process.add('sample', time.time()-t_sample)
                shared.state.end(sammplejob)

                t_vae = time.time()
                if history_pixels is None:
                    history_pixels = framepack_vae.vae_decode(real_history_latents, vae_type=vae_type).cpu()
                else:
                    overlapped_frames = latent_window_size * 4 - 3
                    if is_f1:
                        section_latent_frames = latent_window_size * 2
                        current_pixels = framepack_vae.vae_decode(real_history_latents[:, :, -section_latent_frames:], vae_type=vae_type).cpu()
                        history_pixels = utils.soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)
                    else:
                        section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                        current_pixels = framepack_vae.vae_decode(real_history_latents[:, :, :section_latent_frames], vae_type=vae_type).cpu()
                        history_pixels = utils.soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
                sd_models.apply_balanced_offload(MODELDATA.sd_model)
                timer.process.add('vae', time.time()-t_vae)

                if is_last_section:
                    break

            total_generated_frames, _video_filename = save_video(
                None,
                history_pixels,
                mp4_fps,
                mp4_codec,
                mp4_opt,
                mp4_ext,
                mp4_sf,
                mp4_video,
                mp4_frames,
                mp4_interpolate,
                pbar=pbar,
                stream=stream,
                metadata=metadata,
            )

    except AssertionError:
        shared.log.info('FramePack: interrupted')
        if shared.opts.keep_incomplete:
            save_video(None, history_pixels, mp4_fps, mp4_codec, mp4_opt, mp4_ext, mp4_sf, mp4_video, mp4_frames, mp4_interpolate=0, stream=stream, metadata=metadata)
    except Exception as e:
        shared.log.error(f'FramePack: {e}')
        errors.display(e, 'FramePack')

    sd_models.apply_balanced_offload(MODELDATA.sd_model)
    stream.output_queue.push(('end', None))
    t1 = time.time()
    shared.log.info(f'Processed: frames={total_generated_frames} fps={total_generated_frames/(t1-t0):.2f} its={(shared.state.sampling_step)/(t1-t0):.2f} time={t1-t0:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')
    shared.state.end(videojob)
