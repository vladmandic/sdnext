import os
from installer import git_commit
from modules import shared, sd_samplers_common, sd_vae, generation_parameters_copypaste
from modules.processing_class import StableDiffusionProcessing


args = {} # maintain history
infotext = '' # maintain history
debug = shared.log.trace if os.environ.get('SD_PROCESS_DEBUG', None) is not None else lambda *args, **kwargs: None


def get_last_args():
    return args, infotext


def create_infotext(p: StableDiffusionProcessing, all_prompts=None, all_seeds=None, all_subseeds=None, comments=None, iteration=0, position_in_batch=0, index=None, all_negative_prompts=None, grid=None):
    global args, infotext # pylint: disable=global-statement
    if p is None:
        shared.log.warning('Processing info: no data')
        return ''
    if not hasattr(shared.sd_model, 'sd_checkpoint_info'):
        return ''
    if index is None:
        index = position_in_batch + iteration * p.batch_size
    if all_prompts is None:
        all_prompts = p.all_prompts or [p.prompt]
    if all_negative_prompts is None:
        all_negative_prompts = p.all_negative_prompts or [p.negative_prompt]
    if all_seeds is None:
        all_seeds = p.all_seeds or [p.seed]
    if all_subseeds is None:
        all_subseeds = p.all_subseeds or [p.subseed]
    while len(all_prompts) <= index:
        all_prompts.insert(0, p.prompt)
    while len(all_seeds) <= index:
        all_seeds.insert(0, int(p.seed))
    while len(all_subseeds) <= index:
        all_subseeds.insert(0, int(p.subseed))
    while len(all_negative_prompts) <= index:
        all_negative_prompts.insert(0, p.negative_prompt)
    comment = ', '.join(comments) if comments is not None and type(comments) is list else None
    ops = list(set(p.ops))
    args = {
        # basic
        "Steps": p.steps,
        "Size": f"{p.width}x{p.height}" if hasattr(p, 'width') and hasattr(p, 'height') else None,
        "Sampler": p.sampler_name if p.sampler_name != 'Default' else None,
        "Seed": all_seeds[index],
        "Seed resize from": None if p.seed_resize_from_w <= 0 or p.seed_resize_from_h <= 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}",
        "CFG scale": p.cfg_scale if p.cfg_scale > 1.0 else 1.0,
        "CFG rescale": p.diffusers_guidance_rescale if p.diffusers_guidance_rescale > 0 else None,
        "CFG end": p.cfg_end if p.cfg_end < 1.0 else None,
        "CFG true": p.pag_scale if p.pag_scale > 0 else None,
        "CFG adaptive": p.pag_adaptive if p.pag_adaptive != 0.5 else None,
        "Clip skip": p.clip_skip if p.clip_skip > 1 else None,
        "Batch": f'{p.n_iter}x{p.batch_size}' if p.n_iter > 1 or p.batch_size > 1 else None,
        "Refiner prompt": p.refiner_prompt if len(p.refiner_prompt) > 0 else None,
        "Refiner negative": p.refiner_negative if len(p.refiner_negative) > 0 else None,
        "Styles": "; ".join(p.styles) if p.styles is not None and len(p.styles) > 0 else None,
        "App": 'SD.Next',
        "Version": git_commit,
        "Parser": shared.opts.prompt_attention if shared.opts.prompt_attention != 'native' else None,
        "Comment": comment,
        "Pipeline": shared.sd_model.__class__.__name__,
        "TE": None if (shared.opts.sd_text_encoder is None or shared.opts.sd_text_encoder == 'Default') else shared.opts.sd_text_encoder,
        "UNet": None if (shared.opts.sd_unet is None or shared.opts.sd_unet == 'Default') else shared.opts.sd_unet,
        "Operations": '; '.join(ops).replace('"', '') if len(p.ops) > 0 else 'none',
    }
    if shared.opts.add_model_name_to_info:
        if getattr(shared.sd_model, 'sd_checkpoint_info', None) is not None:
            args["Model"] = shared.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')
    if shared.opts.add_model_hash_to_info:
        if getattr(p, 'sd_model_hash', None) is not None:
            args["Model hash"] = p.sd_model_hash
        elif getattr(shared.sd_model, 'sd_model_hash', None) is not None:
            args["Model hash"] = shared.sd_model.sd_model_hash
    if p.vae_type == 'Full':
        args["VAE"] = (None if not shared.opts.add_model_name_to_info or sd_vae.loaded_vae_file is None else os.path.splitext(os.path.basename(sd_vae.loaded_vae_file))[0])
    elif p.vae_type == 'Tiny':
        args["VAE"] = 'TAESD'
    elif p.vae_type == 'REPA-E':
        args["VAE"] = 'REPA-E'
    elif p.vae_type == 'Remote':
        args["VAE"] = 'Remote'
    if grid is None and (p.n_iter > 1 or p.batch_size > 1) and index >= 0:
        args['Index'] = f'{p.iteration + 1}x{index + 1}'
    if grid is not None:
        args['Grid'] = grid
    if 'txt2img' in p.ops:
        args["Variation seed"] = all_subseeds[index] if p.subseed_strength > 0 else None
        args["Variation strength"] = p.subseed_strength if p.subseed_strength > 0 else None
    if 'hires' in p.ops or 'upscale' in p.ops:
        is_resize = p.hr_resize_mode > 0 and (p.hr_upscaler != 'None' or p.hr_resize_mode == 5)
        is_fixed = p.hr_resize_x > 0 or p.hr_resize_y > 0
        args["Refine"] = p.enable_hr
        if is_resize:
            args["HiRes mode"] = p.hr_resize_mode
            args["HiRes context"] = p.hr_resize_context if p.hr_resize_mode == 5 else None
            args["Hires upscaler"] = p.hr_upscaler
            if is_fixed:
                args["Hires fixed"] = f"{p.hr_resize_x}x{p.hr_resize_y}"
            else:
                args["Hires scale"] = p.hr_scale
            args["Hires size"] = f"{p.hr_upscale_to_x}x{p.hr_upscale_to_y}"
        if p.hr_force or ('Latent' in p.hr_upscaler):
            args["Hires force"] = p.hr_force
            args["Hires steps"] = p.hr_second_pass_steps
            args["Hires strength"] = p.hr_denoising_strength
            args["Hires sampler"] = p.hr_sampler_name
            args["Hires CFG scale"] = p.image_cfg_scale
    if 'refine' in p.ops:
        args["Refine"] = p.enable_hr
        args["Refiner"] = None if (not shared.opts.add_model_name_to_info) or (not shared.sd_refiner) or (not shared.sd_refiner.sd_checkpoint_info.model_name) else shared.sd_refiner.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')
        args['Hires CFG scale'] = p.image_cfg_scale
        args['Refiner steps'] = p.refiner_steps
        args['Refiner start'] = p.refiner_start
        args["Hires steps"] = p.hr_second_pass_steps
        args["Hires sampler"] = p.hr_sampler_name
    if ('img2img' in p.ops or 'inpaint' in p.ops) and ('txt2img' not in p.ops and 'hires' not in p.ops): # real img2img/inpaint
        args["Init image size"] = f"{getattr(p, 'init_img_width', 0)}x{getattr(p, 'init_img_height', 0)}"
        args["Init image hash"] = getattr(p, 'init_img_hash', None)
        args['Image CFG scale'] = p.image_cfg_scale
        args["Mask weight"] = getattr(p, "inpainting_mask_weight", shared.opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None
        args["Denoising strength"] = getattr(p, 'denoising_strength', None)
        if args["Size"] != args["Init image size"]:
            args['Resize scale'] = float(getattr(p, 'scale_by', None)) if getattr(p, 'scale_by', None) != 1 else None
            args['Resize mode'] = shared.resize_modes[p.resize_mode] if shared.resize_modes[p.resize_mode] != 'None' else None
        if args["Size"] is None:
            args["Size"] = args["Init image size"]
    if p.resize_mode_before != 0 and p.resize_name_before != 'None' and hasattr(p, 'init_images') and p.init_images is not None and len(p.init_images) > 0:
        args['Resize before'] = f"{p.width_before}x{p.height_before}"
        args['Resize mode before'] = p.resize_mode_before
        args['Resize name before'] = p.resize_name_before
        args['Resize scale before'] = float(p.scale_by_before) if p.scale_by_before != 1.0 else None
    if p.resize_mode_after != 0 and p.resize_name_after != 'None':
        args['Resize after'] = f"{p.width_after}x{p.height_after}"
        args['Resize mode after'] = p.resize_mode_after
        args['Resize name after'] = p.resize_name_after
        args['Resize scale after'] = float(p.scale_by_after) if p.scale_by_after != 1.0 else None
    if p.resize_name_mask != 'None' and p.scale_by_mask != 1.0:
        args['Resize mask'] = f"{p.width_mask}x{p.height_mask}"
        args['Resize mode mask'] = p.resize_mode_mask
        args['Resize name mask'] = p.resize_name_mask
        args['Resize scale mask'] = float(p.scale_by_mask)
    if 'detailer' in p.ops:
        args["Detailer"] = ', '.join(shared.opts.detailer_models) if len(shared.opts.detailer_args) == 0 else shared.opts.detailer_args
        args["Detailer steps"] = p.detailer_steps
        args["Detailer strength"] = p.detailer_strength
        args["Detailer resolution"] = p.detailer_resolution if p.detailer_resolution != 1024 else None
        args["Detailer prompt"] = p.detailer_prompt if len(p.detailer_prompt) > 0 else None
        args["Detailer negative"] = p.detailer_negative if len(p.detailer_negative) > 0 else None
    if 'color' in p.ops:
        args["Color correction"] = True
    if shared.opts.token_merging_method == 'ToMe': # tome/todo
        args['ToMe'] = shared.opts.tome_ratio if shared.opts.tome_ratio != 0 else None
    else:
        args['ToDo'] = shared.opts.todo_ratio if shared.opts.todo_ratio != 0 else None
    if hasattr(shared.sd_model, 'embedding_db') and len(shared.sd_model.embedding_db.embeddings_used) > 0: # register used embeddings
        args['Embeddings'] = ', '.join(shared.sd_model.embedding_db.embeddings_used)

    # samplers
    if getattr(p, 'sampler_name', None) is not None and p.sampler_name.lower() != 'default':
        args["Sampler eta delta"] = shared.opts.eta_noise_seed_delta if shared.opts.eta_noise_seed_delta != 0 and sd_samplers_common.is_sampler_using_eta_noise_seed_delta(p) else None
        args["Sampler eta multiplier"] = p.initial_noise_multiplier if getattr(p, 'initial_noise_multiplier', 1.0) != 1.0 else None
        args['Sampler timesteps'] = shared.opts.schedulers_timesteps if shared.opts.schedulers_timesteps != shared.opts.data_labels.get('schedulers_timesteps').default else None
        args['Sampler spacing'] = shared.opts.schedulers_timestep_spacing if shared.opts.schedulers_timestep_spacing != shared.opts.data_labels.get('schedulers_timestep_spacing').default else None
        args['Sampler sigma'] = shared.opts.schedulers_sigma if shared.opts.schedulers_sigma != shared.opts.data_labels.get('schedulers_sigma').default else None
        args['Sampler order'] = shared.opts.schedulers_solver_order if shared.opts.schedulers_solver_order != shared.opts.data_labels.get('schedulers_solver_order').default else None
        args['Sampler type'] = shared.opts.schedulers_prediction_type if shared.opts.schedulers_prediction_type != shared.opts.data_labels.get('schedulers_prediction_type').default else None
        args['Sampler beta schedule'] = shared.opts.schedulers_beta_schedule if shared.opts.schedulers_beta_schedule != shared.opts.data_labels.get('schedulers_beta_schedule').default else None
        args['Sampler low order'] = shared.opts.schedulers_use_loworder if shared.opts.schedulers_use_loworder != shared.opts.data_labels.get('schedulers_use_loworder').default else None
        args['Sampler dynamic'] = shared.opts.schedulers_use_thresholding if shared.opts.schedulers_use_thresholding != shared.opts.data_labels.get('schedulers_use_thresholding').default else None
        args['Sampler rescale'] = shared.opts.schedulers_rescale_betas if shared.opts.schedulers_rescale_betas != shared.opts.data_labels.get('schedulers_rescale_betas').default else None
        args['Sampler beta start'] = shared.opts.schedulers_beta_start if shared.opts.schedulers_beta_start != shared.opts.data_labels.get('schedulers_beta_start').default else None
        args['Sampler beta end'] = shared.opts.schedulers_beta_end if shared.opts.schedulers_beta_end != shared.opts.data_labels.get('schedulers_beta_end').default else None
        args['Sampler range'] = shared.opts.schedulers_timesteps_range if shared.opts.schedulers_timesteps_range != shared.opts.data_labels.get('schedulers_timesteps_range').default else None
        args['Sampler shift'] = shared.opts.schedulers_shift if shared.opts.schedulers_shift != shared.opts.data_labels.get('schedulers_shift').default else None
        args['Sampler dynamic shift'] = shared.opts.schedulers_dynamic_shift if shared.opts.schedulers_dynamic_shift != shared.opts.data_labels.get('schedulers_dynamic_shift').default else None

    # model specific
    if shared.sd_model_type == 'h1':
        args['LLM'] =  None if shared.opts.model_h1_llama_repo == 'Default' else shared.opts.model_h1_llama_repo

    args.update(p.extra_generation_params)
    for k, v in args.copy().items():
        if v is None:
            del args[k]
        if type(v) is float or type(v) is int:
            if v <= -1:
                del args[k]
        if isinstance(v, str):
            if len(v) == 0 or v == '0x0':
                del args[k]
    debug(f'Infotext: args={args}')
    params_text = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in args.items()])

    if hasattr(p, 'original_prompt'):
        args['Original prompt'] = p.original_prompt
    if hasattr(p, 'original_negative'):
        args['Original negative'] = p.original_negative

    negative_prompt_text = f"\nNegative prompt: {all_negative_prompts[index] if all_negative_prompts[index] else ''}"
    infotext = f"{all_prompts[index]}{negative_prompt_text}\n{params_text}".strip()
    debug(f'Infotext: "{infotext}"')
    return infotext
