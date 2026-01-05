import typing
import os
import re
import math
import time
import inspect
import torch
import numpy as np
from PIL import Image
from modules import shared, errors, sd_models, processing, processing_vae, processing_helpers, sd_hijack_hypertile, prompt_parser_diffusers, timer, extra_networks, sd_vae
from modules.processing_callbacks import diffusers_callback_legacy, diffusers_callback, set_callbacks_p
from modules.processing_helpers import resize_hires, fix_prompts, calculate_base_steps, calculate_hires_steps, calculate_refiner_steps, get_generator, set_latents, apply_circular # pylint: disable=unused-import
from modules.api import helpers


debug_enabled = os.environ.get('SD_DIFFUSERS_DEBUG', None)
debug_log = shared.log.trace if debug_enabled else lambda *args, **kwargs: None
disable_pbar = os.environ.get('SD_DISABLE_PBAR', None) is not None


def task_modular_kwargs(p, model): # pylint: disable=unused-argument
    # model_cls = model.__class__.__name__
    task_args = {}
    p.ops.append('modular')

    processing_helpers.resize_init_images(p)
    task_args['width'] = p.width
    task_args['height'] = p.height
    if len(getattr(p, 'init_images', [])) > 0:
        task_args['image'] = p.init_images
        task_args['strength'] = p.denoising_strength
    mask_image = p.task_args.get('image_mask', None) or getattr(p, 'image_mask', None) or getattr(p, 'mask', None)
    if mask_image is not None:
        task_args['mask_image'] = mask_image

    if debug_enabled:
        debug_log(f'Process task specific args: {task_args}')
    return task_args


def task_specific_kwargs(p, model):
    model_cls = model.__class__.__name__
    vae_scale_factor = sd_vae.get_vae_scale_factor(model)
    task_args = {}
    is_img2img_model = bool('Zero123' in model_cls)
    task_type = sd_models.get_diffusers_task(model)
    if len(getattr(p, 'init_images', [])) > 0:
        if isinstance(p.init_images[0], str):
            p.init_images = [helpers.decode_base64_to_image(i, quiet=True) for i in p.init_images]
        if isinstance(p.init_images[0], Image.Image):
            p.init_images = [i.convert('RGB') if i.mode != 'RGB' else i for i in p.init_images if i is not None]
    width, height = processing_helpers.resize_init_images(p)
    if (task_type == sd_models.DiffusersTaskType.TEXT_2_IMAGE or len(getattr(p, 'init_images', [])) == 0) and not is_img2img_model and 'video' not in p.ops:
        p.ops.append('txt2img')
        if hasattr(p, 'width') and hasattr(p, 'height'):
            task_args = {
                'width': width,
                'height': height,
            }
    elif (task_type == sd_models.DiffusersTaskType.IMAGE_2_IMAGE or is_img2img_model) and len(getattr(p, 'init_images', [])) > 0:
        if shared.sd_model_type == 'sdxl' and hasattr(model, 'register_to_config'):
            if model_cls in sd_models.i2i_pipes:
                pass
            else:
                model.register_to_config(requires_aesthetics_score = False)
        if 'hires' not in p.ops:
            p.ops.append('img2img')
        if p.vae_type == 'Remote':
            from modules.sd_vae_remote import remote_encode
            p.init_images = remote_encode(p.init_images)
        task_args = {
            'image': p.init_images,
            'strength': p.denoising_strength,
        }
        if model_cls == 'FluxImg2ImgPipeline' or model_cls == 'FluxKontextPipeline': # needs explicit width/height
            if torch.is_tensor(p.init_images[0]):
                p.width = p.init_images[0].shape[-1] * vae_scale_factor
                p.height = p.init_images[0].shape[-2] * vae_scale_factor
            else:
                p.width = width
                p.height = height
            if model_cls == 'FluxKontextPipeline':
                aspect_ratio = p.width / p.height
                max_area = max(p.width, p.height)**2
                p.width = round((max_area * aspect_ratio) ** 0.5)
                p.height = round((max_area / aspect_ratio) ** 0.5)
                p.width = p.width // vae_scale_factor * vae_scale_factor
                p.height = p.height // vae_scale_factor * vae_scale_factor
                task_args['max_area'] = max_area
            task_args['width'], task_args['height'] = p.width, p.height
        elif model_cls == 'OmniGenPipeline' or model_cls == 'OmniGen2Pipeline':
            p.width = width
            p.height = height
            task_args = {
                'width': p.width,
                'height': p.height,
                'input_images': [p.init_images], # omnigen expects list-of-lists
            }
    elif task_type == sd_models.DiffusersTaskType.INSTRUCT and len(getattr(p, 'init_images', [])) > 0:
        p.ops.append('instruct')
        task_args = {
            'width': width if hasattr(p, 'width') else None,
            'height': height if hasattr(p, 'height') else None,
            'image': p.init_images,
            'strength': p.denoising_strength,
        }
    elif (task_type == sd_models.DiffusersTaskType.INPAINTING or is_img2img_model) and len(getattr(p, 'init_images', [])) > 0:
        if shared.sd_model_type == 'sdxl' and hasattr(model, 'register_to_config'):
            if model_cls in [sd_models.i2i_pipes]:
                pass
            else:
                model.register_to_config(requires_aesthetics_score = False)
        if p.detailer_enabled:
            p.ops.append('detailer')
        else:
            p.ops.append('inpaint')
        mask_image = p.task_args.get('image_mask', None) or getattr(p, 'image_mask', None) or getattr(p, 'mask', None)
        if p.vae_type == 'Remote':
            from modules.sd_vae_remote import remote_encode
            p.init_images = remote_encode(p.init_images)
            # mask_image = remote_encode(mask_image)
        task_args = {
            'image': p.init_images,
            'mask_image': mask_image,
            'strength': p.denoising_strength,
            'height': height,
            'width': width,
        }

    # model specific args
    if ('QwenImageEdit' in model_cls) and (p.init_images is None or len(p.init_images) == 0):
        task_args['image'] = [Image.new('RGB', (p.width, p.height), (0, 0, 0))] # monkey-patch so qwen-image-edit pipeline does not error-out on t2i
    if ('QwenImageEditPlusPipeline' in model_cls) and (p.init_control is not None) and (len(p.init_control) > 0):
        task_args['image'] += p.init_control
    if ('QwenImageLayeredPipeline' in model_cls) and (p.init_images is not None) and (len(p.init_images) > 0):
        task_args['image'] = p.init_images[0].convert('RGBA')
    if ('Flux2' in model_cls) and (p.init_control is not None) and (len(p.init_control) > 0):
        task_args['image'] += p.init_control
    if ('LatentConsistencyModelPipeline' in model_cls) and (len(p.init_images) > 0):
        p.ops.append('lcm')
        init_latents = [processing_vae.vae_encode(image, model=shared.sd_model, vae_type=p.vae_type).squeeze(dim=0) for image in p.init_images]
        init_latent = torch.stack(init_latents, dim=0).to(shared.device)
        init_noise = p.denoising_strength * processing.create_random_tensors(init_latent.shape[1:], seeds=p.all_seeds, subseeds=p.all_subseeds, subseed_strength=p.subseed_strength, p=p)
        init_latent = (1 - p.denoising_strength) * init_latent + init_noise
        task_args = {
            'latents': init_latent.to(model.dtype),
            'width': p.width,
            'height': p.height,
        }
    if ('WanImageToVideoPipeline' in model_cls) or ('ChronoEditPipeline' in model_cls):
        if (p.init_images is not None) and (len(p.init_images) > 0):
            task_args['image'] = p.init_images[0]
        else:
            task_args['image'] = Image.new('RGB', (p.width, p.height), (0, 0, 0)) # monkey-patch so wan-i2i pipeline does not error-out on t2i
    if ('WanVACEPipeline' in model_cls) and (p.init_images is not None) and (len(p.init_images) > 0):
        task_args['reference_images'] = p.init_images
    if ('GoogleNanoBananaPipeline' in model_cls) and (p.init_images is not None) and (len(p.init_images) > 0):
        task_args['image'] = p.init_images[0]
    if 'BlipDiffusionPipeline' in model_cls:
        if len(p.init_images) == 0:
            shared.log.error('BLiP diffusion requires init image')
            return task_args
        task_args = {
            'reference_image': p.init_images[0],
            'source_subject_category': getattr(p, 'negative_prompt', '').split()[-1],
            'target_subject_category': getattr(p, 'prompt', '').split()[-1],
            'output_type': 'pil',
        }

    if debug_enabled:
        debug_log(f'Process task specific args: {task_args}')
    return task_args


def get_params(model):
    if hasattr(model, 'blocks') and hasattr(model.blocks, 'inputs'): # modular pipeline
        possible = [input_param.name for input_param in model.blocks.inputs]
        return possible
    else:
        signature = inspect.signature(type(model).__call__, follow_wrapped=True)
        possible = list(signature.parameters)
        return possible


def set_pipeline_args(p, model, prompts:list, negative_prompts:list, prompts_2:typing.Optional[list]=None, negative_prompts_2:typing.Optional[list]=None, prompt_attention:typing.Optional[str]=None, desc:typing.Optional[str]='', **kwargs):
    t0 = time.time()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    argsid = shared.state.begin('Params')
    apply_circular(p.tiling, model)
    args = {}
    has_vae = hasattr(model, 'vae') or (hasattr(model, 'pipe') and hasattr(model.pipe, 'vae'))
    if hasattr(model, 'pipe') and not hasattr(model, 'no_recurse'): # recurse
        model = model.pipe
        has_vae = has_vae or hasattr(model, 'vae')
    if hasattr(model, "set_progress_bar_config"):
        if disable_pbar:
            model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + desc, ncols=80, colour='#327fba', disable=disable_pbar)
        else:
            model.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m' + desc, ncols=80, colour='#327fba')

    possible = get_params(model)

    if debug_enabled:
        debug_log(f'Process pipeline possible: {possible}')
    prompts, negative_prompts, prompts_2, negative_prompts_2 = fix_prompts(p, prompts, negative_prompts, prompts_2, negative_prompts_2)
    steps = kwargs.get("num_inference_steps", None) or len(getattr(p, 'timesteps', ['1']))
    clip_skip = kwargs.pop("clip_skip", 1)

    extra_networks.activate(p, include=['text_encoder', 'text_encoder_2', 'text_encoder_3'])

    parser = 'fixed'
    prompt_attention = prompt_attention or shared.opts.prompt_attention
    if (prompt_attention != 'fixed') and ('Onnx' not in model.__class__.__name__) and ('prompt' not in p.task_args) and (
        'StableDiffusion' in model.__class__.__name__ or
        'StableCascade' in model.__class__.__name__ or
        ('Flux' in model.__class__.__name__ and 'Flux2' not in model.__class__.__name__) or
        'Chroma' in model.__class__.__name__ or
        'HiDreamImagePipeline' in model.__class__.__name__
    ):
        jobid = shared.state.begin('TE Encode')
        try:
            prompt_parser_diffusers.embedder = prompt_parser_diffusers.PromptEmbedder(prompts, negative_prompts, steps, clip_skip, p)
            parser = shared.opts.prompt_attention
        except Exception as e:
            shared.log.error(f'Prompt parser encode: {e}')
            if os.environ.get('SD_PROMPT_DEBUG', None) is not None:
                errors.display(e, 'Prompt parser encode')
        timer.process.record('prompt', reset=False)
        shared.state.end(jobid)
    else:
        prompt_parser_diffusers.embedder = None

    if 'prompt' in possible:
        if 'OmniGen' in model.__class__.__name__:
            prompts = [p.replace('|image|', '<img><|image_1|></img>') for p in prompts]
        if ('HiDreamImage' in model.__class__.__name__) and (prompt_parser_diffusers.embedder is not None):
            args['pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('positive_pooleds')
            prompt_embeds = prompt_parser_diffusers.embedder('prompt_embeds')
            args['prompt_embeds_t5'] = prompt_embeds[0]
            args['prompt_embeds_llama3'] = prompt_embeds[1]
        elif hasattr(model, 'text_encoder') and hasattr(model, 'tokenizer') and ('prompt_embeds' in possible) and (prompt_parser_diffusers.embedder is not None):
            embeds = prompt_parser_diffusers.embedder('prompt_embeds')
            if embeds is None:
                shared.log.warning('Prompt parser encode: empty prompt embeds')
                prompt_parser_diffusers.embedder = None
                args['prompt'] = prompts
            elif embeds.device == torch.device('meta'):
                shared.log.warning('Prompt parser encode: embeds on meta device')
                prompt_parser_diffusers.embedder = None
                args['prompt'] = prompts
            else:
                args['prompt_embeds'] = embeds
                if 'StableCascade' in model.__class__.__name__:
                    args['prompt_embeds_pooled'] = prompt_parser_diffusers.embedder('positive_pooleds').unsqueeze(0)
                elif 'XL' in model.__class__.__name__:
                    args['pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('positive_pooleds')
                elif 'StableDiffusion3' in model.__class__.__name__:
                    args['pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('positive_pooleds')
                elif 'Flux' in model.__class__.__name__:
                    args['pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('positive_pooleds')
                elif 'Chroma' in model.__class__.__name__:
                    args['prompt_attention_mask'] = prompt_parser_diffusers.embedder('prompt_attention_masks')
        else:
            args['prompt'] = prompts
    if 'negative_prompt' in possible:
        if 'HiDreamImage' in model.__class__.__name__ and prompt_parser_diffusers.embedder is not None:
            args['negative_pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('negative_pooleds')
            negative_prompt_embeds = prompt_parser_diffusers.embedder('negative_prompt_embeds')
            args['negative_prompt_embeds_t5'] = negative_prompt_embeds[0]
            args['negative_prompt_embeds_llama3'] = negative_prompt_embeds[1]
        elif hasattr(model, 'text_encoder') and hasattr(model, 'tokenizer') and 'negative_prompt_embeds' in possible and prompt_parser_diffusers.embedder is not None:
            args['negative_prompt_embeds'] = prompt_parser_diffusers.embedder('negative_prompt_embeds')
            if 'StableCascade' in model.__class__.__name__:
                args['negative_prompt_embeds_pooled'] = prompt_parser_diffusers.embedder('negative_pooleds').unsqueeze(0)
            elif 'XL' in model.__class__.__name__:
                args['negative_pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('negative_pooleds')
            elif 'StableDiffusion3' in model.__class__.__name__:
                args['negative_pooled_prompt_embeds'] = prompt_parser_diffusers.embedder('negative_pooleds')
            elif 'Chroma' in model.__class__.__name__:
                args['negative_prompt_attention_mask'] = prompt_parser_diffusers.embedder('negative_prompt_attention_masks')
        else:
            if 'PixArtSigmaPipeline' in model.__class__.__name__: # pixart-sigma pipeline throws list-of-list for negative prompt
                args['negative_prompt'] = negative_prompts[0]
            else:
                args['negative_prompt'] = negative_prompts

    if 'complex_human_instruction' in possible:
        chi = shared.opts.te_complex_human_instruction
        p.extra_generation_params["CHI"] = chi
        if not chi:
            args['complex_human_instruction'] = None
    if 'use_resolution_binning' in possible:
        args['use_resolution_binning'] = False
    if 'use_mask_in_transformer' in possible:
        args['use_mask_in_transformer'] = shared.opts.te_use_mask
    if prompt_parser_diffusers.embedder is not None and not prompt_parser_diffusers.embedder.scheduled_prompt: # not scheduled so we dont need it anymore
        prompt_parser_diffusers.embedder = None

    if 'clip_skip' in possible and parser == 'fixed':
        if clip_skip == 1:
            pass # clip_skip = None
        else:
            args['clip_skip'] = clip_skip - 1

    timesteps = re.split(',| ', shared.opts.schedulers_timesteps)
    if len(timesteps) > 2:
        if ('timesteps' in possible) and hasattr(model.scheduler, 'set_timesteps') and ("timesteps" in set(inspect.signature(model.scheduler.set_timesteps).parameters.keys())):
            p.timesteps = [int(x) for x in timesteps if x.isdigit()]
            p.steps = len(timesteps)
            args['timesteps'] = p.timesteps
            shared.log.debug(f'Sampler: steps={len(p.timesteps)} timesteps={p.timesteps}')
        elif ('sigmas' in possible) and hasattr(model.scheduler, 'set_timesteps') and ("sigmas" in set(inspect.signature(model.scheduler.set_timesteps).parameters.keys())):
            p.timesteps = [float(x)/1000.0 for x in timesteps if x.isdigit()]
            p.steps = len(p.timesteps)
            args['sigmas'] = p.timesteps
            shared.log.debug(f'Sampler: steps={len(p.timesteps)} sigmas={p.timesteps}')
        else:
            shared.log.warning(f'Sampler: cls={model.scheduler.__class__.__name__} timesteps not supported')

    if hasattr(model, 'scheduler') and hasattr(model.scheduler, 'noise_sampler_seed') and hasattr(model.scheduler, 'noise_sampler'):
        model.scheduler.noise_sampler = None # noise needs to be reset instead of using cached values
        model.scheduler.noise_sampler_seed = p.seeds # some schedulers have internal noise generator and do not use pipeline generator
    if 'seed' in possible and p.seed is not None:
        args['seed'] = p.seed
    if 'noise_sampler_seed' in possible and p.seeds is not None:
        args['noise_sampler_seed'] = p.seeds
    if 'guidance_scale' in possible and p.cfg_scale is not None and p.cfg_scale > 0:
        args['guidance_scale'] = p.cfg_scale
    if 'img_guidance_scale' in possible and hasattr(p, 'image_cfg_scale') and p.image_cfg_scale is not None and p.image_cfg_scale > 0:
        args['img_guidance_scale'] = p.image_cfg_scale
    if 'generator' in possible:
        generator = get_generator(p)
        args['generator'] = generator
    else:
        generator = None
    if 'latents' in possible and getattr(p, "init_latent", None) is not None:
        if sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.TEXT_2_IMAGE:
            args['latents'] = p.init_latent
    if 'output_type' in possible:
        if not has_vae:
            kwargs['output_type'] = 'np' # only set latent if model has vae

    # model specific
    if 'Kandinsky' in model.__class__.__name__ or 'Cosmos2' in model.__class__.__name__ or 'OmniGen2' in model.__class__.__name__:
        kwargs['output_type'] = 'np' # only set latent if model has vae
    if 'StableCascade' in model.__class__.__name__:
        kwargs.pop("guidance_scale") # remove
        kwargs.pop("num_inference_steps") # remove
        if 'prior_num_inference_steps' in possible:
            args["prior_num_inference_steps"] = p.steps
            args["num_inference_steps"] = p.refiner_steps
        if 'prior_guidance_scale' in possible:
            args["prior_guidance_scale"] = p.cfg_scale
        if 'decoder_guidance_scale' in possible:
            args["decoder_guidance_scale"] = p.image_cfg_scale
    if 'Flex2' in model.__class__.__name__:
        if len(getattr(p, 'init_images', [])) > 0:
            args['inpaint_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
            args['inpaint_mask'] = Image.new('L', args['inpaint_image'].size, int(p.denoising_strength * 255))
            args['control_image'] = args['inpaint_image'].convert('L').convert('RGB') # will be interpreted as depth
            args['control_strength'] = p.denoising_strength
            args['width'] = p.width
            args['height'] = p.height
    if 'WanVACEPipeline' in model.__class__.__name__:
        if isinstance(args['prompt'], list):
            args['prompt'] = args['prompt'][0] if len(args['prompt']) > 0 else ''
        if isinstance(args.get('negative_prompt', None), list):
            args['negative_prompt'] = args['negative_prompt'][0] if len(args['negative_prompt']) > 0 else ''
        if isinstance(args['generator'], list) and len(args['generator']) > 0:
            args['generator'] = args['generator'][0]

    # set callbacks
    if 'prior_callback_steps' in possible:  # Wuerstchen / Cascade
        args['prior_callback_steps'] = 1
    elif 'callback_steps' in possible:
        args['callback_steps'] = 1

    set_callbacks_p(p)
    if 'prior_callback_on_step_end' in possible: # Wuerstchen / Cascade
        args['prior_callback_on_step_end'] = diffusers_callback
        if 'prior_callback_on_step_end_tensor_inputs' in possible:
            args['prior_callback_on_step_end_tensor_inputs'] = ['latents']
    elif 'callback_on_step_end' in possible:
        args['callback_on_step_end'] = diffusers_callback
        if 'callback_on_step_end_tensor_inputs' in possible:
            if 'HiDreamImage' in model.__class__.__name__: # uses prompt_embeds_t5 and prompt_embeds_llama3 instead
                args['callback_on_step_end_tensor_inputs'] = model._callback_tensor_inputs # pylint: disable=protected-access
            elif 'prompt_embeds' in possible and 'negative_prompt_embeds' in possible and hasattr(model, '_callback_tensor_inputs'):
                args['callback_on_step_end_tensor_inputs'] = model._callback_tensor_inputs # pylint: disable=protected-access
            else:
                args['callback_on_step_end_tensor_inputs'] = ['latents']
    elif 'callback' in possible:
        args['callback'] = diffusers_callback_legacy

    if 'image' in kwargs:
        if isinstance(kwargs['image'], list) and isinstance(kwargs['image'][0], Image.Image):
            p.init_images = kwargs['image']
        if isinstance(kwargs['image'], Image.Image):
            p.init_images = [kwargs['image']]
        if isinstance(kwargs['image'], torch.Tensor):
            p.init_images = kwargs['image']

    # handle remaining args
    for arg in kwargs:
        if arg in possible: # add kwargs
            if type(kwargs[arg]) == float or type(kwargs[arg]) == int:
                if kwargs[arg] <= -1: # skip -1 as default value
                    continue
            args[arg] = kwargs[arg]

    # optional preprocess
    if hasattr(model, 'preprocess') and callable(model.preprocess):
        model.preprocess(p, args)


    # handle task specific args
    if sd_models.get_diffusers_task(model) == sd_models.DiffusersTaskType.MODULAR:
        task_kwargs = task_modular_kwargs(p, model)
    else:
        task_kwargs = task_specific_kwargs(p, model)

    pipe_args = getattr(p, 'task_args', {})
    model_args = getattr(model, 'task_args', {})
    task_kwargs.update(pipe_args or {})
    task_kwargs.update(model_args or {})
    if debug_enabled:
        debug_log(f'Process task args: {task_kwargs}')
    for k, v in task_kwargs.items():
        if k in possible:
            args[k] = v
        else:
            debug_log(f'Process unknown task args: {k}={v}')

    # handle cross-attention args
    cross_attention_args = getattr(p, 'cross_attention_kwargs', {})
    if debug_enabled:
        debug_log(f'Process cross-attention args: {cross_attention_args}')
    for k, v in cross_attention_args.items():
        if args.get('cross_attention_kwargs', None) is None:
            args['cross_attention_kwargs'] = {}
        args['cross_attention_kwargs'][k] = v

    # handle missing resolution
    if args.get('image', None) is not None and ('width' not in args or 'height' not in args):
        if 'width' in possible and 'height' in possible:
            vae_scale_factor = sd_vae.get_vae_scale_factor(model)
            if isinstance(args['image'], torch.Tensor) or isinstance(args['image'], np.ndarray):
                if args['image'].shape[-1] == 3: # nhwc
                    args['width'] = args['image'].shape[-2]
                    args['height'] = args['image'].shape[-3]
                elif args['image'].shape[-3] == 3: # nchw
                    args['width'] = args['image'].shape[-1]
                    args['height'] = args['image'].shape[-2]
                else: # assume latent
                    args['width'] = vae_scale_factor * args['image'].shape[-1]
                    args['height'] = vae_scale_factor * args['image'].shape[-2]
            elif isinstance(args['image'], Image.Image):
                args['width'] = args['image'].width
                args['height'] = args['image'].height
            elif isinstance(args['image'][0], torch.Tensor) or isinstance(args['image'][0], np.ndarray):
                args['width'] = vae_scale_factor * args['image'][0].shape[-1]
                args['height'] = vae_scale_factor * args['image'][0].shape[-2]
            else:
                args['width'] = vae_scale_factor * math.ceil(args['image'][0].width / vae_scale_factor)
                args['height'] = vae_scale_factor * math.ceil(args['image'][0].height / vae_scale_factor)
    if 'max_area' in possible and 'width' in args and 'height' in args and 'max_area' not in args:
        args['max_area'] = args['width'] * args['height']

    # handle implicit controlnet
    if ('control_image' in possible) and ('control_image' not in args) and ('image' in args):
        if sd_models.get_diffusers_task(model) != sd_models.DiffusersTaskType.MODULAR:
            debug_log('Process: set control image')
            args['control_image'] = args['image']

    sd_hijack_hypertile.hypertile_set(p, hr=len(getattr(p, 'init_images', [])) > 0)

    # debug info
    clean = args.copy()
    clean.pop('cross_attention_kwargs', None)
    clean.pop('callback', None)
    clean.pop('callback_steps', None)
    clean.pop('callback_on_step_end', None)
    clean.pop('callback_on_step_end_tensor_inputs', None)
    if 'prompt' in clean and clean['prompt'] is not None:
        clean['prompt'] = len(clean['prompt'])
    if 'negative_prompt' in clean and clean['negative_prompt'] is not None:
        clean['negative_prompt'] = len(clean['negative_prompt'])
    if generator is not None:
        clean['generator'] = f'{generator[0].device}:{[g.initial_seed() for g in generator]}'
    clean['parser'] = parser
    for k, v in clean.copy().items():
        if v is None:
            clean[k] = None
        elif isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            clean[k] = v.shape
        elif isinstance(v, list) and len(v) > 0 and (isinstance(v[0], torch.Tensor) or isinstance(v[0], np.ndarray)):
            clean[k] = [x.shape for x in v]
        elif not debug_enabled and k.endswith('_embeds'):
            del clean[k]
            clean['prompt'] = 'embeds'
    task = str(sd_models.get_diffusers_task(model)).replace('DiffusersTaskType.', '')
    shared.log.info(f'{desc}: pipeline={model.__class__.__name__} task={task} batch={p.iteration + 1}/{p.n_iter}x{p.batch_size} set={clean}')

    if p.hdr_clamp or p.hdr_maximize or p.hdr_brightness != 0 or p.hdr_color != 0 or p.hdr_sharpen != 0:
        shared.log.debug(f'HDR: clamp={p.hdr_clamp} maximize={p.hdr_maximize} brightness={p.hdr_brightness} color={p.hdr_color} sharpen={p.hdr_sharpen} threshold={p.hdr_threshold} boundary={p.hdr_boundary} max={p.hdr_max_boundary} center={p.hdr_max_center}')
    if shared.cmd_opts.profile:
        t1 = time.time()
        shared.log.debug(f'Profile: pipeline args: {t1-t0:.2f}')
    if debug_enabled:
        debug_log(f'Process pipeline args: {args}')

    _args = {}
    for k, v in args.items(): # pipeline may modify underlying args
        if isinstance(v, Image.Image):
            _args[k] = v.copy()
        elif (isinstance(v, list) and len(v) > 0 and isinstance(v[0], Image.Image)):
            _args[k] = [i.copy() for i in v]
        else:
            _args[k] = v

    shared.state.end(argsid)
    return _args
