import os
import gradio as gr
import diffusers
from safetensors.torch import load_file
from core import MODELDATA
from modules import scripts_manager, processing, shared, devices, sd_models


# config
ADAPTERS = {
    'None': None,
    'Motion 1.5 v3' :'diffusers/animatediff-motion-adapter-v1-5-3',
    'Motion 1.5 v2' :'guoyww/animatediff-motion-adapter-v1-5-2',
    'Motion 1.5 v1': 'guoyww/animatediff-motion-adapter-v1-5',
    'Motion 1.4': 'guoyww/animatediff-motion-adapter-v1-4',
    'TemporalDiff': 'vladmandic/temporaldiff',
    'AnimateFace': 'vladmandic/animateface',
    'Lightning': 'ByteDance/AnimateDiff-Lightning/animatediff_lightning_4step_diffusers.safetensors',
    'SDXL Beta': 'a-r-r-o-w/animatediff-motion-adapter-sdxl-beta',
    'LCM': 'wangfuyun/AnimateLCM',
    # 'SDXL Beta': 'guoyww/animatediff-motion-adapter-sdxl-beta',
    # 'LongAnimateDiff 32': 'vladmandic/longanimatediff-32',
    # 'LongAnimateDiff 64': 'vladmandic/longanimatediff-64',
}
LORAS = {
    'None': None,
    'Zoom-in': 'guoyww/animatediff-motion-lora-zoom-in',
    'Zoom-out': 'guoyww/animatediff-motion-lora-zoom-out',
    'Pan-left': 'guoyww/animatediff-motion-lora-pan-left',
    'Pan-right': 'guoyww/animatediff-motion-lora-pan-right',
    'Tilt-up': 'guoyww/animatediff-motion-lora-tilt-up',
    'Tilt-down': 'guoyww/animatediff-motion-lora-tilt-down',
    'Roll-left': 'guoyww/animatediff-motion-lora-rolling-anticlockwise',
    'Roll-right': 'guoyww/animatediff-motion-lora-rolling-clockwise',
    'LCM': 'wangfuyun/AnimateLCM/AnimateLCM_sd15_t2v_lora.safetensors'
}

# state
motion_adapter = None # instance of diffusers.MotionAdapter
loaded_adapter = None # name of loaded adapter
orig_pipe = None # original sd_model pipeline


def set_adapter(adapter_name: str = 'None'):
    if not MODELDATA.sd_loaded:
        return
    global motion_adapter, loaded_adapter, orig_pipe # pylint: disable=global-statement
    # adapter_name = name if name is not None and isinstance(name, str) else loaded_adapter
    if adapter_name is None or adapter_name == 'None' or not MODELDATA.sd_loaded:
        motion_adapter = None
        loaded_adapter = None
        if orig_pipe is not None:
            shared.log.debug(f'AnimateDiff restore pipeline: adapter="{loaded_adapter}"')
            MODELDATA.sd_model = orig_pipe
            orig_pipe = None
        return
    if MODELDATA.sd_model_type != 'sd' and MODELDATA.sd_model_type != 'sdxl' and not (MODELDATA.sd_model.__class__.__name__ == 'AnimateDiffPipeline' or MODELDATA.sd_model.__class__.__name__ == 'AnimateDiffSDXLPipeline'):
        shared.log.warning(f'AnimateDiff: unsupported model type: {MODELDATA.sd_model.__class__.__name__}')
        return
    if motion_adapter is not None and loaded_adapter == adapter_name and (MODELDATA.sd_model.__class__.__name__ == 'AnimateDiffPipeline' or MODELDATA.sd_model.__class__.__name__ == 'AnimateDiffSDXLPipeline'):
        shared.log.debug(f'AnimateDiff: adapter="{adapter_name}" cached')
        return
    if getattr(MODELDATA.sd_model, 'image_encoder', None) is not None:
        shared.log.debug('AnimateDiff: unloading IP adapter')
        # shared.sd_model.image_encoder = None
        # shared.sd_model.unet.set_default_attn_processor()
        MODELDATA.sd_model.unet.config.encoder_hid_dim_type = None
    if adapter_name.endswith('.ckpt') or adapter_name.endswith('.safetensors'):
        import huggingface_hub as hf
        folder, filename = os.path.split(adapter_name)
        adapter_name = hf.hf_hub_download(repo_id=folder, filename=filename, cache_dir=shared.opts.diffusers_dir)
    try:
        shared.log.info(f'AnimateDiff load: adapter="{adapter_name}"')
        motion_adapter = None
        if adapter_name.endswith('.safetensors'):
            motion_adapter = diffusers.MotionAdapter().to(shared.device, devices.dtype)
            motion_adapter.load_state_dict(load_file(adapter_name))
        elif MODELDATA.sd_model_type == 'sd':
            motion_adapter = diffusers.MotionAdapter.from_pretrained(adapter_name, cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype, low_cpu_mem_usage=False, device_map=None)
        elif MODELDATA.sd_model_type == 'sdxl':
            motion_adapter = diffusers.MotionAdapter.from_pretrained(adapter_name, cache_dir=shared.opts.diffusers_dir, torch_dtype=devices.dtype, low_cpu_mem_usage=False, device_map=None, variant='fp16')
        sd_models.move_model(motion_adapter, devices.device) # move pipeline to device
        sd_models.set_diffuser_options(motion_adapter, vae=None, op='adapter')
        loaded_adapter = adapter_name
        new_pipe = None

        if 'Model' in shared.opts.sdnq_quantize_weights:
            shared.log.debug(f'AnimateDiff: sdnq={shared.opts.sdnq_quantize_weights} reloading model weights')
            prev_opts = shared.opts.sdnq_quantize_weights
            shared.opts.sdnq_quantize_weights = []
            sd_models.reload_model_weights(force=True)
            shared.opts.sdnq_quantize_weights = prev_opts

        if MODELDATA.sd_model_type == 'sd':
            new_pipe = diffusers.AnimateDiffPipeline(
                vae=MODELDATA.sd_model.vae,
                text_encoder=MODELDATA.sd_model.text_encoder,
                tokenizer=MODELDATA.sd_model.tokenizer,
                unet=MODELDATA.sd_model.unet,
                scheduler=MODELDATA.sd_model.scheduler,
                feature_extractor=getattr(MODELDATA.sd_model, 'feature_extractor', None),
                image_encoder=getattr(MODELDATA.sd_model, 'image_encoder', None),
                motion_adapter=motion_adapter,
            )
        elif MODELDATA.sd_model_type == 'sdxl':
            new_pipe = diffusers.AnimateDiffSDXLPipeline(
                vae=MODELDATA.sd_model.vae,
                text_encoder=MODELDATA.sd_model.text_encoder,
                text_encoder_2=MODELDATA.sd_model.text_encoder_2,
                tokenizer=MODELDATA.sd_model.tokenizer,
                tokenizer_2=MODELDATA.sd_model.tokenizer_2,
                unet=MODELDATA.sd_model.unet,
                scheduler=MODELDATA.sd_model.scheduler,
                feature_extractor=getattr(MODELDATA.sd_model, 'feature_extractor', None),
                image_encoder=getattr(MODELDATA.sd_model, 'image_encoder', None),
                motion_adapter=motion_adapter,
            )

        if new_pipe is None:
            motion_adapter = None
            loaded_adapter = None
            shared.log.error(f'AnimateDiff load error: adapter="{adapter_name}"')
            return
        orig_pipe = MODELDATA.sd_model
        MODELDATA.sd_model = new_pipe
        sd_models.move_model(MODELDATA.sd_model, devices.device) # move pipeline to device
        sd_models.copy_diffuser_options(new_pipe, orig_pipe)
        sd_models.set_diffuser_options(MODELDATA.sd_model, vae=None, op='model')
        sd_models.move_model(MODELDATA.sd_model.unet, devices.device) # move pipeline to device
        shared.log.debug(f'AnimateDiff: adapter="{loaded_adapter}"')
    except Exception as e:
        motion_adapter = None
        loaded_adapter = None
        shared.log.error(f'AnimateDiff load error: adapter="{adapter_name}" {e}')
        from modules import errors
        errors.display('e', 'AnimateDiff')


def set_scheduler(p, model, override: bool = False):
    if override:
        p.sampler_name = 'Default'
        if 'LCM' in model:
            MODELDATA.sd_model.scheduler = diffusers.LCMScheduler.from_config(MODELDATA.sd_model.scheduler.config)
        else:
            MODELDATA.sd_model.scheduler = diffusers.DDIMScheduler.from_config(MODELDATA.sd_model.scheduler.config)
    shared.log.debug(f'AnimateDiff: scheduler={MODELDATA.sd_model.scheduler.__class__.__name__}')


def set_prompt(p):
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    shared.prompt_styles.apply_styles_to_extra(p)
    p.styles = []
    prompts = p.prompt.split('\n')
    try:
        prompt = {}
        for line in prompts:
            k, v = line.split(':')
            prompt[int(k.strip())] = v.strip()
    except Exception:
        prompt = p.prompt
    shared.log.debug(f'AnimateDiff prompt: {prompt}')
    p.task_args['prompt'] = prompt
    p.task_args['negative_prompt'] = p.negative_prompt


def set_lora(p, lora, strength):
    if lora is not None and lora != 'None':
        shared.log.debug(f'AnimateDiff: lora="{lora}" strength={strength}')
        if lora.endswith('.safetensors'):
            fn = os.path.basename(lora)
            lora = lora.replace(f'/{fn}', '')
            MODELDATA.sd_model.load_lora_weights(lora, weight_name=fn, adapter_name=lora)
        else:
            MODELDATA.sd_model.load_lora_weights(lora, adapter_name=lora)
        MODELDATA.sd_model.set_adapters([lora], adapter_weights=[strength])
        p.extra_generation_params['AnimateDiff Lora'] = f'{lora}:{strength}'


def set_free_init(method, iters, order, spatial, temporal):
    if hasattr(MODELDATA.sd_model, 'enable_free_init') and method != 'none':
        shared.log.debug(f'AnimateDiff free init: method={method} iters={iters} order={order} spatial={spatial} temporal={temporal}')
        MODELDATA.sd_model.enable_free_init(
            num_iters=iters,
            use_fast_sampling=False,
            method=method,
            order=order,
            spatial_stop_frequency=spatial,
            temporal_stop_frequency=temporal,
        )


def set_free_noise(frames):
    context_length = 16
    context_stride = 4
    if frames >= context_length and hasattr(MODELDATA.sd_model, 'enable_free_noise'):
        shared.log.debug(f'AnimateDiff free noise: frames={frames} context={context_length} stride={context_stride}')
        MODELDATA.sd_model.enable_free_noise(context_length=context_length, context_stride=context_stride)


class Script(scripts_manager.Script):
    def title(self):
        return 'Video: AnimateDiff'

    def show(self, is_img2img):
        return not is_img2img


    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML("<span>&nbsp AnimateDiff</span><br>")
        with gr.Row():
            adapter_index = gr.Dropdown(label='Adapter', choices=list(ADAPTERS), value='None')
            frames = gr.Slider(label='Frames', minimum=1, maximum=256, step=1, value=16)
        with gr.Row():
            override_scheduler = gr.Checkbox(label='Override sampler', value=True)
        with gr.Row():
            lora_index = gr.Dropdown(label='Lora', choices=list(LORAS), value='None')
            strength = gr.Slider(label='Strength', minimum=0.0, maximum=2.0, step=0.05, value=1.0)
        with gr.Row():
            latent_mode = gr.Checkbox(label='Latent mode', value=True, visible=False)
        with gr.Accordion('FreeInit', open=False):
            with gr.Row():
                fi_method = gr.Dropdown(label='Method', choices=['none', 'butterworth', 'ideal', 'gaussian'], value='none')
            with gr.Row():
                # fi_fast = gr.Checkbox(label='Fast sampling', value=False)
                fi_iters = gr.Slider(label='Iterations', minimum=1, maximum=10, step=1, value=3)
                fi_order = gr.Slider(label='Order', minimum=1, maximum=10, step=1, value=4)
            with gr.Row():
                fi_spatial = gr.Slider(label='Spatial frequency', minimum=0.0, maximum=1.0, step=0.05, value=0.25)
                fi_temporal = gr.Slider(label='Temporal frequency', minimum=0.0, maximum=1.0, step=0.05, value=0.25)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [adapter_index, frames, lora_index, strength, latent_mode, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, override_scheduler, fi_method, fi_iters, fi_order, fi_spatial, fi_temporal]

    def run(self, p: processing.StableDiffusionProcessing, adapter_index, frames, lora_index, strength, latent_mode, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, override_scheduler, fi_method, fi_iters, fi_order, fi_spatial, fi_temporal): # pylint: disable=arguments-differ, unused-argument
        adapter = ADAPTERS[adapter_index]
        lora = LORAS[lora_index]
        set_adapter(adapter)
        if motion_adapter is None:
            return None
        set_scheduler(p, adapter, override_scheduler)
        set_lora(p, lora, strength)
        set_free_init(fi_method, fi_iters, fi_order, fi_spatial, fi_temporal)
        set_free_noise(frames)
        processing.fix_seed(p)
        p.extra_generation_params['AnimateDiff'] = loaded_adapter
        p.do_not_save_grid = True
        p.ops.append('video')
        p.task_args['generator'] = None
        p.task_args['num_frames'] = frames
        p.task_args['num_inference_steps'] = p.steps
        p.task_args['output_type'] = 'np'
        shared.log.debug(f'AnimateDiff args: {p.task_args}')
        set_prompt(p)
        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['prompt_attention'] = 'fixed'
        processed: processing.Processed = processing.process_images(p) # runs processing using main loop
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        devices.torch_gc()
        return processed


    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, adapter_index, frames, lora_index, strength, latent_mode, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, override_scheduler, fi_method, fi_iters, fi_order, fi_spatial, fi_temporal): # pylint: disable=arguments-differ, unused-argument
        from modules.images import save_video
        if video_type != 'None':
            shared.log.debug(f'AnimateDiff video: type={video_type} duration={duration} loop={gif_loop} pad={mp4_pad} interpolate={mp4_interpolate}')
            save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
