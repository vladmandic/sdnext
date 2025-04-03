import os
import time
import torch
import gradio as gr
import diffusers
import transformers
from modules import scripts, processing, shared, images, devices, sd_models, sd_checkpoint, model_quant, timer
from modules.teacache.teacache_ltx import teacache_forward


repos = {
    '0.9.0': 'a-r-r-o-w/LTX-Video-diffusers',
    '0.9.1': 'a-r-r-o-w/LTX-Video-0.9.1-diffusers',
    '0.9.5': 'Lightricks/LTX-Video-0.9.5',
    'custom': None,
}


def load_quants(kwargs, repo_id):
    quant_args = model_quant.create_config()
    if not quant_args:
        return kwargs
    model_quant.load_bnb(f'Load model: type=LTX quant={quant_args}')
    if 'transformer' not in kwargs and ('Model' in shared.opts.bnb_quantization or 'Model' in shared.opts.torchao_quantization):
        kwargs['transformer'] = diffusers.LTXVideoTransformer3DModel.from_pretrained(repo_id, subfolder="transformer", cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype, **quant_args)
        shared.log.debug(f'Quantization: module=transformer type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
    if 'text_encoder' not in kwargs and ('TE' in shared.opts.bnb_quantization or 'TE' in shared.opts.torchao_quantization):
        kwargs['text_encoder'] = transformers.T5EncoderModel.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=shared.opts.hfcache_dir, torch_dtype=devices.dtype, **quant_args)
        shared.log.debug(f'Quantization: module=t5 type=bnb dtype={shared.opts.bnb_quantization_type} storage={shared.opts.bnb_quantization_storage}')
    return kwargs


def hijack_decode(*args, **kwargs):
    t0 = time.time()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.log.debug(f'Video: vae={shared.sd_model.vae.__class__.__name__} time={t1-t0:.2f}')
    return res


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    res = shared.sd_model.vae.orig_encode_prompt(*args, **kwargs)
    t1 = time.time()
    timer.process.add('te', t1-t0)
    shared.log.debug(f'Video: te={shared.sd_model.text_encoder.__class__.__name__} time={t1-t0:.2f}')
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


class Script(scripts.Script):
    def title(self):
        return 'Video: LTX Video (Legacy)'

    def show(self, is_img2img):
        return shared.native

    # return signature is array of gradio components
    def ui(self, is_img2img):
        def model_change(model):
            return gr.update(visible=model == 'custom')

        with gr.Row():
            gr.HTML('<a href="https://www.ltxvideo.org/">&nbsp LTX Video</a><br>')
        with gr.Row():
            model = gr.Dropdown(label='LTX Model', choices=list(repos), value='0.9.1')
            decode = gr.Dropdown(label='Decode', choices=['diffusers', 'native'], value='diffusers', visible=False)
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=9, maximum=257, step=1, value=41)
            sampler = gr.Checkbox(label='Override sampler', value=True)
        with gr.Row():
            teacache_enable = gr.Checkbox(label='Enable TeaCache', value=False)
            teacache_threshold = gr.Slider(label='Threshold', minimum=0.01, maximum=0.1, step=0.01, value=0.03)
        with gr.Row():
            model_custom = gr.Textbox(value='', label='Path to model file', visible=False)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        model.change(fn=model_change, inputs=[model], outputs=[model_custom])
        return [model, model_custom, decode, sampler, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, teacache_enable, teacache_threshold]

    def run(self, p: processing.StableDiffusionProcessing, model, model_custom, decode, sampler, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate, teacache_enable, teacache_threshold): # pylint: disable=arguments-differ, unused-argument
        # set params
        image = getattr(p, 'init_images', None)
        image = None if image is None or len(image) == 0 else image[0]
        if (p.width == 0 or p.height == 0) and image is not None:
            p.width = image.width
            p.height = image.height
        num_frames = 8 * int(num_frames // 8) + 1
        p.width = 32 * int(p.width // 32)
        p.height = 32 * int(p.height // 32)
        processing.fix_seed(p)
        if image:
            image = images.resize_image(resize_mode=2, im=image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
            p.task_args['image'] = image
        p.task_args['output_type'] = 'latent' if decode == 'native' else 'pil'
        p.task_args['generator'] = torch.Generator(devices.device).manual_seed(p.seed)
        p.task_args['num_frames'] = num_frames
        p.do_not_save_grid = True
        if sampler:
            p.sampler_name = 'Default'
        p.ops.append('video')

        # load model
        cls = diffusers.LTXPipeline if image is None else diffusers.LTXImageToVideoPipeline
        diffusers.LTXTransformer3DModel = diffusers.LTXVideoTransformer3DModel
        diffusers.AutoencoderKLLTX = diffusers.AutoencoderKLLTXVideo
        repo_id = repos[model]
        if repo_id is None:
            repo_id = model_custom
        if shared.sd_model.__class__ != cls:
            sd_models.unload_model_weights()
            kwargs = model_quant.create_config()
            diffusers.LTXVideoTransformer3DModel.forward = teacache_forward
            if os.path.isfile(repo_id):
                shared.sd_model = cls.from_single_file(
                    repo_id,
                    cache_dir = shared.opts.hfcache_dir,
                    torch_dtype=devices.dtype,
                    **kwargs
                )
            else:
                kwargs = load_quants(kwargs, repo_id)
                shared.sd_model = cls.from_pretrained(
                    repo_id,
                    cache_dir = shared.opts.hfcache_dir,
                    torch_dtype=devices.dtype,
                    **kwargs
                )
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
            shared.sd_model.vae.orig_encode_prompt = shared.sd_model.encode_prompt
            shared.sd_model.vae.decode = hijack_decode
            shared.sd_model.encode_prompt = hijack_encode_prompt
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo_id)
            shared.sd_model.sd_model_hash = None
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        shared.sd_model.vae.enable_slicing()
        shared.sd_model.vae.enable_tiling()
        shared.sd_model.vae.use_framewise_decoding = True
        devices.torch_gc(force=True)

        shared.sd_model.transformer.cnt = 0
        shared.sd_model.transformer.accumulated_rel_l1_distance = 0
        shared.sd_model.transformer.previous_modulated_input = None
        shared.sd_model.transformer.previous_residual = None
        shared.sd_model.transformer.enable_teacache = teacache_enable
        shared.sd_model.transformer.rel_l1_thresh = teacache_threshold
        shared.sd_model.transformer.num_steps = p.steps

        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} args={p.task_args} steps={p.steps} teacache={teacache_enable} threshold={teacache_threshold}')

        # run processing
        t0 = time.time()
        processed = processing.process_images(p)
        t1 = time.time()
        if processed is not None and len(processed.images) > 0:
            shared.log.info(f'Video: frames={len(processed.images)} time={t1-t0:.2f}')
            if video_type != 'None':
                images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
