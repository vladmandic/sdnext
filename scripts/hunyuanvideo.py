import time
import torch
import gradio as gr
import transformers
import diffusers
from modules import scripts_manager, processing, shared, images, devices, sd_models, sd_checkpoint, sd_samplers, model_quant, timer, sd_hijack_te


default_template = """Describe the video by detailing the following aspects:
1. The main content and theme of the video.
2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.
3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.
4. Background environment, light, style and atmosphere.
5. Camera angles, movements, and transitions used in the video.
6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc.
"""

models = {
    'HunyuanVideo': { 'repo': 'tencent/HunyuanVideo', 'revision': 'refs/pr/18' },
    'FastHunyuan': { 'repo': 'FastVideo/FastHunyuan', 'revision': None },
}
loaded_model = None


def get_template(template: str = None):
    # diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video.DEFAULT_PROMPT_TEMPLATE
    base_template_pre = "<|start_header_id|>system<|end_header_id|>\n\n"
    base_template_post = "<|eot_id|>\n"
    base_template_end = "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    if template is None or len(template) == 0:
        template = default_template
    template_lines = '\n'.join([line for line in template.split('\n') if len(line) > 0])
    prompt_template = {
        "crop_start": 95,
        "template": base_template_pre + template_lines + base_template_post + base_template_end
    }
    return prompt_template


def hijack_decode(*args, **kwargs):
    t0 = time.time()
    vae: diffusers.AutoencoderKLHunyuanVideo = shared.sd_model.vae
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.log.debug(f'Video: vae={vae.__class__.__name__} tile={vae.tile_sample_min_width}:{vae.tile_sample_min_height}:{vae.tile_sample_min_num_frames} stride={vae.tile_sample_stride_width}:{vae.tile_sample_stride_height}:{vae.tile_sample_stride_num_frames} time={t1-t0:.2f}')
    return res


class Script(scripts_manager.Script):
    def title(self):
        return 'Video: Hunyuan Video (Legacy)'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    # return signature is array of gradio components
    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/tencent/HunyuanVideo">&nbsp Hunyuan Video</a><br>')
        with gr.Row():
            model = gr.Dropdown(label='Model', choices=list(models.keys()), value=list(models.keys())[0])
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=9, maximum=257, step=1, value=45)
            tile_frames = gr.Slider(label='Tile frames', minimum=1, maximum=64, step=1, value=16)
        with gr.Row():
            with gr.Column():
                override_scheduler = gr.Checkbox(label='Override sampler', value=True)
            with gr.Column():
                scheduler_shift = gr.Slider(label='Sampler shift', minimum=0.0, maximum=20.0, step=0.1, value=7.0)
        with gr.Row():
            template = gr.TextArea(label='Prompt processor', lines=3, value=default_template, visible=False)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [model, num_frames, tile_frames, override_scheduler, scheduler_shift, template, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def load(self, model:str):
        global loaded_model # pylint: disable=global-statement
        if shared.sd_model.__class__ != diffusers.HunyuanVideoPipeline or model != loaded_model:
            sd_models.unload_model_weights()
            t0 = time.time()
            quant_args = model_quant.create_config()
            transformer = diffusers.HunyuanVideoTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path='tencent/HunyuanVideo',
                subfolder="transformer",
                torch_dtype=devices.dtype,
                revision='refs/pr/18',
                cache_dir=shared.opts.hfcache_dir,
                **quant_args
            )
            shared.log.debug(f'Video: module={transformer.__class__.__name__}')
            text_encoder = transformers.LlamaModel.from_pretrained(
                pretrained_model_name_or_path=models.get(model)['repo'],
                subfolder="text_encoder",
                revision=models.get(model)['revision'],
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args
            )
            text_encoder_2 = transformers.CLIPTextModel.from_pretrained(
                pretrained_model_name_or_path=models.get(model)['repo'],
                subfolder="text_encoder_2",
                revision=models.get(model)['revision'],
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
            )
            shared.log.debug(f'Video: module={text_encoder.__class__.__name__}')
            shared.sd_model = diffusers.HunyuanVideoPipeline.from_pretrained(
                pretrained_model_name_or_path='tencent/HunyuanVideo',
                transformer=transformer,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                revision='refs/pr/18',
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args
            )
            t1 = time.time()
            shared.log.debug(f'Video: load cls={shared.sd_model.__class__.__name__} model="{model}" repo={models.get(model)["repo"]} dtype={devices.dtype} time={t1-t0:.2f}')
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(models.get(model)['repo'])
            shared.sd_model.sd_model_hash = None
            shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
            shared.sd_model.orig_encode_prompt = shared.sd_model.encode_prompt
            shared.sd_model.vae.decode = hijack_decode
            shared.sd_model.vae.enable_slicing()
            shared.sd_model.vae.enable_tiling()
            shared.sd_model.vae.use_framewise_decoding = True
            sd_hijack_te.init_hijack(shared.sd_model)
            loaded_model = model

    def run(self, p: processing.StableDiffusionProcessing, model, num_frames, tile_frames, override_scheduler, scheduler_shift, template, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        # set params
        num_frames = int(num_frames)
        p.width = 16 * int(p.width // 16)
        p.height = 16 * int(p.height // 16)
        p.do_not_save_grid = True
        p.ops.append('video')

        # load model
        self.load(model)

        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        devices.torch_gc(force=True)

        if override_scheduler:
            p.sampler_name = 'Default'
        else:
            shared.sd_model.scheduler = sd_samplers.create_sampler(p.sampler_name, shared.sd_model)
            p.sampler_name = 'Default' # avoid double creation
        if hasattr(shared.sd_model.scheduler, '_shift'):
            shared.sd_model.scheduler._shift = scheduler_shift # pylint: disable=protected-access

        # encode prompt
        processing.fix_seed(p)
        p.task_args['num_frames'] = num_frames
        p.task_args['output_type'] = 'pil'
        p.task_args['generator'] = torch.manual_seed(p.seed)
        # p.task_args['prompt'] = None
        # p.task_args['prompt_embeds'], p.task_args['pooled_prompt_embeds'], p.task_args['prompt_attention_mask'] = shared.sd_model.encode_prompt(prompt=p.prompt, prompt_template=get_template(template), device=devices.device)

        # run processing
        t0 = time.time()
        shared.sd_model.vae.tile_sample_min_num_frames = tile_frames
        shared.state.disable_preview = True
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} width={p.width} height={p.height} frames={num_frames}')
        processed = processing.process_images(p)
        shared.state.disable_preview = False
        t1 = time.time()
        if processed is not None and len(processed.images) > 0:
            shared.log.info(f'Video: frames={len(processed.images)} time={t1-t0:.2f}')
            if video_type != 'None':
                images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
