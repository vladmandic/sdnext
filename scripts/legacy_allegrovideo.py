import time
import gradio as gr
import transformers
import diffusers
from modules import scripts, processing, shared, images, devices, sd_models, sd_checkpoint, model_quant, timer


repo_id = 'rhymes-ai/Allegro'


def hijack_decode(*args, **kwargs):
    t0 = time.time()
    vae: diffusers.AutoencoderKLAllegro = shared.sd_model.vae
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model, exclude=['vae'])
    res = shared.sd_model.vae.orig_decode(*args, **kwargs)
    t1 = time.time()
    timer.process.add('vae', t1-t0)
    shared.log.debug(f'Video: vae={vae.__class__.__name__} time={t1-t0:.2f}')
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
        return 'Video: Allegro (Legacy)'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    # return signature is array of gradio components
    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/rhymes-ai/Allegro">&nbsp Allegro Video</a><br>')
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=4, maximum=88, step=1, value=22)
        with gr.Row():
            override_scheduler = gr.Checkbox(label='Override scheduler', value=True)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [num_frames, override_scheduler, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, num_frames, override_scheduler, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        # set params
        num_frames = int(num_frames)
        p.width = 8 * int(p.width // 8)
        p.height = 8 * int(p.height // 8)
        p.do_not_save_grid = True
        p.ops.append('video')

        # load model
        if shared.sd_model.__class__ != diffusers.AllegroPipeline:
            sd_models.unload_model_weights()
            t0 = time.time()
            quant_args = model_quant.create_config()
            transformer = diffusers.AllegroTransformer3DModel.from_pretrained(
                repo_id,
                subfolder="transformer",
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                **quant_args
            )
            shared.log.debug(f'Video: module={transformer.__class__.__name__}')
            text_encoder = transformers.T5EncoderModel.from_pretrained(
                repo_id,
                subfolder="text_encoder",
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args
            )
            shared.log.debug(f'Video: module={text_encoder.__class__.__name__}')
            shared.sd_model = diffusers.AllegroPipeline.from_pretrained(
                repo_id,
                # transformer=transformer,
                # text_encoder=text_encoder,
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args
            )
            t1 = time.time()
            shared.log.debug(f'Video: load cls={shared.sd_model.__class__.__name__} repo="{repo_id}" dtype={devices.dtype} time={t1-t0:.2f}')
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo_id)
            shared.sd_model.sd_model_hash = None
            shared.sd_model.vae.orig_decode = shared.sd_model.vae.decode
            shared.sd_model.vae.orig_encode_prompt = shared.sd_model.encode_prompt
            shared.sd_model.vae.decode = hijack_decode
            shared.sd_model.encode_prompt = hijack_encode_prompt
            shared.sd_model.vae.enable_tiling()
            # shared.sd_model.vae.enable_slicing()

        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        devices.torch_gc(force=True)

        processing.fix_seed(p)
        if override_scheduler:
            p.sampler_name = 'Default'
            p.steps = 100
        p.task_args['num_frames'] = num_frames
        p.task_args['output_type'] = 'pil'
        p.task_args['clean_caption'] = False

        p.all_prompts, p.all_negative_prompts = shared.prompt_styles.apply_styles_to_prompts([p.prompt], [p.negative_prompt], p.styles, [p.seed])
        p.task_args['prompt'] = p.all_prompts[0]
        p.task_args['negative_prompt'] = p.all_negative_prompts[0]

        # w = shared.sd_model.transformer.config.sample_width * shared.sd_model.vae_scale_factor_spatial
        # h = shared.sd_model.transformer.config.sample_height * shared.sd_model.vae_scale_factor_spatial
        # n = shared.sd_model.transformer.config.sample_frames * shared.sd_model.vae_scale_factor_temporal

        # run processing
        t0 = time.time()
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
