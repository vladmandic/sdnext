import time
import torch
import gradio as gr
import diffusers
from modules import scripts_manager, processing, shared, images, devices, sd_models, sd_checkpoint, model_quant


repo_id = 'genmo/mochi-1-preview'


class Script(scripts_manager.Script):
    def title(self):
        return 'Video: Mochi.1 Video (Legacy)'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    # return signature is array of gradio components
    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/genmo/mochi-1-preview">&nbsp Mochi.1 Video</a><br>')
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=9, maximum=257, step=1, value=45)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, num_frames, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        # set params
        num_frames = int(num_frames)
        p.width = 32 * int(p.width // 32)
        p.height = 32 * int(p.height // 32)
        p.task_args['output_type'] = 'pil'
        p.task_args['generator'] = torch.manual_seed(p.seed)
        p.task_args['num_frames'] = num_frames
        p.sampler_name = 'Default'
        p.do_not_save_grid = True
        p.ops.append('video')

        # load model
        cls = diffusers.MochiPipeline
        if shared.sd_model.__class__ != cls:
            sd_models.unload_model_weights()
            kwargs = model_quant.create_config()
            shared.sd_model = cls.from_pretrained(
                repo_id,
                cache_dir = shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **kwargs
            )
            shared.sd_model.scheduler._shift = 7.0 # pylint: disable=protected-access
            sd_models.set_diffuser_options(shared.sd_model)
            shared.sd_model.sd_checkpoint_info = sd_checkpoint.CheckpointInfo(repo_id)
            shared.sd_model.sd_model_hash = None
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        shared.sd_model.vae.enable_slicing()
        shared.sd_model.vae.enable_tiling()
        devices.torch_gc(force=True)
        shared.log.debug(f'Video: cls={shared.sd_model.__class__.__name__} args={p.task_args}')

        # run processing
        t0 = time.time()
        processed = processing.process_images(p)
        t1 = time.time()
        if processed is not None and len(processed.images) > 0:
            shared.log.info(f'Video: frames={len(processed.images)} time={t1-t0:.2f}')
            if video_type != 'None':
                images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
