"""
Additional params for StableVideoDiffusion
"""

import os
import torch
import gradio as gr
from modules import scripts_manager, processing, shared, sd_models, images, modelloader


models = {
    "SVD 1.0": "stabilityai/stable-video-diffusion-img2vid",
    "SVD XT 1.0": "stabilityai/stable-video-diffusion-img2vid-xt",
    "SVD XT 1.1": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
}

class Script(scripts_manager.Script):
    def title(self):
        return 'Video: Stable Video Diffusion'

    def show(self, is_img2img):
        return is_img2img

    # return signature is array of gradio components
    def ui(self, is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid">&nbsp Stable Video Diffusion</a><br>')
        with gr.Row():
            model = gr.Dropdown(label='Model', choices=list(models), value=list(models)[0])
        with gr.Row():
            num_frames = gr.Slider(label='Frames', minimum=1, maximum=50, step=1, value=14)
            min_guidance_scale = gr.Slider(label='Min guidance', minimum=0.0, maximum=10.0, step=0.1, value=1.0)
            max_guidance_scale = gr.Slider(label='Max guidance', minimum=0.0, maximum=10.0, step=0.1, value=3.0)
        with gr.Row():
            decode_chunk_size = gr.Slider(label='Decode chunks', minimum=1, maximum=25, step=1, value=6)
            motion_bucket_id = gr.Slider(label='Motion level', minimum=0, maximum=1, step=0.05, value=0.5)
            noise_aug_strength = gr.Slider(label='Noise strength', minimum=0.0, maximum=1.0, step=0.01, value=0.1)
        with gr.Row():
            override_resolution = gr.Checkbox(label='Override resolution', value=True)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [model, num_frames, override_resolution, min_guidance_scale, max_guidance_scale, decode_chunk_size, motion_bucket_id, noise_aug_strength, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def run(self, p: processing.StableDiffusionProcessing, model, num_frames, override_resolution, min_guidance_scale, max_guidance_scale, decode_chunk_size, motion_bucket_id, noise_aug_strength, video_type, duration, gif_loop, mp4_pad, mp4_interpolate): # pylint: disable=arguments-differ, unused-argument
        image = getattr(p, 'init_images', None)
        if image is None or len(image) == 0:
            shared.log.error('SVD: no init_images')
            return None
        else:
            image = image[0]

        # load/download model on-demand
        model_path = models[model]
        model_name = os.path.basename(model_path)
        has_checkpoint = sd_models.get_closet_checkpoint_match(model_path)
        if has_checkpoint is None:
            shared.log.error(f'SVD: no checkpoint for {model_name}')
            modelloader.load_reference(model_path, variant='fp16')
        c = shared.sd_model.__class__.__name__
        model_loaded = shared.sd_model.sd_checkpoint_info.model_name if shared.sd_loaded else None
        if model_name != model_loaded or c != 'StableVideoDiffusionPipeline':
            shared.opts.sd_model_checkpoint = model_path
            sd_models.reload_model_weights()
            shared.sd_model = shared.sd_model.to(torch.float32) # must run in fp32 due to dtype mismatch

        # set params
        if override_resolution:
            p.width = 1024
            p.height = 576
            image = images.resize_image(resize_mode=2, im=image, width=p.width, height=p.height, upscaler_name=None, output_type='pil')
        else:
            p.width = image.width
            p.height = image.height
        p.ops.append('video')
        p.do_not_save_grid = True
        p.init_images = [image]
        p.sampler_name = 'Default' # svd does not support non-default sampler
        p.task_args['output_type'] = 'pil'
        p.task_args['generator'] = torch.manual_seed(p.seed) # svd does not support gpu based generator
        p.task_args['image'] = image
        p.task_args['width'] = p.width
        p.task_args['height'] = p.height
        p.task_args['num_frames'] = num_frames
        p.task_args['decode_chunk_size'] = decode_chunk_size
        p.task_args['motion_bucket_id'] = round(255 * motion_bucket_id)
        p.task_args['noise_aug_strength'] = noise_aug_strength
        p.task_args['num_inference_steps'] = p.steps
        p.task_args['min_guidance_scale'] = min_guidance_scale
        p.task_args['max_guidance_scale'] = max_guidance_scale
        shared.log.debug(f'SVD: args={p.task_args}')

        # run processing
        processed = processing.process_images(p)
        if video_type != 'None':
            images.save_video(p, filename=None, images=processed.images, video_type=video_type, duration=duration, loop=gif_loop, pad=mp4_pad, interpolate=mp4_interpolate)
        return processed
