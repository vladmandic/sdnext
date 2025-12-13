"""
Additional params for StableVideoDiffusion
"""

import os
import torch
import gradio as gr
from core import MODELDATA
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
            decode_chunk_size = gr.Slider(label='Decode chunks', minimum=1, maximum=25, step=1, value=1)
            motion_bucket_id = gr.Slider(label='Motion level', minimum=0, maximum=1, step=0.05, value=0.5)
            noise_aug_strength = gr.Slider(label='Noise strength', minimum=0.0, maximum=1.0, step=0.01, value=0.1)
        with gr.Row():
            override_resolution = gr.Checkbox(label='Override resolution', value=True)
        with gr.Row():
            from modules.ui_sections import create_video_inputs
            video_type, duration, gif_loop, mp4_pad, mp4_interpolate = create_video_inputs(tab='img2img' if is_img2img else 'txt2img')
        return [model, num_frames, override_resolution, min_guidance_scale, max_guidance_scale, decode_chunk_size, motion_bucket_id, noise_aug_strength, video_type, duration, gif_loop, mp4_pad, mp4_interpolate]

    def _encode_image(self, image: torch.Tensor, device, num_videos_per_prompt, do_classifier_free_guidance):
        image = image.to(device=device, dtype=MODELDATA.sd_model.vae.dtype)
        shared.log.debug(f'Video encode: type=svd input={image.shape} dtype={image.dtype} device={image.device}')
        image_latents = MODELDATA.sd_model.vae.encode(image).latent_dist.mode()
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative_image_latents, image_latents])
        return image_latents

    def _decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        shared.log.debug(f'Video decode: type=svd input={latents.shape} dtype={latents.dtype} device={latents.device} chunk={decode_chunk_size} frames={num_frames}')
        latents = latents.flatten(0, 1)
        latents = 1 / MODELDATA.sd_model.vae.config.scaling_factor * latents
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = { "num_frames": num_frames_in }
            frame = MODELDATA.sd_model.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames = frames.float()
        return frames

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
        has_checkpoint = sd_models.get_closest_checkpoint_match(model_path)
        if has_checkpoint is None:
            shared.log.error(f'SVD: no checkpoint for {model_name}')
            modelloader.load_reference(model_path, variant='fp16')
        c = MODELDATA.sd_model.__class__.__name__
        model_loaded = MODELDATA.sd_model.sd_checkpoint_info.model_name if MODELDATA.sd_loaded else None
        if model_name != model_loaded or c != 'StableVideoDiffusionPipeline':
            from diffusers import StableVideoDiffusionPipeline # pylint: disable=unused-import
            shared.opts.sd_model_checkpoint = model_path
            sd_models.reload_model_weights()
            MODELDATA.sd_model._encode_vae_image = self._encode_image # pylint: disable=protected-access
            MODELDATA.sd_model.decode_latents = self._decode_latents # pylint: disable=protected-access

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
