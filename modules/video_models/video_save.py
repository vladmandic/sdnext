import os
import time
import cv2
import numpy as np
import torch
import einops
from modules import shared, errors ,timer, rife, processing
from modules.video_models.video_utils import check_av


def get_video_filename(p:processing.StableDiffusionProcessingVideo):
    from modules.images_namegen import FilenameGenerator
    namegen = FilenameGenerator(p, seed=p.seed if p is not None else 0, prompt=p.prompt if p is not None else '')
    filename = namegen.apply(shared.opts.samples_filename_pattern if shared.opts.samples_filename_pattern and len(shared.opts.samples_filename_pattern) > 0 else "[seq]-[prompt_words]")
    if shared.opts.save_to_dirs:
        dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]")
        dirfile = os.path.dirname(filename)
        dirname = os.path.join(shared.opts.outdir_video, dirname, dirfile)
    else:
        dirname = shared.opts.outdir_video
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, filename)
    filename = namegen.sequence(filename)
    filename = namegen.sanitize(filename)
    return filename


def save_params(p, filename: str = None):
    from modules.paths import params_path
    if p is None:
        dct = {}
    else:
        # sampler_index, sampler_shift, dynamic_shift, guidance_scale, guidance_true, init_image, init_strength, last_image, vae_type, vae_tile_frames, mp4_fps, mp4_interpolate, mp4_codec, mp4_ext, mp4_opt, mp4_video, mp4_frames, mp4_sf, vlm_enhance, vlm_model, vlm_system_prompt, override_settings = args
        dct = {
            "Prompt": p.prompt,
            "Negative prompt": p.negative_prompt,
            "Steps": p.steps,
            "Sampler": p.sampler_name,
            "Seed": p.seed,
            "Engine": p.video_engine,
            "Model": p.video_model,
            "Frames": p.frames,
            "Size": f"{p.width}x{p.height}",
            "Styles": ','.join(p.styles) if isinstance(p.styles, list) else p.styles,
        }
    params = ', '.join([f'{k}: {v}' for k, v in dct.items() if v is not None and v != ''])
    fn = filename if filename is not None else params_path
    with open(fn, "w", encoding="utf8") as file:
        file.write(params)


def images_to_tensor(images):
    if images is None or len(images) == 0:
        return None
    array = [torch.from_numpy(np.array(image)) for image in images]
    tensor = torch.stack(array, dim=0) # n h w c
    tensor = tensor.unsqueeze(0) # 1, n, h, w, c
    tensor = tensor.permute(0, 4, 1, 2, 3).contiguous() # 1, c, n, h, w
    tensor = (tensor.float() / 127.5) - 1.0 # from [0,255] to [-1,1]
    # shared.log.debug(f'Video output: images={len(images)} tensor={tensor.shape}')
    return tensor


def atomic_save_video(filename, tensor:torch.Tensor, fps:float=24, codec:str='libx264', pix_fmt:str='yuv420p', options:str='', metadata:dict={}, pbar=None):
    av = check_av()
    if av is None or av is False:
        shared.log.error('Video: ffmpeg/av not available')
        return

    savejob = shared.state.begin('Save video')
    frames, height, width, _channels = tensor.shape
    rate = round(fps)
    options_str = options
    options = {}
    for option in [option.strip() for option in options_str.split(',')]:
        if '=' in option:
            key, value = option.split('=', 1)
        elif ':' in option:
            key, value = option.split(':', 1)
        else:
            continue
        options[key.strip()] = value.strip()
    shared.log.info(f'Video: file="{filename}" codec={codec} frames={frames} width={width} height={height} fps={rate} options={options}')
    video_array = torch.as_tensor(tensor, dtype=torch.uint8).numpy(force=True)
    task = pbar.add_task('encoding', total=frames) if pbar is not None else None
    if task is not None:
        pbar.update(task, description='video encoding')
    with av.open(filename, mode="w") as container:
        for k, v in metadata.items():
            container.metadata[k] = v
        stream: av.VideoStream = container.add_stream(codec, rate=rate, options=options)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = pix_fmt
        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode_lazy(frame):
                container.mux(packet)
            if task is not None:
                pbar.update(task, advance=1)
        for packet in stream.encode(): # flush
            container.mux(packet)
    shared.state.outputs(filename)
    shared.state.end(savejob)


def save_video(
        p:processing.StableDiffusionProcessingVideo,
        pixels:torch.Tensor,
        mp4_fps:int=24,
        mp4_codec:str='libx264',
        mp4_opt:str='',
        mp4_ext:str='mp4',
        mp4_sf:bool=False, # save safetensors
        mp4_video:bool=True, # save video
        mp4_frames:bool=False, # save frames
        mp4_interpolate:int=0, # rife interpolation
        stream=None, # async progress reporting stream
        metadata:dict={}, # metadata for video
        pbar=None, # progress bar for video
    ):
    output_video = None
    if pixels is None:
        return 0, output_video
    if not torch.is_tensor(pixels):
        shared.log.error(f'Video: type={type(pixels)} not a tensor')
        return 0, output_video
    t_save = time.time()
    n, _c, t, h, w = pixels.shape
    size = pixels.element_size() * pixels.numel()
    shared.log.debug(f'Video: video={mp4_video} export={mp4_frames} safetensors={mp4_sf} interpolate={mp4_interpolate}')
    shared.log.debug(f'Video: encode={t} raw={size} latent={pixels.shape} fps={mp4_fps} codec={mp4_codec} ext={mp4_ext} options="{mp4_opt}"')
    try:
        preparejob = shared.state.begin('Prepare video')
        if stream is not None:
            stream.output_queue.push(('progress', (None, 'Saving video...')))
        if mp4_interpolate > 0:
            x = pixels.squeeze(0).permute(1, 0, 2, 3)
            interpolated = rife.interpolate_nchw(x, count=mp4_interpolate+1)
            pixels = torch.stack(interpolated, dim=0)
            pixels = pixels.permute(1, 2, 0, 3, 4)

        n, _c, t, h, w = pixels.shape
        x = torch.clamp(pixels.float(), -1., 1.) * 127.5 + 127.5
        x = x.detach().cpu().to(torch.uint8)
        x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=n)
        x = x.contiguous()

        output_filename = get_video_filename(p)
        if shared.opts.save_txt:
            save_params(p, f'{output_filename}.txt')
        save_params(p)

        if mp4_sf:
            fn = f'{output_filename}.safetensors'
            shared.log.info(f'Video export: file="{fn}" type=savetensors shape={x.shape}')
            from safetensors.torch import save_file
            shared.state.outputs(fn)
            save_file({ 'frames': x }, fn, metadata={'format': 'video', 'frames': str(t), 'width': str(w), 'height': str(h), 'fps': str(mp4_fps), 'codec': mp4_codec, 'options': mp4_opt, 'ext': mp4_ext, 'interpolate': str(mp4_interpolate)})

        if mp4_frames:
            shared.log.info(f'Video frames: files="{output_filename}-00000.jpg" frames={t} width={w} height={h}')
            for i in range(t):
                image = cv2.cvtColor(x[i].numpy(), cv2.COLOR_RGB2BGR)
                fn = f'{output_filename}-{i:05d}.jpg'
                shared.state.outputs(fn)
                cv2.imwrite(fn, image)

        shared.state.end(preparejob)

        if mp4_video and (mp4_codec != 'none'):
            output_video = f'{output_filename}.{mp4_ext}'
            atomic_save_video(output_video, tensor=x, fps=mp4_fps, codec=mp4_codec, options=mp4_opt, metadata=metadata, pbar=pbar)
            if stream is not None:
                stream.output_queue.push(('progress', (None, f'Video {os.path.basename(output_video)} | Codec {mp4_codec} | Size {w}x{h}x{t} | FPS {mp4_fps}')))
                stream.output_queue.push(('file', output_video))
        else:
            if stream is not None:
                stream.output_queue.push(('progress', (None, '')))

    except Exception as e:
        shared.log.error(f'Video save: raw={size} {e}')
        errors.display(e, 'video')
    timer.process.add('save', time.time()-t_save)
    return t, output_video
