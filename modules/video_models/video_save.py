import os
import time
import datetime
import cv2
import torch
import einops
from modules import shared, errors ,timer, rife


def get_video_filename(frames:int, codec:str):
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = os.path.join(shared.opts.outdir_video, f'{timestamp}-{codec}-f{frames}')
    return output_filename


def atomic_save_video(filename, tensor:torch.Tensor, fps:float=24, codec:str='libx264', pix_fmt:str='yuv420p', options:str='', metadata:dict={}, pbar=None):
    try:
        import av
        av.logging.set_level(av.logging.ERROR) # pylint: disable=c-extension-no-member
    except Exception as e:
        shared.log.error(f'Video: {e}')
        return

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


def save_video(
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
    t_save = time.time()
    n, _c, t, h, w = pixels.shape
    size = pixels.element_size() * pixels.numel()
    shared.log.debug(f'Video: video={mp4_video} export={mp4_frames} safetensors={mp4_sf} interpolate={mp4_interpolate}')
    shared.log.debug(f'Video: encode={t} raw={size} latent={pixels.shape} fps={mp4_fps} codec={mp4_codec} ext={mp4_ext} options="{mp4_opt}"')
    jobid = shared.state.begin('Save video')
    try:
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

        output_filename = get_video_filename(t, mp4_codec)

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
    shared.state.end(jobid)
    return t, output_video
