import time
import inspect
import torch
from modules.logger import log
from modules import shared, upscaler


def load_upscaler(upscaler_name: str) -> upscaler.UpscalerData | None:
    upscalers = [x for x in shared.sd_upscalers if x.name.lower().replace('-', ' ') == upscaler_name.lower().replace('-', ' ')]
    # use inspect to check if upscaler.scaler method has output_type param, if not, then it is an old upscaler and we should not use it for video
    upscalers = [u for u in upscalers if 'output_type' in inspect.signature(u.scaler.do_upscale).parameters.keys()]
    if len(upscalers) == 0: # do force-refresh before failing
        from modules.modelloader import load_upscalers
        load_upscalers()
        upscalers = [x for x in shared.sd_upscalers if x.name.lower().replace('-', ' ') == upscaler_name.lower().replace('-', ' ')]
        upscalers = [u for u in upscalers if 'output_type' in inspect.signature(u.scaler.do_upscale).parameters.keys()]
    if len(upscalers) > 0:
        return upscalers[0]
    else:
        log.warning(f'Upscaler: invalid="{upscaler_name}"')
        log.debug(f"Upscaler: available={[u.name for u in shared.sd_upscalers]}")
        return None


def upscale_video(pixels: torch.Tensor, scale: float = 1.0, upscaler_name: str = ""):
    if upscaler_name is None or upscaler_name == "" or upscaler_name.lower() == "none":
        return pixels
    model = load_upscaler(upscaler_name)
    if model is None:
        log.warning(f'Video upscale: upscaler="{upscaler_name}" not found')
        return pixels
    log.debug(f'Video upscale: scale={scale} upscaler="{upscaler_name}" cls={model.scaler.__class__.__name__} shape={list(pixels.shape)}')
    # pixels: BCFHW [1, 3, 34, 480, 640]
    if pixels.ndim == 5:
        frames = pixels
    elif pixels.ndim == 4:
        frames = pixels.unsqueeze(0)
    else:
        log.warning(f'Video upscale: shape={list(pixels.shape)} unrecognized')
        return pixels
    if pixels.shape[1] != 3:
        log.warning(f'Video upscale: shape={list(pixels.shape)} unrecognized')
        return pixels
    outputs = []
    t0 = time.time()
    for idx in range(frames.shape[2]):
        frame = frames[:, :, idx, :, :] # BCHW
        w = int(frame.shape[-1] * scale)
        h = int(frame.shape[-2] * scale)
        # upscale
        frame = model.scaler.do_upscale(frame, model.name, output_type='tensor', quiet=True)
        frame = frame * 2.0 - 1.0 # upscaler returns 0:1, need -1:1 for video
        if frame.ndim == 3:
            frame = frame.unsqueeze(0)
        # interpolate to exact size
        if frame.shape[-1] != w or frame.shape[-2] != h:
            frame = torch.nn.functional.interpolate(frame, size=(h, w), mode='lanczos', align_corners=False, antialias=True)
        outputs.append(frame)
    outputs = torch.stack(outputs, dim=2)
    t1 = time.time()
    frames = outputs.shape[2]
    log.debug(f'Video upscale: frames={frames} width={outputs.shape[4]} height={outputs.shape[3]} fps={frames / (t1 - t0):.3f} time={t1 - t0:.3f}')
    return outputs
