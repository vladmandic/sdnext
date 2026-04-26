"""
Video frame interpolation helper.

Used by:
- modules.processing.process_images_inner (after process_samples)
- modules.framepack.framepack_worker (before final save_video)
- modules.ltx.ltx_process (before save_video)
- modules.video_models.video_run (before save_video)

Resolves count and scale from explicit kwargs first, then from
StableDiffusionProcessingVideo.video_interpolate on `p`. Marks the
processing object so save_video can skip its own interpolation pass.

Forwards count straight to the PIL primitive and count+1 to the tensor
primitive to match the legacy interpolate_frames and video_save.py call
shapes.
"""
from typing import Any
import numpy as np
import torch
from PIL import Image
from modules.logger import log


def frames_len(frames: Any):
    if frames is None:
        return None
    if isinstance(frames, list):
        return len(frames)
    try:
        return frames.shape[0]
    except Exception:
        return None


def apply_video_interpolation(
    p: Any = None,
    frames: Any = None,
    count: int = 0,
    scale: float = 0.0,
    pad: int = 1,
    change: float = 0.3,
):
    """Inflate a frame stream by RIFE interpolation.

    Dispatches by frames type:
      list[PIL.Image]            -> rife.interpolate
      4-D torch.Tensor (N,C,H,W) -> rife.interpolate_nchw
      np.ndarray (N,H,W,C)       -> rife.interpolate_nchw via tensor convert
    Sets p.video_interpolated = True after a successful run.
    """
    if frames is None:
        return frames
    if count <= 0:
        count = int(getattr(p, 'video_interpolate', 0) or 0)
    if count <= 0:
        return frames
    if scale <= 0:
        scale = float(getattr(p, 'video_interpolate_scale', 1.0) or 1.0)
    if scale <= 0:
        scale = 1.0

    in_len = frames_len(frames)
    in_type = 'unknown'
    out = frames
    try:
        from modules import rife
        if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], Image.Image):
            in_type = 'pil'
            out = rife.interpolate(frames, count=count, scale=scale, pad=pad, change=change)
        elif torch.is_tensor(frames):
            in_type = 'tensor'
            interpolated = rife.interpolate_nchw(frames, count=count + 1, scale=scale)
            out = torch.cat(interpolated, dim=0) if isinstance(interpolated, list) else interpolated
        elif isinstance(frames, np.ndarray):
            in_type = 'numpy'
            t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
            interpolated = rife.interpolate_nchw(t, count=count + 1, scale=scale)
            t_out = torch.cat(interpolated, dim=0) if isinstance(interpolated, list) else interpolated
            out = (t_out.clamp(0., 1.) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
        else:
            log.warning(f'Video interpolation: unsupported type={type(frames).__name__}')
            return frames
    except Exception as e:
        from modules import errors
        log.error(f'Video interpolation: {e}')
        errors.display(e, 'Video interpolation')
        return frames

    if p is not None:
        try:
            p.video_interpolated = True
        except Exception:
            pass

    log.info(f'Video interpolation: type={in_type} input={in_len} output={frames_len(out)} count={count} scale={scale}')
    return out


def interpolation_factor(p: Any) -> int:
    """Per-source-frame multiplier the helper applied to p, or 1 if it did not run.

    Multiply mp4_fps by this to preserve duration when the helper ran before save.
    """
    if p is None or not getattr(p, 'video_interpolated', False):
        return 1
    n = int(getattr(p, 'video_interpolate', 0) or 0)
    if n <= 0:
        return 1
    return n + 1


def expand_infotexts(infotexts: list, count: int) -> list:
    """Inflate the per-frame infotext list to match apply_video_interpolation output.

    Each interpolated frame inherits the infotext of the prior source frame.
    """
    if not infotexts or count <= 0:
        return infotexts
    out = []
    for txt in infotexts:
        out.append(txt)
        for _ in range(count):
            out.append(txt)
    return out
