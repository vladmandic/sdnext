"""
GPU-accelerated color grading engine using kornia + pillow-lut-tools.
Applied per-image after generation, before mask overlay.
"""

import os
import math
from dataclasses import dataclass, fields
import torch
import numpy as np
from PIL import Image
from modules import devices
from modules.logger import log


debug_enabled = os.environ.get('SD_GRADING_DEBUG', None) is not None
debug = log.trace if debug_enabled else lambda *args, **kwargs: None
debug('Trace: grading')
_kornia = None
_pillow_lut = None


def _ensure_kornia():
    global _kornia  # pylint: disable=global-statement
    if _kornia is not None:
        return _kornia
    from installer import install
    install('kornia', quiet=True)
    import kornia
    _kornia = kornia
    return _kornia


def _ensure_pillow_lut():
    global _pillow_lut  # pylint: disable=global-statement
    if _pillow_lut is not None:
        return _pillow_lut
    from installer import install
    install('pillow_lut', quiet=True)
    import pillow_lut
    _pillow_lut = pillow_lut
    return _pillow_lut


@dataclass
class GradingParams:
    # basic
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    gamma: float = 1.0
    sharpness: float = 0.0
    color_temp: float = 6500
    # tone
    shadows: float = 0.0
    midtones: float = 0.0
    highlights: float = 0.0
    clahe_clip: float = 0.0
    clahe_grid: int = 8
    # split toning
    shadows_tint: str = "#000000"
    highlights_tint: str = "#ffffff"
    split_tone_balance: float = 0.5
    # effects
    vignette: float = 0.0
    grain: float = 0.0
    # lut
    lut_cube_file: str = ""
    lut_strength: float = 1.0

    def __post_init__(self):
        for f in fields(self):
            if f.type is float:
                setattr(self, f.name, float(getattr(self, f.name)))


_defaults = GradingParams()


def is_active(params: GradingParams) -> bool:
    for f in fields(GradingParams):
        if getattr(params, f.name) != getattr(_defaults, f.name):
            return True
    return False


def _hex_to_rgb(hexstr: str) -> tuple[float, float, float]:
    hexstr = hexstr.lstrip('#')
    if len(hexstr) != 6:
        return (0.0, 0.0, 0.0)
    r, g, b = (int(hexstr[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b)


def _kelvin_to_rgb_scale(kelvin: float) -> tuple[float, float, float]:
    """Approximate color temperature as R/B channel multipliers (green=1.0)."""
    temp = max(1000, min(40000, kelvin)) / 100.0
    if temp <= 66:
        r = 1.0
        g = max(0.0, min(1.0, (99.4708025861 * math.log(temp) - 161.1195681661) / 255.0))
        if temp <= 19:
            b = 0.0
        else:
            b = max(0.0, min(1.0, (138.5177312231 * math.log(temp - 10) - 305.0447927307) / 255.0))
    else:
        r = max(0.0, min(1.0, (329.698727446 * ((temp - 60) ** -0.1332047592)) / 255.0))
        g = max(0.0, min(1.0, (288.1221695283 * ((temp - 60) ** -0.0755148492)) / 255.0))
        b = 1.0
    # normalize so the reference (6500K) produces (1,1,1)
    ref_r, ref_g, ref_b = 1.0, 0.9529, 0.9083  # approx 6500K from formula
    return (r / ref_r, g / ref_g, b / ref_b)


def _apply_shadows_midtones_highlights(img: torch.Tensor, shadows: float, midtones: float, highlights: float) -> torch.Tensor:
    """Adjust shadows/midtones/highlights via piecewise gamma on L channel in Lab space."""
    kornia = _ensure_kornia()
    lab = kornia.color.rgb_to_lab(img)
    L = lab[:, 0:1, :, :] / 100.0  # normalize to [0, 1]
    strength = 2.0  # scale slider values for more visible effect
    if shadows != 0:
        s = shadows * strength
        shadow_mask = (1.0 - L).clamp(0, 1) ** 2
        gamma = 1.0 / (1.0 + s) if s > 0 else 1.0 - s
        L = L + shadow_mask * (L.clamp(min=1e-6) ** gamma - L)
    if highlights != 0:
        h = highlights * strength
        highlight_mask = L.clamp(0, 1) ** 2
        gamma = 1.0 / (1.0 + h) if h > 0 else 1.0 - h
        L = L + highlight_mask * (L.clamp(min=1e-6) ** gamma - L)
    if midtones != 0:
        m = midtones * strength
        mid_mask = 1.0 - 2.0 * (L - 0.5).abs()
        mid_mask = mid_mask.clamp(0, 1) ** 2
        gamma = 1.0 / (1.0 + m) if m > 0 else 1.0 - m
        L = L + mid_mask * (L.clamp(min=1e-6) ** gamma - L)
    lab[:, 0:1, :, :] = L.clamp(0, 1) * 100.0
    return kornia.color.lab_to_rgb(lab).clamp(0, 1)


def _apply_split_toning(img: torch.Tensor, shadows_tint: str, highlights_tint: str, balance: float) -> torch.Tensor:
    """Blend tint colors into shadow/highlight regions."""
    kornia = _ensure_kornia()
    lab = kornia.color.rgb_to_lab(img)
    L = lab[:, 0:1, :, :] / 100.0
    shadow_rgb = _hex_to_rgb(shadows_tint)
    highlight_rgb = _hex_to_rgb(highlights_tint)
    shadow_color = torch.tensor(shadow_rgb, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    highlight_color = torch.tensor(highlight_rgb, dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    shadow_mask = ((1.0 - L) * (1.0 - balance)).clamp(0, 1)
    highlight_mask = (L * balance).clamp(0, 1)
    img = img * (1.0 - shadow_mask * 0.3) + shadow_color * shadow_mask * 0.3
    img = img * (1.0 - highlight_mask * 0.3) + highlight_color * highlight_mask * 0.3
    return img.clamp(0, 1)


def _apply_vignette(img: torch.Tensor, strength: float) -> torch.Tensor:
    """Radial darkening vignette via meshgrid."""
    _, _, h, w = img.shape
    y = torch.linspace(-1, 1, h, device=img.device, dtype=img.dtype)
    x = torch.linspace(-1, 1, w, device=img.device, dtype=img.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    dist = (xx ** 2 + yy ** 2).clamp(max=2.0)
    mask = 1.0 - strength * dist * 0.5
    mask = mask.clamp(0, 1).unsqueeze(0).unsqueeze(0)
    return (img * mask).clamp(0, 1)


def _apply_grain(img: torch.Tensor, strength: float) -> torch.Tensor:
    """Film grain via random noise blend."""
    noise = torch.randn_like(img) * strength * 0.1
    return (img + noise).clamp(0, 1)


def _apply_color_temp(img: torch.Tensor, kelvin: float) -> torch.Tensor:
    """Apply color temperature shift via R/B channel scaling."""
    r_scale, g_scale, b_scale = _kelvin_to_rgb_scale(kelvin)
    scales = torch.tensor([r_scale, g_scale, b_scale], dtype=img.dtype, device=img.device).view(1, 3, 1, 1)
    return (img * scales).clamp(0, 1)


def _apply_lut(image: Image.Image, lut_cube_file: str, strength: float) -> Image.Image:
    """Apply .cube LUT file via pillow-lut-tools."""
    if not lut_cube_file or not os.path.isfile(lut_cube_file):
        return image
    pillow_lut = _ensure_pillow_lut()
    try:
        cube = pillow_lut.load_cube_file(lut_cube_file)
        if strength != 1.0:
            cube = pillow_lut.amplify_lut(cube, strength)
        result = image.filter(cube)
        debug(f'Grading LUT: file={os.path.basename(lut_cube_file)} strength={strength}')
        return result
    except Exception as e:
        log.error(f'Grading LUT: {e}')
        return image


def grade_image(image: Image.Image, params: GradingParams) -> Image.Image:
    """Full grading pipeline: PIL -> GPU tensor -> kornia ops -> PIL."""
    log.debug(f"Grading: params={params}")
    kornia = _ensure_kornia()
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=devices.device, dtype=devices.dtype)

    # basic adjustments
    if params.brightness != 0:
        tensor = kornia.enhance.adjust_brightness(tensor, params.brightness)
    if params.contrast != 0:
        tensor = kornia.enhance.adjust_contrast(tensor, 1.0 + params.contrast)
    if params.saturation != 0:
        tensor = kornia.enhance.adjust_saturation(tensor, 1.0 + params.saturation)
    if params.hue != 0:
        tensor = kornia.enhance.adjust_hue(tensor, params.hue * math.pi)
    if params.gamma != 1.0:
        tensor = kornia.enhance.adjust_gamma(tensor, params.gamma)
    if params.sharpness != 0:
        tensor = kornia.enhance.sharpness(tensor, 1.0 + params.sharpness * 4.0)
    if params.color_temp != 6500:
        tensor = _apply_color_temp(tensor, params.color_temp)

    # tone adjustments
    if params.shadows != 0 or params.midtones != 0 or params.highlights != 0:
        tensor = _apply_shadows_midtones_highlights(tensor, params.shadows, params.midtones, params.highlights)
    if params.clahe_clip > 0:
        lab = kornia.color.rgb_to_lab(tensor)
        L = lab[:, 0:1, :, :] / 100.0
        L = kornia.enhance.equalize_clahe(L, clip_limit=params.clahe_clip, grid_size=(params.clahe_grid, params.clahe_grid))
        lab[:, 0:1, :, :] = L * 100.0
        tensor = kornia.color.lab_to_rgb(lab).clamp(0, 1)

    # split toning
    if params.shadows_tint != "#000000" or params.highlights_tint != "#ffffff":
        tensor = _apply_split_toning(tensor, params.shadows_tint, params.highlights_tint, params.split_tone_balance)

    # effects
    if params.vignette > 0:
        tensor = _apply_vignette(tensor, params.vignette)
    if params.grain > 0:
        tensor = _apply_grain(tensor, params.grain)

    # convert back to PIL
    tensor = tensor.clamp(0, 1)
    arr = (tensor.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
    result = Image.fromarray(arr)
    result.info = image.info.copy()  # Image.fromarray drops info; preserve so Process-tab metadata survives grading

    # LUT applied last (CPU, via pillow-lut-tools)
    if params.lut_cube_file:
        result = _apply_lut(result, params.lut_cube_file, params.lut_strength)

    return result
