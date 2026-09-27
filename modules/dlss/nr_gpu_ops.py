import ctypes
import math
import torch
import torch.nn.functional as F

from .nr_features import FEATURE_CHANNELS, NetworkGeometry


def _to_i32(val: int) -> int:
    return ctypes.c_int32(val).value


def deterministic_noise_torch(
    height: int,
    width: int,
    frame_index: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the 3 Gaussian noise channels directly on GPU using 32-bit integer arithmetic."""
    yy = torch.arange(height, dtype=torch.int32, device=device).unsqueeze(1).expand(height, width)
    xx = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0).expand(height, width)

    C1 = _to_i32(0xD8163841)
    C2 = _to_i32(0x8DA6B343)
    C3 = _to_i32(int(frame_index) * 0x9E3779B9)
    C4 = _to_i32(0x243F6A88)

    seed = yy * C1
    seed = seed ^ (xx * C2)
    seed = seed ^ C3
    seed = seed ^ C4

    # dynamic shift mix
    shift = (seed.bitwise_right_shift(28)) + 4
    shift = shift.bitwise_and(31)
    mixed = seed ^ (seed.bitwise_right_shift(shift))
    mixed = mixed * _to_i32(0x108EF2D9)
    mixed = mixed ^ (mixed.bitwise_right_shift(22))

    def uniform24(val):
        b1 = val.bitwise_right_shift(30).bitwise_and(0x03)
        b2 = val.bitwise_right_shift(8).bitwise_and(0x00FFFFFF)
        bits = b1 ^ b2
        return (bits + 1).to(torch.float32) * 5.960464477539063e-8

    u_ra = uniform24(mixed * _to_i32(0xCAA5B80D) + _to_i32(0x21DD796B))
    u_ab = uniform24(mixed * _to_i32(0x83232C31) + _to_i32(0x3463E0AC))
    u_rb = uniform24(mixed * _to_i32(0x2C9277B5) + _to_i32(0xAC564B05))
    u_aa = uniform24(mixed * _to_i32(0xFA6DC5F9) + _to_i32(0x4712A88E))

    radius_a = torch.sqrt(-2.0 * torch.log(u_ra))
    radius_b = torch.sqrt(-2.0 * torch.log(u_rb))
    tau = 6.2831854820251465
    angle_a = tau * u_aa
    angle_b = tau * u_ab

    c_a = torch.cos(angle_a)
    s_a = torch.sin(angle_a)
    c_b = torch.cos(angle_b)

    noise = torch.stack([
        (radius_b * c_a).to(torch.float16).to(dtype),
        (radius_b * s_a).to(torch.float16).to(dtype),
        (radius_a * c_b).to(torch.float16).to(dtype),
    ], dim=-1)
    return noise


def make_features_torch(
    color_tensor: torch.Tensor,
    *,
    frame_index: int = 0,
    geometry: NetworkGeometry,
    normalized_style: float = 0.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.0,
    skin_structure_strength: float | None = None,
    mask_structure_strength: float | None = None,
    control_mask: torch.Tensor | None = None,
    history_reprojected: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the (1, network_height, network_width, 16) feature tensor entirely on GPU."""
    # color_tensor: (height, width, 3) in [0, 1] on device
    device = color_tensor.device

    style = torch.tensor(normalized_style, device=device, dtype=torch.float16).to(dtype)
    tone = torch.tensor(local_tone_strength, device=device, dtype=torch.float16).to(dtype)

    if control_mask is not None:
        structure = torch.tensor(0.0, device=device, dtype=dtype)
        skin_structure = torch.tensor(0.0, device=device, dtype=dtype)
        automatic_structure = torch.tensor(0.0, device=device, dtype=dtype)
    elif skin_structure_strength is not None or mask_structure_strength is not None:
        skin_val = skin_structure_strength if skin_structure_strength is not None else -1.0
        auto_val = mask_structure_strength if mask_structure_strength is not None else -1.0
        enabled = max(skin_val, auto_val) >= 0
        structure = torch.tensor(1.0 if enabled else local_structure_strength, device=device, dtype=torch.float16).to(dtype)
        s_val = (skin_val if skin_val >= 0 else local_structure_strength) if enabled else -1.0
        skin_structure = torch.tensor(s_val, device=device, dtype=torch.float16).to(dtype)
        a_val = (auto_val if auto_val >= 0 else local_structure_strength) if enabled else -1.0
        automatic_structure = torch.tensor(a_val, device=device, dtype=torch.float16).to(dtype)
    else:
        structure = torch.tensor(local_structure_strength, device=device, dtype=torch.float16).to(dtype)
        skin_structure = torch.tensor(-1.0, device=device, dtype=dtype)
        automatic_structure = torch.tensor(-1.0, device=device, dtype=dtype)

    rows = torch.from_numpy(geometry.source_rows()).to(device=device, dtype=torch.long)
    cols = torch.from_numpy(geometry.source_columns()).to(device=device, dtype=torch.long)

    # Extended image
    extended = color_tensor[rows[:, None], cols[None, :], :]  # (net_h, net_w, 3)

    # Scaled color: (half(c) - 0.5) * 0.125
    sampled = extended.to(torch.float16)
    centered = (sampled - 0.5).to(torch.float16)
    scaled = (centered * 0.125).to(torch.float16).to(dtype)

    features = torch.empty((1, geometry.network_height, geometry.network_width, FEATURE_CHANNELS), device=device, dtype=dtype)

    # Noise
    features[0, ..., 0:3] = deterministic_noise_torch(geometry.network_height, geometry.network_width, frame_index, device=device, dtype=dtype)
    features[0, ..., 3] = 1.0
    features[0, ..., 4:7] = scaled

    if history_reprojected is not None:
        # History in 7:10, mirrored for network extension
        ext_hist = history_reprojected[rows[:, None], cols[None, :], :]
        sampled_hist = ext_hist.to(torch.float16)
        centered_hist = (sampled_hist - 0.5).to(torch.float16)
        scaled_hist = (centered_hist * 0.125).to(torch.float16).to(dtype)
        features[0, ..., 7:10] = scaled_hist
    else:
        features[0, ..., 7:10] = scaled

    features[0, ..., 10] = style

    if control_mask is not None:
        mask_ext = control_mask[rows[:, None], cols[None, :], :]
        features[0, ..., 11] = (mask_ext[..., 1].to(torch.float16) * tone.to(torch.float16)).to(dtype)
        features[0, ..., 12] = (mask_ext[..., 2].to(torch.float16) * structure.to(torch.float16)).to(dtype)
    else:
        features[0, ..., 11] = tone
        features[0, ..., 12] = structure

    features[0, ..., 13] = skin_structure
    features[0, ..., 14] = automatic_structure
    features[0, ..., 15] = 0.0

    return features


def compose_head_torch(
    head: torch.Tensor,
    color: torch.Tensor,
    *,
    control_mask: torch.Tensor | None = None,
    intensity: float = 1.0,
) -> torch.Tensor:
    """RGB = colour + 0.25 * half(head[..., :3]), blended by mask red * intensity on GPU."""
    residual = head[..., :3].to(torch.float16) * 0.25
    predicted = torch.clamp(color + residual.to(color.dtype), 0.0, 1.0)
    blend = intensity
    if control_mask is not None:
        blend = control_mask[..., :1] * intensity
    if isinstance(blend, torch.Tensor):
        blend = torch.clamp(blend, 0.0, 1.0)
    else:
        blend = min(max(blend, 0.0), 1.0)
    return torch.clamp(color + blend * (predicted - color), 0.0, 1.0)


def gaussian_blur_torch(img_tensor: torch.Tensor, radius: float = 4.0) -> torch.Tensor:
    """Depthwise separable Gaussian blur on GPU: input (H, W, C) -> (H, W, C)."""
    extent = int(math.ceil(3 * radius))
    offsets = torch.arange(-extent, extent + 1, dtype=torch.float32, device=img_tensor.device)
    kernel = torch.exp(-offsets * offsets / (2.0 * radius * radius))
    kernel = kernel / kernel.sum()

    # Convert to NCHW
    x = img_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # (1, C, H, W)
    C = x.shape[1]

    kx = kernel.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(C, 1, -1, 1)

    x_pad_x = F.pad(x, (extent, extent, 0, 0), mode="replicate")
    blurred_x = F.conv2d(x_pad_x, kx, groups=C)

    x_pad_y = F.pad(blurred_x, (0, 0, extent, extent), mode="replicate")
    blurred = F.conv2d(x_pad_y, ky, groups=C)

    return blurred.squeeze(0).permute(1, 2, 0)


def compose_detail_torch(
    source: torch.Tensor,
    output: torch.Tensor,
    *,
    detail_strength: float = 1.0,
    colour_strength: float = 1.0,
    radius: float = 4.0,
) -> torch.Tensor:
    """result = source + colour * lowpass(change) + detail * highpass(change) on GPU."""
    if detail_strength == 1.0 and colour_strength == 1.0:
        return output
    change = output - source
    low = gaussian_blur_torch(change, radius=radius)
    high = change - low
    return torch.clamp(source + colour_strength * low + detail_strength * high, 0.0, 1.0)


def _catmull_coordinates_torch(normalized: torch.Tensor, dimension: int):
    dim = float(dimension)
    pixel = normalized * dim - 0.5
    base_index = torch.floor(pixel)
    t = torch.clamp(pixel - base_index, 0.0, 1.0)
    square = t * t
    cube = square * t
    w0 = -0.5 * t + square - 0.5 * cube
    w1 = 1.0 - 2.5 * square + 1.5 * cube
    w2 = 0.5 * t + 2.0 * square - 1.5 * cube
    w3 = -0.5 * square + 0.5 * cube
    g = w1 + w2
    base = base_index + 0.5
    lower, upper = 0.5, dim - 0.5
    return (
        torch.clamp(base - 1.0, lower, upper),
        torch.clamp(base + w2 / g, lower, upper),
        torch.clamp(base + 2.0, lower, upper),
        w0, w3, g,
    )


def sample_history_torch(history: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Five-tap Catmull-Rom approximation on GPU: history (H, W, C), u/v (H, W) -> (H, W, C)."""
    height, width, _channels = history.shape
    x_outer0, x_middle, x_outer3, x_w0, x_w3, x_g = _catmull_coordinates_torch(u, width)
    y_outer0, y_middle, y_outer3, y_w0, y_w3, y_g = _catmull_coordinates_torch(v, height)

    weights = [x_w0 * y_g, x_g * y_w0, x_g * y_g, x_g * y_w3, x_w3 * y_g]
    coords = [
        (x_outer0, y_middle),
        (x_middle, y_outer0),
        (x_middle, y_middle),
        (x_middle, y_outer3),
        (x_outer3, y_middle),
    ]

    # Convert pixel center coordinates (0.5 .. width - 0.5) to normalized grid_sample coordinates [-1, 1]
    # In PyTorch grid_sample, -1 corresponds to pixel 0's left boundary (coord -0.5) and +1 to pixel width-1's right boundary (coord width-0.5)
    # So normalized coord = (coord / dim) * 2 - 1
    # history in NCHW: (1, C, H, W)
    hist_nchw = history.permute(2, 0, 1).unsqueeze(0)

    total = torch.zeros_like(history)
    weight_sum = torch.zeros_like(u)

    for w, (cx, cy) in zip(weights, coords):
        gx = (cx / float(width)) * 2.0 - 1.0
        gy = (cy / float(height)) * 2.0 - 1.0
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0) # (1, H, W, 2)
        sampled = F.grid_sample(hist_nchw, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled_hwc = sampled.squeeze(0).permute(1, 2, 0)
        total = total + w.unsqueeze(-1) * sampled_hwc
        weight_sum = weight_sum + w

    return (total / weight_sum.unsqueeze(-1)).clamp(0.0, 1.0)


def compose_temporal_torch(
    head: torch.Tensor,
    color: torch.Tensor,
    features: torch.Tensor,
    *,
    blend_scale: float = 0.73974609375,
    control_mask: torch.Tensor | None = None,
    intensity: float = 1.0,
) -> torch.Tensor:
    """predicted + alpha * (history - predicted) on GPU."""
    logit = head[..., 3:4].to(torch.float16)
    alpha = torch.clamp(torch.sigmoid(logit.to(torch.float32)) * float(blend_scale), 0.0, 1.0)
    predicted = torch.clamp(color + head[..., :3].to(torch.float16).to(color.dtype) * 0.25, 0.0, 1.0)
    history = features[0, ..., 7:10] * 8.0 + 0.5
    # History extent might be cropped if features are network-extent vs image-extent
    h, w = color.shape[:2]
    history = history[:h, :w]
    temporal = predicted + alpha * (history - predicted)
    if control_mask is None and intensity == 1.0:
        return temporal
    blend = float(intensity) if control_mask is None else control_mask[..., :1] * float(intensity)
    if isinstance(blend, torch.Tensor):
        blend = torch.clamp(blend, 0.0, 1.0)
    return torch.clamp(color + blend * (temporal - color), 0.0, 1.0)
