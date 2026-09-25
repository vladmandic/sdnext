#!/usr/bin/env python3
import argparse
import importlib
import math
import warnings
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm


# ==============================================================================
# 1. Advanced Quality Loss & Metrics (PSNR, SSIM, LAB, Saturation, Perceptual, FFT)
# ==============================================================================

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.DEFAULT
        vgg_features = vgg16(weights=weights).features
        self.slice1 = nn.Sequential(*[vgg_features[i] for i in range(4)])     # relu1_2
        self.slice2 = nn.Sequential(*[vgg_features[i] for i in range(4, 9)])    # relu2_2
        self.slice3 = nn.Sequential(*[vgg_features[i] for i in range(9, 16)])   # relu3_3
        self.slice4 = nn.Sequential(*[vgg_features[i] for i in range(16, 23)])  # relu4_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = (pred - self.mean) / self.std
        t = (target - self.mean) / self.std
        p1, t1 = self.slice1(p), self.slice1(t)
        p2, t2 = self.slice2(p1), self.slice2(t1)
        p3, t3 = self.slice3(p2), self.slice3(t2)
        p4, t4 = self.slice4(p3), self.slice4(t3)
        loss1 = F.l1_loss(p1, t1, reduction='none').mean(dim=[1, 2, 3])
        loss2 = F.l1_loss(p2, t2, reduction='none').mean(dim=[1, 2, 3])
        loss3 = F.l1_loss(p3, t3, reduction='none').mean(dim=[1, 2, 3])
        loss4 = F.l1_loss(p4, t4, reduction='none').mean(dim=[1, 2, 3])
        return loss1 + loss2 + loss3 + 0.5 * loss4


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Computes Peak Signal-to-Noise Ratio (dB). Higher is better."""
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10((max_val ** 2) / mse)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Computes Structural Similarity Index (SSIM) in [0, 1]. Higher is better."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    gaussian_1d = torch.exp(-torch.arange(window_size, device=pred.device).float() ** 2 / (2 * 1.5 ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    kernel = gaussian_1d.unsqueeze(1) @ gaussian_1d.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    mu_p = F.conv2d(pred, kernel, padding=window_size // 2, groups=3)
    mu_t = F.conv2d(target, kernel, padding=window_size // 2, groups=3)
    mu_p_sq = mu_p.pow(2)
    mu_t_sq = mu_t.pow(2)
    mu_pt = mu_p * mu_t
    sigma_p_sq = F.conv2d(pred * pred, kernel, padding=window_size // 2, groups=3) - mu_p_sq
    sigma_t_sq = F.conv2d(target * target, kernel, padding=window_size // 2, groups=3) - mu_t_sq
    sigma_pt = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=3) - mu_pt
    denom = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / torch.clamp(denom, min=1e-8)
    return ssim_map.mean(dim=[1, 2, 3])


def rgb_to_lab_clamped(rgb: torch.Tensor) -> torch.Tensor:
    """Converts RGB [0, 1] to clamped CIE-LAB space preventing high-gradient blowups."""
    rgb = torch.clamp(rgb, 0.0, 1.0)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    r_safe = torch.clamp(r, min=1e-5)
    g_safe = torch.clamp(g, min=1e-5)
    b_safe = torch.clamp(b, min=1e-5)
    r_lin = torch.where(r > 0.04045, ((r_safe + 0.055) / 1.055) ** 2.4, r / 12.92)
    g_lin = torch.where(g > 0.04045, ((g_safe + 0.055) / 1.055) ** 2.4, g / 12.92)
    b_lin = torch.where(b > 0.04045, ((b_safe + 0.055) / 1.055) ** 2.4, b / 12.92)
    x = (r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375) / 0.95047
    y = (r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750) / 1.00000
    z = (r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041) / 1.08883

    def f(t_val):
        t_safe = torch.clamp(t_val, min=1e-6)
        return torch.where(t_val > 0.008856, torch.pow(t_safe, 1.0 / 3.0), (7.787 * t_val) + (16.0 / 116.0))

    fx, fy, fz = f(x), f(y), f(z)
    L = ((116.0 * fy) - 16.0) / 100.0
    a = (500.0 * (fx - fy)) / 128.0
    b_chan = (200.0 * (fy - fz)) / 128.0
    return torch.cat([L, a, b_chan], dim=1)


def rgb_to_saturation(rgb: torch.Tensor) -> torch.Tensor:
    """Extracts RGB saturation channel to penalize color fading."""
    max_c, _ = torch.max(rgb, dim=1, keepdim=True)
    min_c, _ = torch.min(rgb, dim=1, keepdim=True)
    sat = (max_c - min_c) / torch.clamp(max_c, min=1e-4)
    return sat


class EnhancedQualityLoss(nn.Module):
    def __init__(
        self,
        l1_weight=1.0,
        perceptual_weight=1.5,
        ssim_weight=0.5,
        lab_weight=0.15,
        grad_weight=0.5,
        sat_weight=0.20,
        fft_weight=0.10
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.lab_weight = lab_weight
        self.grad_weight = grad_weight
        self.sat_weight = sat_weight
        self.fft_weight = fft_weight
        self.vgg = VGGPerceptualLoss()

    def _image_gradients(self, x: torch.Tensor):
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
        perceptual = self.vgg(pred, target)
        ssim_val = calculate_ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        pred_lab = rgb_to_lab_clamped(pred)
        target_lab = rgb_to_lab_clamped(target)
        l_loss = F.l1_loss(pred_lab[:, 0:1], target_lab[:, 0:1], reduction='none').mean(dim=[1, 2, 3])
        ab_loss = F.l1_loss(pred_lab[:, 1:3], target_lab[:, 1:3], reduction='none').mean(dim=[1, 2, 3])
        lab = l_loss + 2.0 * ab_loss

        p_dx, p_dy = self._image_gradients(pred)
        t_dx, t_dy = self._image_gradients(target)
        grad = (
            F.l1_loss(p_dx, t_dx, reduction='none').mean(dim=[1, 2, 3]) +
            F.l1_loss(p_dy, t_dy, reduction='none').mean(dim=[1, 2, 3])
        )
        p_sat = rgb_to_saturation(pred)
        t_sat = rgb_to_saturation(target)
        sat_loss = F.l1_loss(p_sat, t_sat, reduction='none').mean(dim=[1, 2, 3])

        p_fft = torch.fft.rfft2(pred, norm="ortho")
        t_fft = torch.fft.rfft2(target, norm="ortho")
        fft_loss = (
            F.l1_loss(torch.real(p_fft), torch.real(t_fft), reduction='none').mean(dim=[1, 2, 3]) +
            F.l1_loss(torch.imag(p_fft), torch.imag(t_fft), reduction='none').mean(dim=[1, 2, 3])
        )

        total = (
            (self.l1_weight * l1) +
            (self.perceptual_weight * perceptual) +
            (self.ssim_weight * ssim_loss) +
            (self.lab_weight * lab) +
            (self.grad_weight * grad) +
            (self.sat_weight * sat_loss) +
            (self.fft_weight * fft_loss)
        )
        components = {
            "l1": l1.detach(),
            "perceptual": perceptual.detach(),
            "ssim": ssim_val.detach(),
            "lab": lab.detach(),
            "grad": grad.detach(),
            "sat": sat_loss.detach(),
            "fft": fft_loss.detach(),
            "total": total.detach()
        }
        return total, components


class DetailedMetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss_components: dict):
        batch_size = preds.shape[0]
        self.count += batch_size

        with torch.no_grad():
            psnr = calculate_psnr(preds, targets).sum().item()

        metrics = {
            "psnr": psnr,
            "ssim": loss_components["ssim"].sum().item(),
            "lpips": loss_components["perceptual"].sum().item(),
            "l1_loss": loss_components["l1"].sum().item(),
            "lab_loss": loss_components["lab"].sum().item(),
            "grad_loss": loss_components["grad"].sum().item(),
            "sat_loss": loss_components["sat"].sum().item(),
            "fft_loss": loss_components["fft"].sum().item(),
            "total_loss": loss_components["total"].sum().item()
        }
        for k, v in metrics.items():
            self.sums[k] = self.sums.get(k, 0.0) + v

    def get_averages(self) -> dict:
        return {k: v / max(self.count, 1) for k, v in self.sums.items()}


# ==============================================================================
# 2. Enhanced Architecture
# ==============================================================================

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, 1) in [0, 1]
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = (t * 1000.0) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 2:
            t = t.view(-1, 1)
        return self.mlp(self.sin_emb(t))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class FiLMSmoothResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.film_proj = nn.Linear(emb_dim, out_channels * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.gn1(h)
        scale_shift = self.film_proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.se(h)
        return F.silu(self.skip(x) + h)


class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            kernel_size=3,
            padding=1
        )
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.gn = nn.GroupNorm(min(4, out_channels), out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.shuffle(h)
        h = self.gn(h)
        return F.silu(h)


class MicroDecoder(nn.Module):
    def __init__(self, in_channels: int = 64, hidden_dim: int = 256, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.hidden_dim = hidden_dim
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.SiLU()
        )
        self.res1 = FiLMSmoothResBlock(hidden_dim, hidden_dim, emb_dim=hidden_dim)
        self.res2 = FiLMSmoothResBlock(hidden_dim, hidden_dim, emb_dim=hidden_dim)

        # Multi-stage progressive upsampling
        curr_scale = scale_factor
        curr_dim = hidden_dim
        self.up_blocks = nn.ModuleList()
        self.up_res_blocks = nn.ModuleList()
        while curr_scale > 1:
            step_scale = 2 if (curr_scale % 2 == 0) else curr_scale
            next_dim = max(curr_dim // 2, 32)
            self.up_blocks.append(PixelShuffleUpsampleBlock(curr_dim, next_dim, scale_factor=step_scale))
            self.up_res_blocks.append(FiLMSmoothResBlock(next_dim, next_dim, emb_dim=hidden_dim))
            curr_dim = next_dim
            curr_scale //= step_scale

        self.out_proj = nn.Sequential(
            nn.Conv2d(curr_dim, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        nn.init.zeros_(self.out_proj[2].bias)

    def forward(self, latents: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        if latents.ndim == 5:
            latents = latents.squeeze(2)
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        if t is None:
            t = torch.zeros((latents.shape[0], 1), device=latents.device, dtype=latents.dtype)
        t_emb = self.t_embedder(t)
        x = self.in_proj(latents)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        for up_block, res_block in zip(self.up_blocks, self.up_res_blocks):
            x = up_block(x)
            x = res_block(x, t_emb)
        return self.out_proj(x)


class EMAModel:
    """Exponential Moving Average of model parameters for smooth inference checkpoints."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach())

    def state_dict(self):
        return self.shadow

    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.shadow)


# ==============================================================================
# 3. Helpers & Dataset
# ==============================================================================

def get_vae_params(vae_model):
    config = getattr(vae_model, "config", {})
    scaling_factor = getattr(config, "scaling_factor", 1.0)
    shift_factor = getattr(config, "shift_factor", None)
    latents_mean = getattr(config, "latents_mean", None)
    latents_std = getattr(config, "latents_std", None)
    return scaling_factor, shift_factor, latents_mean, latents_std


def encode_vae(vae_model, batch_img: torch.Tensor) -> torch.Tensor:
    """Dynamically encodes image batch using 4D or 5D shapes depending on VAE requirements."""
    try:
        out = vae_model.encode(batch_img)
    except (ValueError, RuntimeError, TypeError):
        if batch_img.ndim == 4:
            out = vae_model.encode(batch_img.unsqueeze(2))
        elif batch_img.ndim == 5:
            out = vae_model.encode(batch_img.squeeze(2))
        else:
            raise

    if hasattr(out, "latent_dist"):
        lat = out.latent_dist.mode()
    elif hasattr(out, "latents"):
        lat = out.latents
    elif hasattr(out, "latent"):
        lat = out.latent
    elif hasattr(out, "sample"):
        lat = out.sample
    elif isinstance(out, (tuple, list)):
        lat = out[0]
    else:
        lat = out

    # Squeeze out temporal frame dimension if present
    if lat.ndim == 5:
        lat = lat.squeeze(2)  # Back to [B, C, H_lat, W_lat]
    return lat


class LatentDataset(Dataset):
    def __init__(self, latents: torch.Tensor, target_rgbs: torch.Tensor):
        self.latents = latents
        self.target_rgbs = target_rgbs

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.target_rgbs[idx]


# ==============================================================================
# 4. Main Training Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train MicroDecoder with enhanced metrics & visual logging.")
    parser.add_argument("--folder", type=str, required=True, help="Directory containing dataset images.")
    parser.add_argument("--vae", type=str, required=True, help="VAE class name from Diffusers.")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace model repository path.")
    parser.add_argument("--subfolder", type=str, default="vae", help="Subfolder containing VAE weights.")
    parser.add_argument("--scale", type=int, default=2, help="Upsampling factor for PixelShuffle.")
    parser.add_argument("--resolution", type=int, default=512, help="Target training image resolution.")
    parser.add_argument("--crop", choices=["center", "random"], default="center", help="Cropping mode when resizing non-square images.")
    parser.add_argument("--dim", type=int, default=256, help="Hidden dimension channel capacity.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--ema", type=float, default=0.999, help="EMA decay rate.")
    parser.add_argument("--val", type=float, default=0.10, help="Fraction of dataset for validation.")
    parser.add_argument("--max", type=int, default=500, help="Maximum number of dataset images to encode.")
    parser.add_argument("--output", type=str, default="model-micro.safetensors", help="Path to save output weights.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print('Args:', args)
    device = args.device

    diffusers_module = importlib.import_module("diffusers")
    vae_cls = getattr(diffusers_module, args.vae)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    vae_model = vae_cls.from_pretrained(args.repo, subfolder=args.subfolder, torch_dtype=dtype).to(device)
    vae_model.eval()

    scaling_factor, shift_factor, latents_mean, latents_std = get_vae_params(vae_model)
    in_channels = getattr(vae_model.config, "latent_channels", 64)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = [p for ext in exts for p in Path(args.folder).glob(ext)][:args.max]
    if not image_paths:
        raise ValueError(f"No image files found in {args.folder}")
    crop_op = transforms.CenterCrop(args.resolution) if args.crop == "center" else transforms.RandomCrop(args.resolution)
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        crop_op,
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    images = torch.stack([transform(Image.open(p).convert("RGB")) for p in image_paths])
    latents_list = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), 8), desc="Encoding Latents"):
            batch_img = images[i : i + 8].to(device=device, dtype=dtype)
            # Check expected channels
            expected_in_channels = getattr(vae_model.config, "in_channels", 3)
            if batch_img.shape[1] < expected_in_channels:
                pad_channels = expected_in_channels - batch_img.shape[1]
                padding = torch.ones(
                    (batch_img.shape[0], pad_channels, batch_img.shape[2], batch_img.shape[3]),
                    device=device,
                    dtype=dtype
                )
                batch_img = torch.cat([batch_img, padding], dim=1)
            lat = encode_vae(vae_model, batch_img)
            if latents_mean is not None and latents_std is not None:
                mean_t = torch.tensor(latents_mean[:lat.shape[1]], device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1)
                std_t = torch.tensor(latents_std[:lat.shape[1]], device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1)
                lat = (lat - mean_t) / std_t
            else:
                if shift_factor is not None:
                    if isinstance(shift_factor, (list, tuple, torch.Tensor)):
                        sf = torch.tensor(shift_factor, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1)
                        lat = lat - sf
                    else:
                        lat = lat - float(shift_factor)
                if scaling_factor != 1.0:
                    if isinstance(scaling_factor, (list, tuple, torch.Tensor)):
                        sc = torch.tensor(scaling_factor, device=lat.device, dtype=lat.dtype).view(1, -1, 1, 1)
                        lat = lat * sc
                    else:
                        lat = lat * float(scaling_factor)
            latents_list.append(lat.float().cpu())
    del vae_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    all_latents = torch.cat(latents_list, dim=0)
    in_channels = all_latents.shape[1]
    target_h = all_latents.shape[2] * args.scale
    target_w = all_latents.shape[3] * args.scale
    target_rgbs = (images / 2.0 + 0.5).clamp(0, 1)
    target_rgbs = F.interpolate(target_rgbs, size=(target_h, target_w), mode="area").float()
    total_samples = len(all_latents)
    val_count = max(int(total_samples * args.val), 4) if total_samples >= 8 else 0
    train_count = total_samples - val_count
    # Deterministic split
    indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42))
    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    train_dataset = LatentDataset(all_latents[train_indices], target_rgbs[train_indices])
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, pin_memory=torch.cuda.is_available())

    val_dataset = LatentDataset(all_latents[val_indices], target_rgbs[val_indices]) if val_count > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False) if val_dataset else None

    model = MicroDecoder(in_channels=in_channels, hidden_dim=args.dim, scale_factor=args.scale).to(device)
    ema_model = EMAModel(model, decay=args.ema)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = EnhancedQualityLoss().to(device)
    train_metrics = DetailedMetricsTracker()
    val_metrics = DetailedMetricsTracker()
    eval_model = MicroDecoder(in_channels=in_channels, hidden_dim=args.dim, scale_factor=args.scale).to(device)

    best_val_psnr = -float("inf")
    best_epoch = 0

    print(f"\n--> Training MicroDecoder [{args.epochs} epochs | Channels: {in_channels} | Hidden: {args.dim} | Scale: {args.scale}x]")
    print(f"--> Dataset: {train_count} train samples | {val_count} validation samples | EMA decay: {args.ema}")
    print(f"{'Epoch':<6} | {'Train Tot':<9} | {'Tr PSNR':<7} | {'Tr SSIM':<7} | {'Val Tot':<8} | {'Val PSNR':<8} | {'Val SSIM':<8} | {'Val L1':<7} | {'Val LAB':<7}")
    print("-" * 92)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_metrics.reset()

        for batch_latents, batch_targets in train_loader:
            batch_latents = batch_latents.to(device)
            batch_targets = batch_targets.to(device)

            # Random horizontal flip augmentation for latents and targets
            if torch.rand(1).item() < 0.5:
                batch_latents = torch.flip(batch_latents, dims=[-1])
                batch_targets = torch.flip(batch_targets, dims=[-1])

            clean_mask = torch.rand((batch_latents.shape[0], 1), device=device) < 0.20
            t = torch.rand((batch_latents.shape[0], 1), device=device)
            t = torch.where(clean_mask, torch.zeros_like(t), t)
            t_spatial = t.view(-1, 1, 1, 1)
            noise = torch.randn_like(batch_latents)
            # Variance-Preserving Cosine interpolation (std strictly 1.0 across all t)
            cos_t = torch.cos(0.5 * math.pi * t_spatial)
            sin_t = torch.sin(0.5 * math.pi * t_spatial)
            noisy_latents = cos_t * batch_latents + sin_t * noise
            optimizer.zero_grad(set_to_none=True)
            preds = model(noisy_latents, t=t)
            per_sample_loss, components = criterion(preds, batch_targets)
            # SNR-aware sample weighting: clean/low-noise samples have higher weight for high-freq detail
            weights = (1.0 - 0.4 * t.squeeze(-1)).clamp(0.1, 1.0)
            loss = (per_sample_loss * weights).mean()
            loss.backward()
            optimizer.step()
            ema_model.update(model)
            train_metrics.update(preds, batch_targets, components)

        scheduler.step()
        tr_avg = train_metrics.get_averages()

        # Deterministic Validation using EMA weights
        if val_loader:
            ema_model.apply_to(eval_model)
            eval_model.eval()
            val_metrics.reset()

            with torch.no_grad():
                for v_latents, v_targets in val_loader:
                    v_latents = v_latents.to(device)
                    v_targets = v_targets.to(device)
                    # Evaluate on clean latents (t=0)
                    t_val = torch.zeros((v_latents.shape[0], 1), device=device)
                    v_preds = eval_model(v_latents, t=t_val)
                    _v_loss, v_components = criterion(v_preds, v_targets)
                    val_metrics.update(v_preds, v_targets, v_components)

            val_avg = val_metrics.get_averages()
            val_psnr = val_avg['psnr']
            val_ssim = val_avg['ssim']
            val_tot = val_avg['total_loss']
            val_l1 = val_avg['l1_loss']
            val_lab = val_avg['lab_loss']

            # Checkpoint best model on validation PSNR
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                safetensors.torch.save_file(ema_model.state_dict(), args.output)
                star = " *"
            else:
                star = ""
        else:
            val_tot = val_psnr = val_ssim = val_l1 = val_lab = 0.0
            star = ""
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            safetensors.torch.save_file(ema_model.state_dict(), args.output)

        print(
            f"{epoch:02d}/{args.epochs:02d}  | "
            f"{tr_avg['total_loss']:<9.4f} | "
            f"{tr_avg['psnr']:<7.2f} | "
            f"{tr_avg['ssim']:<7.4f} | "
            f"{val_tot:<8.4f} | "
            f"{val_psnr:<8.2f} | "
            f"{val_ssim:<8.4f} | "
            f"{val_l1:<7.4f} | "
            f"{val_lab:<7.4f}{star}"
        )

    print(f"\n--> Training complete! Best validation PSNR: {best_val_psnr:.2f} dB (Epoch {best_epoch})")
    print(f"--> Saved best EMA model weights to: {args.output}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*", category=UserWarning)
    main()
