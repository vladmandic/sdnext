from functools import wraps
import torch
import diffusers # pylint: disable=import-error

# pylint: disable=protected-access, missing-function-docstring, line-too-long


# Diffusers FreeU
# Diffusers is imported before ipex hijacks so fourier_filter needs hijacking too
original_fourier_filter = diffusers.utils.torch_utils.fourier_filter
@wraps(diffusers.utils.torch_utils.fourier_filter)
def fourier_filter(x_in, threshold, scale):
    return_dtype = x_in.dtype
    return original_fourier_filter(x_in.to(dtype=torch.float32), threshold, scale).to(dtype=return_dtype)


# fp64 error
class FluxPosEmbed(torch.nn.Module):
    def __init__(self, theta: int, axes_dim):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        for i in range(n_axes):
            cos, sin = diffusers.models.embeddings.get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=torch.float32,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def hidream_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    return_device = pos.device
    pos = pos.to("cpu")

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.to(return_device, dtype=torch.float32)


def ipex_diffusers(device_supports_fp64=False, can_allocate_plus_4gb=False):
    # get around lazy imports
    from diffusers.utils import torch_utils # pylint: disable=import-error, unused-import
    diffusers.utils.torch_utils.fourier_filter = fourier_filter
    if not device_supports_fp64:
        # get around lazy imports
        from diffusers.models import transformers as diffusers_transformers # pylint: disable=import-error, unused-import
        from diffusers.models import controlnets as diffusers_controlnets # pylint: disable=import-error, unused-import
        diffusers.models.embeddings.FluxPosEmbed = FluxPosEmbed
        diffusers.models.transformers.transformer_flux.FluxPosEmbed = FluxPosEmbed
        diffusers.models.controlnets.controlnet_flux.FluxPosEmbed = FluxPosEmbed
        diffusers.models.transformers.transformer_hidream_image.rope = hidream_rope
