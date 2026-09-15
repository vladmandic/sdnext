import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from modules import devices
from modules.logger import log


class VectorizedGatedDeltaBranch(nn.Module):
    def __init__(self, d_model: int, num_heads: int, init_gamma: float = 1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.proj_beta = nn.Linear(d_model, num_heads, bias=False)
        nn.init.zeros_(self.proj_beta.weight)
        self.gamma = nn.Parameter(torch.full((1,), float(init_gamma)))
        self.compute_dtype = devices.dtype

    def get_sliding_window_mask(self, num_frames: int, window_size: int, device: torch.device) -> torch.Tensor:
        grid = torch.arange(num_frames, device=device)
        diff = torch.abs(grid.unsqueeze(0) - grid.unsqueeze(1))
        return (diff <= (window_size // 2)).unsqueeze(0).unsqueeze(0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, F_len, S, H, D = q.shape

        if self.proj_beta.weight.device != q.device or self.proj_beta.weight.dtype != self.compute_dtype:
            self.to(device=q.device, dtype=self.compute_dtype)

        q_t = q.to(dtype=self.compute_dtype).permute(0, 2, 3, 1, 4).reshape(B * S, H, F_len, D)
        k_t = k.to(dtype=self.compute_dtype).permute(0, 2, 3, 1, 4).reshape(B * S, H, F_len, D)
        v_t = v.to(dtype=self.compute_dtype).permute(0, 2, 3, 1, 4).reshape(B * S, H, F_len, D)

        q_in = q.reshape(B, F_len, S, -1).to(dtype=self.proj_beta.weight.dtype)
        beta = torch.sigmoid(self.proj_beta(q_in))
        beta = beta.to(dtype=self.compute_dtype).permute(0, 2, 3, 1).reshape(B * S, H, F_len, 1, 1)

        state = torch.zeros((B * S, H, D, D), dtype=self.compute_dtype, device=q.device)
        out_buf = torch.empty((B * S, H, F_len, D), dtype=self.compute_dtype, device=q.device)

        gamma_comp = self.gamma.to(dtype=self.compute_dtype, device=q.device)

        for t in range(F_len):
            q_f = q_t[:, :, t]
            k_f = k_t[:, :, t]
            v_f = v_t[:, :, t]
            b_f = beta[:, :, t]

            pred_v = torch.matmul(state, k_f.unsqueeze(-1)).squeeze(-1)
            error = v_f - pred_v

            # In-place state update avoids intermediate 5D tensor allocations
            state.mul_(1.0 - b_f).addcmul_(b_f, error.unsqueeze(-1), k_f.unsqueeze(-2)) # ty: ignore[too-many-positional-arguments]

            frame_out = torch.matmul(q_f.unsqueeze(-2), state).squeeze(-2)

            # upcast to fp32 before squaring to prevent fp16 overflow
            v_rms = torch.rsqrt(torch.mean(v_f.float().pow(2), dim=-1, keepdim=True) + 1e-6).to(self.compute_dtype)
            out_rms = torch.rsqrt(torch.mean(frame_out.float().pow(2), dim=-1, keepdim=True) + 1e-6).to(self.compute_dtype)

            out_buf[:, :, t] = frame_out * (out_rms / v_rms)

        out = out_buf.reshape(B, S, H, F_len, D).permute(0, 3, 1, 2, 4)
        return (gamma_comp * out).to(dtype=q.dtype)


def vdn_attention_forward(self, hidden_states, *args, **kwargs):
    encoder_hidden_states = kwargs.get("encoder_hidden_states", args[0] if len(args) > 0 else None)

    if encoder_hidden_states is not None or hidden_states.ndim != 3:
        if hasattr(self, "orig_forward"):
            return self.orig_forward(hidden_states, *args, **kwargs)

    B, SeqLen, C = hidden_states.shape
    default_frames = getattr(self, "num_frames", 22)
    num_frames = kwargs.get("num_frames", None) or getattr(self, "num_frames", None)

    if num_frames is None or SeqLen % num_frames != 0:
        if SeqLen % default_frames == 0:
            num_frames = default_frames
        else:
            possible_factors = [f for f in (16, 24, 32, 48, 64) if SeqLen % f == 0]
            if possible_factors:
                num_frames = possible_factors[0]
            else:
                if hasattr(self, "orig_forward"):
                    return self.orig_forward(hidden_states, *args, **kwargs)
                raise ValueError(f"Sequence length {SeqLen} cannot be factorized by frame count.")

    window_size = kwargs.get("window_size", None) or getattr(self, "window_size", 16)
    S = SeqLen // num_frames

    q = self.to_q(hidden_states)
    k = self.to_k(hidden_states)
    v = self.to_v(hidden_states)

    inner_dim = getattr(self, "inner_dim", C)
    head_dim = getattr(self, "slice_able_head_dim", getattr(self, "head_dim", 64))
    num_heads = getattr(self, "heads", inner_dim // head_dim)

    q_5d = q.contiguous().view(B, num_frames, S, num_heads, head_dim)
    k_5d = k.contiguous().view(B, num_frames, S, num_heads, head_dim)
    v_5d = v.contiguous().view(B, num_frames, S, num_heads, head_dim)

    q_loc = q_5d.permute(0, 2, 3, 1, 4).reshape(B * S, num_heads, num_frames, head_dim)
    k_loc = k_5d.permute(0, 2, 3, 1, 4).reshape(B * S, num_heads, num_frames, head_dim)
    v_loc = v_5d.permute(0, 2, 3, 1, 4).reshape(B * S, num_heads, num_frames, head_dim)

    attention_mask = kwargs.get("attention_mask", None)
    if window_size >= num_frames:
        combined_mask = attention_mask
    else:
        window_mask = self.get_sliding_window_mask(num_frames, window_size, hidden_states.device)
        if attention_mask is not None and attention_mask.dtype == torch.bool and attention_mask.shape == window_mask.shape:
            combined_mask = attention_mask & window_mask
        else:
            combined_mask = window_mask

    if self.delta_branch.proj_beta.weight.device != hidden_states.device or self.delta_branch.proj_beta.weight.dtype != devices.dtype:
        self.delta_branch.to(device=hidden_states.device, dtype=devices.dtype)

    try:
        out_local = F.scaled_dot_product_attention(q_loc, k_loc, v_loc, attn_mask=combined_mask, is_causal=False)
    except (RuntimeError, TypeError):
        if combined_mask is not None and combined_mask.dtype == torch.bool:
            mask_val = -1e4 if q_loc.dtype == torch.float16 else -1e9
            float_mask = torch.zeros(combined_mask.shape, device=hidden_states.device, dtype=q_loc.dtype)
            float_mask.masked_fill_(~combined_mask, mask_val)
            out_local = F.scaled_dot_product_attention(q_loc, k_loc, v_loc, attn_mask=float_mask, is_causal=False)
        else:
            raise

    out_local = out_local.reshape(B, S, num_heads, num_frames, head_dim).permute(0, 3, 1, 2, 4)
    out_delta = self.delta_branch(q_5d, k_5d, v_5d)
    out_hybrid = (out_local + out_delta).reshape(B, SeqLen, C)

    to_out = getattr(self, "to_out", None)
    if to_out is not None:
        if isinstance(to_out, (nn.ModuleList, list)):
            for proj in to_out:
                out_hybrid = proj(out_hybrid)
            return out_hybrid
        return to_out(out_hybrid)

    return out_hybrid


def apply_vdn(pipe, window_size=16, num_frames=22, init_gamma=1.0):
    if pipe is None or getattr(pipe, "transformer", None) is None:
        return
    count = 0
    for _, module in pipe.transformer.named_modules():
        if module.__class__.__name__ == "MiniMaxH3Attention":
            if hasattr(module, "delta_branch"):
                # already patched, just update the num_frames and window_size
                module.num_frames = num_frames
                module.window_size = window_size
                continue
            if not hasattr(module, "orig_forward"):
                # backup the original forward method
                module.orig_forward = module.forward

            to_q = getattr(module, "to_q", None)
            if to_q is not None:
                if hasattr(to_q, "out_features"):
                    default_dim = to_q.out_features
                elif hasattr(to_q, "orig_out_features"):
                    default_dim = to_q.orig_out_features
                elif hasattr(to_q, "weight"):
                    default_dim = to_q.weight.shape[0]
                else:
                    default_dim = 1024
            else:
                default_dim = 1024

            inner_dim = getattr(module, "inner_dim", default_dim)
            head_dim = getattr(module, "slice_able_head_dim", getattr(module, "head_dim", 64))
            num_heads = getattr(module, "heads", inner_dim // head_dim)

            delta_branch = VectorizedGatedDeltaBranch(inner_dim, num_heads, init_gamma=init_gamma)
            delta_branch = delta_branch.to(device=devices.device, dtype=devices.dtype)

            module.add_module("delta_branch", delta_branch)
            module.num_frames = num_frames
            module.window_size = window_size

            module.forward = types.MethodType(vdn_attention_forward, module)
            count += 1

    log.info(f'Pipeline: cls={pipe.__class__.__name__} gate={VectorizedGatedDeltaBranch.__name__} patched={count} window={window_size} gamma={init_gamma} frames={num_frames}')


def unapply_vdn(pipe):
    if pipe is None or getattr(pipe, "transformer", None) is None:
        return
    count = 0
    for _, module in pipe.transformer.named_modules():
        if module.__class__.__name__ == "MiniMaxH3Attention":
            if hasattr(module, "orig_forward"):
                count += 1
                module.forward = module.orig_forward
                delattr(module, "orig_forward")
            if hasattr(module, "delta_branch"):
                delattr(module, "delta_branch")
            if hasattr(module, "num_frames"):
                delattr(module, "num_frames")
            if hasattr(module, "window_size"):
                delattr(module, "window_size")
    log.info(f'Pipeline: cls={pipe.__class__.__name__} gate={VectorizedGatedDeltaBranch.__name__} restored={count}')
