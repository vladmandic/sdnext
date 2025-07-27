from typing import Any, Dict, Optional
import random
import torch
import torch.nn.functional as F
from einops import rearrange


def gaussian_kernel(kernel_size=3, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    gaussian_1d = torch.exp(-(x_coord - (kernel_size - 1) / 2) ** 2 / (2 * sigma ** 2))
    gaussian_1d = gaussian_1d / gaussian_1d.sum()
    gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    kernel = gaussian_2d[None, None, :, :].repeat(channels, 1, 1, 1)

    return kernel

def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(latents.device, latents.dtype)
    blurred_latents = F.conv2d(latents, kernel, padding=kernel_size//2, groups=channels)

    return blurred_latents

def get_views(height, width, h_window_size=128, w_window_size=128, scale_factor=8):
    height = int(height)
    width = int(width)
    h_window_stride = h_window_size // 2
    w_window_stride = w_window_size // 2
    h_window_size = int(h_window_size / scale_factor)
    w_window_size = int(w_window_size / scale_factor)
    h_window_stride = int(h_window_stride / scale_factor)
    w_window_stride = int(w_window_stride / scale_factor)
    num_blocks_height = int((height - h_window_size) / h_window_stride - 1e-6) + 2 if height > h_window_size else 1
    num_blocks_width = int((width - w_window_size) / w_window_stride - 1e-6) + 2 if width > w_window_size else 1
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * h_window_stride)
        h_end = h_start + h_window_size
        w_start = int((i % num_blocks_width) * w_window_stride)
        w_end = w_start + w_window_size

        if h_end > height:
            h_start = int(h_start + height - h_end)
            h_end = int(height)
        if w_end > width:
            w_start = int(w_start + width - w_end)
            w_end = int(width)
        if h_start < 0:
            h_end = int(h_end - h_start)
            h_start = 0
        if w_start < 0:
            w_end = int(w_end - w_start)
            w_start = 0

        random_jitter = True
        if random_jitter:
            h_jitter_range = h_window_size // 8
            w_jitter_range = w_window_size // 8
            h_jitter = 0
            w_jitter = 0

            if (w_start != 0) and (w_end != width):
                w_jitter = random.randint(-w_jitter_range, w_jitter_range)
            elif (w_start == 0) and (w_end != width):
                w_jitter = random.randint(-w_jitter_range, 0)
            elif (w_start != 0) and (w_end == width):
                w_jitter = random.randint(0, w_jitter_range)
            if (h_start != 0) and (h_end != height):
                h_jitter = random.randint(-h_jitter_range, h_jitter_range)
            elif (h_start == 0) and (h_end != height):
                h_jitter = random.randint(-h_jitter_range, 0)
            elif (h_start != 0) and (h_end == height):
                h_jitter = random.randint(0, h_jitter_range)
            h_start += (h_jitter + h_jitter_range)
            h_end += (h_jitter + h_jitter_range)
            w_start += (w_jitter + w_jitter_range)
            w_end += (w_jitter + w_jitter_range)

        views.append((h_start, h_end, w_start, w_end))
    return views

def scale_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
):
    # Notice that normalization is always applied before the real computation in the following blocks.
    if self.current_hw:
        current_scale_num_h, current_scale_num_w = max(self.current_hw[0] // 1024, 1), max(self.current_hw[1] // 1024, 1)
    else:
        current_scale_num_h, current_scale_num_w = 1, 1

    # 0. Self-Attention
    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    else:
        norm_hidden_states = self.norm1(hidden_states)

    # 2. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    ratio_hw = current_scale_num_h / current_scale_num_w
    latent_h = int((norm_hidden_states.shape[1] * ratio_hw) ** 0.5)
    latent_w = int(latent_h / ratio_hw)
    scale_factor = 128 * current_scale_num_h / latent_h
    if ratio_hw > 1:
        sub_h = 128
        sub_w = int(128 / ratio_hw)
    else:
        sub_h = int(128 * ratio_hw)
        sub_w = 128

    h_jitter_range = int(sub_h / scale_factor // 8)
    w_jitter_range = int(sub_w / scale_factor // 8)
    views = get_views(latent_h, latent_w, sub_h, sub_w, scale_factor = scale_factor)

    current_scale_num = max(current_scale_num_h, current_scale_num_w)
    global_views = [[h, w] for h in range(current_scale_num_h) for w in range(current_scale_num_w)]

    four_window = True
    fourg_window = False

    if four_window:
        norm_hidden_states_ = rearrange(norm_hidden_states, 'bh (h w) d -> bh h w d', h = latent_h)
        norm_hidden_states_ = F.pad(norm_hidden_states_, (0, 0, w_jitter_range, w_jitter_range, h_jitter_range, h_jitter_range), 'constant', 0)
        value = torch.zeros_like(norm_hidden_states_)
        count = torch.zeros_like(norm_hidden_states_)
        for index, view in enumerate(views):
            h_start, h_end, w_start, w_end = view
            local_states = norm_hidden_states_[:, h_start:h_end, w_start:w_end, :]
            local_states = rearrange(local_states, 'bh h w d -> bh (h w) d')
            local_output = self.attn1(
                local_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            local_output = rearrange(local_output, 'bh (h w) d -> bh h w d', h = int(sub_h / scale_factor))

            value[:, h_start:h_end, w_start:w_end, :] += local_output * 1
            count[:, h_start:h_end, w_start:w_end, :] += 1

        value = value[:, h_jitter_range:-h_jitter_range, w_jitter_range:-w_jitter_range, :]
        count = count[:, h_jitter_range:-h_jitter_range, w_jitter_range:-w_jitter_range, :]
        attn_output = torch.where(count>0, value/count, value)

        gaussian_local = gaussian_filter(attn_output, kernel_size=(2*current_scale_num-1), sigma=1.0)

        attn_output_global = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        attn_output_global = rearrange(attn_output_global, 'bh (h w) d -> bh h w d', h = latent_h)

        gaussian_global = gaussian_filter(attn_output_global, kernel_size=(2*current_scale_num-1), sigma=1.0)

        attn_output = gaussian_local + (attn_output_global - gaussian_global)
        attn_output = rearrange(attn_output, 'bh h w d -> bh (h w) d')

    elif fourg_window:
        norm_hidden_states = rearrange(norm_hidden_states, 'bh (h w) d -> bh h w d', h = latent_h)
        norm_hidden_states_ = F.pad(norm_hidden_states, (0, 0, w_jitter_range, w_jitter_range, h_jitter_range, h_jitter_range), 'constant', 0)
        value = torch.zeros_like(norm_hidden_states_)
        count = torch.zeros_like(norm_hidden_states_)
        for index, view in enumerate(views):
            h_start, h_end, w_start, w_end = view
            local_states = norm_hidden_states_[:, h_start:h_end, w_start:w_end, :]
            local_states = rearrange(local_states, 'bh h w d -> bh (h w) d')
            local_output = self.attn1(
                local_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            local_output = rearrange(local_output, 'bh (h w) d -> bh h w d', h = int(sub_h / scale_factor))

            value[:, h_start:h_end, w_start:w_end, :] += local_output * 1
            count[:, h_start:h_end, w_start:w_end, :] += 1

        value = value[:, h_jitter_range:-h_jitter_range, w_jitter_range:-w_jitter_range, :]
        count = count[:, h_jitter_range:-h_jitter_range, w_jitter_range:-w_jitter_range, :]
        attn_output = torch.where(count>0, value/count, value)

        gaussian_local = gaussian_filter(attn_output, kernel_size=(2*current_scale_num-1), sigma=1.0)

        value = torch.zeros_like(norm_hidden_states)
        count = torch.zeros_like(norm_hidden_states)
        for index, global_view in enumerate(global_views):
            h, w = global_view
            global_states = norm_hidden_states[:, h::current_scale_num_h, w::current_scale_num_w, :]
            global_states = rearrange(global_states, 'bh h w d -> bh (h w) d')
            global_output = self.attn1(
                global_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            global_output = rearrange(global_output, 'bh (h w) d -> bh h w d', h = int(global_output.shape[1] ** 0.5))

            value[:, h::current_scale_num_h, w::current_scale_num_w, :] += global_output * 1
            count[:, h::current_scale_num_h, w::current_scale_num_w, :] += 1

        attn_output_global = torch.where(count>0, value/count, value)

        gaussian_global = gaussian_filter(attn_output_global, kernel_size=(2*current_scale_num-1), sigma=1.0)

        attn_output = gaussian_local + (attn_output_global - gaussian_global)
        attn_output = rearrange(attn_output, 'bh h w d -> bh (h w) d')

    else:
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states

    # 2.5 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
    # 2.5 ends

    # 3. Cross-Attention
    if self.attn2 is not None:
        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
            raise ValueError(
                f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
            )

        num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
        ff_output = torch.cat(
            [
                self.ff(hid_slice)
                for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
            ],
            dim=self._chunk_dim,
        )
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    return hidden_states

def ori_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    timestep: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[torch.LongTensor] = None,
):
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Self-Attention
    if self.use_ada_layer_norm:
        norm_hidden_states = self.norm1(hidden_states, timestep)
    elif self.use_ada_layer_norm_zero:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
    else:
        norm_hidden_states = self.norm1(hidden_states)

    # 2. Prepare GLIGEN inputs
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
    gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

    attn_output = self.attn1(
        norm_hidden_states,
        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )

    if self.use_ada_layer_norm_zero:
        attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = attn_output + hidden_states

    # 2.5 GLIGEN Control
    if gligen_kwargs is not None:
        hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
    # 2.5 ends

    # 3. Cross-Attention
    if self.attn2 is not None:
        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = attn_output + hidden_states

    # 4. Feed-forward
    norm_hidden_states = self.norm3(hidden_states)

    if self.use_ada_layer_norm_zero:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    if self._chunk_size is not None:
        # "feed_forward_chunk_size" can be used to save memory
        if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
            raise ValueError(
                f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
            )

        num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
        ff_output = torch.cat(
            [
                self.ff(hid_slice)
                for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)
            ],
            dim=self._chunk_dim,
        )
    else:
        ff_output = self.ff(norm_hidden_states)

    if self.use_ada_layer_norm_zero:
        ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = ff_output + hidden_states

    return hidden_states
