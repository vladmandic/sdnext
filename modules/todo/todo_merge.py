from collections.abc import Callable
import math
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils.import_utils import is_xformers_available


if is_xformers_available():
    import xformers
    import xformers.ops
    xformers_is_available = True
else:
    xformers_is_available = False


if hasattr(F, "scaled_dot_product_attention"):
    torch2_is_available = True
else:
    torch2_is_available = False


def init_generator(device: torch.device, fallback: torch.Generator = None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.mps.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


def do_nothing(x: torch.Tensor, mode: str | None = None): # pylint: disable=unused-argument
    return x


def mps_gather_workaround(input, dim, index): # pylint: disable=redefined-builtin
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1)
    item = item.permute(0, 3, 1, 2)
    df = cur_h // new_h
    if method in "max_pool":
        item = F.max_pool2d(item, kernel_size=df, stride=df, padding=0)
    elif method in "avg_pool":
        item = F.avg_pool2d(item, kernel_size=df, stride=df, padding=0)
    else:
        item = F.interpolate(item, size=(new_h, new_w), mode=method)
    item = item.permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item


def compute_merge(x: torch.Tensor, tome_info):
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    dim = x.shape[-1]
    if dim == 320:
        cur_level = "level_1"
        downsample_factor = tome_info['args']['downsample_factor']
        ratio = tome_info['args']['ratio']
    elif dim == 640:
        cur_level = "level_2"
        downsample_factor = tome_info['args']['downsample_factor_level_2']
        ratio = tome_info['args']['ratio_level_2']
    else:
        cur_level = "other"
        downsample_factor = 1
        ratio = 0.0

    args = tome_info["args"]

    cur_h, cur_w = original_h // downsample, original_w // downsample
    new_h, new_w = cur_h // downsample_factor, cur_w // downsample_factor

    if tome_info['timestep'] / 1000 > tome_info['args']['timestep_threshold_switch']:
        merge_method = args["merge_method"]
    else:
        merge_method = args["secondary_merge_method"]

    if cur_level != "other" and tome_info['timestep'] / 1000 > tome_info['args']['timestep_threshold_stop']:
        if merge_method == "downsample" and downsample_factor > 1:
            m = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h, args["downsample_method"]) # pylint: disable=unnecessary-lambda-assignment
            u = lambda x: up_or_downsample(x, new_w, new_h, cur_w, cur_h, args["downsample_method"]) # pylint: disable=unnecessary-lambda-assignment
        elif merge_method == "similarity" and ratio > 0.0:
            w = int(math.ceil(original_w / downsample))
            h = int(math.ceil(original_h / downsample))
            r = int(x.shape[1] * ratio)

            # Re-init the generator if it hasn't already been initialized or device has changed.
            if args["generator"] is None:
                args["generator"] = init_generator(x.device)
            elif args["generator"].device != x.device:
                args["generator"] = init_generator(x.device, fallback=args["generator"])

            # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
            # batch, which causes artifacts with use_rand, so force it to be off.
            use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
            m, u = bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r,
                                                    no_rand=not use_rand, generator=args["generator"])
        else:
            m, u = (do_nothing, do_nothing)
    else:
        m, u = (do_nothing, do_nothing)

    merge_fn, unmerge_fn = (m, u)

    return merge_fn, unmerge_fn


def bipartite_soft_matching_random2d(metric: torch.Tensor,
                                     w: int,
                                     h: int,
                                     sx: int,
                                     sy: int,
                                     r: int,
                                     no_rand: bool = False,
                                     generator: torch.Generator = None) -> tuple[Callable, Callable]:
    """
    Partitions the tokens into src and dst and merges r tokens from src to dst.
    Dst tokens are partitioned by choosing one randomy in each (sx, sy) region.

    Args:
     - metric [B, N, C]: metric to use for similarity
     - w: image width in tokens
     - h: image height in tokens
     - sx: stride in the x dimension for dst, must divide w
     - sy: stride in the y dimension for dst, must divide h
     - r: number of tokens to remove (by merging)
     - no_rand: if true, disable randomness (use top left corner only)
     - rand_seed: if no_rand is false, and if not None, sets random seed.
    """
    B, N, _ = metric.shape

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        hsy, wsx = h // sy, w // sx

        # For each sy by sx kernel, randomly assign one token to be dst and the rest src
        if no_rand:
            rand_idx = torch.zeros(hsy, wsx, 1, device=metric.device, dtype=torch.int64)
        else:
            rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(
                metric.device)

        # The image might not divide sx and sy, so we need to work on a view of the top left if the idx buffer instead
        idx_buffer_view = torch.zeros(hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=2, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(hsy, wsx, sy, sx).transpose(1, 2).reshape(hsy * sy, wsx * sx)

        # Image is not divisible by sx or sy so we need to move it into a new buffer
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:(hsy * sy), :(wsx * sx)] = idx_buffer_view
        else:
            idx_buffer = idx_buffer_view

        # We set dst tokens to be -1 and src to be 0, so an argsort gives us dst|src indices
        rand_idx = idx_buffer.reshape(1, -1, 1).argsort(dim=1)

        # We're finished with these
        del idx_buffer, idx_buffer_view

        # rand_idx is currently dst|src, so split them
        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst

        def split(x):
            C = x.shape[-1]
            src = torch.gather(x, dim=1, index=a_idx.expand(B, N - num_dst, C))
            dst = torch.gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity between A and B
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)
        scores = a @ b.transpose(-1, -2)

        # Can't reduce more than the # tokens in src
        r = min(a.shape[1], r)

        # Find the most similar greedily
        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = torch.gather(node_idx[..., None], dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = split(x)
        n, t1, c = src.shape

        unm = torch.gather(src, dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = torch.gather(src, dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape

        src = torch.gather(dst, dim=-2, index=dst_idx.expand(B, r, c))

        # Combine back to the original shape
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2,
                     index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx).expand(B, unm_len, c),
                     src=unm)
        out.scatter_(dim=-2,
                     index=torch.gather(a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx).expand(B, r, c),
                     src=src)

        return out

    return merge, unmerge


class TokenMergeAttentionProcessor:
    def __init__(self):
        # priortize torch2's flash attention, if not fall back to xformers then regular attention
        if torch2_is_available:
            self.attn_method = "torch2"
        elif xformers_is_available:
            self.attn_method = "xformers"
        else:
            self.attn_method = "regular"

    def torch2_attention(self, attn, query, key, value, attention_mask, batch_size):
        inner_dim=key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        return hidden_states

    def xformers_attention(self, attn, query, key, value, attention_mask, batch_size):
        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size * attn.heads, -1, attention_mask.shape[-1])

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, scale=attn.scale
        )

        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states


    def regular_attention(self, attn, query, key, value, attention_mask, batch_size):
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attention_mask is not None:
            attention_mask = attention_mask.reshape(batch_size * attn.heads, -1, attention_mask.shape[-1])

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return hidden_states


    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        temb: torch.FloatTensor | None = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)

        if self._tome_info['args']['merge_tokens'] == "all": # pylint: disable=no-member
            merge_fn, unmerge_fn = compute_merge(hidden_states, self._tome_info) # pylint: disable=no-member
            hidden_states = merge_fn(hidden_states)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self._tome_info['args']['merge_tokens'] == "keys/values": # pylint: disable=no-member
            merge_fn, _ = compute_merge(encoder_hidden_states, self._tome_info) # pylint: disable=no-member
            encoder_hidden_states = merge_fn(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if self.attn_method == "torch2":
            hidden_states = self.torch2_attention(attn, query, key, value, attention_mask, batch_size)
        elif self.attn_method == "xformers":
            hidden_states = self.xformers_attention(attn, query, key, value, attention_mask, batch_size)
        else:
            hidden_states = self.regular_attention(attn, query, key, value, attention_mask, batch_size)

        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if self._tome_info['args']['merge_tokens'] == "all": # pylint: disable=no-member
            hidden_states = unmerge_fn(hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
