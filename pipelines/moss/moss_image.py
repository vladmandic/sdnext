# Adapted from pipeline_moss_image.py in OpenMOSS-Team/MOSS-Image-SFT (Apache-2.0).
# Transformer math and state-dict names follow the release; the pipeline adds step callbacks,
# interrupt, per-step guidance and returns decoder-ready latents like Flux2KleinPipeline.
import math
from collections.abc import Callable
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor


class MOSSImageRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.to(dtype=torch.float32) * torch.rsqrt(x.to(dtype=torch.float32).square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def situ(hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def validate_skip_layers(layers, depth: int, name: str = 'skip_layers') -> frozenset[int]:
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers): # pylint: disable=unidiomatic-typecheck
        raise ValueError(f'{name} must contain integer block indices in [0, {depth}).')
    return frozenset(layers)


def rotate(x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return torch.addcmul(x * cos, torch.cat((-b, a), -1), sin)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rope: torch.Tensor, heads: int) -> torch.Tensor:
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).transpose(1, 2)
    k = rotate(k.reshape(shape), rope).transpose(1, 2)
    v = v.reshape(shape).transpose(1, 2)
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v).transpose(1, 2).flatten(2)


class MOSSImageTextBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = MOSSImageRMSNorm(dim), MOSSImageRMSNorm(dim)
        self.q_norm, self.k_norm = MOSSImageRMSNorm(dim // heads), MOSSImageRMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(attention(q, k, v, rope, self.heads))
        y = self.norm2(x)
        return x + self.ffn_down(situ(self.ffn_up(y), self.ffn_gate(y)))


class MOSSImageTextProjection(nn.Module):
    def __init__(self, text_dim: int, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MOSSImageTextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        for block in self.blocks:
            x = block(x, rope)
        return x


class MOSSImageBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = MOSSImageRMSNorm(dim), MOSSImageRMSNorm(dim)
        self.q_norm, self.k_norm = MOSSImageRMSNorm(dim // heads), MOSSImageRMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ('q', 'k', 'v', 'o')})
        self.proj['ffn_up_gate'] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj['ffn_down'] = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, modulation: torch.Tensor, skip: bool = False) -> torch.Tensor:
        if skip: # the call still runs, so offload hooks see every block in order
            return x
        dtype = x.dtype
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = torch.addcmul(shift_a.unsqueeze(1), self.norm1(x), (1 + scale_a.unsqueeze(1))).to(dtype=dtype)
        q = self.q_norm(self.proj['q'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj['k'](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = torch.addcmul(x, self.proj['o'](attention(q, k, self.proj['v'](y), rope, self.heads)), gate_a.unsqueeze(1))
        y = torch.addcmul(shift_f.unsqueeze(1), self.norm2(x), (1 + scale_f.unsqueeze(1))).to(dtype=dtype)
        hidden, gate = self.proj['ffn_up_gate'](y).chunk(2, -1)
        return torch.addcmul(x, self.proj['ffn_down'](situ(hidden, gate)), gate_f.unsqueeze(1))


class MOSSImageRoPE(nn.Module):
    """4d rope over (r, f, h, w): image tokens first, then text; frequencies stay fp32 whatever the module dtype."""
    def __init__(self, head_dim: int, base: float):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer('bands', self.default_bands(torch.device('cpu')))

    def default_bands(self, device) -> torch.Tensor:
        exponent = torch.arange(self.num_bands, device='cpu', dtype=torch.float32) / self.num_bands
        return (1 / (self.base ** exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.register_buffer('bands', self.default_bands(self.bands.device)) # a bf16 cast cannot be undone by a later float()
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.register_buffer('bands', self.default_bands(self.bands.device))

    def forward(self, height: int, width: int, text_length: int, device) -> torch.Tensor:
        if self.bands.dtype != torch.float32 or self.bands.device.type == 'meta': # accelerate can install tensors past the load hooks
            self.register_buffer('bands', self.default_bands(device))
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).to(dtype=torch.float32)
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands.to(device)
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    """Single-stream DiT; each sample runs at its own unpadded text length."""
    _no_split_modules = ['MOSSImageBlock', 'MOSSImageTextBlock']

    @register_to_config
    def __init__(self, dim: int = 3072, ffn_dim: int = 8192, num_heads: int = 24, num_layers: int = 42, text_dim: int = 4096, freq_dim: int = 256, latent_channels: int = 128, rope_base: float = 10000.0):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError('Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.')
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = MOSSImageTextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = MOSSImageRoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MOSSImageBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = MOSSImageRMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def forward(self, hidden_states: torch.Tensor, timestep, encoder_hidden_states: torch.Tensor, encoder_attention_mask: torch.Tensor | None = None, return_dict: bool = True, skip_layers=None):
        dtype = self.dtype
        skip_layers = validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config['latent_channels']:
            raise ValueError('hidden_states must be [B, latent_channels, H, W].')
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError('encoder_hidden_states must be [B, S, text_dim].')
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2]:
            raise ValueError('encoder_attention_mask must be a [B, S] mask.')
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32).reshape(-1).expand(batch)
        half = self.config['freq_dim'] // 2 # the sinusoid is computed in float64 and fed to the mlp in float32, as in training
        freq = 10000 ** (-torch.arange(half, device=times.device, dtype=torch.float64) / half)
        phase = times.to(dtype=torch.float64).unsqueeze(1) * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).to(dtype=dtype)
        time = self.time_embedding(embedding)
        mod = self.shared_block_modulation(time).view(batch, 6, -1)
        head = self.video_head_modulation(time).view(batch, 2, -1)
        from modules.attention.sparse import layout as sparse_layout # delayed, this module is also importable without the webui
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index:index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError('Each sample needs at least one valid text token.')
            video = self.video_embedder(hidden_states[index:index + 1].flatten(2).transpose(1, 2).to(dtype=dtype, memory_format=torch.contiguous_format))
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(dtype=dtype)
            text = self.text_embedder(context, rope[:, height * width:])
            sparse_layout.publish_segments([('image', height * width), ('text', context.shape[1])], source='moss') # image first, each sample at its own text length
            x = torch.cat((video, text), 1)
            for layer_index, layer in enumerate(self.layers):
                x = layer(x, rope, mod[index:index + 1], skip=layer_index in skip_layers)
            x = self.final_norm_video(x[:, :height * width])
            x = torch.addcmul(head[index:index + 1, 0, None], x, (1 + head[index:index + 1, 1, None]))
            x = self.final_linear_video(x)
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)


class MOSSImageFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler over the training schedule: sigmas shifted, then snapped to the nearest training grid point."""
    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, training_shift: float = 1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps: int = 50, device=None): # pylint: disable=arguments-differ
        shift, training_shift, total = self.config['shift'], self.config['training_shift'], self.config['num_train_timesteps']
        if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
            raise ValueError('Steps and shifts must be positive.')
        def grid(count, shift):
            lower = shift / (total + shift - 1)
            values = torch.linspace(1, lower, count)
            return shift * values / (1 + (shift - 1) * values)
        sigmas = grid(num_inference_steps, shift)
        training = grid(total, training_shift).flip(0)
        upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
        lower = (upper - 1).clamp(min=0)
        nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
        sigmas = torch.unique_consecutive(training[nearest])
        self._shift = 1.0 # sigmas are already shifted and snapped
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


class MOSSImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = 'text_encoder->transformer->vae'
    _optional_components = ['text_encoder', 'tokenizer', 'vae']
    _callback_tensor_inputs = ['latents', 'noise_pred']

    def __init__(self, transformer: MOSSImageTransformer, scheduler: FlowMatchEulerDiscreteScheduler, text_encoder=None, tokenizer=None, vae=None):
        super().__init__()
        self.register_modules(transformer=transformer, scheduler=scheduler, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae)
        self.vae_scale_factor = 8
        self.patch_size = 2 # latents pack 2x2 vae patches into 128 channels
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * self.patch_size)
        self._guidance_scale = 1.0
        self._num_timesteps = 0
        self._interrupt = False

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @torch.no_grad()
    def encode_prompt(self, prompt, max_sequence_length: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError('Text encoder and tokenizer are required for string prompts.')
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError('prompt must be a string or a nonempty list of strings.')
        rows = []
        for value in prompts: # one unpadded forward per prompt: padding changes the gated deltanet layers
            formatted = self.tokenizer.apply_chat_template([{'role': 'user', 'content': value}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            inputs = self.tokenizer(formatted, return_tensors='pt', truncation=True, max_length=max_sequence_length).to(self._execution_device)
            output = self.text_encoder(**inputs, use_cache=False, return_dict=True)
            rows.append(output.last_hidden_state[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, device=embeds.device, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeds[index, :row.shape[0]] = row
            mask[index, :row.shape[0]] = True
        return embeds, mask

    def unpack_latents(self, latents: torch.Tensor, offset: bool = True) -> torch.Tensor:
        """Batch-norm denormalize and unpatchify [B, 128, h, w] into the [B, 32, 2h, 2w] vae latent space; offset=False for differences such as a velocity."""
        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=torch.float32)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1).to(device=latents.device, dtype=torch.float32) + self.vae.config.batch_norm_eps)
        if offset:
            z = torch.addcmul(mean, latents.to(dtype=torch.float32), std)
        else:
            z = latents.to(dtype=torch.float32) * std
        b, c, h, w = z.shape
        return z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        *,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.0,
        negative_prompt='',
        num_images_per_prompt: int = 1,
        generator=None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 4096,
        output_type: str = 'pil',
        return_dict: bool = True,
        uncond_skip_layers=(22,),
        slg_layers=None,
        slg_scale: float = 0.0,
        slg_start: float = 0.0,
        slg_stop: float = 1.0,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ):
        """Generate images; output_type='latent' returns vae latents [B, 32, H/8, W/8].

        uncond_skip_layers (default block 22, zero-based) replaces the negative branch with a skipped-block prediction while CFG is active.
        slg_layers with slg_scale > 0 adds a skipped positive forward and the correction slg_scale * (cond - cond_skip).
        Both apply where slg_start <= step / steps < slg_stop.
        """
        callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs or ['latents']
        uncond_skip_layers = validate_skip_layers(uncond_skip_layers, len(self.transformer.layers), 'uncond_skip_layers')
        slg_layers = validate_skip_layers(slg_layers, len(self.transformer.layers), 'slg_layers')
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError('slg_scale must be finite and nonnegative.')
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError('Require 0 <= slg_start <= slg_stop <= 1.')
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError('height and width must be positive multiples of 16.')
        if num_images_per_prompt < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError('Invalid image count or guidance scale.')
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError('Provide exactly one of prompt and prompt_embeds.')
        if any(k not in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs):
            raise ValueError(f'callback_on_step_end_tensor_inputs must be in {self._callback_tensor_inputs}')
        max_sequence_length = min(max(int(max_sequence_length), 1), 4096)
        self._guidance_scale = guidance_scale
        self._interrupt = False
        device = self._execution_device

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt, max_sequence_length)
        batch = prompt_embeds.shape[0]

        def prepare(embeds, mask):
            if embeds.ndim != 3 or embeds.shape[0] != batch:
                raise ValueError('Prompt embeddings must be [B, S, C] with matching batch size.')
            if mask is None:
                mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device=embeds.device)
            return embeds.to(device).repeat_interleave(num_images_per_prompt, 0), mask.to(device).repeat_interleave(num_images_per_prompt, 0)

        positive, positive_mask = prepare(prompt_embeds, prompt_attention_mask)
        negative, negative_mask = None, None
        if guidance_scale > 1.0:
            if negative_prompt_embeds is None:
                negatives = [negative_prompt or ''] * batch if (negative_prompt is None or isinstance(negative_prompt, str)) else list(negative_prompt)
                if len(negatives) != batch:
                    raise ValueError('negative_prompt must match prompt batch size.')
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(negatives, max_sequence_length)
            negative, negative_mask = prepare(negative_prompt_embeds, negative_prompt_attention_mask)

        shape = (batch * num_images_per_prompt, self.transformer.config.latent_channels, height // 16, width // 16)
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32) # fp32 noise: bf16 noise cannot be promoted back
        elif tuple(latents.shape) != shape:
            raise ValueError(f'latents must have shape {shape}.')
        else:
            latents = latents.to(device=device, dtype=torch.float32)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for step_index, timestep in enumerate(timesteps):
                if self.interrupt:
                    continue
                skip_active = slg_start <= step_index / self._num_timesteps < slg_stop
                do_cfg = negative is not None and self._guidance_scale > 1.0 # the step callback can end guidance early
                # guidance combines fp32 predictions even when the transformer runs in bf16
                noise_pred = self.transformer(latents, timestep, positive, positive_mask).sample.to(dtype=torch.float32)
                conditional = noise_pred
                if do_cfg:
                    skip_kwargs = {'skip_layers': uncond_skip_layers} if uncond_skip_layers and skip_active else {}
                    uncond = self.transformer(latents, timestep, negative, negative_mask, **skip_kwargs).sample.to(dtype=torch.float32)
                    noise_pred = torch.lerp(uncond, noise_pred, self._guidance_scale)
                if slg_layers and slg_scale > 0 and skip_active:
                    skipped = self.transformer(latents, timestep, positive, positive_mask, skip_layers=slg_layers).sample.to(dtype=torch.float32)
                    if not do_cfg:
                        noise_pred = skipped
                    noise_pred = torch.add(noise_pred, (conditional - skipped), alpha=slg_scale)
                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                    latents = callback_outputs.pop('latents', latents)
                progress_bar.update()

        latents = self.unpack_latents(latents)
        if output_type == 'latent':
            images = latents
        else:
            if self.vae is None:
                raise ValueError('A FLUX.2 VAE is required for decoded output.')
            images = self.vae.decode(latents.to(dtype=self.vae.dtype), return_dict=False)[0].clamp(-1, 1)
            images = self.image_processor.postprocess(images, output_type=output_type)
        self.maybe_free_model_hooks()
        return ImagePipelineOutput(images=images) if return_dict else (images,)
