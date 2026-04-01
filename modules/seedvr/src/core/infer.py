from typing import List, Optional, Tuple, Union
import torch
from einops import rearrange
from ..common.diffusion import classifier_free_guidance_dispatcher, create_sampler_from_config, create_sampling_timesteps_from_config, create_schedule_from_config
from ..models.dit_v2 import na


def optimized_channels_to_last(tensor: torch.Tensor) -> torch.Tensor:
    """🚀 Optimized replacement for rearrange(tensor, 'b c ... -> b ... c')
    Moves channels from position 1 to last position using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, channels, spatial]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, channels, height, width]
        return tensor.permute(0, 2, 3, 1)
    elif tensor.ndim == 5:  # [batch, channels, depth, height, width]
        return tensor.permute(0, 2, 3, 4, 1)
    else:
        # Fallback for other dimensions - move channel (dim=1) to last
        dims = list(range(tensor.ndim))
        dims = [dims[0]] + dims[2:] + [dims[1]]  # [0, 2, 3, ..., 1]
        return tensor.permute(*dims)

def optimized_channels_to_second(tensor):
    """🚀 Optimized replacement for rearrange(tensor, 'b ... c -> b c ...')
    Moves channels from last position to position 1 using PyTorch native operations.
    """
    if tensor.ndim == 3:  # [batch, spatial, channels]
        return tensor.permute(0, 2, 1)
    elif tensor.ndim == 4:  # [batch, height, width, channels]
        return tensor.permute(0, 3, 1, 2)
    elif tensor.ndim == 5:  # [batch, depth, height, width, channels]
        return tensor.permute(0, 4, 1, 2, 3)
    else:
        # Fallback for other dimensions - move last dim to position 1
        dims = list(range(tensor.ndim))
        dims = [dims[0], dims[-1]] + dims[1:-1]  # [0, -1, 1, 2, ..., -2]
        return tensor.permute(*dims)


class VideoDiffusionInfer():
    def __init__(self, config, device: str, dtype: torch.dtype):
        from installer import install
        install('omegaconf')
        self.config = config
        self.device = device
        self.dtype = dtype
        self.vae = None
        self.dit = None
        self.sampler = None
        self.schedule = None

    def get_condition(self, latent: torch.Tensor, latent_blur: torch.Tensor, task: str) -> torch.Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    def configure_diffusion(self):
        self.schedule = create_schedule_from_config(
            config=self.config.diffusion.schedule,
        )
        self.sampling_timesteps = create_sampling_timesteps_from_config( # pylint: disable=attribute-defined-outside-init
            config=self.config.diffusion.timesteps.sampling,
            schedule=self.schedule,
            device=self.device,
        )
        self.sampler = create_sampler_from_config(
            config=self.config.diffusion.sampler,
            schedule=self.schedule,
            timesteps=self.sampling_timesteps,
        )

    # -------------------------------- Helper ------------------------------- #

    @torch.no_grad()
    def vae_encode(self, samples: List[torch.Tensor]) -> List[torch.Tensor]:
        from omegaconf import ListConfig
        use_sample = self.config.vae.get("use_sample", True)
        latents = []
        if len(samples) > 0:
            dtype = self.vae.dtype
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=self.device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=self.device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.config.vae.grouping:
                batches, indices = na.pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # Vae process by each group.
            for sample in batches:
                sample = sample.to(self.device, dtype)
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(sample).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = self.vae.encode(sample).posterior.mode().squeeze(2)
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                #latent = optimized_channels_to_last(latent)
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.config.vae.grouping:
                latents = na.unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]
        return latents


    @torch.no_grad()
    def vae_decode(self, latents: List[torch.Tensor], target_dtype: torch.dtype = None) -> List[torch.Tensor]:
        """🚀 VAE decode optimisé - décodage direct sans chunking, compatible avec autocast externe"""
        from omegaconf import ListConfig
        samples = []
        if len(latents) > 0:
            device = self.device
            dtype = self.vae.dtype
            scale = self.config.vae.scaling_factor
            shift = self.config.vae.get("shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)


            # 🚀 OPTIMISATION 1: Group latents intelligemment pour batch processing
            if self.config.vae.grouping:
                latents, indices = na.pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            # 🚀 OPTIMISATION 2: Traitement batch optimisé avec dtype adaptatif
            for _i, latent in enumerate(latents):
                # Préparation optimisée du latent
                # Utiliser target_dtype si fourni (évite double autocast)
                effective_dtype = target_dtype if target_dtype is not None else dtype
                latent = latent.to(device, effective_dtype, non_blocking=True)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                #latent = optimized_channels_to_second(latent)
                latent = latent.squeeze(2)

                # 🚀 OPTIMISATION 3: Décodage direct SANS autocast (utilise l'autocast externe)
                sample = self.vae.decode(latent).sample
                #sample = self.vae.decode(latent).sample
                #sample = self.vae.decode(latent).sample

                # 🚀 OPTIMISATION 4: Post-processing conditionnel
                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)

                samples.append(sample)

            # Ungroup back to individual sample with the original order.
            if self.config.vae.grouping:
                samples = na.unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]
        return samples

    def timestep_transform(self, timesteps: torch.Tensor, latents_shapes: torch.Tensor):
        # Skip if not needed.
        if not self.config.diffusion.timesteps.get("transform", False):
            return timesteps

        # Compute resolution.
        vt = self.config.vae.model.get("temporal_downsample_factor", 4)
        vs = self.config.vae.model.get("spatial_downsample_factor", 8)
        frames = (latents_shapes[:, 0] - 1) * vt + 1
        heights = latents_shapes[:, 1] * vs
        widths = latents_shapes[:, 2] * vs

        # Compute shift factor.
        def get_lin_function(x1, y1, x2, y2):
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return lambda x: m * x + b

        img_shift_fn = get_lin_function(x1=256 * 256, y1=1.0, x2=1024 * 1024, y2=3.2)
        vid_shift_fn = get_lin_function(x1=256 * 256 * 37, y1=1.0, x2=1280 * 720 * 145, y2=5.0)
        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.schedule.T
        return timesteps

    @torch.no_grad()
    def inference(
        self,
        noises: List[torch.Tensor],
        conditions: List[torch.Tensor],
        texts_pos: Union[List[str], List[torch.Tensor], List[Tuple[torch.Tensor]]],
        texts_neg: Union[List[str], List[torch.Tensor], List[Tuple[torch.Tensor]]],
        cfg_scale: Optional[float] = None,
        temporal_overlap: int = 0, # pylint: disable=unused-argument
    ) -> List[torch.Tensor]:
        assert len(noises) == len(conditions) == len(texts_pos) == len(texts_neg)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        # Set cfg scale
        if cfg_scale is None:
            cfg_scale = self.config.diffusion.cfg.scale

        # 🚀 OPTIMISATION: Détecter le dtype du modèle pour performance optimale
        model_dtype = next(self.dit.parameters()).dtype
        # Adapter les dtypes selon le modèle
        if model_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            target_dtype = torch.float16
        elif model_dtype == torch.float16:
            target_dtype = torch.float16
        else:
            target_dtype = torch.bfloat16
        # Text embeddings.
        assert type(texts_pos[0]) is type(texts_neg[0])
        if isinstance(texts_pos[0], str):
            text_pos_embeds, text_pos_shapes = self.text_encode(texts_pos) # pylint: disable=no-member
            text_neg_embeds, text_neg_shapes = self.text_encode(texts_neg) # pylint: disable=no-member
        elif isinstance(texts_pos[0], tuple):
            text_pos_embeds, text_pos_shapes = [], []
            text_neg_embeds, text_neg_shapes = [], []
            for pos in zip(*texts_pos):
                emb, shape = na.flatten(pos)
                text_pos_embeds.append(emb)
                text_pos_shapes.append(shape)
            for neg in zip(*texts_neg):
                emb, shape = na.flatten(neg)
                text_neg_embeds.append(emb)
                text_neg_shapes.append(shape)
        else:
            text_pos_embeds, text_pos_shapes = na.flatten(texts_pos)
            text_neg_embeds, text_neg_shapes = na.flatten(texts_neg)

        # Adapter les embeddings texte au dtype cible (compatible avec FP8)
        if isinstance(text_pos_embeds, torch.Tensor):
            text_pos_embeds = text_pos_embeds.to(target_dtype)
        if isinstance(text_neg_embeds, torch.Tensor):
            text_neg_embeds = text_neg_embeds.to(target_dtype)

        # Flatten.
        latents, latents_shapes = na.flatten(noises)
        latents_cond, _ = na.flatten(conditions)

        # Adapter les latents au dtype cible (compatible avec FP8)
        latents = latents.to(target_dtype) if latents.dtype != target_dtype else latents
        latents_cond = latents_cond.to(target_dtype) if latents_cond.dtype != target_dtype else latents_cond
        self.dit = self.dit.to(device=self.device, dtype=target_dtype)

        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= self.config.diffusion.cfg.get("partial", 1)
                    else 1.0
                ),
                rescale=self.config.diffusion.cfg.rescale,
            ),
        )

        latents = na.unflatten(latents, latents_shapes)

        # 🎯 Pré-calcul des dtypes (une seule fois)
        vae_dtype = self.vae.dtype
        decode_dtype = torch.float16 if (vae_dtype == torch.float16 or target_dtype == torch.float16) else vae_dtype
        samples = self.vae_decode(latents, target_dtype=decode_dtype)

        if samples and len(samples) > 0 and samples[0].dtype != torch.float16:
            samples = [sample.to(torch.float16, non_blocking=True) for sample in samples]

        return samples
