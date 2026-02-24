import cv2
import torch
import numpy as np
from PIL import Image
from modules import devices, masking
from modules.shared import opts


class LotusDetector:
    def __init__(self, unet, vae, text_encoder, tokenizer, scheduler):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="jingheya/lotus-depth-g-v2-1-disparity", cache_dir=None, local_files_only=False, **kwargs):
        from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        load_kwargs = dict(cache_dir=cache_dir, local_files_only=local_files_only)
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_or_path, subfolder="unet", **load_kwargs)
        vae = AutoencoderKL.from_pretrained(pretrained_model_or_path, subfolder="vae", **load_kwargs)
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_or_path, subfolder="text_encoder", **load_kwargs)
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_or_path, subfolder="tokenizer", **load_kwargs)
        scheduler = DDPMScheduler.from_pretrained(pretrained_model_or_path, subfolder="scheduler", **load_kwargs)
        return cls(unet, vae, text_encoder, tokenizer, scheduler)

    def _to(self, device):
        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)

    def __call__(self, image, color_map="none", output_type="pil", **kwargs):
        self._to(devices.device)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        orig_w, orig_h = image.size
        # Resize to processing resolution (768) while maintaining aspect ratio
        processing_res = 768
        scale = processing_res / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale / 8) * 8)
        new_h = int(round(orig_h * scale / 8) * 8)
        image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        # Convert to tensor [-1, 1], matching model dtype
        dtype = next(self.vae.parameters()).dtype
        rgb = torch.from_numpy(np.array(image_resized).astype(np.float32) / 127.5 - 1.0)
        rgb = rgb.permute(2, 0, 1).unsqueeze(0).to(devices.device, dtype=dtype)
        # Encode prompt (empty string for unconditional depth)
        text_inputs = self.tokenizer("", padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        prompt_embeds = self.text_encoder(text_inputs.input_ids.to(devices.device))[0]
        # Task embedding for depth prediction: sin/cos encoding of [1, 0] → shape [1, 4]
        task_emb = torch.tensor([1, 0], dtype=dtype, device=devices.device).unsqueeze(0)
        task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)
        # Single-step direct prediction (Lotus-G)
        self.scheduler.set_timesteps(1, device=devices.device)
        timestep = self.scheduler.timesteps[0:1]
        with devices.inference_context():
            # Encode RGB to latent space
            rgb_latents = self.vae.encode(rgb).latent_dist.sample() * self.vae.config.scaling_factor
            # Random noise latents (half channels of UNet input)
            noise_latents = torch.randn(rgb_latents.shape, device=devices.device, dtype=dtype)
            # Concatenate along channel dim: [rgb_latents, noise_latents]
            latent_input = torch.cat([rgb_latents, noise_latents], dim=1)
            # UNet forward pass with task embedding as class_labels
            prediction = self.unet(latent_input, timestep, encoder_hidden_states=prompt_embeds, class_labels=task_emb).sample
            # Decode prediction
            prediction = prediction / self.vae.config.scaling_factor
            decoded = self.vae.decode(prediction).sample
        if opts.control_move_processor:
            self._to("cpu")
        # Convert from [-1,1] to [0,1]
        depth = (decoded.squeeze(0).permute(1, 2, 0).float().cpu().numpy() + 1.0) * 0.5
        depth = depth.mean(axis=2) if depth.ndim == 3 else depth
        depth = depth - depth.min()
        depth_max = depth.max()
        if depth_max > 0:
            depth = depth / depth_max
        depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
        # Resize back to original
        if depth.shape[:2] != (orig_h, orig_w):
            depth = np.array(Image.fromarray(depth).resize((orig_w, orig_h), Image.Resampling.BILINEAR))
        if color_map != "none":
            colormap_key = color_map if color_map in masking.COLORMAP else "inferno"
            depth = cv2.applyColorMap(depth, masking.COLORMAP.index(colormap_key))[:, :, ::-1]
        if output_type == "pil":
            mode = "RGB" if depth.ndim == 3 else "L"
            depth = Image.fromarray(depth, mode=mode)
        return depth
