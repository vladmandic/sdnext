import os
import torch
import torch.nn.functional as F
import safetensors.torch
import huggingface_hub as hf
from modules import devices, shared, paths, sd_vae
from modules.logger import log
from modules.vae.sd_vae_micro_model import MicroDecoder
from modules.vae.model_map import get_vae_type


decoder = None
decoder_cls = None
repo_id = 'vladmandic/MicroDecoder'


def get_timestep() -> torch.Tensor:
    step = getattr(shared.state, 'sampling_step', 0)
    steps = getattr(shared.state, 'sampling_steps', 0)
    ts = getattr(shared.state, 'timestep', 0)
    try:
        ts_val = float(ts)
    except (TypeError, ValueError):
        ts_val = 0.0
    if steps > 0 and step > 0:
        t_val = max(0.0, min(1.0, 1.0 - (float(step) / float(steps))))
    elif ts_val > 1.0:
        t_val = max(0.0, min(1.0, ts_val / 1000.0))
    elif ts_val > 0.0:
        t_val = max(0.0, min(1.0, ts_val))
    else:
        t_val = 0.0
    return torch.tensor([[t_val]], device=devices.device, dtype=devices.dtype)


def decode(latents: torch.Tensor) -> torch.Tensor:
    global decoder, decoder_cls  # pylint: disable=global-statement
    vae_cls = get_vae_type()
    if vae_cls is None:
        return latents
    if (latents is None) or (latents.ndim == 2): # likely packed latents that we cant handle directly
        return latents
    scale_factor = sd_vae.get_vae_scale_factor(patch=False)
    in_channels = getattr(shared.sd_model.vae.config, "latent_channels", 64)

    if (decoder is None) or (decoder_cls != vae_cls):
        model_folder = os.path.join(paths.models_path, "Preview")
        os.makedirs(model_folder, exist_ok=True)
        model_file = f'microdecoder-{vae_cls}.safetensors'
        if os.path.exists(os.path.join(model_folder, model_file)): # attempt local-first
            model_path = os.path.join(model_folder, model_file)
        else:
            try:
                model_path = hf.hf_hub_download(repo_id=repo_id, filename=model_file, local_dir=model_folder)
            except Exception as e:
                log.error(f'MicroDecoder: repo={repo_id} target={vae_cls} {str(e)}')
                return latents
            if not os.path.exists(model_path):
                log.error(f'MicroDecoder: repo={repo_id} target={vae_cls} file="{model_path}" not found')
                return latents
        state_dict = safetensors.torch.load_file(model_path, device=str(devices.device))
        hidden_dim = state_dict["in_proj.0.weight"].shape[0] if "in_proj.0.weight" in state_dict else 256 # dynamically detect hidden_dim from checkpoint weights if present
        in_ch = state_dict["in_proj.0.weight"].shape[1] if "in_proj.0.weight" in state_dict else in_channels
        num_up = sum(1 for k in state_dict if k.startswith("up_blocks.") and k.endswith(".conv.weight"))
        chk_scale = (2 ** num_up) if num_up > 0 else 1
        decoder = MicroDecoder(in_channels=in_ch, hidden_dim=hidden_dim, scale_factor=chk_scale).to(device=devices.device, dtype=devices.dtype)
        log.info(f'\nDecode: type=Micro file="{model_path}" target={vae_cls} channels={in_ch} dim={hidden_dim} scale={chk_scale}')
        decoder.load_state_dict(state_dict)
        decoder.to(device=devices.device, dtype=devices.dtype)
        decoder.eval()
        del state_dict
        decoder_cls = vae_cls


    with devices.inference_context():
        rgb = decoder(latents.to(device=devices.device, dtype=devices.dtype), t=get_timestep())
        # log.trace(f'MicroDecoder: t={get_timestep():.3f} shape={latents.shape} ch={in_channels} scale={scale_factor} in={latents.min():.2f}:{latents.max():.2f} out={rgb.min():.2f}:{rgb.max():.2f}')

    target_size = (latents.shape[-2] * scale_factor, latents.shape[-1] * scale_factor)
    if rgb.shape[-2:] != target_size:
        rgb = F.interpolate(rgb, size=target_size, mode="bilinear", align_corners=False)

    return rgb.clamp(0.0, 1.0)
