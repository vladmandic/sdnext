import os
from torch import nn
import safetensors
from modules import devices, paths
from modules.logger import log

preview_model = None
dtype = devices.dtype_vae

# Fast Decoder for Stage C latents. E.g. 16 x 24 x 24 -> 3 x 192 x 192
# https://github.com/Stability-AI/StableCascade/blob/master/modules/previewer.py
class Previewer(nn.Module):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(c_in, c_hidden, kernel_size=1),  # 16 channels to 512 channels
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.Conv2d(c_hidden, c_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden),

            nn.ConvTranspose2d(c_hidden, c_hidden // 2, kernel_size=2, stride=2),  # 16 -> 32
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            nn.Conv2d(c_hidden // 2, c_hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 2),

            nn.ConvTranspose2d(c_hidden // 2, c_hidden // 4, kernel_size=2, stride=2),  # 32 -> 64
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.ConvTranspose2d(c_hidden // 4, c_hidden // 4, kernel_size=2, stride=2),  # 64 -> 128
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_hidden // 4, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(c_hidden // 4),

            nn.Conv2d(c_hidden // 4, c_out, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)


def download_model(model_path):
    model_url = 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/previewer.safetensors?download=true'
    if not os.path.exists(model_path):
        import torch
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        log.info(f'Downloading Stable Cascade previewer: {model_path}')
        torch.hub.download_url_to_file(model_url, model_path)

def load_model(model_path):
    checkpoint = {}
    with safetensors.safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)
    return checkpoint

def decode(latents):
    global preview_model # pylint: disable=global-statement
    if preview_model is None:
        model_path = os.path.join(paths.models_path, "VAE-approx", "sd_cascade_previewer.safetensors")
        download_model(model_path)
        if os.path.exists(model_path):
            preview_model = Previewer()
            previewer_checkpoint = load_model(model_path)
            preview_model.load_state_dict(previewer_checkpoint if 'state_dict' not in previewer_checkpoint else previewer_checkpoint['state_dict'])
            preview_model.eval().requires_grad_(False).to(devices.device, dtype)
            del previewer_checkpoint
            log.info(f"Load Stable Cascade previewer: model={model_path}")
    try:
        with devices.inference_context():
            latents = latents.detach().clone().unsqueeze(0).to(devices.device, dtype)
            image = preview_model(latents)[0].clamp(0, 1).float()
            return image
    except Exception as e:
        log.error(f'Stable Cascade previewer: {e}')
        return latents
