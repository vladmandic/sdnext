import os
import torch
import numpy as np
from PIL import Image
from modules import devices
from modules.shared import opts


class StableNormalDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="Stable-X/StableNormal", cache_dir=None, local_files_only=False):
        hub_dir = os.path.join(cache_dir, 'torch_hub') if cache_dir else None
        old_hub_dir = torch.hub.get_dir()
        if hub_dir:
            os.makedirs(hub_dir, exist_ok=True)
            torch.hub.set_dir(hub_dir)
        try:
            # StableNormal's custom pipeline imports from the old diffusers path
            import diffusers.models
            if not hasattr(diffusers.models, 'controlnet'):
                import diffusers.models.controlnets.controlnet as _cn_compat
                diffusers.models.controlnet = _cn_compat
                import sys
                sys.modules['diffusers.models.controlnet'] = _cn_compat
            model = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
        finally:
            torch.hub.set_dir(old_hub_dir)
        model = model.to(devices.device)
        return cls(model)

    def __call__(self, image, output_type="pil", **kwargs):
        self.model.to(devices.device)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        with devices.inference_context():
            normals = self.model(image)
        if opts.control_move_processor:
            self.model.to("cpu")
        if isinstance(normals, torch.Tensor):
            normals = normals.squeeze(0).permute(1, 2, 0).cpu().numpy()
            normals = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        elif isinstance(normals, np.ndarray):
            if normals.max() <= 1.0:
                normals = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        if output_type == "pil":
            if isinstance(normals, Image.Image):
                return normals
            normals = Image.fromarray(normals)
        return normals
