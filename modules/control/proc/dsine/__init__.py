import os
import torch
import numpy as np
from PIL import Image
from modules import devices
from modules.shared import opts


class DSINEDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="hugoycj/DSINE-hub", cache_dir=None, local_files_only=False):
        hub_dir = os.path.join(cache_dir, 'torch_hub') if cache_dir else None
        old_hub_dir = torch.hub.get_dir()
        if hub_dir:
            os.makedirs(hub_dir, exist_ok=True)
            torch.hub.set_dir(hub_dir)
        try:
            model = torch.hub.load(pretrained_model_or_path, "DSINE", trust_repo=True, source="github")
        finally:
            torch.hub.set_dir(old_hub_dir)
        model = model.to(devices.device).eval()
        return cls(model)

    def __call__(self, image, output_type="pil", **kwargs):
        self.model.to(devices.device)
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        img = image.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(devices.device)
        with devices.inference_context():
            normals = self.model(img_tensor)
        if opts.control_move_processor:
            self.model.to("cpu")
        normals = normals[0].permute(1, 2, 0).cpu().numpy()
        normals = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        if normals.shape[:2] != (h, w):
            normals = np.array(Image.fromarray(normals).resize((w, h), Image.Resampling.BILINEAR))
        if output_type == "pil":
            normals = Image.fromarray(normals)
        return normals
