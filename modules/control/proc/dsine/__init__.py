import os
import torch
import numpy as np
from PIL import Image
from modules import devices
from modules.shared import opts


class DSINEDetector:
    def __init__(self, predictor):
        self.predictor = predictor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="hugoycj/DSINE-hub", cache_dir=None, local_files_only=False):
        from installer import install
        install('geffnet', quiet=True)
        hub_dir = os.path.join(cache_dir, 'torch_hub') if cache_dir else None
        old_hub_dir = torch.hub.get_dir()
        if hub_dir:
            os.makedirs(hub_dir, exist_ok=True)
            torch.hub.set_dir(hub_dir)
        try:
            predictor = torch.hub.load(pretrained_model_or_path, "DSINE", trust_repo=True, source="github")
        finally:
            torch.hub.set_dir(old_hub_dir)
        # Override hardcoded cuda device with project device
        predictor.device = devices.device
        predictor.model = predictor.model.to(devices.device).eval()
        return cls(predictor)

    def __call__(self, image, output_type="pil", **kwargs):
        self.predictor.device = devices.device
        self.predictor.model.to(devices.device)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        with devices.inference_context():
            normals = self.predictor.infer_pil(image)
        if opts.control_move_processor:
            self.predictor.model.to("cpu")
        normals = normals[0].permute(1, 2, 0).cpu().numpy()
        normals = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        if output_type == "pil":
            normals = Image.fromarray(normals)
        return normals
