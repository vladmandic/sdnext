import torch
import numpy as np
from PIL import Image
from modules import devices
from modules.shared import opts


class MarigoldNormalsDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="prs-eth/marigold-normals-v1-1", cache_dir=None, **load_config):
        from diffusers import MarigoldNormalsPipeline
        model = MarigoldNormalsPipeline.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, **load_config)
        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, denoising_steps=4, ensemble_size=4, processing_res=768, match_input_res=True, output_type=None):
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        self.model.to(device=devices.device, dtype=torch.float16)
        res = self.model(
            input_image,
            denoising_steps=denoising_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=match_input_res,
            batch_size=1,
            show_progress_bar=True,
        )
        normal_map = res.prediction
        if opts.control_move_processor:
            self.model.to("cpu")
        if isinstance(normal_map, np.ndarray):
            normal_map = ((normal_map + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
            if normal_map.ndim == 4:
                normal_map = normal_map[0]
            if normal_map.shape[0] == 3:
                normal_map = np.transpose(normal_map, (1, 2, 0))
        elif isinstance(normal_map, torch.Tensor):
            normal_map = normal_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
            normal_map = ((normal_map + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        if output_type == "pil" or output_type is None:
            normal_map = Image.fromarray(normal_map)
        return normal_map
