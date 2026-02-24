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
        # Load in float32 to avoid NaN from SD.Next global fp16 precision settings
        model = MarigoldNormalsPipeline.from_pretrained(pretrained_model_or_path, torch_dtype=torch.float32, cache_dir=cache_dir, **load_config)
        return cls(model)

    def to(self, device):
        self.model.to(device)
        return self

    def __call__(self, input_image, denoising_steps=4, ensemble_size=4, processing_res=768, match_input_res=True, output_type=None):
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
        self.model.to(device=devices.device)
        res = self.model(
            input_image,
            num_inference_steps=denoising_steps,
            ensemble_size=ensemble_size,
            processing_resolution=processing_res,
            match_input_resolution=match_input_res,
            batch_size=1,
            output_type="pt",
        )
        normal_images = self.model.image_processor.visualize_normals(res.prediction)
        if opts.control_move_processor:
            self.model.to("cpu")
        return normal_images[0]
