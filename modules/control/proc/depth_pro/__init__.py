import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from modules import devices, masking
from modules.shared import opts


class DepthProDetector:
    """Apple DepthPro detector (aligned with Depth Anything style)."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str = "apple/DepthPro-hf", cache_dir: str = None, local_files_only = False) -> "DepthProDetector":
        from transformers import AutoImageProcessor, DepthProForDepthEstimation

        processor = AutoImageProcessor.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only)
        model = DepthProForDepthEstimation.from_pretrained(
            pretrained_model_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).to(devices.device).eval()
        return cls(model, processor)

    def __call__(self, image, color_map: str = "none", output_type: str = "pil"):
        self.model.to(devices.device)
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(devices.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with devices.inference_context():
            outputs = self.model(**inputs)
        results = self.processor.post_process_depth_estimation(outputs, target_sizes=[(h, w)])
        depth_tensor = results[0]["predicted_depth"].to(devices.device, dtype=torch.float32)

        if opts.control_move_processor:
            self.model.to("cpu")

        depth_tensor = F.interpolate(depth_tensor[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
        depth_tensor = 1.0 / torch.clamp(depth_tensor, min=1e-6)
        depth_tensor -= depth_tensor.min()
        depth_max = depth_tensor.max()
        if depth_max > 0:
            depth_tensor /= depth_max
        depth = (depth_tensor * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()

        if color_map != "none":
            colormap_key = color_map if color_map in masking.COLORMAP else "inferno"
            depth = cv2.applyColorMap(depth, masking.COLORMAP.index(colormap_key))[:, :, ::-1]
        if output_type == "pil":
            mode = "RGB" if depth.ndim == 3 else "L"
            depth = Image.fromarray(depth, mode=mode)
        return depth
