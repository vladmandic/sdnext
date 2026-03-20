import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from modules import devices, masking
from modules.shared import opts


class DepthAnythingV2Detector:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="depth-anything/Depth-Anything-V2-Small-hf", cache_dir=None, local_files_only=False):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only)
        model = AutoModelForDepthEstimation.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only, use_safetensors=True).to(devices.device).eval()
        return cls(model, processor)

    def __call__(self, image, color_map="none", output_type="pil"):
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
        predicted_depth = outputs.predicted_depth
        depth = F.interpolate(predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False).squeeze()
        if opts.control_move_processor:
            self.model.to("cpu")
        depth = depth - depth.min()
        depth_max = depth.max()
        if depth_max > 0:
            depth = depth / depth_max
        depth = (depth * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
        if color_map != "none":
            colormap_key = color_map if color_map in masking.COLORMAP else "inferno"
            depth = cv2.applyColorMap(depth, masking.COLORMAP.index(colormap_key))[:, :, ::-1]
        if output_type == "pil":
            mode = "RGB" if depth.ndim == 3 else "L"
            depth = Image.fromarray(depth, mode=mode)
        return depth
