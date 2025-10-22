import cv2
import numpy as np
import torch
from PIL import Image

from modules import devices, masking
from modules.shared import opts


class DepthProDetector:
    """Wrapper around Apple's DepthPro depth estimation model."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path: str, cache_dir: str, use_fast_processor: bool = False, **kwargs):
        from transformers import AutoImageProcessor, DepthProForDepthEstimation

        processor_kwargs = {"cache_dir": cache_dir}
        processor_kwargs.update(kwargs)
        if use_fast_processor:
            from transformers.models.depth_pro.image_processing_depth_pro_fast import DepthProImageProcessorFast

            processor = DepthProImageProcessorFast.from_pretrained(
                pretrained_model_or_path,
                **processor_kwargs,
            )
        else:
            processor = AutoImageProcessor.from_pretrained(
                pretrained_model_or_path,
                **processor_kwargs,
            )

        model = DepthProForDepthEstimation.from_pretrained(
            pretrained_model_or_path,
            cache_dir=cache_dir,
        )
        model = model.to(device=devices.device).eval()
        return cls(model, processor)

    def _prepare_inputs(self, image: Image.Image) -> dict:
        inputs = self.processor(images=image, return_tensors="pt")
        tensor_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                tensor_inputs[key] = value.to(device=devices.device)
            else:
                tensor_inputs[key] = value
        return tensor_inputs

    def __call__(
        self,
        image,
        color_map: str = "inferno",
        output_type: str = "pil",
    ):
        if isinstance(image, list):
            image = image[0]
        if image is None:
            return image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        original_size = (image.height, image.width)
        inputs = self._prepare_inputs(image)
        with devices.inference_context():
            outputs = self.model(**inputs)
        results = self.processor.post_process_depth_estimation(outputs, target_sizes=[original_size])
        depth_tensor = results[0]["predicted_depth"].to(torch.float32)
        if opts.control_move_processor:
            self.model.to("cpu")

        # Invert to align with other depth processors that render near as bright
        depth_tensor = 1.0 / torch.clamp(depth_tensor, min=1e-6)
        depth_tensor -= depth_tensor.min()
        max_val = depth_tensor.max()
        if max_val > 0:
            depth_tensor /= max_val
        depth_tensor = (depth_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        depth = depth_tensor.cpu().numpy()

        if color_map and color_map.lower() != "none":
            color = color_map.lower()
            if color not in masking.COLORMAP:
                color = "inferno"
            processed = cv2.applyColorMap(depth, masking.COLORMAP.index(color))[:, :, ::-1]
        else:
            processed = depth

        if output_type == "pil":
            mode = "RGB" if processed.ndim == 3 else "L"
            processed = Image.fromarray(processed, mode=mode)
        return processed
