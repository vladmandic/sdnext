import cv2
import torch
import numpy as np
from PIL import Image
from modules import devices, masking
from modules.shared import opts


class LotusDetector:
    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="jingheya/lotus-depth-g-v2-1-disparity", cache_dir=None, local_files_only=False, **kwargs):
        from diffusers import DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            pretrained_model_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        return cls(pipe)

    def __call__(self, image, color_map="none", output_type="pil", **kwargs):
        self.pipe.to(devices.device)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        h, w = image.size[1], image.size[0]
        with devices.inference_context():
            result = self.pipe(image, processing_res=768)
        if opts.control_move_processor:
            self.pipe.to("cpu")
        depth = result.prediction
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().cpu().numpy()
        elif isinstance(depth, Image.Image):
            depth = np.array(depth)
        if depth.ndim == 3 and depth.shape[0] in (1, 3):
            depth = depth.squeeze(0) if depth.shape[0] == 1 else np.transpose(depth, (1, 2, 0))
        if depth.dtype == np.float32 or depth.dtype == np.float64:
            depth = depth - depth.min()
            depth_max = depth.max()
            if depth_max > 0:
                depth = depth / depth_max
            depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY) if depth.shape[2] == 3 else depth[:, :, 0]
        if depth.shape[:2] != (h, w):
            depth = np.array(Image.fromarray(depth).resize((w, h), Image.Resampling.BILINEAR))
        if color_map != "none":
            colormap_key = color_map if color_map in masking.COLORMAP else "inferno"
            depth = cv2.applyColorMap(depth, masking.COLORMAP.index(colormap_key))[:, :, ::-1]
        if output_type == "pil":
            mode = "RGB" if depth.ndim == 3 else "L"
            depth = Image.fromarray(depth, mode=mode)
        return depth
