import cv2
import numpy as np
import torch
from PIL import Image
from modules import devices
from modules.shared import opts
from modules.control.util import HWC3, resize_image


class Sam2Detector:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="facebook/sam2.1-hiera-large", cache_dir=None, local_files_only=False):
        from transformers import AutoProcessor, AutoModelForMaskGeneration
        processor = AutoProcessor.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only)
        model = AutoModelForMaskGeneration.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only).to(devices.device).eval()
        return cls(model, processor)

    def _generate_grid_points(self, h, w, n_points_per_side=32):
        ys = np.linspace(0, h, n_points_per_side + 2)[1:-1]
        xs = np.linspace(0, w, n_points_per_side + 2)[1:-1]
        points = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
        labels = np.ones(len(points), dtype=np.int64)
        return points, labels

    def _colorize_masks(self, masks, h, w):
        from numpy.random import default_rng
        gen = default_rng(42)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if len(masks) == 0:
            return canvas
        sorted_masks = sorted(enumerate(masks), key=lambda x: x[1].sum(), reverse=True)
        for _idx, mask in sorted_masks:
            color = gen.integers(50, 256, size=3, dtype=np.uint8)
            canvas[mask] = color
        return canvas

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        self.model.to(devices.device)
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        h, w = input_image.shape[:2]
        pil_image = Image.fromarray(input_image)
        points, labels = self._generate_grid_points(h, w, n_points_per_side=16)
        inputs = self.processor(images=pil_image, input_points=[points.tolist()], input_labels=[labels.tolist()], return_tensors="pt")
        inputs = {k: v.to(devices.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        with devices.inference_context():
            outputs = self.model(**inputs)
        masks = self.processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
        )
        if opts.control_move_processor:
            self.model.to("cpu")
        if len(masks) > 0:
            masks_np = masks[0].squeeze(1).cpu().numpy() > 0.5
            scores = outputs.iou_scores[0].squeeze(-1).cpu().numpy() if hasattr(outputs, 'iou_scores') else np.ones(len(masks_np))
            good = scores > 0.7
            masks_np = masks_np[good]
        else:
            masks_np = []
        detected_map = self._colorize_masks(masks_np, h, w)
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        out_h, out_w = img.shape[:2]
        detected_map = cv2.resize(detected_map, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
