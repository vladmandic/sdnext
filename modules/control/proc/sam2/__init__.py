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
        model = AutoModelForMaskGeneration.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only, use_safetensors=True).to(devices.device).eval()
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
        import torch.nn.functional as F
        self.model.to(devices.device)
        if not isinstance(input_image, np.ndarray):
            input_image = np.array(input_image, dtype=np.uint8)
        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        h, w = input_image.shape[:2]
        pil_image = Image.fromarray(input_image)
        points, labels = self._generate_grid_points(h, w, n_points_per_side=16)
        # Process grid points in batches to avoid OOM
        all_masks = []
        all_scores = []
        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch_pts = points[i:i + batch_size]
            batch_lbl = labels[i:i + batch_size]
            # SAM2 expects 4-level nesting: [image, object, point, coords]
            pts_nested = [[[pt] for pt in batch_pts.tolist()]]
            lbl_nested = [[[lb] for lb in batch_lbl.tolist()]]
            inputs = self.processor(images=pil_image, input_points=pts_nested, input_labels=lbl_nested, return_tensors="pt")
            inputs = {k: v.to(devices.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with devices.inference_context():
                outputs = self.model(**inputs)
            # pred_masks: [1, N_objects, N_masks_per_obj, mask_h, mask_w] (low-res logits)
            pred_masks = outputs.pred_masks[0]  # [N_objects, N_masks, mask_h, mask_w]
            iou_scores = outputs.iou_scores[0] if hasattr(outputs, 'iou_scores') else torch.ones(pred_masks.shape[:2], device=pred_masks.device)
            for obj_idx in range(pred_masks.shape[0]):
                best = iou_scores[obj_idx].argmax()
                # Upscale low-res mask logits to original image size
                mask_logits = pred_masks[obj_idx, best].unsqueeze(0).unsqueeze(0).float()  # [1,1,mh,mw]
                mask_upscaled = F.interpolate(mask_logits, size=(h, w), mode="bilinear", align_corners=False)
                all_masks.append(mask_upscaled.squeeze().cpu().numpy() > 0.0)
                all_scores.append(iou_scores[obj_idx, best].item())
        if opts.control_move_processor:
            self.model.to("cpu")
        # Filter by IoU score
        if len(all_masks) > 0:
            scores = np.array(all_scores)
            good = scores > 0.7
            masks_np = [m for m, g in zip(all_masks, good) if g]
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
