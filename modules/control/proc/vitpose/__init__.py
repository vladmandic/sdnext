import numpy as np
import torch
from PIL import Image
from modules import devices
from modules.shared import opts
from modules.control.proc.rtmlib_pose import draw_skeleton


class ViTPoseDetector:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="usyd-community/vitpose-plus-base", cache_dir=None, local_files_only=False):
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        processor = AutoProcessor.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only)
        model = VitPoseForPoseEstimation.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir, local_files_only=local_files_only, use_safetensors=True).to(devices.device).eval()
        return cls(model, processor)

    def __call__(self, image, min_confidence=0.3, output_type="pil", **kwargs):
        self.model.to(devices.device)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        w, h = image.size
        boxes = [[[0, 0, w, h]]]
        inputs = self.processor(images=image, boxes=boxes, return_tensors="pt")
        inputs = {k: v.to(devices.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        # vitpose-plus models use MoE backbone; dataset_index=0 selects the COCO expert
        if hasattr(self.model.config, 'backbone_config') and getattr(self.model.config.backbone_config, 'num_experts', 1) > 1:
            inputs["dataset_index"] = torch.tensor([0], device=devices.device)
        with devices.inference_context():
            outputs = self.model(**inputs)
        if opts.control_move_processor:
            self.model.to("cpu")
        pose_results = self.processor.post_process_pose_estimation(outputs, boxes=boxes)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if pose_results and len(pose_results) > 0:
            for result in pose_results:
                if isinstance(result, list):
                    for person in result:
                        keypoints = person.get("keypoints", person.get("coordinates", None))
                        scores = person.get("scores", person.get("confidence", None))
                        if keypoints is not None and scores is not None:
                            if isinstance(keypoints, torch.Tensor):
                                keypoints = keypoints.cpu().numpy()
                            if isinstance(scores, torch.Tensor):
                                scores = scores.cpu().numpy()
                            canvas = draw_skeleton(canvas, keypoints, scores, min_confidence)
                elif isinstance(result, dict):
                    keypoints = result.get("keypoints", result.get("coordinates", None))
                    scores = result.get("scores", result.get("confidence", None))
                    if keypoints is not None and scores is not None:
                        if isinstance(keypoints, torch.Tensor):
                            keypoints = keypoints.cpu().numpy()
                        if isinstance(scores, torch.Tensor):
                            scores = scores.cpu().numpy()
                        canvas = draw_skeleton(canvas, keypoints, scores, min_confidence)
        if output_type == "pil":
            canvas = Image.fromarray(canvas)
        return canvas
