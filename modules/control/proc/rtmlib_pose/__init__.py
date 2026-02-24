import os
import numpy as np
from PIL import Image
from modules.logger import log


OPENPOSE_BODY_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6],
]

OPENPOSE_BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0),
]


def draw_skeleton(canvas, keypoints, scores, min_confidence=0.3):
    import cv2
    for i, (x, y) in enumerate(keypoints):
        if i < len(scores) and scores[i] >= min_confidence:
            color = OPENPOSE_BODY_COLORS[i % len(OPENPOSE_BODY_COLORS)]
            cv2.circle(canvas, (int(x), int(y)), 4, color, -1)
    for idx, (start, end) in enumerate(OPENPOSE_BODY_SKELETON):
        if start < len(keypoints) and end < len(keypoints):
            if start < len(scores) and end < len(scores):
                if scores[start] >= min_confidence and scores[end] >= min_confidence:
                    pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                    pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                    color = OPENPOSE_BODY_COLORS[idx % len(OPENPOSE_BODY_COLORS)]
                    cv2.line(canvas, pt1, pt2, color, 2)
    return canvas


class RtmlibPoseDetector:
    def __init__(self, pose_model, mode):
        self.pose_model = pose_model
        self.mode = mode

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="DWPose", cache_dir=None, local_files_only=False, **kwargs):
        from installer import install
        install('rtmlib', quiet=True)
        # rtmlib reads TORCH_HOME to locate its cache at <TORCH_HOME>/hub/checkpoints
        old_torch_home = os.environ.get('TORCH_HOME')
        if cache_dir:
            rtmlib_cache = os.path.join(cache_dir, 'rtmlib')
            os.makedirs(rtmlib_cache, exist_ok=True)
            os.environ['TORCH_HOME'] = rtmlib_cache
        try:
            import rtmlib
            mode = pretrained_model_or_path
            model_map = {
                'DWPose': ('RTMPose', {'to_openpose': False}),
                'RTMW-l': ('RTMW', {'to_openpose': False}),
                'RTMO-l': ('RTMO', {'to_openpose': False}),
            }
            if mode not in model_map:
                log.warning(f'RtmlibPose: unknown mode "{mode}", falling back to DWPose')
                mode = 'DWPose'
            model_name, model_kwargs = model_map[mode]
            if model_name == 'RTMPose':
                body = rtmlib.Body(mode='lightweight', backend='onnxruntime', device='cpu', **model_kwargs)
            elif model_name == 'RTMW':
                body = rtmlib.Wholebody(mode='lightweight', backend='onnxruntime', device='cpu', to_openpose=False)
            elif model_name == 'RTMO':
                body = rtmlib.Body(mode='balanced', backend='onnxruntime', device='cpu', **model_kwargs)
            else:
                body = rtmlib.Body(mode='lightweight', backend='onnxruntime', device='cpu')
        finally:
            if old_torch_home is not None:
                os.environ['TORCH_HOME'] = old_torch_home
            elif 'TORCH_HOME' in os.environ:
                del os.environ['TORCH_HOME']
        return cls(body, mode)

    def __call__(self, image, min_confidence=0.3, output_type="pil", **kwargs):
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        keypoints, scores = self.pose_model(image)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if keypoints is not None and len(keypoints) > 0:
            for person_idx in range(len(keypoints)):
                kps = keypoints[person_idx]
                sc = scores[person_idx] if person_idx < len(scores) else np.ones(len(kps))
                canvas = draw_skeleton(canvas, kps, sc, min_confidence)
        if output_type == "pil":
            canvas = Image.fromarray(canvas)
        return canvas
