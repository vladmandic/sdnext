import os
import numpy as np
from PIL import Image
from modules.logger import log


# ---------------------------------------------------------------------------
# COCO-WholeBody 133-keypoint layout
#   0-16:    body (17 COCO keypoints)
#   17-22:   feet (6 points)
#   23-90:   face (68 landmarks)
#   91-111:  left hand (21 points)
#   112-132: right hand (21 points)
# ---------------------------------------------------------------------------

BODY_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6],
]

BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85), (255, 0, 0),
]

FOOT_SKELETON = [
    [0, 1], [1, 2],   # left ankle -> big toe -> small toe
    [3, 4], [4, 5],   # right ankle -> big toe -> small toe
]
FOOT_OFFSET = 17

HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]
LEFT_HAND_OFFSET = 91
RIGHT_HAND_OFFSET = 112
FACE_OFFSET = 23
FACE_COUNT = 68


def _hsv_to_rgb(h, s, v):
    """Convert HSV [0-1] to BGR tuple for cv2."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_body(canvas, keypoints, scores, min_conf):
    import cv2
    n_kps = min(len(keypoints), 17)
    for i in range(n_kps):
        if scores[i] >= min_conf:
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            color = BODY_COLORS[i % len(BODY_COLORS)]
            cv2.circle(canvas, (x, y), 4, color, -1)
    for idx, (start, end) in enumerate(BODY_SKELETON):
        if start < n_kps and end < n_kps:
            if scores[start] >= min_conf and scores[end] >= min_conf:
                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                color = BODY_COLORS[idx % len(BODY_COLORS)]
                cv2.line(canvas, pt1, pt2, color, 2)


def draw_feet(canvas, keypoints, scores, min_conf):
    import cv2
    for start_local, end_local in FOOT_SKELETON:
        start = start_local + FOOT_OFFSET
        end = end_local + FOOT_OFFSET
        if start < len(keypoints) and end < len(keypoints):
            if scores[start] >= min_conf and scores[end] >= min_conf:
                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                cv2.line(canvas, pt1, pt2, (0, 255, 170), 2)
    for i in range(FOOT_OFFSET, min(FOOT_OFFSET + 6, len(keypoints))):
        if scores[i] >= min_conf:
            cv2.circle(canvas, (int(keypoints[i][0]), int(keypoints[i][1])), 3, (0, 255, 170), -1)


def draw_hand(canvas, keypoints, scores, offset, min_conf):
    import cv2
    n_edges = len(HAND_EDGES)
    for ie, (start_local, end_local) in enumerate(HAND_EDGES):
        start = start_local + offset
        end = end_local + offset
        if start < len(keypoints) and end < len(keypoints):
            if scores[start] >= min_conf and scores[end] >= min_conf:
                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                color = _hsv_to_rgb(ie / n_edges, 1.0, 1.0)
                cv2.line(canvas, pt1, pt2, color, 2)
    for i in range(offset, min(offset + 21, len(keypoints))):
        if scores[i] >= min_conf:
            cv2.circle(canvas, (int(keypoints[i][0]), int(keypoints[i][1])), 3, (0, 0, 255), -1)


def draw_hands(canvas, keypoints, scores, min_conf):
    draw_hand(canvas, keypoints, scores, LEFT_HAND_OFFSET, min_conf)
    draw_hand(canvas, keypoints, scores, RIGHT_HAND_OFFSET, min_conf)


def draw_face(canvas, keypoints, scores, min_conf):
    import cv2
    for i in range(FACE_OFFSET, min(FACE_OFFSET + FACE_COUNT, len(keypoints))):
        if scores[i] >= min_conf:
            cv2.circle(canvas, (int(keypoints[i][0]), int(keypoints[i][1])), 2, (255, 255, 255), -1)


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

    def __call__(self, image, min_confidence=0.3, draw_body_pose=True, draw_hand_pose=True, draw_face_pose=True, output_type="pil", **kwargs):
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w = image.shape[:2]
        keypoints, scores = self.pose_model(image)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if keypoints is not None and len(keypoints) > 0:
            for person_idx in range(len(keypoints)):
                kps = keypoints[person_idx]
                sc = scores[person_idx] if person_idx < len(scores) else np.ones(len(kps))
                is_wholebody = len(kps) >= 133
                if draw_body_pose:
                    draw_body(canvas, kps, sc, min_confidence)
                    if is_wholebody:
                        draw_feet(canvas, kps, sc, min_confidence)
                if draw_hand_pose and is_wholebody:
                    draw_hands(canvas, kps, sc, min_confidence)
                if draw_face_pose and is_wholebody:
                    draw_face(canvas, kps, sc, min_confidence)
        if output_type == "pil":
            canvas = Image.fromarray(canvas)
        return canvas
