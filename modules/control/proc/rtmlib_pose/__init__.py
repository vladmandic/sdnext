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


def draw_skeleton(canvas, keypoints, scores, min_conf):
    """Draw a full pose skeleton for one person. Used by ViTPoseDetector."""
    is_wholebody = len(keypoints) >= 133
    draw_body(canvas, keypoints, scores, min_conf)
    if is_wholebody:
        draw_feet(canvas, keypoints, scores, min_conf)
        draw_hands(canvas, keypoints, scores, min_conf)
        draw_face(canvas, keypoints, scores, min_conf)
    return canvas


# Original DWPose-l 384x288 (distilled from RTMPose-x on COCO-WholeBody+UBody), ONNX export hosted by OpenMMLab.
# Same model as dw-ll_ucoco_384 and the original default of rtmlib.Wholebody.
DWPOSE_MODEL = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip'
DWPOSE_INPUT_SIZE = (288, 384)

# YOLOX person detectors trained on HumanArt: (onnx model, input size)
DETECTORS = {
    'tiny': ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip', (416, 416)),
    'm': ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip', (640, 640)),
    'x': ('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip', (640, 640)),
}

# OpenPose-134 layout returned by rtmlib with to_openpose=True for whole-body models:
#   0-17 body (with neck), 18-23 feet, 24-91 face, 92-112 left hand, 113-133 right hand
OPENPOSE_BODY = slice(0, 24)
OPENPOSE_FACE = slice(24, 92)
OPENPOSE_HANDS = slice(92, 134)


class RtmlibPoseDetector:
    def __init__(self, pose_model, mode, openpose=True):
        self.pose_model = pose_model
        self.mode = mode
        self.openpose = openpose

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="DWPose", cache_dir=None, local_files_only=False, detector='m', **kwargs):
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
            if mode not in ('DWPose', 'RTMW-l', 'RTMO-l'):
                log.warning(f'RtmlibPose: unknown mode "{mode}", falling back to DWPose')
                mode = 'DWPose'
            if mode == 'DWPose':
                if detector not in DETECTORS:
                    log.warning(f'RtmlibPose: unknown detector "{detector}", falling back to "m"')
                    detector = 'm'
                det, det_input_size = DETECTORS[detector]
                body = rtmlib.Wholebody(det=det, det_input_size=det_input_size, pose=DWPOSE_MODEL, pose_input_size=DWPOSE_INPUT_SIZE, backend='onnxruntime', device='cpu', to_openpose=True)
            elif mode == 'RTMW-l':
                # balanced loads rtmw-dw-x-l 256x192 (RTMW-l); lightweight would load rtmw-dw-l-m (RTMW-m)
                body = rtmlib.Wholebody(mode='balanced', backend='onnxruntime', device='cpu', to_openpose=True)
            else:
                # rtmlib.Body only switches to one-stage RTMO when the pose argument contains 'rtmo'; performance loads rtmo-l
                body = rtmlib.Body(pose='rtmo', mode='performance', backend='onnxruntime', device='cpu', to_openpose=True)
        finally:
            if old_torch_home is not None:
                os.environ['TORCH_HOME'] = old_torch_home
            elif 'TORCH_HOME' in os.environ:
                del os.environ['TORCH_HOME']
        return cls(body, mode)

    def __call__(self, image, min_confidence=0.3, draw_body_pose=True, draw_hand_pose=True, draw_face_pose=True, output_type="pil", **kwargs):
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        h, w = image.shape[:2]
        keypoints, scores = self.pose_model(image)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if keypoints is not None and len(keypoints) > 0:
            import rtmlib
            # rtmlib.draw_skeleton draws everything; hide disabled parts by zeroing their scores so they fall below kpt_thr
            scores = np.array(scores, copy=True)
            is_wholebody = scores.shape[-1] >= 134
            if not draw_body_pose:
                scores[..., OPENPOSE_BODY if is_wholebody else slice(None)] = 0
            if is_wholebody and not draw_face_pose:
                scores[..., OPENPOSE_FACE] = 0
            if is_wholebody and not draw_hand_pose:
                scores[..., OPENPOSE_HANDS] = 0
            canvas = rtmlib.draw_skeleton(canvas, keypoints, scores, openpose_skeleton=self.openpose, kpt_thr=min_confidence)
        if output_type == "pil":
            canvas = Image.fromarray(canvas)
        return canvas
