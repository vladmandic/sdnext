import warnings
import cv2
import numpy as np
from PIL import Image
from modules.control.util import HWC3, resize_image

class CannyDetector:
    def __call__(self, input_image=None, low_threshold=100, high_threshold=200, detect_resolution=512, image_resolution=512, output_type=None, **kwargs):
        # Track Canny edge detection operation
        from modules import pipeline_viz
        pipeline_viz.safe_track_operation('canny_detect', {
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'detect_resolution': detect_resolution,
            'image_resolution': image_resolution,
            'input_type': type(input_image).__name__ if input_image is not None else None
        })
        
        if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
        if input_image is None:
            pipeline_viz.safe_track_operation_fail('canny_detect', 'input_image must be defined')
            raise ValueError("input_image must be defined.")

        try:
            if not isinstance(input_image, np.ndarray):
                input_image = np.array(input_image, dtype=np.uint8)
                output_type = output_type or "pil"
            else:
                output_type = output_type or "np"
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, detect_resolution)
            detected_map = cv2.Canny(input_image, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, _C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)
                
            # Complete Canny edge detection tracking
            pipeline_viz.safe_track_operation_complete('canny_detect', {
                'success': True,
                'output_size': f"{W}x{H}",
                'output_type': output_type,
                'edge_pixels_detected': int(np.sum(detected_map > 0) if isinstance(detected_map, np.ndarray) else 0)
            })
            
            return detected_map
        except Exception as e:
            pipeline_viz.safe_track_operation_fail('canny_detect', str(e))
            raise
