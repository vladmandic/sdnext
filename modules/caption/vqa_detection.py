# VQA Detection Utilities
# Parsing, formatting, and drawing functions for detection results (points, bboxes, gaze)

from PIL import Image, ImageDraw, ImageFont
from modules import shared


def parse_points(result) -> list:
    """Parse and validate point coordinates from model result.

    Args:
        result: Model output, typically dict with 'points' key or list of coordinates

    Returns:
        List of (x, y) tuples with coordinates clamped to 0-1 range.
    """
    points = []

    # Dict format: {'points': [{'x': 0.5, 'y': 0.5}, ...]}
    if isinstance(result, dict) and 'points' in result:
        points_list = result['points']
        if points_list and len(points_list) > 0:
            for point_data in points_list:
                if isinstance(point_data, dict) and 'x' in point_data and 'y' in point_data:
                    x = max(0.0, min(1.0, float(point_data['x'])))
                    y = max(0.0, min(1.0, float(point_data['y'])))
                    points.append((x, y))

    # Fallback for simple [x, y] format
    elif isinstance(result, (list, tuple)) and len(result) == 2:
        try:
            x = max(0.0, min(1.0, float(result[0])))
            y = max(0.0, min(1.0, float(result[1])))
            points.append((x, y))
        except (ValueError, TypeError):
            pass

    return points


def parse_detections(result, label: str, max_objects: int = None) -> list:
    """Parse and validate detection bboxes from model result.

    Args:
        result: Model output, typically dict with 'objects' key
        label: Label to assign to detected objects
        max_objects: Maximum number of objects to return (None for all)

    Returns:
        List of {'bbox': [x1,y1,x2,y2], 'label': str, 'confidence': float}
        with coordinates clamped to 0-1 range.
    """
    detections = []

    if isinstance(result, dict) and 'objects' in result:
        objects = result['objects']
        if max_objects is not None:
            objects = objects[:max_objects]

        for obj in objects:
            if all(k in obj for k in ['x_min', 'y_min', 'x_max', 'y_max']):
                bbox = [
                    max(0.0, min(1.0, float(obj['x_min']))),
                    max(0.0, min(1.0, float(obj['y_min']))),
                    max(0.0, min(1.0, float(obj['x_max']))),
                    max(0.0, min(1.0, float(obj['y_max'])))
                ]
                detections.append({
                    'bbox': bbox,
                    'label': label,
                    'confidence': obj.get('confidence', 1.0)
                })

    return detections


def parse_florence_detections(response, image_size: tuple = None) -> list:
    """Parse Florence-style detection response into standard detection format.

    Florence returns detection data in two possible formats:

    1. Dict format (from post_process_generation with proper task):
        {'<OD>': {'bboxes': [[x1,y1,x2,y2], ...], 'labels': ['label1', ...]}}

    2. String format (raw output):
        'label1<loc_x1><loc_y1><loc_x2><loc_y2>label2<loc_x1>...'

    Coordinates are on a 1000-point scale (0-999).

    Args:
        response: Florence model response (dict or string)
        image_size: Optional (width, height) - not used for normalization since
                   Florence coordinates are already normalized to 1000-point scale

    Returns:
        List of {'bbox': [x1,y1,x2,y2], 'label': str, 'confidence': float}
        with coordinates normalized to 0-1 range.
    """
    import re
    detections = []

    def parse_loc_string(text: str) -> list:
        """Parse string format: label<loc_X><loc_Y><loc_X2><loc_Y2>..."""
        results = []
        # Pattern matches: label followed by 4 <loc_N> tags
        pattern = r'([^<]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
        matches = re.findall(pattern, text)

        for match in matches:
            label, x1, y1, x2, y2 = match
            # Florence uses 0-999 scale, normalize to 0-1
            results.append({
                'bbox': [
                    max(0.0, min(1.0, float(x1) / 1000)),
                    max(0.0, min(1.0, float(y1) / 1000)),
                    max(0.0, min(1.0, float(x2) / 1000)),
                    max(0.0, min(1.0, float(y2) / 1000))
                ],
                'label': label.strip(),
                'confidence': 1.0
            })
        return results

    # Handle string format directly
    if isinstance(response, str):
        return parse_loc_string(response)

    # Handle dict format
    if not isinstance(response, dict):
        return detections

    # Check for 'task' key containing loc string (common Florence output format)
    if 'task' in response and isinstance(response['task'], str):
        detections = parse_loc_string(response['task'])
        if detections:
            return detections

    # Florence detection task keys
    detection_keys = ['<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<OPEN_VOCABULARY_DETECTION>']

    for key in detection_keys:
        if key in response:
            data = response[key]
            if isinstance(data, dict) and 'bboxes' in data:
                bboxes = data.get('bboxes', [])
                labels = data.get('labels', [])

                # Florence uses 1000x1000 coordinate space by default
                scale_w = image_size[0] if image_size else 1000
                scale_h = image_size[1] if image_size else 1000

                for i, bbox in enumerate(bboxes):
                    if len(bbox) >= 4:
                        # Normalize to 0-1 range
                        x1 = max(0.0, min(1.0, float(bbox[0]) / scale_w))
                        y1 = max(0.0, min(1.0, float(bbox[1]) / scale_h))
                        x2 = max(0.0, min(1.0, float(bbox[2]) / scale_w))
                        y2 = max(0.0, min(1.0, float(bbox[3]) / scale_h))

                        label = labels[i] if i < len(labels) else 'object'
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': label,
                            'confidence': 1.0
                        })
            break  # Only process first matching key

    return detections


def format_florence_response(response: dict) -> str:
    """Format Florence response dict as human-readable text.

    Handles various Florence task outputs and formats them appropriately.

    Args:
        response: Florence processor post_process_generation output

    Returns:
        Formatted string representation
    """
    if not isinstance(response, dict):
        return str(response)

    # Caption tasks - return text directly
    caption_keys = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']
    for key in caption_keys:
        if key in response:
            return str(response[key])

    # Detection tasks - format with label<loc> syntax for raw output
    detection_keys = ['<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>']
    for key in detection_keys:
        if key in response:
            data = response[key]
            if isinstance(data, dict) and 'bboxes' in data:
                bboxes = data.get('bboxes', [])
                labels = data.get('labels', [])
                parts = []
                for i, bbox in enumerate(bboxes):
                    label = labels[i] if i < len(labels) else 'object'
                    if len(bbox) >= 4:
                        parts.append(f"{label}<loc_{int(bbox[0])}><loc_{int(bbox[1])}><loc_{int(bbox[2])}><loc_{int(bbox[3])}>")
                return ''.join(parts) if parts else 'No objects detected'

    # OCR task
    if '<OCR>' in response:
        return str(response['<OCR>'])

    # OCR with regions
    if '<OCR_WITH_REGION>' in response:
        data = response['<OCR_WITH_REGION>']
        if isinstance(data, dict) and 'labels' in data:
            return ' '.join(data['labels'])

    # Tags
    if '<GENERATE_TAGS>' in response:
        return str(response['<GENERATE_TAGS>'])

    # Fallback - check for 'task' key or stringify
    if 'task' in response:
        return str(response['task'])

    return str(response)


def format_points_text(points: list) -> str:
    """Format point coordinates as human-readable text.

    Args:
        points: List of (x, y) tuples with normalized coordinates

    Returns:
        Formatted text string describing the points.
    """
    if not points:
        return "Object not found"

    if len(points) == 1:
        return f"Found at: ({points[0][0]:.3f}, {points[0][1]:.3f})"

    lines = [f"Found {len(points)} instances:"]
    for i, (x, y) in enumerate(points, 1):
        lines.append(f"  {i}. ({x:.3f}, {y:.3f})")
    return '\n'.join(lines)


def format_detections_text(detections: list, include_confidence: bool = True) -> str:
    """Format detections with bboxes as human-readable text.

    Args:
        detections: List of detection dicts with 'bbox', 'label', 'confidence'
        include_confidence: Whether to include confidence scores in output

    Returns:
        Formatted text string describing the detections.
    """
    if not detections:
        return "No objects detected"

    lines = []
    for det in detections:
        bbox = det['bbox']
        label = det.get('label', 'object')
        confidence = det.get('confidence', 1.0)

        if include_confidence and confidence < 1.0:
            lines.append(f"{label}: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}] (confidence: {confidence:.2f})")
        else:
            lines.append(f"{label}: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")

    return '\n'.join(lines)


def calculate_eye_position(face_bbox: dict) -> tuple:
    """Calculate approximate eye position from face bounding box.

    Args:
        face_bbox: Dict with 'x_min', 'y_min', 'x_max', 'y_max' keys

    Returns:
        (eye_x, eye_y) tuple with normalized coordinates.
    """
    eye_x = (face_bbox['x_min'] + face_bbox['x_max']) / 2
    eye_y = face_bbox['y_min'] + (face_bbox['y_max'] - face_bbox['y_min']) * 0.3  # Approximate eye level
    return (eye_x, eye_y)


def draw_bounding_boxes(image: Image.Image, detections: list, points: list = None) -> Image.Image:
    """
    Draw bounding boxes and/or points on an image.

    Args:
        image: PIL Image to annotate
        detections: List of detection dicts with format:
            [{'label': str, 'bbox': [x1, y1, x2, y2], 'confidence': float}, ...]
            where coordinates are normalized 0-1
        points: Optional list of (x, y) tuples with normalized 0-1 coordinates

    Returns:
        Annotated PIL Image with boxes and labels drawn, or None if no annotations
    """
    if not detections and not points:
        return None

    # Create a copy to avoid modifying original
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width, height = image.size

    # Try to load a font, fall back to default if unavailable
    try:
        font_size = max(12, int(min(width, height) * 0.02))
        font_path = shared.opts.font or "javascript/notosans-nerdfont-regular.ttf"
        font = ImageFont.truetype(font_path, size=font_size)
    except Exception:
        font = ImageFont.load_default()

    # Draw bounding boxes
    if detections:
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#800080']
        for idx, det in enumerate(detections):
            bbox = det['bbox']
            label = det.get('label', 'object')
            confidence = det.get('confidence', 1.0)

            # Convert normalized coordinates to pixel coordinates
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            # Choose color
            color = colors[idx % len(colors)]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, int(min(width, height) * 0.003)))

            # Draw label with background
            label_text = f"{label} {confidence:.2f}" if confidence < 1.0 else label
            bbox_font = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox_font[2] - bbox_font[0]
            text_height = bbox_font[3] - bbox_font[1]
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)

    # Draw points
    if points:
        point_radius = max(3, int(min(width, height) * 0.01))
        for px, py in points:
            x = int(px * width)
            y = int(py * height)
            # Draw point as a circle
            draw.ellipse([x - point_radius, y - point_radius, x + point_radius, y + point_radius],
                        fill='#FF0000', outline='#FFFFFF', width=2)

    return annotated
