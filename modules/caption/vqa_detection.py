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
