# VQA Image Annotation Utilities
# Drawing functions for bounding boxes, points, and other visual annotations

from PIL import Image, ImageDraw, ImageFont
from modules import shared


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
