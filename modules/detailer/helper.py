import os
import re
from PIL import Image
from modules.logger import log


class_tag_re = re.compile(r'^\[class\s*=\s*([^\]]+)\]\s*(.*)$', re.IGNORECASE)


def list_models(self):
    from modules.detailer import detailer_models
    from modules import shared
    self.list.clear()
    files = []
    downloaded = 0
    for m in detailer_models:
        name = os.path.splitext(os.path.basename(m))[0]
        self.list[name] = m
        files.append(name)
    if os.path.exists(shared.opts.yolo_dir):
        for f in os.listdir(shared.opts.yolo_dir):
            if f.endswith('.pt'):
                downloaded += 1
                name = os.path.splitext(os.path.basename(f))[0]
                if name not in files:
                    self.list[name] = os.path.join(shared.opts.yolo_dir, f)
    log.info(f'Available Detailer: path="{shared.opts.yolo_dir}" items={len(list(self.list))} downloaded={downloaded}')
    return list(self.list)


def detailer_opt(p, attr, opts_attr=None):
    """Read detailer param from processing object if set, otherwise fall back to shared.opts."""
    from modules import shared
    if p is not None:
        val = getattr(p, attr, None)
        if val is not None:
            return val
    return getattr(shared.opts, opts_attr or attr, None)


def parse_prompt_lines(text: str):
    """Split a detailer prompt into class-tagged templates and positional fallback lines.

    A line starting with '[CLASS=name]' or '[CLASS=name1,name2]' assigns its text to every
    detection whose label matches one of the given class names (case-insensitive). All
    other non-empty lines are kept, in order, as the legacy positional fallback used for
    detections that don't match any class tag.
    """
    class_map: dict[str, str] = {}
    fallback: list[str] = []
    for line in (text or '').split('\n'):
        line = line.strip()
        m = class_tag_re.match(line)
        if m:
            names = [n.strip().lower() for n in m.group(1).split(',') if n.strip()]
            for name in names:
                class_map[name] = m.group(2).strip()
        else:
            fallback.append(line)
    return class_map, fallback


def assign_prompts(text: str, items: list) -> list[str]:
    """Resolve a detailer prompt/negative-prompt string into one entry per detection.

    Detections whose YOLO label matches a '[CLASS=name]' tag get that tag's text.
    Remaining detections fall back to the untagged lines, applied positionally in
    detection order and cycling if there are more detections than fallback lines
    (matching prior behavior when no class tags are used).
    """
    class_map, fallback = parse_prompt_lines(text)
    if len(fallback) == 0:
        fallback = ['']
    resolved = []
    fallback_idx = 0
    for item in items:
        label = (getattr(item, 'label', None) or '').strip().lower()
        if label in class_map:
            resolved.append(class_map[label])
        else:
            resolved.append(fallback[fallback_idx % len(fallback)])
            fallback_idx += 1
    return resolved


class DetailerResult:
    def __init__(self, cls: int, label: str, score: float, box: list[int], mask: Image.Image = None, item: Image.Image = None, width = 0, height = 0, args = None):
        if args is None:
            args = {}
        self.cls = cls
        self.label = label
        self.score = score
        self.box = box
        self.mask = mask
        self.item = item
        self.width = width
        self.height = height
        self.args = args

    def __str__(self):
        return f'DetailerResult(cls={self.cls} label={self.label} score={self.score:.2f} box={self.box} size={self.width}x{self.height} args={self.args})'
