import os
from PIL import Image
from modules.logger import log


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
