from typing import TYPE_CHECKING
import os
import threading
import numpy as np
from PIL import Image, ImageDraw
from modules.detailer import DetailerResult, detailer_opt
from modules.logger import log
from modules import shared, devices


load_lock = threading.Lock()


def dependencies():
    from installer import install
    install('ultralytics==8.4.67', ignore=True, quiet=True)
    install('omegaconf')
    install('antlr4-python3-runtime')


def load(self, model_name: str | None = None):
    with load_lock:
        from modules import modelloader
        model = None
        if model_name is None:
            model_name = list(self.list)[0]
        if model_name in self.models:
            return model_name, self.models[model_name]
        else:
            model_url = self.list.get(model_name, None)
            if model_url is None:
                log.error(f'Load: type=Detailer name="{model_name}" error="model not found"')
                return None, None
            file_name = os.path.basename(model_url)
            model_file = None
            try:
                model_file = modelloader.load_file_from_url(url=model_url, model_dir=shared.opts.yolo_dir, file_name=file_name)
                if model_file is None:
                    log.error(f'Load: type=Detailer name="{model_name}" url="{model_url}" error="failed to fetch model"')
                elif model_file.endswith('.onnx'):
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session = ort.InferenceSession(model_file, sess_options=options, providers=devices.onnx)
                    self.models[model_name] = session
                    return model_name, session
                else:
                    dependencies()
                    import ultralytics
                    model = ultralytics.YOLO(model_file)
                    classes = list(model.names.values())
                    log.info(f'Load: type=Detailer name="{model_name}" model="{model_file}" ultralytics={ultralytics.__version__} classes={classes}')
                    self.models[model_name] = model
                    return model_name, model
            except Exception as e:
                log.error(f'Load: type=Detailer name="{model_name}" error="{e}"')
    return None, None


def predict(
        self,
        model,
        image: Image.Image,
        imgsz: int = 640,
        half: bool = True,
        device = devices.device,
        agnostic: bool = False,
        retina: bool = False,
        mask: bool = True,
        augment: bool | None = None,
        offload: bool | None = None,
        p = None,
    ) -> list[DetailerResult]:
    if augment is None:
        augment = detailer_opt(p, 'detailer_augment')
    if offload is None:
        offload = shared.opts.detailer_unload

    if model is None or (isinstance(model, str) and len(model) == 0):
        model = 'yolo11m'
    result = []
    if isinstance(model, str):
        cached = self.models.get(model, None)
        if cached is None:
            _, model = self.load(model)
        else:
            model = cached
    if model is None:
        return result
    args = {
        'conf': detailer_opt(p, 'detailer_conf'),
        'iou': detailer_opt(p, 'detailer_iou'),
        # 'max_det': detailer_opt(p, 'detailer_max'),
    }
    try:
        if TYPE_CHECKING:
            from ultralytics import YOLO # pylint: disable=import-outside-toplevel, unused-import
        model: YOLO = model.to(device)
        predictions = model.predict(
            source=[image],
            stream=False,
            verbose=False,
            imgsz=imgsz,
            half=half,
            device=device,
            augment=augment,
            agnostic_nms=agnostic,
            retina_masks=retina,
            **args
        )
        if offload:
            model.to('cpu')
    except Exception as e:
        log.error(f'Detailer predict: {e}')
        return result

    classes = detailer_opt(p, 'detailer_classes') or ''
    desired = classes.split(',')
    desired = [d.lower().strip() for d in desired]
    desired = [d for d in desired if len(d) > 0]

    for prediction in predictions:
        boxes = prediction.boxes.xyxy.detach().int().cpu().numpy() if prediction.boxes is not None else []
        scores = prediction.boxes.conf.detach().float().cpu().numpy() if prediction.boxes is not None else []
        classes = prediction.boxes.cls.detach().float().cpu().numpy() if prediction.boxes is not None else []
        masks = prediction.masks.data.cpu().float().numpy() if prediction.masks is not None else []
        if len(masks) < len(classes):
            masks = len(classes) * [None]
        for score, box, cls, seg in zip(scores, boxes, classes, masks, strict=False):
            if seg is not None:
                try:
                    seg = (255 * seg).astype(np.uint8)
                    seg = Image.fromarray(seg).resize(image.size).convert('L')
                except Exception:
                    seg = None
            cls = int(cls)
            label = prediction.names[cls] if cls < len(prediction.names) else f'cls{cls}'
            if len(desired) > 0 and label.lower() not in desired:
                continue
            box = box.tolist()
            w, h = box[2] - box[0], box[3] - box[1]
            x_size, y_size = w/image.width, h/image.height
            opt_min = detailer_opt(p, 'detailer_min_size') or 0
            opt_max = detailer_opt(p, 'detailer_max_size') or 1
            min_size = opt_min if 0 <= opt_min <= 1 else 0
            max_size = opt_max if 0 < opt_max <= 1 else 1
            if x_size >= min_size and y_size >=min_size and x_size <= max_size and y_size <= max_size:
                if mask:
                    if detailer_opt(p, 'detailer_segmentation') and seg is not None:
                        masked = seg
                    else:
                        masked = Image.new('L', image.size, 0)
                        draw = ImageDraw.Draw(masked)
                        draw.rectangle(box, fill="white", outline=None, width=0)
                    cropped = image.crop(box)
                    res = DetailerResult(
                        cls=cls,
                        label=label,
                        score=round(score, 2),
                        box=box,
                        mask=masked,
                        item=cropped,
                        width=w,
                        height=h,
                        args=args,
                    )
                    result.append(res)
            if len(result) >= (detailer_opt(p, 'detailer_max') or 2):
                break
    return result
