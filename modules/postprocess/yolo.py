from typing import TYPE_CHECKING
import os
import re
import threading
from copy import copy
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from modules import shared, processing, devices, processing_class, ui_common, ui_components, ui_symbols
from modules.detailer import Detailer


predefined = [ # <https://huggingface.co/vladmandic/yolo-detailers/tree/main>
    'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/face-yolo8n.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/face-yolo8m.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/hand_yolov8n.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/person_yolov8n-seg.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/eyes-v1.pt',
    'https://huggingface.co/vladmandic/yolo-detailers/resolve/main/eyes-full-v1.pt',
    'https://huggingface.co/netrunner-exe/Face-Upscalers-onnx/resolve/main/codeformer.fp16.onnx',
    'https://huggingface.co/netrunner-exe/Face-Upscalers-onnx/resolve/main/restoreformer.fp16.onnx',
    'https://huggingface.co/netrunner-exe/Face-Upscalers-onnx/resolve/main/GFPGANv1.4.fp16.onnx',
    'https://huggingface.co/netrunner-exe/Face-Upscalers-onnx/resolve/main/GPEN-BFR-512.fp16.onnx',
]
load_lock = threading.Lock()


class YoloResult:
    def __init__(self, cls: int, label: str, score: float, box: list[int], mask: Image.Image = None, item: Image.Image = None, width = 0, height = 0, args = {}):
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
        return f'cls={self.cls} label={self.label} score={self.score} box={self.box} mask={self.mask} item={self.item} size={self.width}x{self.height} args={self.args}'


class YoloRestorer(Detailer):
    def __init__(self):
        super().__init__()
        self.models = {} # cache loaded models
        self.list = {}
        self.ui_mode = True
        self.enumerate()

    def name(self):
        return "Detailer"

    def enumerate(self):
        self.list.clear()
        files = []
        downloaded = 0
        for m in predefined:
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
        shared.log.info(f'Available Detailer: path="{shared.opts.yolo_dir}" items={len(list(self.list))} downloaded={downloaded}')
        return list(self.list)

    def dependencies(self):
        import installer
        installer.install('ultralytics==8.3.40', ignore=True, quiet=True)

    def predict(
            self,
            model,
            image: Image.Image,
            imgsz: int = 640,
            half: bool = True,
            device = devices.device,
            augment: bool = shared.opts.detailer_augment,
            agnostic: bool = False,
            retina: bool = False,
            mask: bool = True,
            offload: bool = shared.opts.detailer_unload,
        ) -> list[YoloResult]:

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
            'conf': shared.opts.detailer_conf,
            'iou': shared.opts.detailer_iou,
            # 'max_det': shared.opts.detailer_max,
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
            shared.log.error(f'Detailer predict: {e}')
            return result

        desired = shared.opts.detailer_classes.split(',')
        desired = [d.lower().strip() for d in desired]
        desired = [d for d in desired if len(d) > 0]

        for prediction in predictions:
            boxes = prediction.boxes.xyxy.detach().int().cpu().numpy() if prediction.boxes is not None else []
            scores = prediction.boxes.conf.detach().float().cpu().numpy() if prediction.boxes is not None else []
            classes = prediction.boxes.cls.detach().float().cpu().numpy() if prediction.boxes is not None else []
            for score, box, cls in zip(scores, boxes, classes):
                cls = int(cls)
                label = prediction.names[cls] if cls < len(prediction.names) else f'cls{cls}'
                if len(desired) > 0 and label.lower() not in desired:
                    continue
                box = box.tolist()
                mask_image = None
                w, h = box[2] - box[0], box[3] - box[1]
                x_size, y_size = w/image.width, h/image.height
                min_size = shared.opts.detailer_min_size if shared.opts.detailer_min_size >= 0 and shared.opts.detailer_min_size <= 1 else 0
                max_size = shared.opts.detailer_max_size if shared.opts.detailer_max_size >= 0 and shared.opts.detailer_max_size <= 1 else 1
                if x_size >= min_size and y_size >=min_size and x_size <= max_size and y_size <= max_size:
                    if mask:
                        mask_image = image.copy()
                        mask_image = Image.new('L', image.size, 0)
                        draw = ImageDraw.Draw(mask_image)
                        draw.rectangle(box, fill="white", outline=None, width=0)
                        cropped = image.crop(box)
                        res = YoloResult(cls=cls, label=label, score=round(score, 2), box=box, mask=mask_image, item=cropped, width=w, height=h, args=args)
                        result.append(res)
                if len(result) >= shared.opts.detailer_max:
                    break
        return result

    def load(self, model_name: str = None):
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
                    shared.log.error(f'Load: type=Detailer name="{model_name}" error="model not found"')
                    return None, None
                file_name = os.path.basename(model_url)
                model_file = None
                try:
                    model_file = modelloader.load_file_from_url(url=model_url, model_dir=shared.opts.yolo_dir, file_name=file_name)
                    if model_file is None:
                        shared.log.error(f'Load: type=Detailer name="{model_name}" url="{model_url}" error="failed to fetch model"')
                    elif model_file.endswith('.onnx'):
                        import onnxruntime as ort
                        options = ort.SessionOptions()
                        # options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                        session = ort.InferenceSession(model_file, sess_options=options, providers=devices.onnx)
                        self.models[model_name] = session
                        return model_name, session
                    else:
                        self.dependencies()
                        import ultralytics
                        model = ultralytics.YOLO(model_file)
                        classes = list(model.names.values())
                        shared.log.info(f'Load: type=Detailer name="{model_name}" model="{model_file}" ultralytics={ultralytics.__version__} classes={classes}')
                        self.models[model_name] = model
                        return model_name, model
                except Exception as e:
                    shared.log.error(f'Load: type=Detailer name="{model_name}" error="{e}"')
        return None, None

    def merge(self, items: list[YoloResult]) -> list[YoloResult]:
        if items is None or len(items) == 0:
            return None
        box=[min(item.box[0] for item in items), min(item.box[1] for item in items), max(item.box[2] for item in items), max(item.box[3] for item in items)]
        mask = Image.new('L', items[0].mask.size, 0)
        for item in items:
            mask = Image.fromarray(np.maximum(np.array(mask), np.array(item.mask)))
        merged = YoloResult(
            cls=items[0].cls,
            label=items[0].label,
            score=sum(item.score for item in items) / len(items),
            box=box,
            mask=mask,
            item=None,
            width=box[2] - box[0],
            height=box[3] - box[1],
        )
        return [merged]

    def restore(self, np_image, p: processing.StableDiffusionProcessing = None):
        if hasattr(p, 'recursion'):
            return np_image
        if not hasattr(p, 'detailer_active'):
            p.detailer_active = 0
        if np_image is None or p.detailer_active >= p.batch_size * p.n_iter:
            return np_image
        models = []
        if len(shared.opts.detailer_args) > 0:
            models = [m.strip() for m in re.split(r'\n|,|;', shared.opts.detailer_args)]
            models = [m for m in models if len(m) > 0]
        if len(models) == 0:
            models = shared.opts.detailer_models
        if len(models) == 0:
            shared.log.warning('Detailer: model=None')
            return np_image
        shared.log.debug(f'Detailer: models={models}')

        # create backups
        orig_apply_overlay = shared.opts.mask_apply_overlay
        orig_p = p.__dict__.copy()
        orig_cls = p.__class__
        models_used = []

        for i, model_val in enumerate(models):
            if ':' in model_val:
                model_name, model_args = model_val.split(':', 1)
            else:
                model_name, model_args = model_val, ''
            model_args = [m.strip() for m in model_args.split(':')]
            model_args = {k.strip(): v.strip() for k, v in (arg.split('=') for arg in model_args if '=' in arg)}

            name, model = self.load(model_name)
            if model is None:
                shared.log.warning(f'Detailer: model="{name}" not loaded')
                continue

            if name.endswith('.fp16'):
                from modules.postprocess import restorer
                np_image = restorer.restore(np_image, name, model, p.detailer_strength)
                continue

            image = Image.fromarray(np_image)
            items = self.predict(model, image)

            if len(items) == 0:
                shared.log.info(f'Detailer: model="{name}" no items detected')
                continue

            if shared.opts.detailer_merge and len(items) > 1:
                shared.log.debug(f'Detailer: model="{name}" items={len(items)} merge')
                items = self.merge(items)

            shared.opts.data['mask_apply_overlay'] = True
            resolution = 512 if shared.sd_model_type in ['none', 'sd', 'lcm', 'unknown'] else 1024
            orig_prompt: str = orig_p.get('all_prompts', [''])[0]
            orig_negative: str = orig_p.get('all_negative_prompts', [''])[0]
            prompt: str = orig_p.get('detailer_prompt', '')
            negative: str = orig_p.get('detailer_negative', '')
            if prompt is None or len(prompt) == 0:
                prompt = orig_prompt
            else:
                prompt = prompt.replace('[PROMPT]', orig_prompt)
                prompt = prompt.replace('[prompt]', orig_prompt)
            if len(negative) == 0:
                negative = orig_negative
            else:
                negative = negative.replace('[PROMPT]', orig_negative)
                negative = negative.replace('[prompt]', orig_negative)
            prompt_lines = prompt.split('\n')
            negative_lines = negative.split('\n')
            prompt = prompt_lines[i % len(prompt_lines)]
            negative = negative_lines[i % len(negative_lines)]

            args = {
                'detailer': True,
                'batch_size': 1,
                'n_iter': 1,
                'prompt': prompt,
                'negative_prompt': negative,
                'denoising_strength': p.detailer_strength,
                'sampler_name': orig_p.get('hr_sampler_name', 'default'),
                'steps': p.detailer_steps,
                'styles': [],
                'inpaint_full_res': True,
                'inpainting_mask_invert': 0,
                'mask_blur': shared.opts.detailer_blur,
                'inpaint_full_res_padding': shared.opts.detailer_padding,
                'width': resolution,
                'height': resolution,
                'vae_type': orig_p.get('vae_type', 'Full'),
            }
            args.update(model_args)
            if args['denoising_strength'] == 0:
                shared.log.debug(f'Detailer: model="{name}" strength=0 skip')
                return np_image
            control_pipeline = None
            orig_class = shared.sd_model.__class__
            if getattr(p, 'is_control', False):
                from modules.control import run
                control_pipeline = shared.sd_model
                run.restore_pipeline()

            p = processing_class.switch_class(p, processing.StableDiffusionProcessingImg2Img, args)
            if hasattr(shared.sd_model, 'restore_pipeline'):
                shared.sd_model.restore_pipeline()
            p.detailer_active += 1 # set flag to avoid recursion

            if p.steps < 1:
                p.steps = orig_p.get('steps', 0)

            report = [{'label': i.label, 'score': i.score, 'size': f'{i.width}x{i.height}' } for i in items]
            shared.log.info(f'Detailer: model="{name}" items={report} args={args}')
            models_used.append(name)

            mask_all = []
            p.state = ''
            prev_state = shared.state.job
            pc = copy(p)

            orig_sigma_adjust: float = shared.opts.schedulers_sigma_adjust
            orig_sigma_end: float = shared.opts.schedulers_sigma_adjust_max
            shared.opts.schedulers_sigma_adjust = shared.opts.detailer_sigma_adjust
            shared.opts.schedulers_sigma_adjust_max = shared.opts.detailer_sigma_adjust_max

            for item in items:
                if item.mask is None:
                    continue
                pc.init_images = [image]
                pc.image_mask = [item.mask]
                pc.overlay_images = []
                pc.recursion = True
                shared.state.job = 'Detailer'
                pp = processing.process_images_inner(pc)
                del pc.recursion
                if pp is not None and pp.images is not None and len(pp.images) > 0:
                    image = pp.images[0] # update image to be reused for next item
                    if len(pp.images) > 1:
                        mask_all.append(pp.images[1])

            shared.opts.schedulers_sigma_adjust = orig_sigma_adjust
            shared.opts.schedulers_sigma_adjust_max = orig_sigma_end

            # restore pipeline
            if control_pipeline is not None:
                shared.sd_model = control_pipeline
            else:
                shared.sd_model.__class__ = orig_class
            p = processing_class.switch_class(p, orig_cls, orig_p)
            p.init_images = orig_p.get('init_images', None)
            p.image_mask = orig_p.get('image_mask', None)
            p.state = orig_p.get('state', None)
            p.ops = orig_p.get('ops', [])
            shared.state.job = prev_state
            shared.opts.data['mask_apply_overlay'] = orig_apply_overlay
            np_image = np.array(image)

            if len(mask_all) > 0 and shared.opts.include_mask:
                from modules.control.util import blend
                p.image_mask = blend([np.array(m) for m in mask_all])
                p.image_mask = Image.fromarray(p.image_mask)

        return np_image

    def change_mode(self, dropdown, text):
        self.ui_mode = not self.ui_mode
        if self.ui_mode:
            value = [val.split(':')[0].strip() for val in text.split(',')]
            return gr.update(visible=True, value=value), gr.update(visible=False), gr.update(visible=True)
        else:
            value = ', '.join(dropdown)
            return gr.update(visible=False), gr.update(visible=True, value=value), gr.update(visible=False)

    def ui(self, tab: str):
        def ui_settings_change(merge, detailers, text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end):
            shared.opts.detailer_merge = merge
            shared.opts.detailer_models = detailers
            shared.opts.detailer_args = text if not self.ui_mode else ''
            shared.opts.detailer_classes = classes
            shared.opts.detailer_padding = padding
            shared.opts.detailer_blur = blur
            shared.opts.detailer_conf = min_confidence
            shared.opts.detailer_max = max_detected
            shared.opts.detailer_min_size = min_size
            shared.opts.detailer_max_size = max_size
            shared.opts.detailer_iou = iou
            shared.opts.detailer_sigma_adjust = renoise_value
            shared.opts.detailer_sigma_adjust_max = renoise_end
            shared.opts.save(shared.config_filename, silent=True)
            shared.log.debug(f'Detailer settings: models={detailers} classes={classes} strength={strength} conf={min_confidence} max={max_detected} iou={iou} size={min_size}-{max_size} padding={padding} steps={steps}')

        with gr.Accordion(open=False, label="Detailer", elem_id=f"{tab}_detailer_accordion", elem_classes=["small-accordion"]):
            with gr.Row():
                enabled = gr.Checkbox(label="Enable detailer pass", elem_id=f"{tab}_detailer_enabled", value=False)
                merge = gr.Checkbox(label="Merge detailers", elem_id=f"{tab}_detailer_merge", value=shared.opts.detailer_merge, visible=True)
            with gr.Row():
                detailers = gr.Dropdown(label="Detailer models", elem_id=f"{tab}_detailers", choices=list(self.list), value=shared.opts.detailer_models, multiselect=True, visible=True)
                detailers_text = gr.Textbox(label="Detailer list", elem_id=f"{tab}_detailers_text", placeholder="Comma separated list of detailer models", lines=2, visible=False, interactive=True)
                refresh_btn = ui_common.create_refresh_button(detailers, self.enumerate, lambda: {"choices": self.enumerate()}, 'yolo_refresh_models')
                ui_mode = ui_components.ToolButton(value=ui_symbols.view)
                ui_mode.click(fn=self.change_mode, inputs=[detailers, detailers_text], outputs=[detailers, detailers_text, refresh_btn])
            with gr.Row():
                classes = gr.Textbox(label="Detailer classes", placeholder="Classes", elem_id=f"{tab}_detailer_classes")
            with gr.Row():
                prompt = gr.Textbox(label="Detailer prompt", value='', placeholder='detailer prompt or leave empty to use main prompt', lines=2, elem_id=f"{tab}_detailer_prompt")
            with gr.Row():
                negative = gr.Textbox(label="Detailer negative prompt", value='', placeholder='detailer prompt or leave empty to use main prompt', lines=2, elem_id=f"{tab}_detailer_negative")
            with gr.Row():
                steps = gr.Slider(label="Detailer steps", elem_id=f"{tab}_detailer_steps", value=10, minimum=0, maximum=99, step=1)
                strength = gr.Slider(label="Detailer strength", elem_id=f"{tab}_detailer_strength", value=0.3, minimum=0, maximum=1, step=0.01)
            with gr.Row():
                max_detected = gr.Slider(label="Max detected", elem_id=f"{tab}_detailer_max", value=shared.opts.detailer_max, minimum=1, maximum=10, step=1)
            with gr.Row():
                padding = gr.Slider(label="Edge padding", elem_id=f"{tab}_detailer_padding", value=shared.opts.detailer_padding, minimum=0, maximum=100, step=1)
                blur = gr.Slider(label="Edge blur", elem_id=f"{tab}_detailer_blur", value=shared.opts.detailer_blur, minimum=0, maximum=100, step=1)
            with gr.Row():
                min_confidence = gr.Slider(label="Min confidence", elem_id=f"{tab}_detailer_conf", value=shared.opts.detailer_conf, minimum=0.0, maximum=1.0, step=0.05)
                iou = gr.Slider(label="Max overlap", elem_id=f"{tab}_detailer_iou", value=shared.opts.detailer_iou, minimum=0, maximum=1.0, step=0.05)
            with gr.Row():
                min_size = shared.opts.detailer_min_size if shared.opts.detailer_min_size < 1 else 0.0
                min_size = gr.Slider(label="Min size", elem_id=f"{tab}_detailer_min_size", value=min_size, minimum=0.0, maximum=1.0, step=0.05)
                max_size = shared.opts.detailer_max_size if shared.opts.detailer_max_size < 1 and shared.opts.detailer_max_size > 0 else 1.0
                max_size = gr.Slider(label="Max size", elem_id=f"{tab}_detailer_max_size", value=max_size, minimum=0.0, maximum=1.0, step=0.05)
            with gr.Row(elem_classes=['flex-break']):
                renoise_value = gr.Slider(minimum=0.5, maximum=1.5, step=0.01, label='Renoise', value=shared.opts.detailer_sigma_adjust, elem_id=f"{tab}_detailer_renoise")
                renoise_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Renoise end', value=shared.opts.detailer_sigma_adjust_max, elem_id=f"{tab}_detailer_renoise_end")

            merge.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            detailers.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            detailers_text.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            classes.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            padding.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            blur.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            min_confidence.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            max_detected.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            min_size.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            max_size.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            iou.change(fn=ui_settings_change, inputs=[merge, detailers, detailers_text, classes, strength, padding, blur, min_confidence, max_detected, min_size, max_size, iou, steps, renoise_value, renoise_end], outputs=[])
            return enabled, prompt, negative, steps, strength


def initialize():
    shared.yolo = YoloRestorer()
    shared.detailers.append(shared.yolo)
