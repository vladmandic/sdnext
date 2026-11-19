import os
from functools import lru_cache
import gradio as gr
from PIL import Image
from modules import processing, scripts_manager, scripts_postprocessing
from modules.logger import log


positions = ["none", "top/left", "top/center", "top/right", "bottom/left", "bottom/center", "bottom/right", "center", "random"]


def create_ui(parent):
    with gr.Accordion('Watermark', open=False, elem_id=f'{parent}_watermark_accordion'):
        with gr.Row():
            position = gr.Dropdown(label="Watermark location", choices=positions, value="none", elem_id=f"{parent}_watermark_position")
        with gr.Row():
            auto_mask = gr.Checkbox(label="Image mask", value=False, elem_id=f"{parent}_watermark_auto_mask")
            scale = gr.Slider(label="Image scale", minimum=0.1, maximum=2.0, value=1.0, step=0.1, elem_id=f"{parent}_watermark_scale")
        with gr.Row():
            gr_image = gr.Image(elem_id=f"{parent}_watermark_preview", type='pil', height=256, visible=True, image_mode='RGB', show_label=False)
        with gr.Row():
            wm_image = gr.File(label="Image watermark", file_types=["image"], type="file", elem_id=f"{parent}_watermark_image")
        with gr.Row():
            wm_text = gr.Textbox(label="Invisible watermark text", value="", elem_id=f"{parent}_watermark_text")
        wm_image.change(fn=_load_watermark,inputs=[wm_image, auto_mask, scale], outputs=[gr_image])
    return {
        "position": position,
        "wm_image": wm_image,
        "auto_mask": auto_mask,
        "scale": scale,
        "wm_text": wm_text,
    }


@lru_cache(maxsize=1)
def _load_watermark(path: str, auto_mask: bool, scale: float): # pylint: disable=unused-argument
    try:
        if path is None:
            return None
        if hasattr(path, "name"): # gr.tempfile
            path = path.name
        with Image.open(path) as source:
            has_alpha = "A" in source.getbands() or "transparency" in source.info
            image = source.copy()
        if auto_mask and not has_alpha:
            from modules.rembg import lucida
            image = lucida.remove(image.convert("RGB"))
        if scale != 1.0:
            width, height = image.size
            image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        log.error(f"Watermark: {e}")
        return None


def apply_watermark(image, position, wm_image, auto_mask, scale, wm_text):
    if image is None:
        return image
    if position is None or position.lower() == 'none':
        return image
    if wm_image:
        wm_path = wm_image if isinstance(wm_image, str) else getattr(wm_image, "name", None)
        if wm_path is None:
            raise TypeError("Image watermark must be an uploaded file or a path string")
        wm_path = os.path.realpath(os.fspath(wm_path))
        wm_image = _load_watermark(wm_path, auto_mask, scale)
        if wm_image is None:
            return image
        from modules.image.watermark import set_image_watermark
        image = set_image_watermark(image, wm_image, position)
    if wm_text:
        from modules.image.watermark import set_text_watermark
        image = set_text_watermark(image, wm_text)
    return image


class ScriptWatermark(scripts_manager.Script):
    def title(self):
        return "Watermark"

    def show(self, *args, **kwargs): # pylint: disable=unused-argument
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        return list(create_ui(self.parent).values())

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts_manager.PostprocessImageArgs, position, wm_image, auto_mask, scale, wm_text): # pylint: disable=arguments-differ,unused-argument
        pp.image = apply_watermark(pp.image, position, wm_image, auto_mask, scale, wm_text)


class ScriptPostprocessingWatermark(scripts_postprocessing.ScriptPostprocessing):
    name = "Watermark"
    order = 40000

    def ui(self):
        return create_ui("postprocess")

    def process(self, pp: scripts_postprocessing.PostprocessedImage, position, wm_image, auto_mask, scale, wm_text): # pylint: disable=arguments-differ
        if pp.image is not None:
            pp.image = apply_watermark(pp.image, position, wm_image, auto_mask, scale, wm_text)
        return pp
