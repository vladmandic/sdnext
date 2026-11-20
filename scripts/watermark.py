import gradio as gr
from modules import processing, scripts_manager, scripts_postprocessing
from modules.image.watermark import load_watermark, set_image_watermark, set_text_watermark, set_video_watermark


positions = ["none", "top/left", "top/center", "top/right", "bottom/left", "bottom/center", "bottom/right", "center", "random"]


def create_ui(parent):
    with gr.Accordion('Watermark', open=False, elem_id=f'{parent}_watermark_accordion'):
        with gr.Row():
            position = gr.Dropdown(label="Watermark location", choices=positions, value="none", elem_id=f"{parent}_watermark_position")
            auto_mask = gr.Checkbox(label="Auto mask", value=True, elem_id=f"{parent}_watermark_auto_mask")
        with gr.Row():
            scale = gr.Slider(label="WM scale", minimum=0.1, maximum=2.0, value=1.0, step=0.1, elem_id=f"{parent}_watermark_scale")
            opacity = gr.Slider(label="WM opacity", minimum=0.0, maximum=1.0, value=1.0, step=0.1, elem_id=f"{parent}_watermark_opacity")
            padding = gr.Slider(label="WM padding", minimum=0, maximum=100, value=0, step=1, elem_id=f"{parent}_watermark_padding")
        with gr.Row():
            gr_image = gr.Image(elem_id=f"{parent}_watermark_preview", type='pil', height=256, visible=True, image_mode='RGB', show_label=False)
        with gr.Row():
            wm_image = gr.File(label="", file_types=["image"], type="file", elem_id=f"{parent}_watermark_image", show_label=False)
        with gr.Row():
            wm_text = gr.Textbox(label="Invisible watermark text", value="", elem_id=f"{parent}_watermark_text")
        wm_image.change(fn=load_watermark,inputs=[wm_image, auto_mask, scale], outputs=[gr_image])
    return {
        "position": position,
        "wm_image": wm_image,
        "auto_mask": auto_mask,
        "scale": scale,
        "opacity": opacity,
        "padding": padding,
        "wm_text": wm_text,
    }


class ScriptWatermark(scripts_manager.Script):
    def title(self):
        return "Watermark"

    def show(self, *args, **kwargs): # pylint: disable=unused-argument
        return scripts_manager.AlwaysVisible

    def ui(self, _is_img2img):
        return list(create_ui(self.parent).values())

    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts_manager.PostprocessImageArgs, position, wm_image, auto_mask, scale, opacity, padding, wm_text): # pylint: disable=arguments-differ,unused-argument
        if pp is None:
            return
        if pp.image is not None:
            wm_path = wm_image if isinstance(wm_image, str) else getattr(wm_image, "name", None)
            if wm_path:
                wm = set_image_watermark(pp.image, wm_path, position, scale=scale, opacity=opacity, pad=padding)
                if wm:
                    pp.image = wm
            if wm_text:
                wm = set_text_watermark(pp.image, wm_text)
                if wm:
                    pp.image = wm
        if pp.video is not None:
            wm = set_video_watermark(pp.video, wm_image=wm_path, position=position, mask=auto_mask, scale=scale, opacity=opacity, pad=padding)
            if wm:
                pp.video = wm


class ScriptPostprocessingWatermark(scripts_postprocessing.ScriptPostprocessing):
    name = "Watermark"
    order = 40000

    def ui(self):
        return create_ui("postprocess")

    def process(self, pp: scripts_postprocessing.PostprocessedImage, position, wm_image, auto_mask, scale, opacity, padding, wm_text): # pylint: disable=arguments-differ
        if pp is None:
            return pp
        if pp.image is not None:
            wm_path = wm_image if isinstance(wm_image, str) else getattr(wm_image, "name", None)
            if wm_path:
                wm = set_image_watermark(pp.image, wm_path, position, mask=auto_mask, scale=scale, opacity=opacity, pad=padding)
                if wm:
                    pp.image = wm
            if wm_text:
                wm = set_text_watermark(pp.image, wm_text)
                if wm:
                    pp.image = wm
        if pp.video is not None:
            wm_path = wm_image if isinstance(wm_image, str) else getattr(wm_image, "name", None)
            if wm_path:
                wm = set_video_watermark(pp.video, wm_image=wm_path, position=position, mask=auto_mask, scale=scale, opacity=opacity, pad=padding, quiet=False)
                if wm:
                    pp.video = wm
        return pp
