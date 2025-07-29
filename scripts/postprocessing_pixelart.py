import gradio as gr
from modules import scripts_postprocessing, devices

class ScriptPixelArt(scripts_postprocessing.ScriptPostprocessing):
    name = "PixelArt"
    order = 30000

    def ui(self):
        with gr.Accordion('PixelArt', open = False, elem_id="postprocess_pixelart_accordion"):
            with gr.Row():
                pixelart_enabled = gr.Checkbox(label="Enable PixelArt", value=False, elem_id="extras_pixelart_enabled")
                pixelart_use_edge_detection = gr.Checkbox(label="Enable edge detection", value=True, elem_id="extras_pixelart_use_edge_detection")
            with gr.Row():
                pixelart_block_size = gr.Slider(minimum=2, maximum=64, step=1, value=8, label="PixelArt block size", elem_id="extras_pixelart_block_size")
                pixelart_edge_block_size = gr.Slider(minimum=2, maximum=64, step=1, value=4, label="Edge block size", elem_id="extras_pixelart_edge_block_size")
                pixelart_image_weight = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=1.0, label="Edge image weight", elem_id="extras_pixelart_image_weight")
                pixelart_sharpen_amount = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, label="PixelArt sharpen", elem_id="extras_pixelart_sharpen_amount")
        return {
            "pixelart_enabled": pixelart_enabled,
            "pixelart_block_size": pixelart_block_size,
            "pixelart_edge_block_size": pixelart_edge_block_size,
            "pixelart_use_edge_detection": pixelart_use_edge_detection,
            "pixelart_image_weight": pixelart_image_weight,
            "pixelart_sharpen_amount": pixelart_sharpen_amount,
        }

    def process(self, pp: scripts_postprocessing.PostprocessedImage, pixelart_enabled: bool, pixelart_use_edge_detection: bool, pixelart_block_size: int, pixelart_edge_block_size: int, pixelart_image_weight: float, pixelart_sharpen_amount: float): # pylint: disable=arguments-differ
        if not pixelart_enabled:
            return
        from modules.postprocess.pixelart import img_to_pixelart, edge_detect_for_pixelart
        pixel_image = pp.image

        if pixelart_use_edge_detection:
            pixel_image = edge_detect_for_pixelart(pixel_image, image_weight=pixelart_image_weight, block_size=pixelart_edge_block_size, device=devices.device)
            pp.info["PixelArt edge block size"] = pixelart_edge_block_size

        pixel_image = img_to_pixelart(pixel_image, sharpen=pixelart_sharpen_amount, block_size=pixelart_block_size, device=devices.device)
        if len(pixel_image) == 1:
            pixel_image = pixel_image[0]
        pp.image = pixel_image
        pp.info["PixelArt block size"] = pixelart_block_size
