"""
downloads: https://luts.iwltbap.com/
lib: https://github.com/homm/pillow-lut-tools
"""
import os
import gradio as gr
from installer import install
from modules import scripts_manager, shared, processing


class Script(scripts_manager.Script):
    def title(self):
        return 'LUT Color grading'

    def show(self, is_img2img): # pylint: disable=unused-argument
        return shared.native

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML("<span>&nbsp LUT Color grading</span><br>")
        with gr.Row():
            original = gr.Checkbox(label='Include original image', value=True)
        with gr.Row():
            cube_file = gr.File(label='LUT .cube file', help='Download LUTs from https://luts.iwltbap.com/')
            # cube_file = gr.File(label='LUT .cube file')
        with gr.Row():
            gr.HTML("<br>Enhance LUT")
        with gr.Row():
            cube_scale = gr.Slider(label='Amplify LUT', minimum=0.0, maximum=5.0, step=0.05, value=1.0)
            brightness = gr.Slider(label='Brightness', minimum=-1, maximum=1, step=0.05, value=0)
            exposure = gr.Slider(label='Exposure', minimum=-5, maximum=5, step=0.05, value=0)
            contrast = gr.Slider(label='Contrast', minimum=-1, maximum=1, step=0.05, value=0)
            warmth = gr.Slider(label='Warmth', minimum=-1, maximum=1, step=0.05, value=0)
            saturation = gr.Slider(label='Saturation', minimum=-1, maximum=5, step=0.05, value=0)
            vibrance = gr.Slider(label='Vibrance', minimum=-1, maximum=5, step=0.05, value=0)
            hue = gr.Slider(label='Hue', minimum=0, maximum=1, step=0.05, value=0)
            gamma = gr.Slider(label='Gamma', minimum=0, maximum=10.0, step=0.1, value=1.0)
        return [original, cube_file, cube_scale, brightness, exposure, contrast, warmth, saturation, vibrance, hue, gamma]

    # auto-executed by the script-callback
    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, original, cube_file, cube_scale, brightness, exposure, contrast, warmth, saturation, vibrance, hue, gamma): # pylint: disable=arguments-differ, unused-argument
        install('pillow_lut', quiet=True)
        import pillow_lut

        cube = None
        name = os.path.splitext(os.path.basename(cube_file.name))[0] if cube_file is not None else None
        shared.log.info(f'Color grading: cube="{name}" scale={cube_scale} brightness={brightness} exposure={exposure} contrast={contrast} warmth={warmth} saturation={saturation} vibrance={vibrance} hue={hue} gamma={gamma}')
        if cube_file is not None:
            try:
                cube = pillow_lut.load_cube_file(cube_file.name)
                cube = pillow_lut.amplify_lut(cube, cube_scale)
                cube = pillow_lut.rgb_color_enhance(source=cube, brightness=brightness, exposure=exposure, contrast=contrast, warmth=warmth, saturation=saturation, vibrance=vibrance, hue=hue, gamma=gamma)
            except Exception as e:
                shared.log.error(f'Color grading: {e}')

        images = []
        if processed is not None and len(processed.images) > 0:
            for image in processed.images:
                info = image.info.get('parameters', '')
                if original:
                    images.append(image)
                if cube is not None:
                    filtered = image.filter(cube)
                    filtered.info['parameters'] = f'{info}, LUT: {name}'
                    images.append(filtered)
        processed.images = images

        return processed
