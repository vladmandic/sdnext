# https://github.com/genforce/ctrl-x

import gradio as gr
from installer import install
from modules import shared, scripts_manager, processing


COLORMAP = ['autumn', 'bone', 'jet', 'winter', 'rainbow', 'ocean', 'summer', 'spring', 'cool', 'hsv', 'pink', 'hot', 'parula', 'magma', 'inferno', 'plasma', 'viridis', 'cividis', 'twilight', 'shifted', 'turbo', 'deepgreen']


class Script(scripts_manager.Script):
    def title(self):
        return 'DAAM: Diffusion Attentive Attribution Maps'

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/castorini/daam">&nbsp DAAM: Diffusion Attentive Attribution Maps</a><br>')
        with gr.Row():
            append_images = gr.Checkbox(label='Append heatmaps to results', value=True, elem_id='daam_append_images')
            colormap = gr.Dropdown(label='Colormap', choices=COLORMAP, value='jet', type='value', elem_id='daam_colormap')
        return append_images, colormap

    def run(self, p: processing.StableDiffusionProcessing, append_images, colormap): # pylint: disable=arguments-differ
        c = shared.sd_model.__class__.__name__ if shared.sd_loaded else ''
        if shared.sd_model_type != 'sdxl':
            shared.log.warning(f'DAAM: pipeline={c} required=StableDiffusionXLPipeline')
            return None

        install('thinc==8.3.4')
        install('spacy==3.8.4')

        from scripts import daam # pylint: disable=no-name-in-module
        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['prompt_attention'] = 'fixed'

        # process
        with daam.trace(shared.sd_model) as tc:
            processed: processing.Processed = processing.process_images(p)
            global_heat_map = tc.compute_global_heat_map()
            shared.log.info(f'DAAM: prompt="{global_heat_map.prompt}" heatmaps={global_heat_map.heat_maps.shape}')

            # word_heat_map = global_heat_map.compute_word_heat_map('woman')
            parsed_heat_maps = global_heat_map.parsed_heat_maps()
            if append_images:
                image = processed.images[0]
                for parsed_heat_map in parsed_heat_maps:
                    if len(parsed_heat_map.token.text) > 1:
                        shared.log.debug(f'DAAM: token="{parsed_heat_map.token.text}"')
                        overlay = parsed_heat_map.word_heat_map.plot_overlay(image=image, color_normalize=True, cmap=colormap)
                        processed.images.append(overlay)

        # restore and return
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        return processed
