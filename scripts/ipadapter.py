import json
from PIL import Image
import gradio as gr
from modules import scripts_manager, processing, shared, ipadapter, ui_common


MAX_ADAPTERS = 4


class Script(scripts_manager.Script):
    standalone = True

    def title(self):
        return 'IP Adapters'

    def show(self, is_img2img):
        return scripts_manager.AlwaysVisible if shared.native else False

    def load_images(self, files):
        init_images = []
        for file in files or []:
            try:
                if isinstance(file, str):
                    from modules.api.api import decode_base64_to_image
                    image = decode_base64_to_image(file)
                elif isinstance(file, Image.Image):
                    image = file
                elif isinstance(file, dict) and 'name' in file:
                    image = Image.open(file['name']) # _TemporaryFileWrapper from gr.Files
                elif hasattr(file, 'name'):
                    image = Image.open(file.name) # _TemporaryFileWrapper from gr.Files
                else:
                    raise ValueError(f'IP adapter unknown input: {file}')
                init_images.append(image)
            except Exception as e:
                shared.log.warning(f'IP adapter failed to load image: {e}')
        return gr.update(value=init_images, visible=len(init_images) > 0)

    def display_units(self, num_units):
        num_units = num_units or 1
        return (num_units * [gr.update(visible=True)]) + ((MAX_ADAPTERS - num_units) * [gr.update(visible=False)])

    def display_advanced(self, advanced):
        return [gr.update(visible=advanced), gr.update(visible=advanced)]

    def ui(self, _is_img2img):
        with gr.Accordion('IP Adapters', open=False, elem_id='ipadapter', elem_classes=['ipadapter']):
            units = []
            adapters = []
            scales = []
            starts = []
            ends = []
            files = []
            crops = []
            masks = []
            image_galleries = []
            mask_galleries = []
            with gr.Row():
                num_adapters = gr.Slider(label="Active IP adapters", minimum=1, maximum=MAX_ADAPTERS, step=1, value=1, scale=1)
                unload_adapter = gr.Checkbox(label='Unload adapter', value=False, interactive=True)
            for i in range(MAX_ADAPTERS):
                with gr.Accordion(f'Adapter {i+1}', visible=i==0) as unit:
                    with gr.Row():
                        adapter = gr.Dropdown(label='Adapter', choices=list(ipadapter.get_adapters()), value='None')
                        adapters.append(adapter)
                        ui_common.create_refresh_button(adapter, ipadapter.get_adapters)
                    with gr.Row():
                        scales.append(gr.Slider(label='Strength', minimum=0.0, maximum=1.0, step=0.01, value=0.5))
                        crops.append(gr.Checkbox(label='Crop to portrait', value=False, interactive=True))
                    with gr.Row():
                        starts.append(gr.Slider(label='Start', minimum=0.0, maximum=1.0, step=0.1, value=0))
                        ends.append(gr.Slider(label='End', minimum=0.0, maximum=1.0, step=0.1, value=1))
                    with gr.Row():
                        files.append(gr.File(label='Input images', file_count='multiple', file_types=['image'], interactive=True, height=100))
                    with gr.Row():
                        image_galleries.append(gr.Gallery(show_label=False, value=[], visible=False, container=False, rows=1))
                    with gr.Row():
                        masks.append(gr.File(label='Input masks', file_count='multiple', file_types=['image'], interactive=True, height=100))
                    with gr.Row():
                        mask_galleries.append(gr.Gallery(show_label=False, value=[], visible=False))
                    files[i].change(fn=self.load_images, inputs=[files[i]], outputs=[image_galleries[i]])
                    masks[i].change(fn=self.load_images, inputs=[masks[i]], outputs=[mask_galleries[i]])
                units.append(unit)
            num_adapters.change(fn=self.display_units, inputs=[num_adapters], outputs=units)
            layers_active = gr.Checkbox(label='Layer options', value=False, interactive=True)
            layers_label = gr.HTML('<a href="https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter#style--layout-control" target="_blank">InstantStyle: advanced layer activation</a>', visible=False)
            layers = gr.Textbox(label='Layer scales', placeholder='{\n"down": {"block_2": [0.0, 1.0]},\n"up": {"block_0": [0.0, 1.0, 0.0]}\n}', type='text', interactive=True, lines=5, visible=False, show_label=False)
            layers_active.change(fn=self.display_advanced, inputs=[layers_active], outputs=[layers_label, layers])
        return [num_adapters] + [unload_adapter] + adapters + scales + files + crops + starts + ends + masks + [layers_active] + [layers]

    def process(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        if not shared.native:
            return
        args = list(args) if args is not None else []
        if len(args) == 0:
            return
        units = args.pop(0)
        unload = args.pop(0)
        if getattr(p, 'ip_adapter_names', []) == []:
            p.ip_adapter_names = args[:MAX_ADAPTERS][:units]
        if getattr(p, 'ip_adapter_scales', [0.0]) == [0.0]:
            p.ip_adapter_scales = args[MAX_ADAPTERS:MAX_ADAPTERS*2][:units]
        if getattr(p, 'ip_adapter_images', []) == []:
            p.ip_adapter_images = args[MAX_ADAPTERS*2:MAX_ADAPTERS*3][:units]
        if getattr(p, 'ip_adapter_crops', []) == []:
            p.ip_adapter_crops = args[MAX_ADAPTERS*3:MAX_ADAPTERS*4][:units]
        if getattr(p, 'ip_adapter_starts', [0.0]) == [0.0]:
            p.ip_adapter_starts = args[MAX_ADAPTERS*4:MAX_ADAPTERS*5][:units]
        if getattr(p, 'ip_adapter_ends', [1.0]) == [1.0]:
            p.ip_adapter_ends = args[MAX_ADAPTERS*5:MAX_ADAPTERS*6][:units]
        if getattr(p, 'ip_adapter_masks', []) == []:
            p.ip_adapter_masks = args[MAX_ADAPTERS*6:MAX_ADAPTERS*7][:units]
            p.ip_adapter_masks = [x for x in p.ip_adapter_masks if x]
        layers_active, layers = args[MAX_ADAPTERS*7:MAX_ADAPTERS*8]
        p.ip_adapter_unload = unload
        if layers_active and len(layers) > 0:
            try:
                layers = json.loads(layers)
                p.ip_adapter_layers = layers
            except Exception as e:
                shared.log.error(f'IP adapter: failed to parse layer scales: {e}')
        # ipadapter.apply(shared.sd_model, p, p.ip_adapter_names, p.ip_adapter_scales, p.ip_adapter_starts, p.ip_adapter_ends, p.ip_adapter_images) # called directly from processing.process_images_inner
