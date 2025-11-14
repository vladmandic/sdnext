from PIL import Image
import gradio as gr
import gradio.processing_utils
from modules import scripts_manager, patches, gr_tempdir


hijacked = False
original_IOComponent_init = None
original_Block_get_config = None
original_BlockContext_init = None
original_Blocks_get_config_file = None


def gr_image_preprocess(self, x):
    if x is None:
        return x
    mask = None
    if isinstance(x, dict):
        x, mask = x["image"], x["mask"]
    im = gradio.processing_utils.decode_base64_to_image(x)
    im = im.convert(self.image_mode)
    if self.shape is not None:
        im = gradio.processing_utils.resize_and_crop(im, self.shape)
    if self.tool == "sketch" and self.source in ["upload"]:
        if mask is not None:
            mask_im = gradio.processing_utils.decode_base64_to_image(mask)
            if mask_im.mode == "RGBA":  # whiten any opaque pixels in the mask
                alpha_data = mask_im.getchannel("A").convert("L")
                mask_im = Image.merge("RGB", [alpha_data, alpha_data, alpha_data])
        else:
            mask_im = Image.new("L", im.size, 0)
        return { "image": self._format_image(im), "mask": self._format_image(mask_im) } # pylint: disable=protected-access
    return self._format_image(im) # pylint: disable=protected-access


def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """
    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]
    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    self.webui_tooltip = kwargs.pop('tooltip', None)
    if scripts_manager.scripts_current is not None:
        scripts_manager.scripts_current.before_component(self, **kwargs)
    scripts_manager.script_callbacks.before_component_callback(self, **kwargs)
    res = original_IOComponent_init(self, *args, **kwargs) # pylint: disable=assignment-from-no-return
    add_classes_to_gradio_component(self)
    scripts_manager.script_callbacks.after_component_callback(self, **kwargs)
    if scripts_manager.scripts_current is not None:
        scripts_manager.scripts_current.after_component(self, **kwargs)
    return res


def Block_get_config(self):
    config = original_Block_get_config(self)
    webui_tooltip = getattr(self, 'webui_tooltip', None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip
    config.pop('example_inputs', None)
    return config


def BlockContext_init(self, *args, **kwargs):
    if scripts_manager.scripts_current is not None:
        scripts_manager.scripts_current.before_component(self, **kwargs)
    scripts_manager.script_callbacks.before_component_callback(self, **kwargs)
    res = original_BlockContext_init(self, *args, **kwargs) # pylint: disable=assignment-from-no-return
    add_classes_to_gradio_component(self)
    scripts_manager.script_callbacks.after_component_callback(self, **kwargs)
    if scripts_manager.scripts_current is not None:
        scripts_manager.scripts_current.after_component(self, **kwargs)
    return res


def Blocks_get_config_file(self, *args, **kwargs):
    config = original_Blocks_get_config_file(self, *args, **kwargs)
    for comp_config in config["components"]:
        if "example_inputs" in comp_config:
            comp_config["example_inputs"] = {"serialized": []}
    return config


def patch_gradio():
    def wrap_gradio_js(fn):
        def wrapper(*args, js=None, _js=None, **kwargs):
            if _js is not None:
                js = _js
            return fn(*args, js=js, **kwargs)
        return wrapper

    gradio.components.Button.click = wrap_gradio_js(gradio.components.Button.click)
    gradio.components.Textbox.submit = wrap_gradio_js(gradio.components.Textbox.submit)
    gradio.components.Image.clear = wrap_gradio_js(gradio.components.Image.clear)
    gradio.components.Image.change = wrap_gradio_js(gradio.components.Image.change)
    gradio.components.Image.upload = wrap_gradio_js(gradio.components.Image.upload)
    gradio.components.Video.change = wrap_gradio_js(gradio.components.Video.change)
    gradio.components.Video.clear = wrap_gradio_js(gradio.components.Video.clear)
    gradio.components.Slider.change = wrap_gradio_js(gradio.components.Slider.change)
    gradio.components.Dropdown.change = wrap_gradio_js(gradio.components.Dropdown.change)
    gradio.components.File.change = wrap_gradio_js(gradio.components.File.change)
    gradio.components.File.clear = wrap_gradio_js(gradio.components.File.clear)
    gradio.components.Number.change = wrap_gradio_js(gradio.components.Number.change)
    gradio.components.Textbox.change = wrap_gradio_js(gradio.components.Textbox.change)
    gradio.components.Radio.change = wrap_gradio_js(gradio.components.Radio.change)
    gradio.components.Checkbox.change = wrap_gradio_js(gradio.components.Checkbox.change)
    gradio.components.CheckboxGroup.change = wrap_gradio_js(gradio.components.CheckboxGroup.change)
    gradio.components.ColorPicker.change = wrap_gradio_js(gradio.components.ColorPicker.change)
    gradio.layouts.Tab.select = wrap_gradio_js(gradio.layouts.Tab.select)
    gradio.components.Image.edit = lambda *args, **kwargs: None
    # gradio.components.image.Image.__init__ missing tool, brush_radius, mask_opacity, edit()

def init():
    global hijacked, original_IOComponent_init, original_Block_get_config, original_BlockContext_init, original_Blocks_get_config_file # pylint: disable=global-statement
    if hijacked:
        return
    gr.components.Image.preprocess =  gr_image_preprocess
    if hasattr(gr.components, 'IOComponent'):
        gr.components.IOComponent.pil_to_temp_file =  gr_tempdir.pil_to_temp_file
        original_IOComponent_init = patches.patch(__name__, obj=gr.components.IOComponent, field="__init__", replacement=IOComponent_init)
    original_Block_get_config = patches.patch(__name__, obj=gr.blocks.Block, field="get_config", replacement=Block_get_config)
    original_BlockContext_init = patches.patch(__name__, obj=gr.blocks.BlockContext, field="__init__", replacement=BlockContext_init)
    original_Blocks_get_config_file = patches.patch(__name__, obj=gr.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file)
    if not gr.__version__.startswith('3.43'):
        patch_gradio()
    hijacked = True
