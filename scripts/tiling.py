from typing import Optional
import torch
import gradio as gr
from PIL import Image
from diffusers.models.lora import LoRACompatibleConv
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from modules import scripts_manager, processing, shared


modex = 'constant'
modey = 'constant'


def asymmetricConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]): # pylint: disable=redefined-builtin
    self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0) # pylint: disable=protected-access
    self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3]) # pylint: disable=protected-access
    working = F.pad(input, self.paddingX, mode=modex)
    working = F.pad(working, self.paddingY, mode=modex)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.conv_layers = []
        self.modes = ['constant', 'circular', 'reflect', 'replicate']

    def title(self):
        return 'Asymmetric Tiling'

    def show(self, is_img2img):
        return shared.native

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<b>Asymmetric Tiling</b><br>')
        with gr.Row():
            tilex = gr.Dropdown(label="Mode x-axis", choices=self.modes, value='constant')
            numx = gr.Slider(label="Repeat x-axis", value=1, minimum=1, maximum=10, step=1)
        with gr.Row():
            tiley = gr.Dropdown(label="Mode y-axis", choices=self.modes, value='constant')
            numy = gr.Slider(label="Repeat y-axis", value=1, minimum=1, maximum=10, step=1)
        return [tilex, numx, tiley, numy]

    def run(self, p: processing.StableDiffusionProcessing, tilex:bool=False, numx:int=1, tiley:bool=False, numy:int=1): # pylint: disable=arguments-differ, unused-argument
        global modex, modey # pylint: disable=global-statement
        supported_model_list = ['sd', 'sdxl']
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'Tiling: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        if not tilex and not tiley:
            return None
        self.orig_pipe = shared.sd_model

        modex = tilex
        modey = tiley
        self.conv_layers.clear()
        targets = [shared.sd_model.vae, shared.sd_model.text_encoder, shared.sd_model.unet]
        for target in targets:
            for module in target.modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.conv_layers.append(module)

        for cl in self.conv_layers:
            if isinstance(cl, LoRACompatibleConv) and cl.lora_layer is None:
                cl.lora_layer = lambda *x: 0
            if hasattr(cl, '_conv_forward'):
                cl._orig_conv_forward = cl._conv_forward # pylint: disable=protected-access
            cl._conv_forward = asymmetricConv2DConvForward.__get__(cl, torch.nn.Conv2d) # pylint: disable=protected-access, no-value-for-parameter
        shared.log.info(f'Tiling: x={tilex}:{numx} y={tiley}:{numy}')
        return None

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, tilex:bool=False, numx:int=1, tiley:bool=False, numy:int=1): # pylint: disable=arguments-differ, unused-argument
        if len(self.conv_layers) == 0:
            return processed
        for cl in self.conv_layers:
            if hasattr(cl, '_orig_conv_forward'):
                cl._conv_forward = cl._orig_conv_forward # pylint: disable=protected-access
        if self.orig_pipe is None:
            return processed
        if shared.sd_model_type == "sdxl":
            shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        self.conv_layers.clear()
        if not hasattr(processed, 'images') or processed.images is None:
            return processed
        images = []
        for image in processed.images:
            if tilex and isinstance(image, Image.Image):
                tiled = Image.new('RGB', (image.width * numx, image.height), (0, 0, 0))
                for i in range(numx):
                    tiled.paste(image, (i * image.width, 0))
                image = tiled
            if tiley and isinstance(image, Image.Image):
                tiled = Image.new('RGB', (image.width, image.height * numy), (0, 0, 0))
                for i in range(numy):
                    tiled.paste(image, (0, i * image.height))
                image = tiled
            images.append(image)
        processed.images = images
        return processed
