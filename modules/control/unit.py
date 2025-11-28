from typing import Union
from PIL import Image
import gradio as gr
from installer import log
from modules.control import processors
from modules.control.units import controlnet
from modules.control.units import xs
from modules.control.units import lite
from modules.control.units import t2iadapter
from modules.control.units import reference # pylint: disable=unused-import


default_device = None
default_dtype = None
unit_types = ['t2i adapter', 'controlnet', 'xs', 'lite', 'reference', 'ip']
current = []


class Unit(): # mashup of gradio controls and mapping to actual implementation classes
    def update_choices(self, model_id=None):
        name = model_id or self.model_name
        if name == 'InstantX Union F1':
            self.choices = ['canny', 'tile', 'depth', 'blur', 'pose', 'gray', 'lq']
        elif name == 'Shakker-Labs Union F1':
            self.choices = ['canny', 'tile', 'depth', 'blur', 'pose', 'gray', 'lq']
        elif name == 'Xinsir Union XL':
            self.choices = ['openpose', 'depth', 'scribble', 'canny', 'normal']
        elif name == 'Xinsir ProMax XL':
            self.choices = ['openpose', 'depth', 'scribble', 'canny', 'normal', 'segment', 'tile', 'repaint']
        else:
            self.choices = ['default']

    def __str__(self):
        return f'Unit(index={self.index} enabled={self.enabled} type="{self.type}" strength={self.strength} start={self.start} end={self.end}{self.process}{self.controlnet} override={self.override})'

    def __init__(self,
                 # values
                 index: int = None,
                 enabled: bool = None,
                 strength: float = None,
                 unit_type: str = None,
                 start: float = 0,
                 end: float = 1,
                 # ui bindings
                 enabled_cb = None,
                 reset_btn = None,
                 process_id = None,
                 preview_btn = None,
                 model_id = None,
                 model_strength = None,
                 preview_process = None,
                 image_upload = None,
                 image_reuse = None,
                 image_preview = None,
                 control_start = None,
                 control_end = None,
                 control_mode = None,
                 control_tile = None,
                 result_txt = None,
                 extra_controls: list = [],
        ):
        self.model_id = model_id
        self.process_id = process_id
        self.controls = [gr.Label(value=unit_type, visible=False)] # separator
        self.index = index
        self.enabled = enabled or False
        self.type = unit_type
        self.strength = strength or 1.0
        self.model_strength = model_strength
        self.start = start or 0
        self.end = end or 1
        self.start = min(self.start, self.end)
        self.end = max(self.start, self.end)
        self.mode = None
        # processor always exists, adapter and controlnet are optional
        self.model_name = None
        self.process_name = None
        self.process: processors.Processor = processors.Processor()
        self.adapter: t2iadapter.Adapter = None
        self.controlnet: Union[controlnet.ControlNet, xs.ControlNetXS] = None
        # map to input image
        self.override: Image = None
        # global settings but passed per-unit
        self.factor = 1.0
        self.guess = False
        # reference settings
        self.attention = 'Attention'
        self.fidelity = 0.5
        self.query_weight = 1.0
        self.adain_weight = 1.0
        # control mode
        self.choices = ['default']
        # control tile
        self.tile = '1x1'

        def enabled_change(val):
            self.enabled = val

        def strength_change(val):
            self.strength = val

        def control_change(start, end):
            self.start = min(start, end)
            self.end = max(start, end)

        def control_mode_change(mode):
            self.mode = self.choices.index(mode) if mode is not None and mode in self.choices else 0

        def control_tile_change(tile):
            self.tile = tile

        def control_choices(model_id):
            self.update_choices(model_id)
            mode_visible = 'union' in model_id.lower() or 'promax' in model_id.lower()
            tile_visible = 'union' in model_id.lower() or 'promax' in model_id.lower() or 'tile' in model_id.lower()
            return [gr.update(visible=mode_visible, choices=self.choices), gr.update(visible=tile_visible)]

        def adapter_extra(c1):
            self.factor = c1

        def controlnet_extra(c1):
            self.guess = c1

        def controlnetxs_extra(_c1):
            pass # gr.component passed directly to load method

        def reference_extra(c1, c2, c3, c4):
            self.attention = c1
            self.fidelity = c2
            self.query_weight = c3
            self.adain_weight = c4

        def upload_image(image_file):
            if image_file is None:
                self.process.override = None
                self.override = None
                log.debug('Control process clear image')
                return gr.update(value=None)
            try:
                self.process.override = Image.open(image_file.name)
                self.override = self.process.override
                log.debug(f'Control process upload image: path="{image_file.name}" image={self.process.override}')
                return gr.update(visible=self.process.override is not None, value=self.process.override)
            except Exception as e:
                log.error(f'Control process upload image failed: path="{image_file.name}" error={e}')
                return gr.update(visible=False, value=None)

        def reuse_image(image):
            log.debug(f'Control process reuse image: {image}')
            self.process.override = image
            self.override = self.process.override
            return gr.update(visible=self.process.override is not None, value=self.process.override)

        def set_image(image):
            self.process.override = image
            self.override = image
            return gr.update(visible=image is not None)

        # actual init
        if self.type == 't2i adapter':
            self.adapter = t2iadapter.Adapter(device=default_device, dtype=default_dtype)
        elif self.type == 'controlnet':
            self.controlnet = controlnet.ControlNet(device=default_device, dtype=default_dtype)
        elif self.type == 'xs':
            self.controlnet = xs.ControlNetXS(device=default_device, dtype=default_dtype)
        elif self.type == 'lite':
            self.controlnet = lite.ControlLLLite(device=default_device, dtype=default_dtype)
        elif self.type == 'reference':
            pass
        elif self.type == 'ip':
            pass
        else:
            log.error(f'Control unknown type: unit={unit_type}')
            return

        # bind ui controls to properties if present
        if self.type == 't2i adapter':
            if model_id is not None:
                if isinstance(model_id, str):
                    self.adapter.load(model_id)
                else:
                    self.controls.append(model_id)
                    model_id.change(fn=self.adapter.load, inputs=[model_id], outputs=[result_txt], show_progress='full')
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=adapter_extra, inputs=extra_controls)
        elif self.type == 'controlnet':
            if model_id is not None:
                if isinstance(model_id, str):
                    self.controlnet.load(model_id)
                else:
                    self.controls.append(model_id)
                    model_id.change(fn=self.controlnet.load, inputs=[model_id], outputs=[result_txt], show_progress='full')
                    model_id.change(fn=control_choices, inputs=[model_id], outputs=[control_mode, control_tile], show_progress='hidden')
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=controlnet_extra, inputs=extra_controls)
        elif self.type == 'xs':
            if model_id is not None:
                if isinstance(model_id, str):
                    self.controlnet.load(model_id)
                else:
                    self.controls.append(model_id)
                    model_id.change(fn=self.controlnet.load, inputs=[model_id, extra_controls[0]], outputs=[result_txt], show_progress='full')
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=controlnetxs_extra, inputs=extra_controls)
        elif self.type == 'lite':
            if model_id is not None:
                if isinstance(model_id, str):
                    self.controlnet.load(model_id)
                else:
                    self.controls.append(model_id)
                    model_id.change(fn=self.controlnet.load, inputs=[model_id], outputs=[result_txt], show_progress='full')
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=controlnetxs_extra, inputs=extra_controls)
        elif self.type == 'reference':
            if extra_controls is not None and len(extra_controls) > 0:
                extra_controls[0].change(fn=reference_extra, inputs=extra_controls)
                extra_controls[1].change(fn=reference_extra, inputs=extra_controls)
                extra_controls[2].change(fn=reference_extra, inputs=extra_controls)
                extra_controls[3].change(fn=reference_extra, inputs=extra_controls)

        if enabled_cb is not None:
            self.controls.append(enabled_cb)
            enabled_cb.change(fn=enabled_change, inputs=[enabled_cb])
        if model_strength is not None:
            self.controls.append(model_strength)
            model_strength.change(fn=strength_change, inputs=[model_strength])
        if process_id is not None:
            if isinstance(process_id, str):
                self.process.load(process_id)
            else:
                self.controls.append(process_id)
                process_id.change(fn=self.process.load, inputs=[process_id], outputs=[result_txt], show_progress='full')
        if reset_btn is not None:
            reset_btn.click(fn=self.reset, inputs=[], outputs=[enabled_cb, model_id, process_id, model_strength])
        if preview_btn is not None:
            preview_btn.click(fn=self.process.preview, inputs=[], outputs=[preview_process]) # return list of images for gallery
        if image_upload is not None:
            image_upload.upload(fn=upload_image, inputs=[image_upload], outputs=[image_preview]) # return list of images for gallery
        if image_reuse is not None:
            image_reuse.click(fn=reuse_image, inputs=[preview_process], outputs=[image_preview]) # return list of images for gallery
        if image_preview is not None:
            self.controls.append(image_preview)
            image_preview.change(fn=set_image, inputs=[image_preview], outputs=[image_preview])
        if control_start is not None and control_end is not None:
            self.controls.append(control_start)
            self.controls.append(control_end)
            control_start.change(fn=control_change, inputs=[control_start, control_end])
            control_end.change(fn=control_change, inputs=[control_start, control_end])
        if control_mode is not None:
            self.controls.append(control_mode)
            control_mode.change(fn=control_mode_change, inputs=[control_mode])
        if control_tile is not None:
            self.controls.append(control_tile)
            control_tile.change(fn=control_tile_change, inputs=[control_tile])

    def reset(self):
        if self.process is not None:
            self.process.reset()
        if self.adapter is not None:
            self.adapter.reset()
        if self.controlnet is not None:
            self.controlnet.reset()
        self.override = None
        return [True, 'None', 'None', 1.0] # reset ui values
