import os
import time
import gradio as gr
from PIL import Image
from modules import shared, scripts_manager, masking, video # pylint: disable=ungrouped-imports


gr_height = None
max_units = shared.opts.control_max_units
debug = os.environ.get('SD_CONTROL_DEBUG', None) is not None
debug_log = shared.log.trace if debug else lambda *args, **kwargs: None

# state variables
busy = False # used to synchronize select_input and generate_click
input_source = None
input_init = None
input_mask = None


def initialize():
    from modules import devices
    from modules.control import unit
    from modules.control import processors # patrickvonplaten controlnet_aux
    from modules.control.units import controlnet # lllyasviel ControlNet
    from modules.control.units import xs # vislearn ControlNet-XS
    from modules.control.units import lite # vislearn ControlNet-XS
    from modules.control.units import t2iadapter # TencentARC T2I-Adapter
    shared.log.debug(f'UI initialize: tab=control models="{shared.opts.control_dir}"')
    controlnet.cache_dir = os.path.join(shared.opts.control_dir, 'controlnet')
    xs.cache_dir = os.path.join(shared.opts.control_dir, 'xs')
    lite.cache_dir = os.path.join(shared.opts.control_dir, 'lite')
    t2iadapter.cache_dir = os.path.join(shared.opts.control_dir, 'adapter')
    processors.cache_dir = os.path.join(shared.opts.control_dir, 'processor')
    masking.cache_dir = os.path.join(shared.opts.control_dir, 'segment')
    unit.default_device = devices.device
    unit.default_dtype = devices.dtype
    try:
        os.makedirs(shared.opts.control_dir, exist_ok=True)
        os.makedirs(controlnet.cache_dir, exist_ok=True)
        os.makedirs(xs.cache_dir, exist_ok=True)
        os.makedirs(lite.cache_dir, exist_ok=True)
        os.makedirs(t2iadapter.cache_dir, exist_ok=True)
        os.makedirs(processors.cache_dir, exist_ok=True)
        os.makedirs(masking.cache_dir, exist_ok=True)
    except Exception:
        pass
    scripts_manager.scripts_current = scripts_manager.scripts_control
    scripts_manager.scripts_control.initialize_scripts(is_img2img=False, is_control=True)


def caption():
    prompt = None
    if input_source is None or len(input_source) == 0:
        shared.log.warning('Caption: no input source')
        return prompt
    try:
        from modules.caption.caption import caption as caption_fn
        prompt = caption_fn(input_source[0])
    except Exception as e:
        shared.log.error(f'Caption: {e}')
    return prompt


def display_units(num_units):
    num_units = num_units or 1
    return (num_units * [gr.update(visible=True)]) + ((max_units - num_units) * [gr.update(visible=False)])


def get_video(filepath: str):
    if not os.path.exists(filepath):
        return ''
    try:
        frames, fps, duration, w, h, codec, _cap = video.get_video_params(filepath)
        shared.log.debug(f'Control: input video: path={filepath} frames={frames} fps={fps} size={w}x{h} codec={codec}')
        msg = f'Control input | Video | Size {w}x{h} | Frames {frames} | FPS {fps:.2f} | Duration {duration:.2f} | Codec {codec}'
        return msg
    except Exception as e:
        msg = f'Control: video open failed: path={filepath} {e}'
        shared.log.error(msg)
        return msg


def process_kanvas(x): # only used when kanvas overrides gr.Image object
    image = None
    mask = None
    try: # try base64 decode
        t0 = time.time()
        image_data = x.get('image', '')
        image_bytes = len(image_data)
        if image_bytes > 0:
            from modules.api import helpers
            image = helpers.decode_base64_to_image(image_data)
            image = image.convert('RGB')
        mask_data = x.get('mask', '')
        mask_bytes = len(mask_data)
        if mask_bytes > 0:
            from modules.api import helpers
            mask = helpers.decode_base64_to_image(mask_data)
            mask = mask.convert('L')
        t1 = time.time()
        shared.log.debug(f'Kanvas: image={image}:{image_bytes} mask={mask}:{mask_bytes} time={t1-t0:.2f}')
        return image, mask
    except Exception:
        pass
    try: # try raw pixel data
        import numpy as np
        t0 = time.time()
        image_data = list(x.get('image', {}).values())
        if image_data:
            width = x['imageWidth']
            height = x['imageHeight']
            array = np.array(image_data, dtype=np.uint8).reshape((height, width, 4))
            image = Image.fromarray(array, 'RGBA')
            image = image.convert('RGB')
        mask_data = list(x.get('mask', {}).values())
        if mask_data:
            width = x['maskWidth']
            height = x['maskHeight']
            array = np.array(mask_data, dtype=np.uint8).reshape((height, width, 4))
            mask = Image.fromarray(array, 'RGBA')
            # alpha = mask.getchannel("A").convert("L")
            # mask = Image.merge("RGB", [alpha, alpha, alpha])
            mask = mask.convert('L')
        t1 = time.time()
        shared.log.debug(f'Kanvas: image={image} mask={mask} time={t1-t0:.2f}')
    except Exception:
        pass
    return image, mask


def select_input(input_mode, input_image, init_image, init_type, input_video, input_batch, input_folder):
    global busy, input_source, input_init, input_mask # pylint: disable=global-statement
    t0 = time.time()
    busy = False
    selected_input = input_image # default: Image or Kanvas
    if input_mode == 'Video':
        selected_input = input_video
    elif input_mode == 'Batch':
        selected_input = input_batch
    elif input_mode == 'Folder':
        selected_input = input_folder
    size = [gr.update(), gr.update()]
    if selected_input is None:
        input_source = None
        return [gr.Tabs.update(), None, ''] + size

    busy = True
    input_type = type(selected_input)
    input_mask = None
    status = 'Control input | Unknown'
    res = [gr.Tabs.update(selected='out-gallery'), input_mask, status]
    # control inputs
    if isinstance(selected_input, Image.Image): # image via upload -> image
        if input_mode == 'Outpaint':
            masking.opts.invert = True
            selected_input, input_mask = masking.outpaint(input_image=selected_input)
        input_source = [selected_input]
        input_type = 'PIL.Image'
        status = f'Control input | Image | Size {selected_input.width if selected_input else 0}x{selected_input.height if selected_input else 0} | Mode {selected_input.mode if selected_input else "Unknown"}'
        size = [gr.update(value=selected_input.width), gr.update(value=selected_input.height)]
        res = [gr.Tabs.update(selected='out-gallery'), input_mask, status]
    elif isinstance(selected_input, dict) and 'kanvas' in selected_input: # kanvas via js -> kanvas dict
        selected_input, input_mask = process_kanvas(selected_input)
        input_source = [selected_input]
        input_type = 'Kanvas'
        status = f'Control input | Kanvas | Size {selected_input.width if selected_input else 0}x{selected_input.height if selected_input else 0} | Mode {selected_input.mode if selected_input else "Unknown"}'
        if selected_input:
            size = [gr.update(value=selected_input.width), gr.update(value=selected_input.height)]
        res = [gr.Tabs.update(selected='out-gallery'), input_mask, status]
    elif isinstance(selected_input, dict) and 'mask' in selected_input: # inpaint -> dict image+mask
        input_mask = selected_input['mask']
        selected_input = selected_input['image']
        input_source = [selected_input]
        input_type = 'PIL.Image'
        status = f'Control input | Image | Size {selected_input.width if selected_input else 0}x{selected_input.height if selected_input else 0} | Mode {selected_input.mode if selected_input else "Unknown"}'
        res = [gr.Tabs.update(selected='out-gallery'), input_mask, status]
    elif isinstance(selected_input, gr.components.image.Image): # not likely
        input_source = [selected_input.value]
        input_type = 'gr.Image'
        res = [gr.Tabs.update(selected='out-gallery'), input_mask, status]
    elif isinstance(selected_input, str) and os.path.exists(selected_input): # video via upload > tmp filepath to video
        input_source = selected_input
        input_type = 'gr.Video'
        status = get_video(input_source)
        res = [gr.Tabs.update(selected='out-video'), input_mask, status]
    elif isinstance(selected_input, list): # batch or folder via upload -> list of tmp filepaths
        if hasattr(selected_input[0], 'name'):
            input_type = 'tempfiles'
            input_source = [f.name for f in selected_input] # tempfile
        else:
            input_type = 'files'
            input_source = selected_input
        status = f'Control input | Images | Files {len(input_source)}'
        res = [gr.Tabs.update(selected='out-gallery'), input_mask, status]
    else: # unknown
        input_source = None
    if init_type == 0: # Control only
        input_init = None
    elif init_type == 1: # Init image same as control assigned during runtime
        input_init = None
    elif init_type == 2: # Separate init image
        input_init = [init_image]
    t1 = time.time()
    shared.log.debug(f'Select input: type={input_type} source={input_source} init={input_init} mask={input_mask} mode={input_mode} time={t1-t0:.2f}')
    busy = False
    return res + size


def copy_input(mode_from, mode_to, input_image, input_resize, input_inpaint):
    debug_log(f'Control transfter input: from={mode_from} to={mode_to} image={input_image} resize={input_resize} inpaint={input_inpaint}')
    def getimg(ctrl):
        if ctrl is None:
            return None
        return ctrl.get('image', None) if isinstance(ctrl, dict) else ctrl

    if mode_from == mode_to:
        return [gr.update(), gr.update(), gr.update()]
    elif mode_to == 'Image':
        return [getimg(input_resize) if mode_from == 'Outpaint' else getimg(input_inpaint), None, None]
    elif mode_to == 'Inpaint':
        return [None, None, getimg(input_image) if mode_from == 'Image' else getimg(input_resize)]
    elif mode_to == 'Outpaint':
        return [None, getimg(input_image) if mode_from == 'Image' else getimg(input_inpaint), None]
    else:
        shared.log.error(f'Control transfer unknown input: from={mode_from} to={mode_to}')
        return [gr.update(), gr.update(), gr.update()]


def transfer_input(dst):
    return [gr.update(visible=dst=='Image'), gr.update(visible=dst=='Outpaint'), gr.update(visible=dst=='Inpaint'), gr.update(interactive=dst!='Image'), gr.update(interactive=dst!='Inpaint'), gr.update(interactive=dst!='Outpaint')]
