import os
import time
import hashlib
import numpy as np
from PIL import Image
from modules.processing_class import StableDiffusionProcessingControl
from modules import shared, images, masking, sd_models
from modules.timer import process as process_timer
from modules.control import util
from modules.control import processors as control_processors


debug = os.environ.get('SD_CONTROL_DEBUG', None) is not None
debug_log = shared.log.trace if debug else lambda *args, **kwargs: None
processors = [
    'None',
    'OpenPose',
    'DWPose',
    'MediaPipe Face',
    'Canny',
    'Edge',
    'LineArt Realistic',
    'LineArt Anime',
    'HED',
    'PidiNet',
    'Midas Depth Hybrid',
    'Leres Depth',
    'Zoe Depth',
    'Marigold Depth',
    'Normal Bae',
    'SegmentAnything',
    'MLSD',
    'Shuffle',
    'DPT Depth Hybrid',
    'GLPN Depth',
    'Depth Anything',
    'Depth Pro',
]


def preprocess_image(
        p:StableDiffusionProcessingControl,
        pipe,
        input_image:Image.Image = None,
        init_image:Image.Image = None,
        input_mask:Image.Image = None,
        input_type:str = 0,
        unit_type:str = 'controlnet',
        active_process:list = [],
        active_model:list = [],
        selected_models:list = [],
        has_models:bool = False,
    ):
    t0 = time.time()
    jobid = shared.state.begin('Preprocess')

    # run resize before
    if (p.resize_mode_before != 0) and (p.resize_name_before != 'None'):
        if (p.selected_scale_tab_before == 1) and (input_image is not None):
            p.width_before, p.height_before = int(input_image.width * p.scale_by_before), int(input_image.height * p.scale_by_before)
        if input_image is not None:
            debug_log(f'Control resize: op=before image={input_image} width={p.width_before} height={p.height_before} mode={p.resize_mode_before} name={p.resize_name_before} context="{p.resize_context_before}"')
            p.init_img_hash = getattr(p, 'init_img_hash', hashlib.sha256(input_image.tobytes()).hexdigest()[0:8]) # pylint: disable=attribute-defined-outside-init
            p.init_img_width = getattr(p, 'init_img_width', input_image.width) # pylint: disable=attribute-defined-outside-init
            p.init_img_height = getattr(p, 'init_img_height', input_image.height) # pylint: disable=attribute-defined-outside-init
            input_image = images.resize_image(p.resize_mode_before, input_image, p.width_before, p.height_before, p.resize_name_before, context=p.resize_context_before)
    if (input_image is not None) and (init_image is not None) and (init_image.size != input_image.size):
        debug_log(f'Control resize init: image={init_image} target={input_image}')
        init_image = images.resize_image(resize_mode=1, im=init_image, width=input_image.width, height=input_image.height)
    if (input_image is not None) and (p.override is not None) and (p.override.size != input_image.size):
        debug_log(f'Control resize override: image={p.override} target={input_image}')
        p.override = images.resize_image(resize_mode=1, im=p.override, width=input_image.width, height=input_image.height)
    if input_image is not None:
        p.width = input_image.width
        p.height = input_image.height
        debug_log(f'Control: input image={input_image}')

    # run masking
    if input_mask is not None:
        p.extra_generation_params["Mask only"] = masking.opts.mask_only if masking.opts.mask_only else None
        p.extra_generation_params["Mask auto"] = masking.opts.auto_mask if masking.opts.auto_mask != 'None' else None
        p.extra_generation_params["Mask invert"] = masking.opts.invert if masking.opts.invert else None
        p.extra_generation_params["Mask blur"] = masking.opts.mask_blur if masking.opts.mask_blur > 0 else None
        p.extra_generation_params["Mask erode"] = masking.opts.mask_erode if masking.opts.mask_erode > 0 else None
        p.extra_generation_params["Mask dilate"] = masking.opts.mask_dilate if masking.opts.mask_dilate > 0 else None
        p.extra_generation_params["Mask model"] = masking.opts.model if masking.opts.model is not None else None
        masked_image = masking.run_mask(input_image=input_image, input_mask=input_mask, return_type='Masked', invert=p.inpainting_mask_invert==1) if input_mask is not None else input_image
    else:
        masked_image = input_image

    # resize mask
    if input_mask is not None and p.resize_mode_mask != 0 and p.resize_name_mask != 'None':
        if p.selected_scale_tab_mask == 1:
            p.width_mask, p.height_mask = int(input_image.width * p.scale_by_mask), int(input_image.height * p.scale_by_mask)
        p.width, p.height = p.width_mask, p.height_mask
        debug_log(f'Control resize: op=mask image={input_mask} width={p.width_mask} height={p.height_mask} mode={p.resize_mode_mask} name={p.resize_name_mask} context="{p.resize_context_mask}"')

    # run image processors
    processed_images = []
    blended_image = None
    for i, process in enumerate(active_process): # list[image]
        debug_log(f'Control: i={i+1} process="{process.processor_id}" input={masked_image} override={process.override}')
        if p.resize_mode_before != 0:
            resize_mode = p.resize_mode_before
        else:
            resize_mode = 3 if shared.opts.control_aspect_ratio else 1
        processed_image = process(
            image_input=masked_image,
            width=p.width,
            height=p.height,
            mode='RGB',
            resize_mode=resize_mode,
            resize_name=p.resize_name_before,
            scale_tab=p.selected_scale_tab_before,
            scale_by=p.scale_by_before,
        )
        if processed_image is not None:
            processed_images.append(processed_image)
        if shared.opts.control_unload_processor and process.processor_id is not None:
            control_processors.config[process.processor_id]['dirty'] = True # to force reload
            process.model = None

    # blend processed images
    debug_log(f'Control processed: {len(processed_images)}')
    if len(processed_images) > 0:
        try:
            if len(p.extra_generation_params["Control process"]) == 0:
                p.extra_generation_params["Control process"] = None
            else:
                p.extra_generation_params["Control process"] = ';'.join([p.processor_id for p in active_process if p.processor_id is not None])
        except Exception:
            pass
        if any(img is None for img in processed_images):
            shared.log.error('Control: one or more processed images are None')
            processed_images = [img for img in processed_images if img is not None]
        if len(processed_images) > 1 and len(active_process) != len(active_model):
            processed_image = [np.array(i) for i in processed_images]
            processed_image = util.blend(processed_image) # blend all processed images into one
            processed_image = Image.fromarray(processed_image)
            blended_image = processed_image
        elif len(processed_images) == 1:
            processed_image = processed_images
            blended_image = processed_image[0]
        else:
            blended_image = [np.array(i) for i in processed_images]
            blended_image = util.blend(blended_image) # blend all processed images into one
            blended_image = Image.fromarray(blended_image)
        if isinstance(selected_models, list) and len(processed_images) == len(selected_models) and len(processed_images) > 0:
            debug_log(f'Control: inputs match: input={len(processed_images)} models={len(selected_models)}')
            p.init_images = processed_images
        elif isinstance(selected_models, list) and len(processed_images) != len(selected_models):
            shared.log.error(f'Control: number of inputs does not match: input={len(processed_images)} models={len(selected_models)}')
        elif selected_models is not None:
            p.init_images = processed_image
    else:
        debug_log('Control processed: using input direct')
        processed_image = input_image

    # conditional assignment
    possible = sd_models.get_call(pipe).keys()
    if unit_type == 'reference' and has_models:
        p.ref_image = p.override or input_image
        p.task_args.pop('image', None)
        p.task_args['ref_image'] = p.ref_image
        debug_log(f'Control: process=None image={p.ref_image}')
        if p.ref_image is None:
            shared.log.error('Control: reference mode without image')
    elif unit_type == 'controlnet' and has_models:
        if input_type == 0: # Control only
            if 'control_image' in possible:
                p.task_args['control_image'] = [p.init_images] if isinstance(p.init_images, Image.Image) else p.init_images
            elif 'image' in possible:
                p.task_args['image'] = [p.init_images] if isinstance(p.init_images, Image.Image) else p.init_images
            if 'control_mode' in possible:
                p.task_args['control_mode'] = getattr(p, 'control_mode', None)
            if 'strength' in possible:
                p.task_args['strength'] = p.denoising_strength
            p.init_images = None
        elif input_type == 1: # Init image same as control
            p.init_images = [p.override or input_image] * max(1, len(active_model))
            if 'inpaint_image' in possible: # flex
                p.task_args['inpaint_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                p.task_args['inpaint_mask'] = Image.new('L', p.task_args['inpaint_image'].size, int(p.denoising_strength * 255))
                p.task_args['control_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                p.task_args['width'] = p.width
                p.task_args['height'] = p.height
            elif 'control_image' in possible:
                p.task_args['control_image'] = p.init_images # switch image and control_image
            if 'control_mode' in possible:
                p.task_args['control_mode'] = getattr(p, 'control_mode', None)
            if 'strength' in possible:
                p.task_args['strength'] = p.denoising_strength
        elif input_type == 2: # Separate init image
            if init_image is None:
                shared.log.warning('Control: separate init image not provided')
                init_image = input_image
            if 'inpaint_image' in possible: # flex
                p.task_args['inpaint_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                p.task_args['inpaint_mask'] = Image.new('L', p.task_args['inpaint_image'].size, int(p.denoising_strength * 255))
                p.task_args['control_image'] = p.init_images[0] if isinstance(p.init_images, list) else p.init_images
                p.task_args['width'] = p.width
                p.task_args['height'] = p.height
            elif 'control_image' in possible:
                p.task_args['control_image'] = p.init_images # switch image and control_image
            if 'control_mode' in possible:
                p.task_args['control_mode'] = getattr(p, 'control_mode', None)
            if 'strength' in possible:
                p.task_args['strength'] = p.denoising_strength
            p.init_images = [init_image] * len(active_model)
        if hasattr(shared.sd_model, 'controlnet') and hasattr(p.task_args, 'control_image') and len(p.task_args['control_image']) > 1 and (shared.sd_model.__class__.__name__ == 'StableDiffusionXLControlNetUnionPipeline'): # special case for controlnet-union
            p.task_args['control_image'] = [[x] for x in p.task_args['control_image']]
            p.task_args['control_mode'] = [[x] for x in p.task_args['control_mode']]

    # determine txt2img, img2img, inpaint pipeline
    if unit_type == 'reference' and has_models: # special case
        p.is_control = True
        shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
    elif not has_models: # run in txt2img/img2img/inpaint mode
        if input_mask is not None:
            p.task_args['strength'] = p.denoising_strength
            p.image_mask = input_mask
            p.init_images = input_image if isinstance(input_image, list) else [input_image]
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING)
        elif processed_image is not None:
            p.init_images = processed_image if isinstance(processed_image, list) else [processed_image]
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE)
        else:
            p.init_hr(p.scale_by, p.resize_name, force=True)
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
    elif has_models: # actual control
        p.is_control = True
        if input_mask is not None:
            p.task_args['strength'] = p.denoising_strength
            p.image_mask = input_mask
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.INPAINTING) # only controlnet supports inpaint
        if hasattr(p, 'init_images') and p.init_images is not None:
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.IMAGE_2_IMAGE) # only controlnet supports img2img
        else:
            shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
            if hasattr(p, 'init_images') and p.init_images is not None and 'image' in possible:
                p.task_args['image'] = p.init_images # need to set explicitly for txt2img
                p.init_images = None
        if unit_type == 'lite':
            if input_type == 0:
                shared.sd_model = sd_models.set_diffuser_pipe(shared.sd_model, sd_models.DiffusersTaskType.TEXT_2_IMAGE)
                shared.sd_model.no_task_switch = True
            elif input_type == 1:
                p.init_images = [input_image]
            elif input_type == 2:
                if init_image is None:
                    shared.log.warning('Control: separate init image not provided')
                    init_image = input_image
                p.init_images = [init_image]

    t1 = time.time()
    process_timer.add('proc', t1-t0)
    shared.state.end(jobid)
    return processed_image, blended_image
