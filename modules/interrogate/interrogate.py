import time
from PIL import Image
from modules import shared


def interrogate(image):
    if isinstance(image, list):
        image = image[0] if len(image) > 0 else None
    if isinstance(image, dict) and 'name' in image:
        image = Image.open(image['name'])
    if image is None:
        shared.log.error('Interrogate: no image provided')
        return ''
    t0 = time.time()
    if shared.opts.interrogate_default_type == 'OpenCLiP':
        shared.log.info(f'Interrogate: type={shared.opts.interrogate_default_type} clip="{shared.opts.interrogate_clip_model}" blip="{shared.opts.interrogate_blip_model}" mode="{shared.opts.interrogate_clip_mode}"')
        from modules.interrogate import openclip
        openclip.load_interrogator(clip_model=shared.opts.interrogate_clip_model, blip_model=shared.opts.interrogate_blip_model)
        openclip.update_interrogate_params()
        prompt = openclip.interrogate(image, mode=shared.opts.interrogate_clip_mode)
        shared.log.debug(f'Interrogate: time={time.time()-t0:.2f} answer="{prompt}"')
        return prompt
    elif shared.opts.interrogate_default_type == 'DeepBooru':
        shared.log.info(f'Interrogate: type={shared.opts.interrogate_default_type}')
        from modules.interrogate import deepbooru
        prompt = deepbooru.model.tag(image)
        shared.log.debug(f'Interrogate: time={time.time()-t0:.2f} answer="{prompt}"')
        return prompt
    elif shared.opts.interrogate_default_type == 'VLM':
        shared.log.info(f'Interrogate: type={shared.opts.interrogate_default_type} vlm="{shared.opts.interrogate_vlm_model}" prompt="{shared.opts.interrogate_vlm_prompt}"')
        from modules.interrogate import vqa
        prompt = vqa.interrogate(image=image, model_name=shared.opts.interrogate_vlm_model, question=shared.opts.interrogate_vlm_prompt, prompt=None, system_prompt=shared.opts.interrogate_vlm_system)
        shared.log.debug(f'Interrogate: time={time.time()-t0:.2f} answer="{prompt}"')
        return prompt
    else:
        shared.log.error(f'Interrogate: type="{shared.opts.interrogate_default_type}" unknown')
        return ''
