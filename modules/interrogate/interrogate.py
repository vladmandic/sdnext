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
    elif shared.opts.interrogate_default_type == 'Tagger':
        shared.log.info(f'Interrogate: type={shared.opts.interrogate_default_type} model="{shared.opts.waifudiffusion_model}"')
        from modules.interrogate import tagger
        prompt = tagger.tag(
            image=image,
            model_name=shared.opts.waifudiffusion_model,
            general_threshold=shared.opts.tagger_threshold,
            character_threshold=shared.opts.waifudiffusion_character_threshold,
            include_rating=shared.opts.tagger_include_rating,
            exclude_tags=shared.opts.tagger_exclude_tags,
            max_tags=shared.opts.tagger_max_tags,
            sort_alpha=shared.opts.tagger_sort_alpha,
            use_spaces=shared.opts.tagger_use_spaces,
            escape_brackets=shared.opts.tagger_escape_brackets,
        )
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
