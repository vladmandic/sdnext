import time
from PIL import Image
from modules import shared


def caption(image):
    if isinstance(image, list):
        image = image[0] if len(image) > 0 else None
    if isinstance(image, dict) and 'name' in image:
        image = Image.open(image['name'])
    if image is None:
        shared.log.error('Caption: no image provided')
        return ''
    t0 = time.time()
    if shared.opts.caption_default_type == 'OpenCLiP':
        shared.log.info(f'Caption: type={shared.opts.caption_default_type} clip="{shared.opts.caption_openclip_model}" blip="{shared.opts.caption_openclip_blip_model}" mode="{shared.opts.caption_openclip_mode}"')
        from modules.caption import openclip
        openclip.load_captioner(clip_model=shared.opts.caption_openclip_model, blip_model=shared.opts.caption_openclip_blip_model)
        openclip.update_caption_params()
        prompt = openclip.caption(image, mode=shared.opts.caption_openclip_mode)
        if shared.opts.caption_offload:
            openclip.unload_clip_model()
        shared.log.debug(f'Caption: time={time.time()-t0:.2f} answer="{prompt}"')
        return prompt
    elif shared.opts.caption_default_type == 'Tagger':
        shared.log.info(f'Caption: type={shared.opts.caption_default_type} model="{shared.opts.waifudiffusion_model}"')
        from modules.caption import tagger
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
        shared.log.debug(f'Caption: time={time.time()-t0:.2f} answer="{prompt}"')
        return prompt
    elif shared.opts.caption_default_type == 'VLM':
        shared.log.info(f'Caption: type={shared.opts.caption_default_type} vlm="{shared.opts.caption_vlm_model}" prompt="{shared.opts.caption_vlm_prompt}"')
        from modules.caption import vqa
        prompt = vqa.caption(image=image, model_name=shared.opts.caption_vlm_model, question=shared.opts.caption_vlm_prompt, prompt=None, system_prompt=shared.opts.caption_vlm_system)
        shared.log.debug(f'Caption: time={time.time()-t0:.2f} answer="{prompt}"')
        return prompt
    else:
        shared.log.error(f'Caption: type="{shared.opts.caption_default_type}" unknown')
        return ''
