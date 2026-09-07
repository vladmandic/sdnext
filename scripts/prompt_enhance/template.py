import os
from PIL import Image
from modules.logger import log
from .options import Options
from .helpers import b64, is_cloud_model


debug_enabled = os.environ.get('SD_LLM_DEBUG', None) is not None
debug_log = log.trace if debug_enabled else lambda *args, **kwargs: None


def get_system_prompt(system: str | None, options: Options, nsfw: bool, has_prompt: bool, is_video: bool, is_image: bool) -> str:
    if system is not None and len(system) > 4:
        return system
    if is_video:
        system = options.t2v_prompt if has_prompt else options.t2v_noprompt
    elif is_image:
        system = options.i2i_prompt if has_prompt else options.i2i_noprompt
    else:
        system = options.t2i_prompt if has_prompt else options.t2i_noprompt
    system += options.nsfw_ok if nsfw else options.nsfw_no
    system += options.details_prompt
    system += options.details_format
    debug_log(f'Prompt enhance: system="{system}"')
    return system


def get_text_template(system, prompt, options, nsfw, has_prompt, has_processor, is_video, _image) -> list[dict]:
    system = get_system_prompt(system, options, nsfw, has_prompt, is_video, is_image=False)
    if not has_prompt:
        prompt = 'be creative!'
    if not has_processor:
        chat_template = [
            { "role": "system", "content": system },
            { "role": "user",   "content": prompt },
        ]
    else:
        chat_template = [
            { "role": "system", "content": [
                {"type": "text", "text": system }
            ] },
            { "role": "user",   "content": [
                {"type": "text", "text": prompt},
            ] },
        ]
    return chat_template


def get_image_template(system, prompt, options, nsfw, has_prompt, _has_processor, is_video, image) -> list[dict]:
    system = get_system_prompt(system, options, nsfw, has_prompt, is_video, is_image=True)
    if has_prompt:
        chat_template = [
            { "role": "system", "content": [
                {"type": "text", "text": system }
            ] },
            { "role": "user",   "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": b64(image)}
            ] },
        ]
    else:
        chat_template = [
            { "role": "system", "content": [
                {"type": "text", "text": system }
            ] },
            { "role": "user",   "content": [
                {"type": "image", "image": b64(image)}
            ] },
        ]
    return chat_template


def set_template(
        system: str | None,
        prompt: str | None,
        image: Image.Image | None,
        options: Options,
        model: str,
        nsfw: bool = True,
        has_processor: bool = False,
        module: str | None = None,
) -> list[dict] | str:
    chat_template = []
    has_prompt = prompt is not None and len(prompt) > 4
    has_image = image is not None and isinstance(image, Image.Image)
    is_video = module == 'video'

    debug_log(f'Prompt enhance template: module={module} prompt={has_prompt} image={has_image} video={is_video} model="{model}" nsfw={nsfw} processor={has_processor}')

    if has_image:
        if is_cloud_model(model):
            pass
        elif options.processor is None:
            log.error('Prompt enhance: image not supported by model')
            return prompt if prompt is not None else '' # Return original text part if image cannot be processed

    if has_image:
        chat_template = get_image_template(system, prompt, options, nsfw, has_prompt, has_processor, is_video, image)
    else:
        chat_template = get_text_template(system, prompt, options, nsfw, has_prompt, has_processor, is_video, image)

    return chat_template
