from modules import shared, extra_networks, ui_video_vlm


def prepare_prompt(p, init_image, prompt:str, vlm_enhance:bool, vlm_model:str, vlm_system_prompt:str):
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    shared.prompt_styles.apply_styles_to_extra(p)
    p.prompts, p.network_data = extra_networks.parse_prompts([p.prompt])
    extra_networks.activate(p)
    prompt = p.prompts[0]

    new_prompt = ui_video_vlm.enhance_prompt(
        enable=vlm_enhance,
        model=vlm_model,
        image=init_image,
        prompt=prompt,
        system_prompt=vlm_system_prompt,
    )
    if new_prompt is not None and len(new_prompt) > 0:
        prompt = new_prompt
    return prompt
