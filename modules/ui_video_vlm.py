import gradio as gr
from PIL import Image
from modules import shared


models = [
    'Google Gemma 3 4B',
]
system_prompts = {
    'prefix': 'You are a video director.',
    'desc': 'You will provide a prompt that can be used to create short, creative and dynamic video. Describe the subject of the video, add movement that fits the overall tone of the video, add camera movement.',
    'nsfw_ok': 'NSFW content and nudity are allowed and must be fully described if present. ',
    'nsfw_no': 'NSFW content and nudity are not allowed. ',
    'suffix': 'Movement should be dynamic and creative. Do not specify duration and assume video is short. Avoid slow-motion and prefer faster movements. Output should be a single short paragraph without explanations',
    'example': 'Example: "Short video of beautiful blonde woman in her 20ies wearing a long flowing red dress. She is briskly walking on the beach during sunset and performing a pirouette ending with her hand pointing at the camera as she smiles. Camera is moving around her and zooming to her face. Sun is setting in the background causing changes in colors and shadows to move dynamically."',

    't2v-prompt': 'You are a given short prompt with basic instructions.',
    't2v-noprompt': '',
    'i2v-prompt': 'You are given an image as a starting point and a short prompt with basic instructions.',
    'i2v-noprompt': 'You are given an image as a starting point.',
}


def enhance_prompt(enable:bool, model:str=None, image=None, prompt:str='', system_prompt:str='', nsfw:bool=True):
    from modules.caption import vqa
    if not enable:
        return prompt
    if model is None or len(model) < 4:
        model = models[0]
    if image is not None and not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if prompt is None or len(prompt) < 4:
        prompt = '  '
    if system_prompt is None or len(system_prompt) < 4:
        if image is not None:
            if prompt is not None and len(prompt) > 4:
                core_prompt = system_prompts['i2v-prompt']
            else:
                core_prompt = system_prompts['i2v-noprompt']
        else:
            if prompt is not None and len(prompt) > 4:
                core_prompt = system_prompts['t2v-prompt']
            else:
                core_prompt = system_prompts['t2v-noprompt']
        system_prompt = f"{system_prompts['prefix']} {core_prompt} {system_prompts['desc']}' "
        system_prompt += system_prompts['nsfw_ok'] if nsfw else system_prompts['nsfw_no']
        system_prompt += f" {system_prompts['suffix']} {system_prompts['example']}"
    shared.log.debug(f'Video prompt enhance: model="{model}" image={image} nsfw={nsfw} prompt="{prompt}"')
    answer = vqa.caption(question='', prompt=prompt, system_prompt=system_prompt, image=image, model_name=model, quiet=False)
    shared.log.debug(f'Video prompt enhance: answer="{answer}"')
    return answer


def create_ui(prompt_element:gr.Textbox, image_element:gr.Image):
    with gr.Accordion('Prompt enhance', open=False):
        with gr.Row():
            enable = gr.Checkbox(label='Enable', value=False)
            nsfw = gr.Checkbox(label='NSFW allowed', value=True)
            btn_enhance = gr.Button(value='Enhance now', elem_id='btn_enhance')
        with gr.Row():
            model = gr.Dropdown(label='LLM Model', choices=models, value=models[0])
        with gr.Row():
            system_prompt = gr.Textbox(label='System prompt', placeholder='override system prompt with user-provided prompt', lines=3)
        btn_enhance.click(
            fn=enhance_prompt,
            inputs=[enable, model, image_element, prompt_element, system_prompt, nsfw],
            outputs=prompt_element,
            show_progress='full',
        )
    return enable, model, system_prompt
