import io
import os
from modules import shared
from modules.logger import log

debug_enabled = os.environ.get('SD_CAPTION_DEBUG', None) is not None
debug_log = log.trace if debug_enabled else lambda *args, **kwargs: None


class GoogleGeminiPipeline():
    def __init__(self, model_name: str):
        self.model = model_name.split(' (')[0]
        from installer import install
        install('google-genai==1.52.0')
        from google import genai # pylint: disable=no-name-in-module
        args = self.get_args()
        self.client = genai.Client(**args)
        log.debug(f'Load model: type=GoogleGemini model="{self.model}"')

    def get_args(self):
        from modules.shared import opts
        # Use UI settings only - env vars are intentionally ignored
        api_key = opts.google_api_key
        project_id = opts.google_project_id
        location_id = opts.google_location_id
        use_vertexai = opts.google_use_vertexai

        has_api_key = api_key and len(api_key) > 0
        has_project = project_id and len(project_id) > 0
        has_location = location_id and len(location_id) > 0

        if use_vertexai:
            if has_api_key and (has_project or has_location): # invalid: can't have both api_key AND project/location
                log.error(f'Cloud: model="{self.model}" API key and project/location are mutually exclusive')
                return None
            elif has_api_key: # vertex AI Express Mode: api_key + vertexai, no project/location
                args = {'api_key': api_key, 'vertexai': True}
            elif has_project and has_location: # standard Vertex AI: project/location, no api_key
                args = {'vertexai': True, 'project': project_id, 'location': location_id}
            else:
                log.error(f'Cloud: model="{self.model}" Vertex AI requires either API key (Express Mode) or project ID + location ID')
                return None
        else:
            if not has_api_key: # Gemini Developer API: api_key only
                log.error(f'Cloud: model="{self.model}" API key not provided')
                return None
            args = {'api_key': api_key}

        # Debug logging
        args_log = args.copy()
        if args_log.get('api_key'):
            args_log['api_key'] = '...' + args_log['api_key'][-4:]
        log.debug(f'Cloud: model="{self.model}" args={args_log}')
        return args

    def __call__(self, question, image, model, instructions, prefill, thinking, kwargs):
        from google.genai import types # pylint: disable=no-name-in-module
        config = {
            'system_instruction': instructions or shared.opts.caption_vlm_system,
            'thinking_config': types.ThinkingConfig(thinking_level="high" if thinking else "low")
        }
        if 'temperature' in kwargs:
            config['temperature'] = kwargs['temperature']
        if 'max_output_tokens' in kwargs:
            config['max_output_tokens'] = kwargs['max_output_tokens']
        debug_log(f'Gemini config: {config}')
        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        if prefill:
            question += prefill
        debug_log(f'Gemini question: "{question}"')

        if image:
            data = io.BytesIO()
            image.save(data, format='JPEG')
            data = data.getvalue()
            contents = [types.Part.from_bytes(data=data, mime_type='image/jpeg'), question]
        else:
            contents = [question]

        answer = ''
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            debug_log(f'Gemini response: {response}')
            answer = response.text
        except Exception as e:
            log.error(f'Gemini: {e}')
            answer = f'Error: {e}'
        return answer


ai = None

def predict(question, image, vqa_model, system_prompt, model_name, prefill, thinking, gen_kwargs):
    global ai # pylint: disable=global-statement
    if ai is None:
        ai = GoogleGeminiPipeline(model_name)
    return ai(question, image, vqa_model, system_prompt, prefill, thinking, gen_kwargs)
