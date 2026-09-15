import io
import os
import base64
from modules import shared
from modules.logger import log


debug_enabled = os.environ.get('SD_CAPTION_DEBUG', None) is not None
debug_log = log.trace if debug_enabled else lambda *args, **kwargs: None


class XAIGrokPipeline():
    def __init__(self, model_name: str):
        self.url = 'https://api.x.ai/v1'
        self.model = model_name.split(' (')[0].replace('xai/', '')
        from installer import install
        install('openai')
        from openai import OpenAI # pylint: disable=no-name-in-module
        args = self.get_args()
        if not args:
            return
        self.client = OpenAI(**args)
        log.debug(f'Load model: type=XAIGrok model="{self.model}"')

    def get_args(self):
        from modules.shared import opts
        # Use UI settings only - env vars are intentionally ignored
        api_key = opts.xai_api_key
        has_api_key = api_key and len(api_key) > 0
        if not has_api_key: # Gemini Developer API: api_key only
            log.error(f'Cloud: model="{self.model}" API key not provided')
            return None
        args = {
            'api_key': api_key,
            'base_url': self.url,
        }
        # Debug logging
        args_log = args.copy()
        if args_log.get('api_key'):
            args_log['api_key'] = '...' + args_log['api_key'][-4:]
        log.debug(f'Cloud: model="{self.model}" args={args_log}')
        return args

    def __call__(self, question, image, model, instructions, prefill, thinking, kwargs):
        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        if prefill:
            question += prefill
        debug_log(f'LLM instructions: "{instructions}"')
        debug_log(f'LLM question: "{question}"')
        debug_log(f'LLM image: {image}')
        answer = ''
        temperature = kwargs.get('temperature', 0.0)
        try:
            if image is not None:
                image_data = image.convert('RGB')
                image_bytes = io.BytesIO()
                image_data.save(image_bytes, format='JPEG')
                image_bytes.seek(0)
                content = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image_bytes.getvalue()).decode('utf-8')}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": question,
                    },
                ]
            else:
                content = question
            response = self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role": "system", "content": instructions or shared.opts.caption_vlm_system},
                    {"role": "user", "content": content},
                ],
                stream = False,
                temperature = temperature,
                reasoning_effort = "high" if thinking else "low"
            )
            text = (response.choices[0].message.content or "").strip()
            debug_log(f'Grok response: {response}')
            answer = text
        except Exception as e:
            log.error(f'Grok: {e}')
            answer = f'Error: {e}'
        return answer


ai = None

def predict(question, image, model_name, system_prompt, prefill, thinking, gen_kwargs):
    global ai # pylint: disable=global-statement
    if ai is None:
        ai = XAIGrokPipeline(model_name)
    return ai(question, image, model_name, system_prompt, prefill, thinking, gen_kwargs)
