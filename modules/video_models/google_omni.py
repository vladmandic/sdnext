import io
import os
import base64
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PIL import Image
from modules.logger import log


image_size_buckets = {
    '360p': 640*360,
    '720p': 1280*720,
    '1080p': 1920*1080,
    '4k': 3840*2160,
}
aspect_ratios_buckets = {
    '16:9': 16/9,
    '9:16': 9/16,
}


def google_requirements():
    from installer import install
    install('google-genai==2.22.0')
    # install('pydantic==2.11.7', ignore=True, quiet=True)
    # reload('pydantic', '2.11.7')


def get_size_buckets(width: int, height: int) -> tuple[str, str]:
    aspect_ratio = width / height
    pixel_count = width * height
    closest_size = min(image_size_buckets.items(), key=lambda x: abs(x[1] - pixel_count))[0]
    closest_aspect_ratio = min(aspect_ratios_buckets.items(), key=lambda x: abs(x[1] - aspect_ratio))[0]
    return closest_size, closest_aspect_ratio


class GoogleOmniVideoPipeline:
    def __init__(self, model_name: str):
        self.model = model_name
        self.client = None
        google_requirements()
        log.debug(f'Load model: type=GoogleOmni model="{model_name}"')

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
            if has_api_key and (has_project or has_location):
                # Invalid: can't have both api_key AND project/location
                log.error(f'Cloud: model="{self.model}" API key and project/location are mutually exclusive')
                return None
            elif has_api_key:
                # Vertex AI Express Mode: api_key + vertexai, no project/location
                args = {'api_key': api_key, 'vertexai': True}
            elif has_project and has_location:
                # Standard Vertex AI: project/location, no api_key
                args = {'vertexai': True, 'project': project_id, 'location': location_id}
            else:
                log.error(f'Cloud: model="{self.model}" Vertex AI requires either API key (Express Mode) or project ID + location ID')
                return None
        else:
            # Gemini Developer API: api_key only
            if not has_api_key:
                log.error(f'Cloud: model="{self.model}" API key not provided')
                return None
            args = {'api_key': api_key}

        # Debug logging
        args_log = args.copy()
        if args_log.get('api_key'):
            args_log['api_key'] = '...' + args_log['api_key'][-4:]
        log.debug(f'Cloud: model="{self.model}" args={args_log}')
        return args

    def __call__(self, prompt: list[str], width: int, height: int, image: Image.Image = None):
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt = prompt[0]
        if self.client is None:
            args = self.get_args()
            if args is None:
                return None
            from google import genai # pylint: disable=no-name-in-module
            self.client = genai.Client(**args)

        resolution, aspect_ratio = get_size_buckets(width, height)
        response_format = {
            'type': 'video',
            'aspect_ratio': aspect_ratio,
            'resolution': resolution,
        }
        if image is not None:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG')
            input_content = [
                {'type': 'image', 'data': base64.b64encode(image_bytes.getvalue()).decode('utf-8'), 'mime_type': 'image/jpeg'},
                {'type': 'text', 'text': prompt},
            ]
        else:
            input_content = prompt
        log.debug(f'Cloud: prompt="{prompt}" size={resolution} ar={aspect_ratio} image={image} model="{self.model}" genai={genai.__version__}')

        t0 = time.time()
        try:
            interaction = self.client.interactions.create(
                model=self.model,
                input=input_content,
                response_format=response_format,
            )
        except Exception as e:
            log.error(f'Cloud video: model="{self.model}" {e}')
            return None
        t1 = time.time()
        log.debug(f'Cloud processing: model="{self.model}" elapsed={t1-t0:.2f}')

        try:
            video_bytes = base64.b64decode(interaction.output_video.data)
            return { 'bytes': video_bytes, 'images': [] }
        except Exception as e:
            log.error(f'Cloud download: model="{self.model}" {e}')
            return None


def load_omni(model_name): # pylint: disable=unused-argument
    pipe = GoogleOmniVideoPipeline(model_name = model_name)
    return pipe
