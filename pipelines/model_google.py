import io
import os
import time
from PIL import Image
from installer import install, reload, log


image_size_buckets = {
    '1K': 1024*1024,
    '2K': 2048*1024,
    '4K': 4096*1024,
}
aspect_ratios_buckets = {
    '1:1': 1/1,
    '2:3': 2/3,
    '3:2': 3/2,
    '4:3': 4/3,
    '3:4': 3/4,
    '4:5': 4/5,
    '5:4': 5/4,
    '16:9': 16/9,
    '9:16': 9/16,
    '21:9': 21/9,
    '9:21': 9/21,
}


def google_requirements():
    install('google-genai==1.52.0')
    install('pydantic==2.11.7', ignore=True, quiet=True)
    reload('pydantic', '2.11.7')


def get_size_buckets(width: int, height: int) -> str:
    aspect_ratio = width / height
    closest_aspect_ratio = min(aspect_ratios_buckets.items(), key=lambda x: abs(x[1] - aspect_ratio))[0]
    pixel_count = width * height
    closest_size = min(image_size_buckets.items(), key=lambda x: abs(x[1] - pixel_count))[0]
    closest_aspect_ratio = min(aspect_ratios_buckets.items(), key=lambda x: abs(x[1] - aspect_ratio))[0]
    return closest_size, closest_aspect_ratio


class GoogleNanoBananaPipeline():
    def __init__(self, model_name: str):
        self.model = model_name
        self.client = None
        self.config = None
        google_requirements()
        log.debug(f'Load model: type=NanoBanana model="{model_name}"')

    def txt2img(self, prompt):
        return self.client.models.generate_content(
            model=self.model,
            config=self.config,
            contents=prompt,
        )

    def img2img(self, prompt, image):
        from google import genai
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        return self.client.models.generate_content(
            model=self.model,
            config=self.config,
            contents=[
                genai.types.Part.from_bytes(data=image_bytes.getvalue(), mime_type='image/jpeg'),
                prompt,
            ],
        )

    def get_args(self):
        from modules.shared import opts
        # Use UI settings only - env vars are intentionally ignored to prevent unexpected API charges
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
        from google import genai
        if self.client is None:
            args = self.get_args()
            if args is None:
                return None
            self.client = genai.Client(**args)

        image_size, aspect_ratio = get_size_buckets(width, height)
        if 'gemini-3' in self.model:
            image_config=genai.types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size)
        else:
            image_config=genai.types.ImageConfig(aspect_ratio=aspect_ratio)
        self.config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=image_config
        )
        log.debug(f'Cloud: model="{self.model}" prompt="{prompt}" size={image_size} ar={aspect_ratio} image={image}')
        # log.debug(f'Cloud: config={self.config}')

        try:
            t0 = time.time()
            if image is not None:
                response = self.img2img(prompt, image)
            else:
                response = self.txt2img(prompt)
            t1 = time.time()
            try:
                tokens = response.usage_metadata.total_token_count
            except Exception:
                tokens = 0
            log.debug(f'Cloud: model="{self.model}" tokens={tokens} time={(t1 - t0):.2f}')
        except Exception as e:
            log.error(f'Cloud: model="{self.model}" {e}')
            return None

        image = None
        if getattr(response, 'prompt_feedback', None) is not None:
            log.error(f'Cloud: model="{self.model}" {response.prompt_feedback}')

        parts = []
        try:
            for candidate in response.candidates:
                parts.extend(candidate.content.parts)
        except Exception:
            log.error(f'Cloud: model="{self.model}" no images received')
            return None

        for part in parts:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
        return image


def load_nanobanana(checkpoint_info, diffusers_load_config): # pylint: disable=unused-argument
    pipe = GoogleNanoBananaPipeline(model_name = checkpoint_info.filename)
    return pipe


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    log.info('test')
    model = GoogleNanoBananaPipeline('gemini-3-pro-image-preview')
    img = model(['A beautiful landscape with mountains and a river'], 1024, 1024)
    img.save('test.png')
