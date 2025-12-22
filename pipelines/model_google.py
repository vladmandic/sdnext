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
        api_key = os.getenv("GOOGLE_API_KEY") or opts.google_api_key
        vertex_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if (api_key is None or len(api_key) == 0) and (vertex_credentials is None or len(vertex_credentials) == 0):
            log.error(f'Cloud: model="{self.model}" API key not provided')
            return None
        use_vertexai = (os.getenv("GOOGLE_GENAI_USE_VERTEXAI") is not None) or opts.google_use_vertexai
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or opts.google_project_id
        location_id = os.getenv("GOOGLE_CLOUD_LOCATION") or opts.google_location_id
        args = {
            'api_key': api_key,
            'vertexai': use_vertexai,
            'project': project_id if len(project_id) > 0 else None,
            'location': location_id if len(location_id) > 0 else None,
        }
        args_copy = args.copy()
        args_copy['api_key'] = '...' + args_copy['api_key'][-4:] # last 4 chars
        args_copy['credentials'] = vertex_credentials
        log.debug(f'Cloud: model="{self.model}" args={args_copy}')
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
