import io
import os
from PIL import Image
from installer import install, reload, log


image_size_buckets = {
    '1M': 1024*1024,
    '2M': 2048*1024,
    '4M': 4096*1024,
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
        install('google-genai')
        install('pydantic==2.11.7', ignore=True, quiet=True)
        reload('pydantic', '2.11.7')
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

    def __call__(self, prompt: list[str], width: int, height: int, image: Image.Image = None):
        from google import genai
        if self.client is None:
            api_key = os.getenv("GOOGLE_API_KEY", None)
            if api_key is None:
                log.error(f'Cloud: model="{self.model}" GOOGLE_API_KEY environment variable not set')
                return None
            self.client = genai.Client(api_key=api_key, vertexai=False)

        image_size, aspect_ratio = get_size_buckets(width, height)
        log.debug(f'Cloud: prompt={prompt} size={image_size} ar={aspect_ratio} image={image} model="{self.model}"')
        self.config=genai.types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=genai.types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size)
        )

        try:
            if image is not None:
                response = self.img2img(prompt, image)
            else:
                response = self.txt2img(prompt)
        except Exception as e:
            log.error(f'Cloud: model="{self.model}" {e}')
            return None

        image = None
        if getattr(response, 'prompt_feedback', None) is not None:
            log.error(f'Cloud: model="{self.model}" {response.prompt_feedback}')
        if not hasattr(response, 'candidates') or (response.candidates is None) or (len(response.candidates) == 0):
            log.error(f'Cloud: model="{self.model}" no images received')
            return None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(io.BytesIO(part.inline_data.data))
        return image


def load_nanobanana(checkpoint_info, diffusers_load_config): # pylint: disable=unused-argument
    pipe = GoogleNanoBananaPipeline(model_name = checkpoint_info.filename)
    return pipe
