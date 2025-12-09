import io
import os
import time

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from PIL import Image
from installer import install, reload, log


image_size_buckets = {
    '720p': 1280*720,
    '1080p': 1920*1080,
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


class GoogleVeoVideoPipeline():
    def __init__(self, model_name: str):
        self.model = model_name
        self.client = None
        self.config = None
        google_requirements()
        log.debug(f'Load model: type=GoogleVeo model="{model_name}"')

    def txt2vid(self, prompt):
        return self.client.models.generate_videos(
            model=self.model,
            prompt=prompt,
            config=self.config,
        )

    def img2vid(self, prompt, image):
        from google import genai
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        return self.client.models.generate_videos(
            model=self.model,
            prompt=prompt,
            config=self.config,
            image=genai.types.Image(image_bytes=image_bytes.getvalue(), mime_type='image/jpeg'),
        )

    def __call__(self, prompt: list[str], width: int, height: int, image: Image.Image = None, num_frames: int = 4*24):
        from google import genai

        if isinstance(prompt, list) and len(prompt) > 0:
            prompt = prompt[0]
        if self.client is None:
            api_key = os.getenv("GOOGLE_API_KEY", None)
            if api_key is None:
                log.error(f'Cloud: model="{self.model}" GOOGLE_API_KEY environment variable not set')
                return None
            self.client = genai.Client(api_key=api_key, vertexai=False)

        resolution, aspect_ratio = get_size_buckets(width, height)
        duration = num_frames // 24
        if duration < 4:
            duration = 4
        if duration > 8:
            duration = 8
        self.config=genai.types.GenerateVideosConfig(
            # seed=42,
            # fps=24,
            duration_seconds=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            # person_generation='ALLOW_ALL',
            # safety_filter_level='BLOCK_NONE',
            # negative_prompt=None,
            # enhance_prompt=True,
            # generate_audio=True,
        )
        log.debug(f'Cloud: prompt="{prompt}" size={resolution} ar={aspect_ratio} image={image} model="{self.model}" frames={num_frames} duration={duration}')

        operation = None
        try:
            if image is not None:
                operation = self.img2vid(prompt, image)
            else:
                operation = self.txt2vid(prompt)
            while not operation.done:
                log.debug(f"Cloud processing: {operation}")
                time.sleep(10)
                operation = self.client.operations.get(operation)
        except Exception as e:
            log.error(f'Cloud video: model="{self.model}" {operation} {e}')
            return None

        if operation is None or operation.response is None or operation.response.generated_videos is None or len(operation.response.generated_videos) == 0:
            log.error(f'Cloud video: model="{self.model}" no response {operation}')
            return None
        try:
            response: genai.types.GeneratedVideo = operation.response.generated_videos[0]
            self.client.files.download(file=response.video)
            video_bytes = response.video.video_bytes
            return { 'bytes': video_bytes, 'images': [] }
        except Exception as e:
            log.error(f'Cloud download: model="{self.model}" {e}')
            return None


def load_veo(model_name): # pylint: disable=unused-argument
    pipe = GoogleVeoVideoPipeline(model_name = model_name)
    return pipe


if __name__ == "__main__":
    from installer import setup_logging
    setup_logging()
    log.info('test')
    model = GoogleVeoVideoPipeline('veo-3.1-generate-preview')
    img = Image.open('C:\\Users\\mandi\\OneDrive\\Generative\\Samples\\cartoon.png')
    vid = model(['A beautiful young woman walking through the fantasy city'], 1280, 720, image=img)
    if vid is not None:
        with open("veo.mp4", "wb") as f:
            f.write(vid['video'])
