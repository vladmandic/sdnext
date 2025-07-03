import sys
sys.path.append("./")

# import torch
# from torchvision import transforms
from meissonic.transformer import Transformer2DModel as TransformerMeissonic
from meissonic.pipeline import Pipeline as PipelineMeissonic
from meissonic.scheduler import Scheduler as MeissonicScheduler
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import VQModel

device = 'cuda'
model_path = 'MeissonFlow/Meissonic'
cache_dir = '/mnt/models/Diffusers'

# diffusers_load_config['variant'] = fp16

model = TransformerMeissonic.from_pretrained(model_path, subfolder="transformer", cache_dir=cache_dir)
vq_model = VQModel.from_pretrained(model_path, subfolder="vqvae", cache_dir=cache_dir)
# text_encoder = CLIPTextModelWithProjection.from_pretrained(model_path,subfolder="text_encoder",)
text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=cache_dir)
tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
scheduler = MeissonicScheduler.from_pretrained(model_path, subfolder="scheduler")
pipe = PipelineMeissonic(vq_model, tokenizer=tokenizer, text_encoder=text_encoder, transformer=model, scheduler=scheduler)
pipe = pipe.to(device)

steps = 64
guidance_scale = 9
resolution = 1024 
negative = "worst quality, low quality, low res, blurry, distortion, watermark, logo, signature, text, jpeg artifacts, signature, sketch, duplicate, ugly, identifying mark"
prompt = "Beautiful young woman posing on a lake with snow covered mountains in the background"
image = pipe(prompt=prompt, negative_prompt=negative, height=resolution, width=resolution, guidance_scale=guidance_scale, num_inference_steps=steps).images[0]
image.save('/tmp/meissonic.png')
