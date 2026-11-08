
To load a custom pipeline you just need to pass the `custom_pipeline` argument to `DiffusionPipeline`, as one of the files in `diffusers/examples/community`. Feel free to send a PR with your own pipelines, we will merge them quickly.


## Example Usage

# Lumina-DiMOO
[Project](https://synbol.github.io/Lumina-DiMOO/) / [GitHub](https://github.com/Alpha-VLLM/Lumina-DiMOO/) / [Model](https://huggingface.co/Alpha-VLLM/Lumina-DiMOO)

Lumina-DiMOO is a discrete-diffusion omni-modal foundation model unifying generation and understanding. This implementation integrates a Lumina-DiMOO switch for T2I, I2I editing, and MMU.

## Key features

- **Unified Discrete Diffusion Architecture**: Employs a fully discrete diffusion framework to process inputs and outputs across diverse modalities.
- **Versatile Multimodal Capabilities**: Supports a wide range of multimodal tasks, including text-to-image generation (arbitrary and high-resolution), image-to-image generation (e.g., image editing, subject-driven generation, inpainting), and advanced image understanding.
- **Higher Sampling Efficiency**:  Outperforms previous autoregressive (AR) or hybrid AR-diffusion models with significantly faster sampling. A custom caching mechanism further boosts sampling speed by up to 2×.


## Example Usage

The Lumina-DiMOO pipeline provides three core functions — T2I, I2I, and MMU.
For detailed implementation examples and creative applications, please visit the [GitHub](https://github.com/Alpha-VLLM/Lumina-DiMOO)


### Text-to-Image
**prompt**             |  **image** 
:-------------------------:|:-------------------------:
| "A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort." | <img alt="Orange juice splash forming the word Smile" src="https://github-production-user-asset-6210df.s3.amazonaws.com/73575386/500095460-5490df4b-5ed9-4db7-bbca-3768a32ac840.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251011%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251011T015852Z&X-Amz-Expires=300&X-Amz-Signature=c9dc21e9414ed9069d35c72e21c4226aaea86142ab18dc9571aefb9bbdce9642&X-Amz-SignedHeaders=host">

```python
import torch

from diffusers import VQModel, DiffusionPipeline
from transformers import AutoTokenizer

vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)

pipe = DiffusionPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-DiMOO",
    vqvae=vqvae,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    custom_pipeline="lumina_dimoo",
)
pipe.to("cuda")

prompt = '''A striking photograph of a glass of orange juice on a wooden kitchen table, capturing a playful moment. The orange juice splashes out of the glass and forms the word \"Smile\" in a whimsical, swirling script just above the glass. The background is softly blurred, revealing a cozy, homely kitchen with warm lighting and a sense of comfort.'''

img = pipe(
    prompt=prompt,
    task="text_to_image",
    height=768,
    width=1536,
    num_inference_steps=64,
    cfg_scale=4.0,     
    use_cache=True,
    cache_ratio=0.9, 
    warmup_ratio=0.3,
    refresh_interval=5
).images[0]

img.save("t2i_test_output.png")
```

### Image-to-Image
**prompt**             |  **image_before**   |  **image_after**  
:-------------------------:|:-------------------------:|:-------------------------:
| "A functional wooden printer stand.Nestled next to a brick wall in a bustling city street, it stands firm as pedestrians hustle by, illuminated by the warm glow of vintage street lamps." | <img alt="Wooden printer stand, before" src="https://github-production-user-asset-6210df.s3.amazonaws.com/73575386/500095462-7b451c2f-ec15-4cb9-8a6a-4eab9d2f5760.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251011%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251011T015605Z&X-Amz-Expires=300&X-Amz-Signature=a0e4be32af521b555747b794e660eb0ab116a7ced1cded55448f324fca9fa901&X-Amz-SignedHeaders=host"> | <img alt="Wooden printer stand, after" src="https://github-production-user-asset-6210df.s3.amazonaws.com/73575386/500095459-1d6631ef-8286-4402-a74b-85fb1ea684c3.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251011%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251011T015729Z&X-Amz-Expires=300&X-Amz-Signature=4fb8f2e5eb9645b5e8efb1022e01c329f4a54e793a9e3c4e36fdd357e076964a&X-Amz-SignedHeaders=host"> |

```python
import torch

from diffusers import VQModel, DiffusionPipeline
from transformers import AutoTokenizer
from diffusers.utils import load_image

vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)

pipe = DiffusionPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-DiMOO",
    vqvae=vqvae,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    custom_pipeline="lumina_dimoo",
)
pipe.to("cuda")

input_image = load_image(
    "https://raw.githubusercontent.com/Alpha-VLLM/Lumina-DiMOO/main/examples/example_2.jpg"
).convert("RGB")

prompt = "A functional wooden printer stand.Nestled next to a brick wall in a bustling city street, it stands firm as pedestrians hustle by, illuminated by the warm glow of vintage street lamps."

img = pipe(
    prompt=prompt,
    image=input_image,
    edit_type="depth_control",
    num_inference_steps=64,
    temperature=1.0,
    cfg_scale=2.5,
    cfg_img=4.0,
    task="image_to_image"
).images[0]

img.save("i2i_test_output.png")

```


### Multimodal Understanding
**question**       |   **image**      |   **answer** 
:-------------------------:|:-------------------------:|:-------------------------:
| "Please describe the image." | <img alt="Orange sports car in a showroom" src="https://github-production-user-asset-6210df.s3.amazonaws.com/73575386/500095461-9ae63fc0-b992-4652-9af5-b87be647048f.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251011%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251011T015701Z&X-Amz-Expires=300&X-Amz-Signature=acd3b53d43c81324b2135fde46a0ad25d04dfb71f987456a0b02d1b9af2f2e2e&X-Amz-SignedHeaders=host"> | "The image shows a vibrant orange sports car parked in a showroom. The car has a sleek, aerodynamic design with a prominent front grille and side vents. The body is adorned with black and orange racing stripes, creating a striking contrast against the orange paint. The car is equipped with black alloy wheels and a low-profile body style. The background features a white wall with a large emblem that reads "BREITZEN" and includes a silhouette of a horse and text. The floor is tiled with dark tiles, and the showroom is well-lit, highlighting the car. The overall setting suggests a high-end, possibly luxury, automotive environment."|


```python
import torch
result.images[0].save(f"flux_fill_controlnet_inpaint_depth{timestamp}.jpg")
```


```python
import torch

from diffusers import VQModel, DiffusionPipeline
from transformers import AutoTokenizer
from diffusers.utils import load_image

vqvae = VQModel.from_pretrained("Alpha-VLLM/Lumina-DiMOO", subfolder="vqvae").to(device='cuda', dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Alpha-VLLM/Lumina-DiMOO", trust_remote_code=True)

pipe = DiffusionPipeline.from_pretrained(
    "Alpha-VLLM/Lumina-DiMOO",
    vqvae=vqvae,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    custom_pipeline="lumina_dimoo",
)
pipe.to("cuda")

question = "Please describe the image."

input_image = load_image(
    "https://raw.githubusercontent.com/Alpha-VLLM/Lumina-DiMOO/main/examples/example_8.png"
).convert("RGB")

out = pipe(
    prompt=question,
    image=input_image,
    task="multimodal_understanding",
    num_inference_steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
)

text = getattr(out, "text", out)
with open("mmu_answer.txt", "w", encoding="utf-8") as f:
    f.write(text.strip() + "\n")
```
