from rich import print as rprint
rprint('starting chroma conversion')

import os
import torch
import transformers
import diffusers
import huggingface_hub as hf
from rich.traceback import install as install_traceback


convert = True
test = False
upload = True
input_files = [
    'chroma-unlocked-v48.safetensors',
    'chroma-unlocked-v48-detail-calibrated.safetensors',
    'chroma-unlocked-v46-flash.safetensors',
]
input_folder = '/mnt/models/UNET'
output_folder = '/mnt/models/Diffusers'
cache_dir = '/mnt/models/huggingface'
hf_token = ''
dtype = torch.bfloat16
device = torch.device('cuda')


install_traceback(show_locals=False)
rprint(f'torch={torch.__version__} diffusers={diffusers.__version__} transformers={transformers.__version__}')
for input_file in input_files:
    input_basename = os.path.splitext(input_file)[0]
    input_model = os.path.join(input_folder, input_file)
    output_model = os.path.join(output_folder, input_basename)

    if convert:
        rprint(f'load transformer: {input_model}')
        transformer = diffusers.ChromaTransformer2DModel.from_single_file(
            input_model,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(device)

        rprint(f'load text-encoder')
        text_encoder = transformers.T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="text_encoder_2",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(device)

        rprint(f'load tokenizer')
        tokenizer = transformers.T5Tokenizer.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="tokenizer_2",
            cache_dir=cache_dir,
        )

        rprint(f'load pipeline')
        pipe = diffusers.ChromaPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        ).to(device)


        rprint(f'save pipeline: {output_model}')
        pipe.save_pretrained(
            output_model,
        )

    if test:
        rprint('test load')
        pipe = diffusers.ChromaPipeline.from_pretrained(
            output_model,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )

    if upload:
        rprint('hf login')
        hf.logout()
        hf.login(token=hf_token, add_to_git_credential=False, write_permission=True)
        rprint('upload model')
        pipe.push_to_hub(
            input_basename,
            private=False,
            token=hf_token,
        )
    
    pipe = None

rprint(f'done')
