import io
import os
import json
import base64
from datetime import datetime
from PIL import Image
import torch
from safetensors.torch import load_file
import diffusers
import transformers
from modules import shared, devices, errors


class Recipe:
    author = ''
    name = ''
    version = ''
    desc = ''
    hint = ''
    license = ''
    prediction = ''
    thumbnail = None
    base = None
    unet = None
    vae = None
    te1 = None
    te2 = None
    scheduler = 'UniPCMultistepScheduler'
    dtype = torch.float16
    diffusers = True
    safetensors = True
    debug = False
    lora = {
    }
    fuse = 1.0

    def __repr__(self):
        return f'Recipe(name="{self.name}" version="{self.version}" author="{self.author}" desc="{self.desc}" hint="{self.hint}" license="{self.license}" dtype="{self.dtype}" fuse={self.fuse} diffusers={self.diffusers} safetensors={self.safetensors})'


class Test:
    generate = True
    prompt = 'astronaut in a diner drinking coffee with burger and french fries on the table'
    negative = 'ugly, blurry'
    width = 1024
    height = 1024
    guidance = 4
    steps = 20


recipe = Recipe()
test = Test()
pipeline: diffusers.StableDiffusionXLPipeline = None
status = ''


def msg(text, err:bool=False):
    global status # pylint: disable=global-statement
    if err:
        shared.log.error(f'Modules merge: {text}')
    else:
        shared.log.info(f'Modules merge: {text}')
    status += text + '<br>'
    return status


def load_base(override:str=None):
    global pipeline # pylint: disable=global-statement
    fn = override or recipe.base
    yield msg(f'base={fn}')
    if os.path.isfile(fn):
        pipeline = diffusers.StableDiffusionXLPipeline.from_single_file(fn, cache_dir=shared.opts.hfcache_dir, torch_dtype=recipe.dtype, add_watermarker=False)
    elif os.path.isdir(fn):
        pipeline = diffusers.StableDiffusionXLPipeline.from_pretrained(fn, cache_dir=shared.opts.hfcache_dir, torch_dtype=recipe.dtype, add_watermarker=False)
    else:
        yield msg('base: not found')
        return
    pipeline.vae.register_to_config(force_upcast = False)


def load_unet(pipe: diffusers.StableDiffusionXLPipeline, override:str=None):
    if (recipe.unet is None or len(recipe.unet) == 0) and override is None:
        return
    fn = override or recipe.unet
    if not os.path.isabs(fn):
        fn = os.path.join(shared.opts.unet_dir, fn)
    if not fn.endswith('.safetensors'):
        fn += '.safetensors'
    yield msg(f'unet={fn}')
    if recipe.debug:
        yield msg(f'config={pipe.unet.config}')
    try:
        unet = diffusers.UNet2DConditionModel.from_config(pipe.unet.config).to(recipe.dtype)
        state_dict = load_file(fn)
        unet.load_state_dict(state_dict)
        pipe.unet = unet.to(device=devices.device, dtype=recipe.dtype)
    except Exception as e:
        yield msg(f'unet: {e}')


def load_scheduler(pipe: diffusers.StableDiffusionXLPipeline, override:str=None):
    if recipe.scheduler is None and override is None:
        return
    config = pipe.scheduler.config.__dict__
    scheduler = override or recipe.scheduler
    yield msg(f'scheduler={scheduler}')
    if recipe.debug:
        yield msg(f'config={config}')
    try:
        pipe.scheduler = getattr(diffusers, scheduler).from_config(config)
    except Exception as e:
        yield msg(f'scheduler: {e}')



def load_vae(pipe: diffusers.StableDiffusionXLPipeline, override:str=None):
    if (recipe.vae is None or len(recipe.vae) == 0)and override is None:
        return
    fn = override or recipe.vae
    if not os.path.isabs(fn):
        fn = os.path.join(shared.opts.vae_dir, fn)
    if not fn.endswith('.safetensors'):
        fn += '.safetensors'
    try:
        vae = diffusers.AutoencoderKL.from_single_file(fn, cache_dir=shared.opts.hfcache_dir, torch_dtype=recipe.dtype)
        vae.config.force_upcast = False
        vae.config.scaling_factor = 0.13025
        vae.config.sample_size = 1024
        yield msg(f'vae={fn}')
        if recipe.debug:
            yield msg(f'config={pipe.vae.config}')
        pipe.vae = vae.to(device=devices.device, dtype=recipe.dtype)
    except Exception as e:
        yield msg(f'vae: {e}')


def load_te1(pipe: diffusers.StableDiffusionXLPipeline, override:str=None):
    if (recipe.te1 is None or len(recipe.te1) == 0) and override is None:
        return
    config = pipe.text_encoder.config.__dict__
    pretrained_config = transformers.PretrainedConfig.from_dict(config)
    fn = override or recipe.te1
    if not os.path.isabs(fn):
        fn = os.path.join(shared.opts.te_dir, fn)
    if not fn.endswith('.safetensors'):
        fn += '.safetensors'
    yield msg(f'te1={fn}')
    if recipe.debug:
        yield msg(f'config={config}')
    try:
        state_dict = load_file(fn)
        te1 = transformers.CLIPTextModel.from_pretrained(pretrained_model_name_or_path=None, state_dict=state_dict, config=pretrained_config, cache_dir=shared.opts.hfcache_dir)
        pipe.text_encoder = te1.to(device=devices.device, dtype=recipe.dtype)
    except Exception as e:
        yield msg(f'te1: {e}')


def load_te2(pipe: diffusers.StableDiffusionXLPipeline, override:str=None):
    if (recipe.te2 is None or len(recipe.te2) == 0) and override is None:
        return
    config = pipe.text_encoder_2.config.__dict__
    pretrained_config = transformers.PretrainedConfig.from_dict(config)
    fn = override or recipe.te2
    if not os.path.isabs(fn):
        fn = os.path.join(shared.opts.te_dir, fn)
    if not fn.endswith('.safetensors'):
        fn += '.safetensors'
    yield msg(f'te2={recipe.te2}')
    if recipe.debug:
        yield msg(f'config={config}')
    try:
        state_dict = load_file(fn)
        te2 = transformers.CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path=None, state_dict=state_dict, config=pretrained_config, cache_dir=shared.opts.hfcache_dir)
        pipe.text_encoder_2 = te2.to(device=devices.device, dtype=recipe.dtype)
    except Exception as e:
        yield msg(f'te2: {e}')


def load_lora(pipe: diffusers.StableDiffusionXLPipeline, override: dict=None, fuse: float=None):
    if recipe.lora is None and override is None:
        return
    names = []
    pipe.unfuse_lora()
    pipe.unload_lora_weights()
    loras = override or recipe.lora
    for lora, weight in loras.items():
        try:
            fn = lora
            if not os.path.isabs(fn):
                fn = os.path.join(shared.opts.lora_dir, fn)
            if not fn.endswith('.safetensors'):
                fn += '.safetensors'
            yield msg(f'lora={fn} weight={weight} fuse={fuse or recipe.fuse}')
            name = os.path.splitext(os.path.basename(lora))[0].replace('.', '').replace(' ', '').replace('-', '').replace('_', '')
            names.append(name)
            pipe.load_lora_weights(fn, name)
        except Exception as e:
            yield msg(f'lora: {e}')
    if len(names) > 0:
        pipe.set_adapters(adapter_names=names, adapter_weights=list(loras.values()))
        pipe.fuse_lora(adapter_names=names, lora_scale=fuse or recipe.fuse, components=["unet", "text_encoder", "text_encoder_2"])
        pipe.unload_lora_weights()


def test_model(pipe: diffusers.StableDiffusionXLPipeline, fn: str, **kwargs):
    if not test.generate:
        return
    try:
        generator = torch.Generator(devices.device).manual_seed(int(4242))
        args = {
            'prompt': test.prompt,
            'negative_prompt': test.negative,
            'num_inference_steps': test.steps,
            'width': test.width,
            'height': test.height,
            'guidance_scale': test.guidance,
            'generator': generator,
        }
        args.update(kwargs)
        yield msg(f'test={args}')
        image = pipe(**args).images[0]
        yield msg(f'image={fn} {image}')
        image.save(fn)
    except Exception as e:
        yield msg(f'test: {e}')


def get_thumbnail():
    if recipe.thumbnail is None:
        return ''
    image = Image.open(recipe.thumbnail)
    image = image.convert('RGB')
    image.thumbnail((512, 512), resample=Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=50)
    b64encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f'data:image/jpeg;base64,{b64encoded}'


def get_metadata():
    return {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": "stable-diffusion-xl-v1-base",
        "modelspec.implementation": "diffusers",
        "modelspec.title": recipe.name,
        "modelspec.version": recipe.version,
        "modelspec.description": recipe.desc,
        "modelspec.author": recipe.author,
        "modelspec.date": datetime.now().isoformat(timespec='minutes'),
        "modelspec.license": recipe.license,
        "modelspec.usage_hint": recipe.hint,
        "modelspec.prediction_type": recipe.prediction,
        "modelspec.dtype": str(recipe.dtype).split('.')[1],
        "modelspec.hash_sha256": "",
        "modelspec.thumbnail": get_thumbnail(),
        "recipe": json.dumps({
            "base": os.path.basename(recipe.base) if recipe.base else "default",
            "unet": os.path.basename(recipe.unet) if recipe.unet else "default",
            "vae": os.path.basename(recipe.vae) if recipe.vae else "default",
            "te1": os.path.basename(recipe.te1) if recipe.te1 else "default",
            "te2": os.path.basename(recipe.te2) if recipe.te2 else "default",
            "scheduler": recipe.scheduler or "default",
            "lora": [f'{os.path.basename(k)}:{v}' for k, v in recipe.lora.items()],
        }),
    }


def save_model(pipe: diffusers.StableDiffusionXLPipeline):
    author = recipe.author if len(recipe.author) > 0 else 'anonymous'
    folder = os.path.join(shared.opts.diffusers_dir, f'models--{author}--{recipe.name}')
    if len(recipe.version) > 0:
        folder += f'-{recipe.version}'
    if not (recipe.diffusers or recipe.safetensors):
        shared.log.debug(f'Modules merge: type=sdxl {recipe} skipping save')
        return
    try:
        yield msg('save')
        yield msg(f'pretrained={folder}')
        shared.log.info(f'Modules merge save: type=sdxl diffusers="{folder}"')
        pipe.save_pretrained(folder, safe_serialization=True, push_to_hub=False)
        with open(os.path.join(folder, 'vae', 'config.json'), 'r', encoding='utf8') as f:
            vae_config = json.load(f)
            vae_config['force_upcast'] = False
            vae_config['scaling_factor'] = 0.13025
            vae_config['sample_size'] = 1024
        with open(os.path.join(folder, 'vae', 'config.json'), 'w', encoding='utf8') as f:
            json.dump(vae_config, f, indent=2)
        if recipe.safetensors:
            fn = recipe.name
            if len(recipe.version) > 0:
                fn += f'-{recipe.version}'
            if not os.path.isabs(fn):
                fn = os.path.join(shared.opts.ckpt_dir, fn)
            if not fn.endswith('.safetensors'):
                fn += '.safetensors'
            shared.log.info(f'Modules merge save: type=sdxl safetensors="{fn}"')
            yield msg(f'safetensors={fn}')
            from modules.merging import convert_sdxl
            metadata = convert_sdxl.convert(model_path=folder, checkpoint_path=fn, metadata=get_metadata())
            if 'modelspec.thumbnail' in metadata:
                metadata['modelspec.thumbnail'] = f"{metadata['modelspec.thumbnail'].split(',')[0]}:{len(metadata['modelspec.thumbnail'])}" # pylint: disable=use-maxsplit-arg
            yield msg(f'metadata={metadata}')
    except Exception as e:
        shared.log.error(f'Modules merge save: {e}')
        errors.display(e, 'merge')
        yield msg(f'save: {e}')


def merge():
    global pipeline # pylint: disable=global-statement
    yield from load_base()
    if pipeline is None:
        return
    shared.log.info(f'Modules merge: type=sdxl {recipe}')
    pipeline = pipeline.to(device=devices.device, dtype=recipe.dtype)
    yield from load_scheduler(pipeline)
    yield from load_unet(pipeline)
    yield from load_vae(pipeline)
    yield from load_te1(pipeline)
    yield from load_te2(pipeline)
    yield from load_lora(pipeline)
    yield from save_model(pipeline)
    # pipeline = pipeline.to(device=devices.device, dtype=recipe.dtype)
    # test_model(pipeline, '/tmp/merge.png')
