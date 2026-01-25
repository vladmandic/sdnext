import os
import time
from collections import namedtuple
import threading
import re
import gradio as gr
from PIL import Image
from modules import devices, shared, errors, sd_models


debug_enabled = os.environ.get('SD_INTERROGATE_DEBUG', None) is not None
debug_log = shared.log.trace if debug_enabled else lambda *args, **kwargs: None


def _apply_blip2_fix(model, processor):
    """Apply compatibility fix for BLIP2 models with newer transformers versions."""
    from transformers import AddedToken
    if not hasattr(model.config, 'num_query_tokens'):
        return
    processor.num_query_tokens = model.config.num_query_tokens
    image_token = AddedToken("<image>", normalized=False, special=True)
    processor.tokenizer.add_tokens([image_token], special_tokens=True)
    model.resize_token_embeddings(len(processor.tokenizer), pad_to_multiple_of=64)
    model.config.image_token_index = len(processor.tokenizer) - 1
    debug_log(f'CLIP load: applied BLIP2 tokenizer fix num_query_tokens={model.config.num_query_tokens}')


caption_models = {
    'blip-base': 'Salesforce/blip-image-captioning-base',
    'blip-large': 'Salesforce/blip-image-captioning-large',
    'blip2-opt-2.7b': 'Salesforce/blip2-opt-2.7b-coco',
    'blip2-opt-6.7b': 'Salesforce/blip2-opt-6.7b',
    'blip2-flip-t5-xl': 'Salesforce/blip2-flan-t5-xl',
    'blip2-flip-t5-xxl': 'Salesforce/blip2-flan-t5-xxl',
}
caption_types = [
    'best',
    'fast',
    'classic',
    'caption',
    'negative',
]
clip_models = []
ci = None
blip_image_eval_size = 384
clip_model_name = 'ViT-L/14'
Category = namedtuple("Category", ["name", "topn", "items"])
re_topn = re.compile(r"\.top(\d+)\.")
load_lock = threading.Lock()


class BatchWriter:
    def __init__(self, folder, mode='w'):
        self.folder = folder
        self.csv = None
        self.file = None
        self.mode = mode

    def add(self, file, prompt):
        txt_file = os.path.splitext(file)[0] + ".txt"
        if self.mode == 'a':
            prompt = '\n' + prompt
        with open(os.path.join(self.folder, txt_file), self.mode, encoding='utf-8') as f:
            f.write(prompt)

    def close(self):
        if self.file is not None:
            self.file.close()


def update_interrogate_params():
    if ci is not None:
        ci.caption_max_length=shared.opts.interrogate_clip_max_length
        ci.chunk_size=shared.opts.interrogate_clip_chunk_size
        ci.flavor_intermediate_count=shared.opts.interrogate_clip_flavor_count
        ci.clip_offload=shared.opts.interrogate_offload
        ci.caption_offload=shared.opts.interrogate_offload


def get_clip_models():
    return clip_models


def refresh_clip_models():
    global clip_models # pylint: disable=global-statement
    import open_clip
    models = sorted(open_clip.list_pretrained())
    shared.log.debug(f'Interrogate: pkg=openclip version={open_clip.__version__} models={len(models)}')
    clip_models = ['/'.join(x) for x in models]
    return clip_models


def load_interrogator(clip_model, blip_model):
    from installer import install
    install('clip_interrogator==0.6.0')
    import clip_interrogator
    clip_interrogator.clip_interrogator.CAPTION_MODELS = caption_models
    global ci # pylint: disable=global-statement
    if ci is None:
        t0 = time.time()
        device = devices.get_optimal_device()
        cache_path = shared.opts.clip_models_path
        shared.log.info(f'CLIP load: clip="{clip_model}" blip="{blip_model}" device={device}')
        debug_log(f'CLIP load: cache_path="{cache_path}" max_length={shared.opts.interrogate_clip_max_length} chunk_size={shared.opts.interrogate_clip_chunk_size} flavor_count={shared.opts.interrogate_clip_flavor_count} offload={shared.opts.interrogate_offload}')
        interrogator_config = clip_interrogator.Config(
            device=device,
            cache_path=cache_path,
            clip_model_name=clip_model,
            caption_model_name=blip_model,
            quiet=True,
            caption_max_length=shared.opts.interrogate_clip_max_length,
            chunk_size=shared.opts.interrogate_clip_chunk_size,
            flavor_intermediate_count=shared.opts.interrogate_clip_flavor_count,
            clip_offload=shared.opts.interrogate_offload,
            caption_offload=shared.opts.interrogate_offload,
        )
        ci = clip_interrogator.Interrogator(interrogator_config)
        if blip_model.startswith('blip2-'):
            _apply_blip2_fix(ci.caption_model, ci.caption_processor)
        shared.log.debug(f'CLIP load: time={time.time()-t0:.2f}')
    elif clip_model != ci.config.clip_model_name or blip_model != ci.config.caption_model_name:
        t0 = time.time()
        if clip_model != ci.config.clip_model_name:
            shared.log.info(f'CLIP load: clip="{clip_model}" reloading')
            debug_log(f'CLIP load: previous clip="{ci.config.clip_model_name}"')
            ci.config.clip_model_name = clip_model
            ci.config.clip_model = None
            ci.load_clip_model()
        if blip_model != ci.config.caption_model_name:
            shared.log.info(f'CLIP load: blip="{blip_model}" reloading')
            debug_log(f'CLIP load: previous blip="{ci.config.caption_model_name}"')
            ci.config.caption_model_name = blip_model
            ci.config.caption_model = None
            ci.load_caption_model()
            if blip_model.startswith('blip2-'):
                _apply_blip2_fix(ci.caption_model, ci.caption_processor)
        shared.log.debug(f'CLIP load: time={time.time()-t0:.2f}')
    else:
        debug_log(f'CLIP: models already loaded clip="{clip_model}" blip="{blip_model}"')


def unload_clip_model():
    if ci is not None and shared.opts.interrogate_offload:
        shared.log.debug('CLIP unload: offloading models to CPU')
        sd_models.move_model(ci.caption_model, devices.cpu)
        sd_models.move_model(ci.clip_model, devices.cpu)
        ci.caption_offloaded = True
        ci.clip_offloaded = True
        devices.torch_gc()
        debug_log('CLIP unload: complete')


def interrogate(image, mode, caption=None):
    if isinstance(image, list):
        image = image[0] if len(image) > 0 else None
    if isinstance(image, dict) and 'name' in image:
        image = Image.open(image['name'])
    if image is None:
        return ''
    image = image.convert("RGB")
    t0 = time.time()
    debug_log(f'CLIP: mode="{mode}" image_size={image.size} caption={caption is not None} min_flavors={shared.opts.interrogate_clip_min_flavors} max_flavors={shared.opts.interrogate_clip_max_flavors}')
    if mode == 'best':
        prompt = ci.interrogate(image, caption=caption, min_flavors=shared.opts.interrogate_clip_min_flavors, max_flavors=shared.opts.interrogate_clip_max_flavors, )
    elif mode == 'caption':
        prompt = ci.generate_caption(image) if caption is None else caption
    elif mode == 'classic':
        prompt = ci.interrogate_classic(image, caption=caption, max_flavors=shared.opts.interrogate_clip_max_flavors)
    elif mode == 'fast':
        prompt = ci.interrogate_fast(image, caption=caption, max_flavors=shared.opts.interrogate_clip_max_flavors)
    elif mode == 'negative':
        prompt = ci.interrogate_negative(image, max_flavors=shared.opts.interrogate_clip_max_flavors)
    else:
        raise RuntimeError(f"Unknown mode {mode}")
    debug_log(f'CLIP: mode="{mode}" time={time.time()-t0:.2f} result="{prompt[:100]}..."' if len(prompt) > 100 else f'CLIP: mode="{mode}" time={time.time()-t0:.2f} result="{prompt}"')
    return prompt


def interrogate_image(image, clip_model, blip_model, mode):
    jobid = shared.state.begin('Interrogate CLiP')
    t0 = time.time()
    shared.log.info(f'CLIP: mode="{mode}" clip="{clip_model}" blip="{blip_model}" image_size={image.size if image else None}')
    try:
        if shared.sd_loaded:
            from modules.sd_models import apply_balanced_offload # prevent circular import
            apply_balanced_offload(shared.sd_model)
            debug_log('CLIP: applied balanced offload to sd_model')
        load_interrogator(clip_model, blip_model)
        image = image.convert('RGB')
        prompt = interrogate(image, mode)
        devices.torch_gc()
        shared.log.debug(f'CLIP: complete time={time.time()-t0:.2f}')
    except Exception as e:
        prompt = f"Exception {type(e)}"
        shared.log.error(f'CLIP: {e}')
        errors.display(e, 'Interrogate')
    shared.state.end(jobid)
    return prompt


def interrogate_batch(batch_files, batch_folder, batch_str, clip_model, blip_model, mode, write, append, recursive):
    files = []
    if batch_files is not None:
        files += [f.name for f in batch_files]
    if batch_folder is not None:
        files += [f.name for f in batch_folder]
    if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
        from modules.files_cache import list_files
        files += list(list_files(batch_str, ext_filter=['.png', '.jpg', '.jpeg', '.webp', '.jxl'], recursive=recursive))
    if len(files) == 0:
        shared.log.warning('CLIP batch: no images found')
        return ''
    t0 = time.time()
    shared.log.info(f'CLIP batch: mode="{mode}" images={len(files)} clip="{clip_model}" blip="{blip_model}" write={write} append={append}')
    debug_log(f'CLIP batch: recursive={recursive} files={files[:5]}{"..." if len(files) > 5 else ""}')
    jobid = shared.state.begin('Interrogate batch')
    prompts = []

    load_interrogator(clip_model, blip_model)
    if write:
        file_mode = 'w' if not append else 'a'
        writer = BatchWriter(os.path.dirname(files[0]), mode=file_mode)
        debug_log(f'CLIP batch: writing to "{os.path.dirname(files[0])}" mode="{file_mode}"')
    import rich.progress as rp
    pbar = rp.Progress(rp.TextColumn('[cyan]Caption:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
    with pbar:
        task = pbar.add_task(total=len(files), description='starting...')
        for file in files:
            pbar.update(task, advance=1, description=file)
            try:
                if shared.state.interrupted:
                    shared.log.info('CLIP batch: interrupted')
                    break
                image = Image.open(file).convert('RGB')
                prompt = interrogate(image, mode)
                prompts.append(prompt)
                if write:
                    writer.add(file, prompt)
            except OSError as e:
                shared.log.error(f'CLIP batch: file="{file}" error={e}')
    if write:
        writer.close()
    ci.config.quiet = False
    unload_clip_model()
    shared.state.end(jobid)
    shared.log.info(f'CLIP batch: complete images={len(prompts)} time={time.time()-t0:.2f}')
    return '\n\n'.join(prompts)


def analyze_image(image, clip_model, blip_model):
    t0 = time.time()
    shared.log.info(f'CLIP analyze: clip="{clip_model}" blip="{blip_model}" image_size={image.size if image else None}')
    load_interrogator(clip_model, blip_model)
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)
    debug_log(f'CLIP analyze: features shape={image_features.shape if hasattr(image_features, "shape") else "unknown"}')
    top_mediums = ci.mediums.rank(image_features, 5)
    top_artists = ci.artists.rank(image_features, 5)
    top_movements = ci.movements.rank(image_features, 5)
    top_trendings = ci.trendings.rank(image_features, 5)
    top_flavors = ci.flavors.rank(image_features, 5)
    medium_ranks = dict(sorted(zip(top_mediums, ci.similarities(image_features, top_mediums)), key=lambda x: x[1], reverse=True))
    artist_ranks = dict(sorted(zip(top_artists, ci.similarities(image_features, top_artists)), key=lambda x: x[1], reverse=True))
    movement_ranks = dict(sorted(zip(top_movements, ci.similarities(image_features, top_movements)), key=lambda x: x[1], reverse=True))
    trending_ranks = dict(sorted(zip(top_trendings, ci.similarities(image_features, top_trendings)), key=lambda x: x[1], reverse=True))
    flavor_ranks = dict(sorted(zip(top_flavors, ci.similarities(image_features, top_flavors)), key=lambda x: x[1], reverse=True))
    shared.log.debug(f'CLIP analyze: complete time={time.time()-t0:.2f}')

    # Format labels as text
    def format_category(name, ranks):
        lines = [f"{name}:"]
        for item, score in ranks.items():
            lines.append(f"  • {item} - {score*100:.1f}%")
        return '\n'.join(lines)

    formatted_text = '\n\n'.join([
        format_category("Medium", medium_ranks),
        format_category("Artist", artist_ranks),
        format_category("Movement", movement_ranks),
        format_category("Trending", trending_ranks),
        format_category("Flavor", flavor_ranks),
    ])

    return [
        gr.update(value=medium_ranks, visible=True),
        gr.update(value=artist_ranks, visible=True),
        gr.update(value=movement_ranks, visible=True),
        gr.update(value=trending_ranks, visible=True),
        gr.update(value=flavor_ranks, visible=True),
        gr.update(value=formatted_text, visible=True),  # New text output for the textbox
    ]
