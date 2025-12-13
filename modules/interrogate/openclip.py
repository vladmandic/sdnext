import os
from collections import namedtuple
import threading
import re
import gradio as gr
from PIL import Image
from core import MODELDATA
from modules import devices, paths, shared, errors, sd_models


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
        shared.log.debug(f'Interrogate load: clip="{clip_model}" blip="{blip_model}"')
        interrogator_config = clip_interrogator.Config(
            device=devices.get_optimal_device(),
            cache_path=os.path.join(paths.models_path, 'Interrogator'),
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
    elif clip_model != ci.config.clip_model_name or blip_model != ci.config.caption_model_name:
        ci.config.clip_model_name = clip_model
        ci.config.clip_model = None
        ci.load_clip_model()
        ci.config.caption_model_name = blip_model
        ci.config.caption_model = None
        ci.load_caption_model()


def unload_clip_model():
    if ci is not None and shared.opts.interrogate_offload:
        sd_models.move_model(ci.caption_model, devices.cpu)
        sd_models.move_model(ci.clip_model, devices.cpu)
        ci.caption_offloaded = True
        ci.clip_offloaded = True
        devices.torch_gc()


def interrogate(image, mode, caption=None):
    if isinstance(image, list):
        image = image[0] if len(image) > 0 else None
    if isinstance(image, dict) and 'name' in image:
        image = Image.open(image['name'])
    if image is None:
        return ''
    image = image.convert("RGB")
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
    return prompt


def interrogate_image(image, clip_model, blip_model, mode):
    jobid = shared.state.begin('Interrogate CLiP')
    try:
        if MODELDATA.sd_loaded:
            from modules.sd_models import apply_balanced_offload # prevent circular import
            apply_balanced_offload(MODELDATA.sd_model)
        load_interrogator(clip_model, blip_model)
        image = image.convert('RGB')
        prompt = interrogate(image, mode)
        devices.torch_gc()
    except Exception as e:
        prompt = f"Exception {type(e)}"
        shared.log.error(f'Interrogate: {e}')
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
        shared.log.warning('Interrogate batch: type=clip no images')
        return ''
    jobid = shared.state.begin('Interrogate batch')
    prompts = []

    load_interrogator(clip_model, blip_model)
    if write:
        file_mode = 'w' if not append else 'a'
        writer = BatchWriter(os.path.dirname(files[0]), mode=file_mode)
    import rich.progress as rp
    pbar = rp.Progress(rp.TextColumn('[cyan]Caption:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
    with pbar:
        task = pbar.add_task(total=len(files), description='starting...')
        for file in files:
            pbar.update(task, advance=1, description=file)
            try:
                if shared.state.interrupted:
                    break
                image = Image.open(file).convert('RGB')
                prompt = interrogate(image, mode)
                prompts.append(prompt)
                if write:
                    writer.add(file, prompt)
            except OSError as e:
                shared.log.error(f'Interrogate batch: {e}')
    if write:
        writer.close()
    ci.config.quiet = False
    unload_clip_model()
    shared.state.end(jobid)
    return '\n\n'.join(prompts)


def analyze_image(image, clip_model, blip_model):
    load_interrogator(clip_model, blip_model)
    image = image.convert('RGB')
    image_features = ci.image_to_features(image)
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
