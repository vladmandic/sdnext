import os
import sys
from collections import namedtuple
from pathlib import Path
import threading
import re
import torch
import torch.hub # pylint: disable=ungrouped-imports
import gradio as gr
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from modules import devices, paths, shared, lowvram, errors


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


def category_types():
    return [f.stem for f in Path(interrogator.content_dir).glob('*.txt')]


def download_default_clip_interrogate_categories(content_dir):
    shared.log.info("Interrogate: downloading CLIP categories...")
    tmpdir = f"{content_dir}_tmp"
    cat_types = ["artists", "flavors", "mediums", "movements"]
    try:
        os.makedirs(tmpdir, exist_ok=True)
        for category_type in cat_types:
            torch.hub.download_url_to_file(f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{category_type}.txt", os.path.join(tmpdir, f"{category_type}.txt"))
        os.rename(tmpdir, content_dir)
    except Exception as e:
        errors.display(e, "downloading default CLIP interrogate categories")
    finally:
        if os.path.exists(tmpdir):
            os.removedirs(tmpdir)


class InterrogateModels:
    blip_model = None
    clip_model = None
    clip_preprocess = None
    dtype = None
    running_on_cpu = None

    def __init__(self, content_dir: str = None):
        self.loaded_categories = None
        self.skip_categories = []
        self.content_dir = content_dir or os.path.join(paths.models_path, "interrogate")
        self.running_on_cpu = False

    def categories(self):
        if not os.path.exists(self.content_dir):
            download_default_clip_interrogate_categories(self.content_dir)
        if self.loaded_categories is not None and self.skip_categories == shared.opts.interrogate_clip_skip_categories:
            return self.loaded_categories
        self.loaded_categories = []

        if os.path.exists(self.content_dir):
            self.skip_categories = shared.opts.interrogate_clip_skip_categories
            cat_types = []
            for filename in Path(self.content_dir).glob('*.txt'):
                cat_types.append(filename.stem)
                if filename.stem in self.skip_categories:
                    continue
                m = re_topn.search(filename.stem)
                topn = 1 if m is None else int(m.group(1))
                with open(filename, "r", encoding="utf8") as file:
                    lines = [x.strip() for x in file.readlines()]
                self.loaded_categories.append(Category(name=filename.stem, topn=topn, items=lines))
        return self.loaded_categories

    def create_fake_fairscale(self):
        class FakeFairscale:
            def checkpoint_wrapper(self):
                pass
        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    def load_blip_model(self):
        with load_lock:
            self.create_fake_fairscale()
            from repositories.blip import models # pylint: disable=unused-import
            from repositories.blip.models import blip
            import modules.modelloader as modelloader
            model_path = os.path.join(paths.models_path, "BLIP")
            download_name='model_base_caption_capfilt_large.pth'
            shared.log.debug(f'Interrogate load: module=BLiP model="{download_name}" path="{model_path}"')
            files = modelloader.load_models(
                model_path=model_path,
                model_url='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
                ext_filter=[".pth"],
                download_name=download_name,
            )
            blip_model = blip.blip_decoder(pretrained=files[0], image_size=blip_image_eval_size, vit='base', med_config=os.path.join(paths.paths["BLIP"], "configs", "med_config.json")) # pylint: disable=c-extension-no-member
            blip_model.eval()
            return blip_model

    def load_clip_model(self):
        with load_lock:
            shared.log.debug(f'Interrogate load: module=CLiP model="{clip_model_name}" path="{shared.opts.clip_models_path}"')
            import clip
            if self.running_on_cpu:
                model, preprocess = clip.load(clip_model_name, device="cpu", download_root=shared.opts.clip_models_path)
            else:
                model, preprocess = clip.load(clip_model_name, download_root=shared.opts.clip_models_path)
            model.eval()
            model = model.to(devices.device)
            return model, preprocess

    def load(self):
        if self.blip_model is None:
            self.blip_model = self.load_blip_model()
            if not shared.opts.no_half and not self.running_on_cpu:
                self.blip_model = self.blip_model.half()
        self.blip_model = self.blip_model.to(devices.device)
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = self.load_clip_model()
            if not shared.opts.no_half and not self.running_on_cpu:
                self.clip_model = self.clip_model.half()
        self.clip_model = self.clip_model.to(devices.device)
        self.dtype = next(self.clip_model.parameters()).dtype

    def send_clip_to_ram(self):
        if shared.opts.interrogate_offload:
            if self.clip_model is not None:
                self.clip_model = self.clip_model.to(devices.cpu)

    def send_blip_to_ram(self):
        if shared.opts.interrogate_offload:
            if self.blip_model is not None:
                self.blip_model = self.blip_model.to(devices.cpu)

    def unload(self):
        self.send_clip_to_ram()
        self.send_blip_to_ram()
        devices.torch_gc()

    def rank(self, image_features, text_array, top_count=1):
        import clip
        devices.torch_gc()
        if shared.opts.interrogate_clip_dict_limit != 0:
            text_array = text_array[0:int(shared.opts.interrogate_clip_dict_limit)]
        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize(list(text_array), truncate=True).to(devices.device)
        text_features = self.clip_model.encode_text(text_tokens).type(self.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = torch.zeros((1, len(text_array))).to(devices.device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]
        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(devices.device)
        with devices.inference_context():
            min_length = min(shared.opts.interrogate_clip_min_length, shared.opts.interrogate_clip_max_length)
            max_length = max(shared.opts.interrogate_clip_min_length, shared.opts.interrogate_clip_max_length)
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=shared.opts.interrogate_clip_num_beams, min_length=min_length, max_length=max_length)
        return caption[0]

    def interrogate(self, image):
        res = ""
        shared.state.begin('Interrogate')
        try:
            self.load()
            if isinstance(image, list):
                image = image[0] if len(image) > 0 else None
            if isinstance(image, dict) and 'name' in image:
                image = Image.open(image['name'])
            if image is None:
                return ''
            image = image.convert("RGB")
            caption = self.generate_caption(image)
            res = caption
            clip_image = self.clip_preprocess(image).unsqueeze(0).type(self.dtype).to(devices.device)
            with devices.inference_context(), devices.autocast():
                image_features = self.clip_model.encode_image(clip_image).type(self.dtype)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                for _name, topn, items in self.categories():
                    matches = self.rank(image_features, items, top_count=topn)
                    for match, score in matches:
                        if shared.opts.interrogate_score:
                            res += f", ({match}:{score/100:.2f})"
                        else:
                            res += f", {match}"
        except Exception as e:
            errors.display(e, 'interrogate')
            res += "<error>"
        self.unload()
        shared.state.end()
        return res

# --------- interrrogate ui

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
        ci.caption_model = ci.caption_model.to(devices.cpu)
        ci.clip_model = ci.clip_model.to(devices.cpu)
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
    shared.state.begin('Interrogate')
    try:
        if not shared.native and (shared.cmd_opts.lowvram or shared.cmd_opts.medvram):
            lowvram.send_everything_to_cpu()
            devices.torch_gc()
        if shared.native and shared.sd_loaded:
            from modules.sd_models import apply_balanced_offload # prevent circular import
            apply_balanced_offload(shared.sd_model)
        load_interrogator(clip_model, blip_model)
        image = image.convert('RGB')
        prompt = interrogate(image, mode)
        devices.torch_gc()
    except Exception as e:
        prompt = f"Exception {type(e)}"
        shared.log.error(f'Interrogate: {e}')
        errors.display(e, 'Interrogate')
    shared.state.end()
    return prompt


def interrogate_batch(batch_files, batch_folder, batch_str, clip_model, blip_model, mode, write, append, recursive):
    files = []
    if batch_files is not None:
        files += [f.name for f in batch_files]
    if batch_folder is not None:
        files += [f.name for f in batch_folder]
    if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
        from modules.files_cache import list_files
        files += list(list_files(batch_str, ext_filter=['.png', '.jpg', '.jpeg', '.webp'], recursive=recursive))
    if len(files) == 0:
        shared.log.warning('Interrogate batch: type=clip no images')
        return ''
    shared.state.begin('Interrogate batch')
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
    shared.state.end()
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
    medium_ranks = dict(zip(top_mediums, ci.similarities(image_features, top_mediums)))
    artist_ranks = dict(zip(top_artists, ci.similarities(image_features, top_artists)))
    movement_ranks = dict(zip(top_movements, ci.similarities(image_features, top_movements)))
    trending_ranks = dict(zip(top_trendings, ci.similarities(image_features, top_trendings)))
    flavor_ranks = dict(zip(top_flavors, ci.similarities(image_features, top_flavors)))
    return [
        gr.update(value=medium_ranks, visible=True),
        gr.update(value=artist_ranks, visible=True),
        gr.update(value=movement_ranks, visible=True),
        gr.update(value=trending_ranks, visible=True),
        gr.update(value=flavor_ranks, visible=True),
    ]


interrogator = InterrogateModels()
