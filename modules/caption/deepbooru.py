import os
import re
import threading
import torch
import numpy as np
from PIL import Image
from modules import modelloader, devices, shared, paths
from modules.logger import log, console

re_special = re.compile(r'([\\()])')
load_lock = threading.Lock()


class DeepDanbooru:
    def __init__(self):
        self.model = None

    def load(self):
        with load_lock:
            if self.model is not None:
                return
            model_path = os.path.join(paths.models_path, "DeepDanbooru")
            log.debug(f'Caption load: module=DeepDanbooru folder="{model_path}"')
            files = modelloader.load_models(
                model_path=model_path,
                model_url='https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt',
                ext_filter=[".pt"],
                download_name='model-resnet_custom_v3.pt',
            )

            from modules.caption.deepbooru_model import DeepDanbooruModel
            self.model = DeepDanbooruModel()
            self.model.load_state_dict(torch.load(files[0], map_location="cpu"))
            self.model.eval()  # required: loaded via torch.load + load_state_dict
            self.model.to(devices.cpu, devices.dtype)

    def start(self):
        self.load()
        self.model.to(devices.device)

    def stop(self):
        if shared.opts.caption_offload:
            self.model.to(devices.cpu)
        devices.torch_gc()

    def tag(self, pil_image, **kwargs):
        self.start()
        res = self.tag_multi(pil_image, **kwargs)
        self.stop()

        return res

    def tag_multi(
        self,
        pil_image,
        general_threshold: float | None = None,
        include_rating: bool | None = None,
        exclude_tags: str | None = None,
        max_tags: int | None = None,
        sort_alpha: bool | None = None,
        use_spaces: bool | None = None,
        escape_brackets: bool | None = None,
    ):
        """Run inference and return formatted tag string.

        Args:
            pil_image: PIL Image to tag
            general_threshold: Threshold for tag scores (0-1)
            include_rating: Whether to include rating tags
            exclude_tags: Comma-separated tags to exclude
            max_tags: Maximum number of tags to return
            sort_alpha: Sort tags alphabetically vs by confidence
            use_spaces: Use spaces instead of underscores
            escape_brackets: Escape parentheses/brackets in tags

        Returns:
            Formatted tag string
        """
        # Use settings defaults if not specified
        general_threshold = general_threshold or shared.opts.tagger_threshold
        include_rating = include_rating if include_rating is not None else shared.opts.tagger_include_rating
        exclude_tags = exclude_tags or shared.opts.tagger_exclude_tags
        max_tags = max_tags or shared.opts.tagger_max_tags
        sort_alpha = sort_alpha if sort_alpha is not None else shared.opts.tagger_sort_alpha
        use_spaces = use_spaces if use_spaces is not None else shared.opts.tagger_use_spaces
        escape_brackets = escape_brackets if escape_brackets is not None else shared.opts.tagger_escape_brackets

        if isinstance(pil_image, list):
            pil_image = pil_image[0] if len(pil_image) > 0 else None
        if isinstance(pil_image, dict) and 'name' in pil_image:
            pil_image = Image.open(pil_image['name'])
        if pil_image is None:
            return ''
        pic = pil_image.resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        with devices.inference_context():
            x = torch.from_numpy(a).to(device=devices.device, dtype=devices.dtype)
            y = self.model(x)[0].detach().float().cpu().numpy()
        probability_dict = {}
        for current, probability in zip(self.model.tags, y, strict=False):
            if probability < general_threshold:
                continue
            if current.startswith("rating:") and not include_rating:
                continue
            probability_dict[current] = probability
        if sort_alpha:
            tags = sorted(probability_dict)
        else:
            tags = [tag for tag, _ in sorted(probability_dict.items(), key=lambda x: -x[1])]
        res = []
        filtertags = {x.strip().replace(' ', '_') for x in exclude_tags.split(",")}
        for filtertag in [x for x in tags if x not in filtertags]:
            probability = probability_dict[filtertag]
            tag_outformat = filtertag
            if use_spaces:
                tag_outformat = tag_outformat.replace('_', ' ')
            if escape_brackets:
                tag_outformat = re.sub(re_special, r'\\\1', tag_outformat)
            if shared.opts.tagger_show_scores:
                tag_outformat = f"({tag_outformat}:{probability:.2f})"
            res.append(tag_outformat)
        if max_tags > 0 and len(res) > max_tags:
            res = res[:max_tags]
        return ", ".join(res)


model = DeepDanbooru()




def get_models() -> list:
    """Return list of available DeepBooru models (just one)."""
    return ["DeepBooru"]


def load_model(model_name: str | None = None) -> bool: # pylint: disable=unused-argument
    """Load the DeepBooru model."""
    try:
        model.load()
        return model.model is not None
    except Exception as e:
        log.error(f'DeepBooru load: {e}')
        return False


def unload_model():
    """Unload the DeepBooru model and free memory."""
    if model.model is not None:
        log.debug('DeepBooru unload')
        model.model.to(devices.cpu)
        model.model = None
        devices.torch_gc(force=True)


def tag(image, **kwargs) -> str:
    """Tag an image using DeepBooru.

    Args:
        image: PIL Image to tag
        **kwargs: Tagger parameters (general_threshold, include_rating, exclude_tags,
                  max_tags, sort_alpha, use_spaces, escape_brackets)

    Returns:
        Formatted tag string
    """
    import time
    t0 = time.time()
    jobid = shared.state.begin('DeepBooru Tag')
    log.info(f'DeepBooru: image_size={image.size if image else None}')

    try:
        result = model.tag(image, **kwargs)
        log.debug(f'DeepBooru: complete time={time.time()-t0:.2f} tags={len(result.split(", ")) if result else 0}')
    except Exception as e:
        result = f"Exception {type(e)}"
        log.error(f'DeepBooru: {e}')

    shared.state.end(jobid)
    return result


def batch(
    model_name: str, # pylint: disable=unused-argument
    batch_files: list,
    batch_folder: str,
    batch_str: str,
    save_output: bool = True,
    save_append: bool = False,
    recursive: bool = False,
    **kwargs
) -> str:
    """Process multiple images in batch mode.

    Args:
        model_name: Model name (ignored, only DeepBooru available)
        batch_files: List of file paths
        batch_folder: Folder path from file picker
        batch_str: Folder path as string
        save_output: Save caption to .txt files
        save_append: Append to existing caption files
        recursive: Recursively process subfolders
        **kwargs: Additional arguments (for interface compatibility)

    Returns:
        Combined tag results
    """
    import os
    import time
    from pathlib import Path
    import rich.progress as rp

    # Load model
    model.load()

    # Collect image files
    image_files = []
    if batch_files is not None:
        image_files += [f.name for f in batch_files]
    if batch_folder is not None:
        image_files += [f.name for f in batch_folder]
    if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        folder_path = Path(batch_str.strip())
        for ext in image_extensions:
            image_files.extend(str(p) for p in (folder_path.rglob(f'*{ext}') if recursive else folder_path.glob(f'*{ext}')))

    if not image_files:
        log.warning('DeepBooru batch: no images found')
        return ''

    t0 = time.time()
    jobid = shared.state.begin('DeepBooru Batch')
    log.info(f'DeepBooru batch: images={len(image_files)} write={save_output} append={save_append} recursive={recursive}')

    results = []
    model.start()

    # Progress bar
    pbar = rp.Progress(rp.TextColumn('[cyan]DeepBooru:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=console)

    with pbar:
        task = pbar.add_task(total=len(image_files), description='starting...')
        for file in image_files:
            file_name = os.path.basename(file)
            pbar.update(task, advance=1, description=file_name)
            try:
                if shared.state.interrupted:
                    log.info('DeepBooru batch: interrupted')
                    break

                image = Image.open(file)
                tags_str = model.tag_multi(image, **kwargs)

                if save_output:
                    from modules.caption import tagger
                    tagger.save_tags_to_file(Path(file), tags_str, save_append)

                results.append(f'{file_name}: {tags_str[:100]}...' if len(tags_str) > 100 else f'{file_name}: {tags_str}')

            except Exception as e:
                log.error(f'DeepBooru batch: file="{file}" error={e}')
                results.append(f'{file_name}: ERROR - {e}')

    model.stop()
    elapsed = time.time() - t0
    log.info(f'DeepBooru batch: complete images={len(results)} time={elapsed:.1f}s')
    shared.state.end(jobid)

    return '\n'.join(results)
