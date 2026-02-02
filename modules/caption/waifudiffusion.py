# WaifuDiffusion Tagger - ONNX-based anime/illustration tagging
# Based on SmilingWolf's tagger models: https://huggingface.co/SmilingWolf

import os
import re
import time
import threading
import numpy as np
from PIL import Image
from modules import shared, devices, errors


# Debug logging - enable with SD_CAPTION_DEBUG environment variable
debug_enabled = os.environ.get('SD_CAPTION_DEBUG', None) is not None
debug_log = shared.log.trace if debug_enabled else lambda *args, **kwargs: None

re_special = re.compile(r'([\\()])')
load_lock = threading.Lock()

# WaifuDiffusion model repository mappings
WAIFUDIFFUSION_MODELS = {
    # v3 models (latest, recommended)
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-convnext-tagger-v3": "SmilingWolf/wd-convnext-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    # v2 models
    "wd-v1-4-moat-tagger-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-vit-tagger-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
}

# Tag categories from selected_tags.csv
CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9


class WaifuDiffusionTagger:
    """WaifuDiffusion Tagger using ONNX inference."""

    def __init__(self):
        self.session = None
        self.tags = None
        self.tag_categories = None
        self.model_name = None
        self.model_path = None
        self.image_size = 448  # Standard for WD models

    def load(self, model_name: str = None):
        """Load the ONNX model and tags from HuggingFace."""
        import huggingface_hub

        if model_name is None:
            model_name = shared.opts.waifudiffusion_model
        if model_name not in WAIFUDIFFUSION_MODELS:
            shared.log.error(f'WaifuDiffusion: unknown model "{model_name}"')
            return False

        with load_lock:
            if self.session is not None and self.model_name == model_name:
                debug_log(f'WaifuDiffusion: model already loaded model="{model_name}"')
                return True  # Already loaded

            # Unload previous model if different
            if self.model_name != model_name and self.session is not None:
                debug_log(f'WaifuDiffusion: switching model from "{self.model_name}" to "{model_name}"')
                self.unload()

            repo_id = WAIFUDIFFUSION_MODELS[model_name]
            t0 = time.time()
            shared.log.info(f'WaifuDiffusion load: model="{model_name}" repo="{repo_id}"')

            try:
                # Download only ONNX model and tags CSV (skip safetensors/msgpack variants)
                debug_log(f'WaifuDiffusion load: downloading from HuggingFace cache_dir="{shared.opts.hfcache_dir}"')
                self.model_path = huggingface_hub.snapshot_download(
                    repo_id,
                    cache_dir=shared.opts.hfcache_dir,
                    allow_patterns=["model.onnx", "selected_tags.csv"],
                )
                debug_log(f'WaifuDiffusion load: model_path="{self.model_path}"')

                # Load ONNX model
                model_file = os.path.join(self.model_path, "model.onnx")
                if not os.path.exists(model_file):
                    shared.log.error(f'WaifuDiffusion load: model file not found: {model_file}')
                    return False

                import onnxruntime as ort

                debug_log(f'WaifuDiffusion load: onnxruntime version={ort.__version__}')

                self.session = ort.InferenceSession(model_file, providers=devices.onnx)
                self.model_name = model_name

                # Get actual providers used
                actual_providers = self.session.get_providers()
                debug_log(f'WaifuDiffusion load: active providers={actual_providers}')

                # Load tags from CSV
                self._load_tags()

                load_time = time.time() - t0
                shared.log.debug(f'WaifuDiffusion load: time={load_time:.2f} tags={len(self.tags)}')
                debug_log(f'WaifuDiffusion load: input_name={self.session.get_inputs()[0].name} output_name={self.session.get_outputs()[0].name}')
                return True

            except Exception as e:
                shared.log.error(f'WaifuDiffusion load: failed error={e}')
                errors.display(e, 'WaifuDiffusion load')
                self.unload()
                return False

    def _load_tags(self):
        """Load tags and categories from selected_tags.csv."""
        import csv

        csv_path = os.path.join(self.model_path, "selected_tags.csv")
        if not os.path.exists(csv_path):
            shared.log.error(f'WaifuDiffusion load: tags file not found: {csv_path}')
            return

        self.tags = []
        self.tag_categories = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.tags.append(row['name'])
                self.tag_categories.append(int(row['category']))

        # Count tags by category
        category_counts = {}
        for cat in self.tag_categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        debug_log(f'WaifuDiffusion load: tag categories={category_counts}')

    def unload(self):
        """Unload the model and free resources."""
        if self.session is not None:
            shared.log.debug(f'WaifuDiffusion unload: model="{self.model_name}"')
            self.session = None
            self.tags = None
            self.tag_categories = None
            self.model_name = None
            self.model_path = None
            devices.torch_gc(force=True)
            debug_log('WaifuDiffusion unload: complete')
        else:
            debug_log('WaifuDiffusion unload: no model loaded')

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for WaifuDiffusion model input.

        - Resize to 448x448 (standard for WD models)
        - Pad to square with white background
        - Normalize to [0, 1] range
        - BGR channel order (as used by these models)
        """
        original_size = image.size
        original_mode = image.mode

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Pad to square with white background
        w, h = image.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2

        padded = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))

        # Resize to model input size
        if max_dim != self.image_size:
            padded = padded.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(padded, dtype=np.float32)

        # Convert RGB to BGR (model expects BGR)
        img_array = img_array[:, :, ::-1]

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        debug_log(f'WaifuDiffusion preprocess: original_size={original_size} mode={original_mode} padded_size={max_dim} output_shape={img_array.shape}')
        return img_array

    def predict(
        self,
        image: Image.Image,
        general_threshold: float = None,
        character_threshold: float = None,
        include_rating: bool = None,
        exclude_tags: str = None,
        max_tags: int = None,
        sort_alpha: bool = None,
        use_spaces: bool = None,
        escape_brackets: bool = None,
    ) -> str:
        """Run inference and return formatted tag string.

        Args:
            image: PIL Image to tag
            general_threshold: Threshold for general tags (0-1)
            character_threshold: Threshold for character tags (0-1)
            include_rating: Whether to include rating tags
            exclude_tags: Comma-separated tags to exclude
            max_tags: Maximum number of tags to return
            sort_alpha: Sort tags alphabetically vs by confidence
            use_spaces: Use spaces instead of underscores
            escape_brackets: Escape parentheses/brackets in tags

        Returns:
            Formatted tag string
        """
        t0 = time.time()

        # Use settings defaults if not specified
        general_threshold = general_threshold or shared.opts.tagger_threshold
        character_threshold = character_threshold or shared.opts.waifudiffusion_character_threshold
        include_rating = include_rating if include_rating is not None else shared.opts.tagger_include_rating
        exclude_tags = exclude_tags or shared.opts.tagger_exclude_tags
        max_tags = max_tags or shared.opts.tagger_max_tags
        sort_alpha = sort_alpha if sort_alpha is not None else shared.opts.tagger_sort_alpha
        use_spaces = use_spaces if use_spaces is not None else shared.opts.tagger_use_spaces
        escape_brackets = escape_brackets if escape_brackets is not None else shared.opts.tagger_escape_brackets

        debug_log(f'WaifuDiffusion predict: general_threshold={general_threshold} character_threshold={character_threshold} max_tags={max_tags} include_rating={include_rating} sort_alpha={sort_alpha}')

        # Handle input variations
        if isinstance(image, list):
            image = image[0] if len(image) > 0 else None
        if isinstance(image, dict) and 'name' in image:
            image = Image.open(image['name'])
        if image is None:
            shared.log.error('WaifuDiffusion predict: no image provided')
            return ''

        # Load model if needed
        if self.session is None:
            if not self.load():
                return ''

        # Preprocess image
        img_input = self.preprocess_image(image)

        # Run inference
        t_infer = time.time()
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        probs = self.session.run([output_name], {input_name: img_input})[0][0]
        infer_time = time.time() - t_infer
        debug_log(f'WaifuDiffusion predict: inference time={infer_time:.3f}s output_shape={probs.shape}')

        # Build tag list with probabilities
        tag_probs = {}
        exclude_set = {x.strip().replace(' ', '_').lower() for x in exclude_tags.split(',') if x.strip()}
        if exclude_set:
            debug_log(f'WaifuDiffusion predict: exclude_tags={exclude_set}')

        general_count = 0
        character_count = 0
        rating_count = 0

        for i, (tag_name, prob) in enumerate(zip(self.tags, probs)):
            category = self.tag_categories[i]
            tag_lower = tag_name.lower()

            # Skip excluded tags
            if tag_lower in exclude_set:
                continue

            # Apply category-specific thresholds
            if category == CATEGORY_RATING:
                if not include_rating:
                    continue
                # Always include rating if enabled
                tag_probs[tag_name] = float(prob)
                rating_count += 1
            elif category == CATEGORY_CHARACTER:
                if prob >= character_threshold:
                    tag_probs[tag_name] = float(prob)
                    character_count += 1
            elif category == CATEGORY_GENERAL:
                if prob >= general_threshold:
                    tag_probs[tag_name] = float(prob)
                    general_count += 1
            else:
                # Other categories use general threshold
                if prob >= general_threshold:
                    tag_probs[tag_name] = float(prob)

        debug_log(f'WaifuDiffusion predict: matched tags general={general_count} character={character_count} rating={rating_count} total={len(tag_probs)}')

        # Sort tags
        if sort_alpha:
            sorted_tags = sorted(tag_probs.keys())
        else:
            sorted_tags = [t for t, _ in sorted(tag_probs.items(), key=lambda x: -x[1])]

        # Limit number of tags
        if max_tags > 0 and len(sorted_tags) > max_tags:
            sorted_tags = sorted_tags[:max_tags]
            debug_log(f'WaifuDiffusion predict: limited to max_tags={max_tags}')

        # Format output
        result = []
        for tag_name in sorted_tags:
            formatted_tag = tag_name
            if use_spaces:
                formatted_tag = formatted_tag.replace('_', ' ')
            if escape_brackets:
                formatted_tag = re.sub(re_special, r'\\\1', formatted_tag)
            if shared.opts.tagger_show_scores:
                formatted_tag = f"({formatted_tag}:{tag_probs[tag_name]:.2f})"
            result.append(formatted_tag)

        output = ", ".join(result)
        total_time = time.time() - t0
        debug_log(f'WaifuDiffusion predict: complete tags={len(result)} time={total_time:.2f} result="{output[:100]}..."' if len(output) > 100 else f'WaifuDiffusion predict: complete tags={len(result)} time={total_time:.2f} result="{output}"')

        return output

    def tag(self, image: Image.Image, **kwargs) -> str:
        """Alias for predict() to match deepbooru interface."""
        return self.predict(image, **kwargs)


# Global tagger instance
tagger = WaifuDiffusionTagger()




def get_models() -> list:
    """Return list of available WaifuDiffusion model names."""
    return list(WAIFUDIFFUSION_MODELS.keys())


def refresh_models() -> list:
    """Refresh and return list of available models."""
    # For now, just return the static list
    # Could be extended to check for locally cached models
    return get_models()


def load_model(model_name: str = None) -> bool:
    """Load the specified WaifuDiffusion model."""
    return tagger.load(model_name)


def unload_model():
    """Unload the current WaifuDiffusion model."""
    tagger.unload()


def tag(image: Image.Image, model_name: str = None, **kwargs) -> str:
    """Tag an image using WaifuDiffusion tagger.

    Args:
        image: PIL Image to tag
        model_name: Model to use (loads if needed)
        **kwargs: Additional arguments passed to predict()

    Returns:
        Formatted tag string
    """
    t0 = time.time()
    jobid = shared.state.begin('WaifuDiffusion Tag')
    shared.log.info(f'WaifuDiffusion: model="{model_name or tagger.model_name or shared.opts.waifudiffusion_model}" image_size={image.size if image else None}')

    try:
        if model_name and model_name != tagger.model_name:
            tagger.load(model_name)
        result = tagger.predict(image, **kwargs)
        shared.log.debug(f'WaifuDiffusion: complete time={time.time()-t0:.2f} tags={len(result.split(", ")) if result else 0}')
        # Offload model if setting enabled
        if shared.opts.caption_offload:
            tagger.unload()
    except Exception as e:
        result = f"Exception {type(e)}"
        shared.log.error(f'WaifuDiffusion: {e}')
        errors.display(e, 'WaifuDiffusion Tag')

    shared.state.end(jobid)
    return result


def batch(
    model_name: str,
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
        model_name: Model to use
        batch_files: List of file paths
        batch_folder: Folder path from file picker
        batch_str: Folder path as string
        save_output: Save caption to .txt files
        save_append: Append to existing caption files
        recursive: Recursively process subfolders
        **kwargs: Additional arguments passed to predict()

    Returns:
        Combined tag results
    """
    from pathlib import Path

    # Load model
    if model_name:
        tagger.load(model_name)
    elif tagger.session is None:
        tagger.load()

    # Collect image files
    image_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

    # From file picker
    if batch_files:
        for f in batch_files:
            if isinstance(f, dict):
                image_files.append(Path(f['name']))
            elif hasattr(f, 'name'):
                image_files.append(Path(f.name))
            else:
                image_files.append(Path(f))

    # From folder picker
    if batch_folder:
        folder_path = None
        if isinstance(batch_folder, list) and len(batch_folder) > 0:
            f = batch_folder[0]
            if isinstance(f, dict):
                folder_path = Path(f['name']).parent
            elif hasattr(f, 'name'):
                folder_path = Path(f.name).parent
        if folder_path and folder_path.is_dir():
            if recursive:
                for ext in image_extensions:
                    image_files.extend(folder_path.rglob(f'*{ext}'))
            else:
                for ext in image_extensions:
                    image_files.extend(folder_path.glob(f'*{ext}'))

    # From string path
    if batch_str and batch_str.strip():
        folder_path = Path(batch_str.strip())
        if folder_path.is_dir():
            if recursive:
                for ext in image_extensions:
                    image_files.extend(folder_path.rglob(f'*{ext}'))
            else:
                for ext in image_extensions:
                    image_files.extend(folder_path.glob(f'*{ext}'))

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in image_files:
        f_resolved = f.resolve()
        if f_resolved not in seen:
            seen.add(f_resolved)
            unique_files.append(f)
    image_files = unique_files

    if not image_files:
        shared.log.warning('WaifuDiffusion batch: no images found')
        return ''

    t0 = time.time()
    jobid = shared.state.begin('WaifuDiffusion Batch')
    shared.log.info(f'WaifuDiffusion batch: model="{tagger.model_name}" images={len(image_files)} write={save_output} append={save_append} recursive={recursive}')
    debug_log(f'WaifuDiffusion batch: files={[str(f) for f in image_files[:5]]}{"..." if len(image_files) > 5 else ""}')

    results = []

    # Progress bar
    import rich.progress as rp
    pbar = rp.Progress(rp.TextColumn('[cyan]WaifuDiffusion:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)

    with pbar:
        task = pbar.add_task(total=len(image_files), description='starting...')
        for img_path in image_files:
            pbar.update(task, advance=1, description=str(img_path.name))
            try:
                if shared.state.interrupted:
                    shared.log.info('WaifuDiffusion batch: interrupted')
                    break

                image = Image.open(img_path)
                tags_str = tagger.predict(image, **kwargs)

                if save_output:
                    from modules.caption import tagger as tagger_module
                    tagger_module.save_tags_to_file(img_path, tags_str, save_append)

                results.append(f'{img_path.name}: {tags_str[:100]}...' if len(tags_str) > 100 else f'{img_path.name}: {tags_str}')

            except Exception as e:
                shared.log.error(f'WaifuDiffusion batch: file="{img_path}" error={e}')
                results.append(f'{img_path.name}: ERROR - {e}')

    elapsed = time.time() - t0
    shared.log.info(f'WaifuDiffusion batch: complete images={len(results)} time={elapsed:.1f}s')
    shared.state.end(jobid)

    return '\n'.join(results)
