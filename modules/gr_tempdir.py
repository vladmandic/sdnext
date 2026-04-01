import os
import tempfile
from collections import namedtuple
from pathlib import Path
from PIL import Image, PngImagePlugin
from modules import shared, paths
from modules.logger import log


Savedfile = namedtuple("Savedfile", ["name"])
debug = log.trace if os.environ.get('SD_PATH_DEBUG', None) is not None else lambda *args, **kwargs: None


def register_tmp_file(gradio, filename):
    if hasattr(gradio, 'temp_file_sets'):
        gradio.temp_file_sets[0] = gradio.temp_file_sets[0] | {os.path.abspath(filename)}


def check_tmp_file(gradio, filename):
    ok = False
    if hasattr(gradio, 'temp_file_sets'):
        ok = ok or any(filename in fileset for fileset in gradio.temp_file_sets)
    # Check resolved output paths (base + specific)
    base_samples = shared.opts.outdir_samples
    base_grids = shared.opts.outdir_grids
    resolved_paths = [
        paths.resolve_output_path(base_samples, shared.opts.outdir_txt2img_samples),
        paths.resolve_output_path(base_samples, shared.opts.outdir_img2img_samples),
        paths.resolve_output_path(base_samples, shared.opts.outdir_extras_samples),
        paths.resolve_output_path(base_samples, shared.opts.outdir_control_samples),
        paths.resolve_output_path(base_samples, shared.opts.outdir_save),
        paths.resolve_output_path(base_samples, shared.opts.outdir_video),
        paths.resolve_output_path(base_samples, shared.opts.outdir_init_images),
        paths.resolve_output_path(base_grids, shared.opts.outdir_txt2img_grids),
        paths.resolve_output_path(base_grids, shared.opts.outdir_img2img_grids),
        paths.resolve_output_path(base_grids, shared.opts.outdir_control_grids),
    ]
    # Also check base folders directly if set
    if base_samples:
        resolved_paths.append(base_samples)
    if base_grids:
        resolved_paths.append(base_grids)
    for path in resolved_paths:
        if path:
            try:
                ok = ok or Path(path).resolve() in Path(filename).resolve().parents
            except Exception:
                pass
    return ok


def pil_to_temp_file(self, img: Image, dir: str, format="png") -> str: # pylint: disable=redefined-builtin,unused-argument
    """
    # original gradio implementation
    bytes_data = gr.processing_utils.encode_pil_to_bytes(img, format)
    temp_dir = Path(dir) / self.hash_bytes(bytes_data)
    temp_dir.mkdir(exist_ok=True, parents=True)
    filename = str(temp_dir / f"image.{format}")
    img.save(filename, pnginfo=gr.processing_utils.get_pil_metadata(img))
    """
    folder = dir
    already_saved_as = getattr(img, 'already_saved_as', None)
    exists = os.path.isfile(already_saved_as) if already_saved_as is not None else False
    debug(f'Image lookup: {already_saved_as} exists={exists}')
    if already_saved_as and exists:
        register_tmp_file(shared.demo, already_saved_as)
        file_obj = Savedfile(already_saved_as)
        name = file_obj.name
        debug(f'Image registered: {name}')
        return name
    if shared.opts.temp_dir != "":
        folder = shared.opts.temp_dir
    use_metadata = False
    metadata = PngImagePlugin.PngInfo()
    for key, value in img.info.items():
        if isinstance(key, str) and isinstance(value, str):
            metadata.add_text(key, value)
            use_metadata = True
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        log.debug(f'Created temp folder: path="{folder}"')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png", dir=folder) as tmp:
        name = tmp.name
        img.save(name, pnginfo=(metadata if use_metadata else None))
        img.already_saved_as = name
        size = os.path.getsize(name)
        log.debug(f'Save temp: image="{name}" width={img.width} height={img.height} size={size}')
        shared.state.image_history += 1
    params = ', '.join([f'{k}: {v}' for k, v in img.info.items()])
    params = params[12:] if params.startswith('parameters: ') else params
    if len(params) > 2:
        with open(paths.params_path, "w", encoding="utf8") as file:
            file.write(params)
    return name


# override save to file function so that it also writes PNG info

def on_tmpdir_changed():
    if shared.opts.temp_dir == "":
        return
    register_tmp_file(shared.demo, os.path.join(shared.opts.temp_dir, "x"))


def cleanup_tmpdr():
    temp_dir = shared.opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        temp_dir = os.path.join(paths.temp_dir, "gradio")
    log.debug(f'Temp folder: path="{temp_dir}"')
    if not os.path.isdir(temp_dir):
        return
    for root, _dirs, files in os.walk(temp_dir, topdown=False):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension not in {".png", ".jpg", ".webp", ".jxl"}:
                continue
            filename = os.path.join(root, name)
            os.remove(filename)
