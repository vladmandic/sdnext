import io
import os
import base64
from pathlib import Path
from urllib.parse import unquote
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper
from fastapi.exceptions import HTTPException
from modules import shared, sd_samplers
from modules.logger import log


_upload_store_getter = None
MAX_B64_BYTES = 256 * 1024 * 1024 # base64 expands ~4/3 and the response is built in memory; larger artifacts are fetched by path instead


def register_upload_store(getter_fn):
    global _upload_store_getter # pylint: disable=global-statement
    _upload_store_getter = getter_fn


def validate_sampler_name(name):
    if sd_samplers.is_separator(name):  # dropdown divider, not a selectable sampler
        raise HTTPException(status_code=404, detail="Sampler not found")
    config = sd_samplers.all_samplers_map.get(name, None)
    if config is not None:
        return name
    # accept case-insensitive and alias variants, returning the canonical name so the
    # exact-match lookup in create_sampler resolves instead of silently using the model default
    if isinstance(name, str) and name not in ('', 'None'):
        sampler = sd_samplers.find_sampler(name)
        if sampler is not None:
            return sampler.name
    raise HTTPException(status_code=404, detail="Sampler not found")


def decode_base64_to_image(encoding, quiet=False):
    if encoding is None:
        return None
    if isinstance(encoding, str) and encoding.startswith("upload:"):
        return _resolve_upload_ref(encoding, quiet)
    if encoding.startswith("data:image/"):
        parts = encoding.split(";", 1)
        if len(parts) == 2:
            parts2 = parts[1].split(",", 1)
            encoding = parts2[1] if len(parts2) == 2 else parts2[0]
    try:
        decoded = base64.b64decode(encoding)
        data = io.BytesIO(decoded)
        image = Image.open(data)
        image = image.convert('RGB')
        return image
    except Exception as e:
        log.warning(f'API cannot decode image: {e}')
        # from modules import errors
        # errors.display(e, 'API cannot decode image')
        if not quiet:
            raise HTTPException(status_code=500, detail="Invalid encoded image") from e
        return None


def _resolve_upload_ref(encoding: str, quiet: bool = False):
    ref_id = encoding[len("upload:"):]
    try:
        if _upload_store_getter is None:
            raise RuntimeError("Upload store not registered")
        store = _upload_store_getter()
        image = store.resolve_to_image(ref_id)
        if image is not None:
            return image
    except Exception as e:
        log.warning(f'API cannot resolve upload ref={ref_id}: {e}')
        if not quiet:
            raise HTTPException(status_code=400, detail=f"Upload reference not found: {encoding}") from e
        return None
    if not quiet:
        raise HTTPException(status_code=400, detail=f"Upload reference not found: {encoding}")
    return None


def encode_pil_to_base64(image):
    """
    with io.BytesIO() as output_bytes:
        images.save_image(image, output_bytes, shared.opts.samples_format)
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data)
    """
    if not isinstance(image, Image.Image):
        log.error('API cannot encode image: not a PIL image')
        return ''
    buffered = io.BytesIO()
    save_image(image, fn=buffered, ext=shared.opts.samples_format)
    b64 = base64.b64encode(buffered.getvalue())
    return b64


def encode_file_to_base64(fn: str, max_bytes: int = MAX_B64_BYTES) -> str | None:
    try:
        if fn is None or not os.path.isfile(fn):
            return None
        size = os.path.getsize(fn)
        if size > max_bytes:
            log.warning(f'API cannot encode file: fn="{fn}" size={size} max={max_bytes}')
            return None
        with open(fn, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except Exception as e:
        log.warning(f'API cannot encode file: fn="{fn}" {e}')
        return None


def upscaler_to_index(name: str):
    try:
        return [x.name.lower() for x in shared.sd_upscalers].index(name.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid upscaler, needs to be one of these: {' , '.join([x.name for x in shared.sd_upscalers])}") from e


def save_image(image, fn, ext):
    # actual save
    parameters = image.info.get('parameters', None)
    image_format = Image.registered_extensions()[f'.{ext}']
    if image_format == 'PNG':
        pnginfo_data = PngImagePlugin.PngInfo()
        for k, v in image.info.items():
            pnginfo_data.add_text(k, str(v))
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, pnginfo=pnginfo_data)
    elif image_format == 'JPEG':
        if image.mode == 'RGBA':
            log.warning('Save: RGBA image as JPEG - removed alpha channel')
            image = image.convert("RGB")
        elif image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("L")
        elif image.mode == 'P':
            image = image.convert("RGB")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, exif=exif_bytes)
    elif image_format == 'WEBP':
        if image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, lossless=shared.opts.webp_lossless, exif=exif_bytes)
    elif image_format == 'JXL':
        if image.mode == 'I;16':
            image = image.point(lambda p: p * 0.0038910505836576).convert("RGB")
        elif image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGBA")
        exif_bytes = piexif.dump({ "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") } })
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality, lossless=shared.opts.webp_lossless, exif=exif_bytes)
    else:
        # log.warning(f'Unrecognized image format: {extension} attempting save as {image_format}')
        image.save(fn, format=image_format, quality=shared.opts.jpeg_quality)


def sanitize_filename(filename):
    import unicodedata
    # starting reference: <https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file>
    invalid_chars = (
        "#<>\"'`"                         # ASCII quote and backtick
        "’‚‛\u2018\u2019\u201B"           # smart single quotes and variants # noqa: RUF001
        "\u02BB"                          # modifier letter turned comma
        "\u201C\u201D\u201F"              # smart double quotes and variants
        "|?*^%$\u00A0\u2013\u2014\n\t\r"  # pipes, wildcards, percent, currency, NBSP, dashes, control chars
    )
    invalid_folder = ':'
    invalid_files = ['CON', 'PRN', 'AUX', 'NUL', 'NULL', 'COM0', 'COM1', 'LPT0', 'LPT1']
    invalid_prefix = ', '
    invalid_suffix = '.,_ '
    fn, ext = os.path.splitext(unicodedata.normalize('NFKC', filename))
    fn = fn.strip()
    ext = ext.strip()
    parts = Path(fn).parts
    newparts = []
    for i, part in enumerate(parts):
        part = part.translate({ ord(x): '_' for x in invalid_chars })
        if i > 0 or (len(part) >= 2 and part[1] != invalid_folder): # skip drive, otherwise remove
            part = part.translate({ ord(x): '_' for x in invalid_folder })
        part = part.lstrip(invalid_prefix).rstrip(invalid_suffix)
        if part in invalid_files: # reserved names
            [part := part.replace(word, '_') for word in invalid_files] # pylint: disable=expression-not-assigned
        newparts.append(part)
    fn = str(Path(*newparts))
    fn = fn.replace('  ', ' ').strip()
    max_length = max(256 - len(ext), os.statvfs(__file__).f_namemax - 32 if hasattr(os, 'statvfs') else 256 - len(ext))
    while len(os.path.abspath(fn)) > max_length:
        fn = fn[:-1]
    fn += ext
    return fn


def validate_path(fn: str, allowed_dirs: list[str] | None = None, allowed_folder: bool = False, allowed_file: bool = False):
    if shared.demo is None:
        raise HTTPException(status_code=503, detail="server not ready")
    if not fn.strip():
        raise HTTPException(status_code=400, detail="file path is required")

    allowed = [Path(folder).absolute() for folder in shared.demo.allowed_paths]
    if allowed_dirs is not None:
        allowed.extend([Path(folder).absolute() for folder in allowed_dirs])
    decoded = unquote(fn).replace('%3A', ':')
    sanitized = sanitize_filename(decoded)
    resolved = Path(sanitized).resolve()
    # log.trace(f'API validate: fn="{fn}" sanitized="{sanitized}" resolved="{resolved}" parents={resolved.parents} allowed={allowed}')
    if not any(folder in resolved.parents for folder in allowed):
        raise HTTPException(status_code=403, detail=f"file not allowed: {resolved}")
    if resolved.is_dir():
        if allowed_folder:
            return str(resolved)
        raise HTTPException(status_code=403, detail=f"directory not allowed: {resolved}")
    if not resolved.is_file():
        if allowed_file:
            return str(resolved)
        raise HTTPException(status_code=404, detail=f"file not found: {resolved}")
    return str(resolved)
