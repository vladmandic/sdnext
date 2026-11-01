"""Private storage and validation for Chat image artifacts."""

import base64
import json
import mimetypes
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules import images, infotext, paths, shared
from modules.chat import db
from modules.logger import log


TEXT_SUFFIXES = {'.txt', '.md'}
TEXT_LIMIT = 1024 * 1024
COPY_CHUNK_SIZE = 1024 * 1024


def storage_root() -> Path:
    configured_root = shared.opts.chat_upload_dir
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
    else:
        root = Path(shared.opts.temp_dir or paths.temp_dir).expanduser().resolve() / 'sdnext-chat-artifacts'
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except OSError:
        pass
    return root


def safe_display_name(filename: str | None) -> str:
    name = str(filename or 'upload').replace('\\', '/').rsplit('/', 1)[-1].strip()
    name = ''.join(character for character in name if character.isprintable() and character not in '\r\n\x00')
    return name[:255] or 'upload'


def _new_storage_path(filename: str) -> Path:
    root = storage_root()
    prompt = Path(safe_display_name(filename)).stem
    params = SimpleNamespace(
        n_iter=1,
        batch_size=1,
        iteration=0,
        batch_index=0,
        seed=0,
        width=0,
        height=0,
        prompt=prompt,
        styles=[],
        sampler_name='',
        steps=0,
        cfg_scale=0,
        clip_skip=1,
        denoising_strength=0,
    )
    namegen = images.FilenameGenerator(params, seed=0, prompt=prompt, image=None)
    pattern = shared.opts.samples_filename_pattern or '[seq]-[prompt_words]'
    if '[seq]' not in pattern:
        pattern = f'[seq]-{pattern}'
    basename = Path(namegen.apply(pattern)).name
    suffix = Path(safe_display_name(filename)).suffix
    destination = Path(namegen.sanitize(str(root / f'{basename}{suffix}'), base=str(root)))
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return Path(namegen.sequence(str(destination)))


def _image_extension(image_format: str) -> str:
    media_type = Image.MIME.get(image_format)
    extension = mimetypes.guess_extension(media_type, strict=False) if media_type else None
    if extension:
        return extension
    extensions = sorted(
        suffix for suffix, registered_format in Image.registered_extensions().items()
        if registered_format == image_format
    )
    if extensions:
        return extensions[0]
    raise ValueError('Unsupported image format')


def _add_image_extension(path: Path, extension: str) -> Path:
    destination = path.with_suffix(extension)
    if destination == path:
        return path
    path.replace(destination)
    return destination


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_allowed_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip() or shared.demo is None:
        raise ValueError('File is unavailable or not allowed')
    source = Path(value.strip()).expanduser()
    roots = []
    for folder in shared.demo.allowed_paths:
        try:
            root = Path(folder).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if root not in roots:
            roots.append(root)
    candidates = [source] if source.is_absolute() else [root / source for root in roots]
    matches = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            if not stat.S_ISREG(resolved.stat().st_mode):
                continue
            if any(_inside(resolved, root) for root in roots) and os.access(resolved, os.R_OK):
                matches.add(resolved)
        except (OSError, RuntimeError):
            continue
    if len(matches) > 1:
        raise ValueError('Relative file path is ambiguous; use an absolute path')
    if not matches:
        raise ValueError('File is unavailable or not allowed')
    return matches.pop()


def inspect_file(path: str | Path, filename: str) -> dict:
    path = Path(path)
    display_name = safe_display_name(filename)
    if Path(display_name).suffix.lower() in TEXT_SUFFIXES:
        try:
            if path.stat().st_size <= TEXT_LIMIT:
                content = path.read_bytes().decode('utf-8-sig')
                if '\x00' not in content:
                    return {'kind': 'text', 'content': content, 'display_name': display_name}
        except (OSError, UnicodeDecodeError):
            pass
        if path.stat().st_size > TEXT_LIMIT:
            raise ValueError('Text files must not exceed 1 MB')
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
        with Image.open(path) as image:
            image.load()
        media_type = Image.MIME.get(image_format or '') or mimetypes.types_map.get(f'.{(image_format or "").lower()}')
        if not media_type or not media_type.startswith('image/'):
            raise ValueError('Unsupported image format')
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc) == 'Unsupported image format':
            raise
        raise ValueError('File contents are not a supported image or UTF-8 text file') from exc
    return {
        'kind': 'image',
        'media_type': media_type,
        'display_name': display_name,
        'extension': _image_extension(image_format or ''),
    }


def copy_allowed_file(value: str) -> dict:
    source = resolve_allowed_path(value)
    destination = _new_storage_path(source.name)
    try:
        with source.open('rb') as source_file, destination.open('xb') as destination_file:
            shutil.copyfileobj(source_file, destination_file, length=COPY_CHUNK_SIZE)
        imported = inspect_file(destination, source.name)
        if imported['kind'] == 'image':
            destination = _add_image_extension(destination, imported['extension'])
        imported['path'] = str(destination)
        return imported
    except Exception:
        destination.unlink(missing_ok=True)
        raise


async def stage_upload(upload, filename: str | None = None) -> dict:
    display_name = safe_display_name(filename or getattr(upload, 'filename', None))
    destination = _new_storage_path(display_name)
    try:
        with destination.open('xb') as output:
            while True:
                chunk = await upload.read(COPY_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
        imported = inspect_file(destination, display_name)
        if imported['kind'] == 'image':
            destination = _add_image_extension(destination, imported['extension'])
        imported['path'] = str(destination)
        return imported
    except Exception:
        destination.unlink(missing_ok=True)
        raise


def create_image_artifact(session_id: str, message_id: str, path: str, display_name: str, role='input', source_artifact_id=None) -> dict:
    inspected = inspect_file(path, display_name)
    if inspected['kind'] != 'image':
        raise ValueError('Artifact is not a supported image')
    return db.create_artifact(
        session_id,
        inspected['media_type'],
        inspected['display_name'],
        role,
        str(path),
        message_id=message_id,
        source_artifact_id=source_artifact_id,
    )


def create_generated_artifact(
    session_id: str,
    message_id: str,
    image_data: str,
    display_name: str,
    saved_path: str | None = None,
    source_artifact_id: str | None = None,
) -> dict:
    if saved_path and Path(saved_path).is_file():
        try:
            return create_image_artifact(
                session_id, message_id, saved_path, display_name, role='result', source_artifact_id=source_artifact_id,
            )
        except (OSError, ValueError):
            pass
    destination = _new_storage_path(display_name)
    try:
        payload = image_data.split(',', 1)[1] if image_data.startswith('data:') and ',' in image_data else image_data
        raw = base64.b64decode(payload, validate=True)
        with destination.open('xb') as output:
            output.write(raw)
        inspected = inspect_file(destination, display_name)
        destination = _add_image_extension(destination, inspected['extension'])
        artifact = create_image_artifact(
            session_id, message_id, str(destination), display_name, role='result', source_artifact_id=source_artifact_id,
        )
        return artifact
    except Exception:
        destination.unlink(missing_ok=True)
        raise ValueError('Generated image could not be stored') from None


def public_artifact(artifact: dict) -> dict:
    path = Path(artifact['storage_ref'])
    return {
        'id': artifact['id'],
        'message_id': artifact.get('message_id'),
        'media_type': artifact['media_type'],
        'display_name': artifact['display_name'],
        'role': artifact['role'],
        'source_artifact_id': artifact.get('source_artifact_id'),
        'source_deleted': bool(artifact.get('source_deleted', 0)),
        'created_at': artifact['created_at'],
        'available': path.is_file(),
    }


def read_image_base64(artifact: dict) -> str:
    path = Path(artifact['storage_ref'])
    if not path.is_file():
        raise ValueError('Selected image is no longer available')
    with path.open('rb') as source:
        return base64.b64encode(source.read()).decode('ascii')


def delete_storage_file(artifact: dict) -> None:
    path = Path(artifact['storage_ref'])
    try:
        if path.is_file() and stat.S_ISREG(path.stat().st_mode):
            path.unlink()
    except OSError as exc:
        raise ValueError('Image file could not be deleted') from exc


def image_metadata(artifact: dict) -> str:
    path = Path(artifact['storage_ref'])
    if not path.is_file():
        log.error(f'Chat: image metadata failed: file unavailable: {path}')
        return 'Selected image is no longer available'
    try:
        with Image.open(path) as image:
            info, items = images.read_info_from_image(image)
    except Exception as exc:
        log.error(f'Chat: image metadata read failed: {type(exc).__name__}: {exc}')
        return 'Image metadata could not be read'
    if not info and not items:
        return 'No image metadata found'
    try:
        parsed = infotext.parse(info) if info else {}
    except Exception as exc:
        log.error(f'Chat: image metadata parse failed: {type(exc).__name__}: {exc}')
        parsed = {}
    details = []
    if parsed:
        details.append('Parsed:\n' + json.dumps(parsed, indent=2, ensure_ascii=False, default=str))
    if items:
        safe_items = {
            str(key): value.decode('utf-8', 'replace') if isinstance(value, bytes) else value
            for key, value in items.items()
            if key not in {'parameters', 'UserComment', 'exif', 'icc_profile', 'photoshop', 'xmp'}
            and key not in parsed
            and isinstance(value, (str, int, float, bool, bytes))
        }
        if safe_items:
            details.append('Raw JSON:\n' + json.dumps(safe_items, indent=2, ensure_ascii=False, default=str))
    return '\n'.join(details) if details else 'No readable image metadata found'
