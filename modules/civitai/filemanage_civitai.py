import os
import re
from pathlib import Path
from modules.logger import log


# Map CivitAI model types to shared.opts directory settings and fallback subfolder
# names. 'Text Encoder' is the file type of bundled companion files, 'TextEncoder'
# the model type. Types absent here land in Stable-diffusion.
TYPE_MAP = {
    'Checkpoint': ('ckpt_dir', 'Stable-diffusion'),
    'Text Encoder': ('te_dir', 'Text-encoder'),
    'TextEncoder': ('te_dir', 'Text-encoder'),
    'UNet': ('unet_dir', 'UNET'),
    'CLIP': ('clip_models_path', 'CLIP'),
    'CLIPVision': ('clip_models_path', 'CLIP'),
    'Detection': ('yolo_dir', 'yolo'),
    'TextualInversion': ('embeddings_dir', 'embeddings'),
    'Hypernetwork': ('hypernetwork_dir', 'hypernetworks'),
    'AestheticGradient': ('ckpt_dir', 'Stable-diffusion'),
    'LORA': ('lora_dir', 'Lora'),
    'LoCon': ('lora_dir', 'Lora'),
    'DoRA': ('lora_dir', 'Lora'),
    'Controlnet': ('control_dir', 'control'),
    'Poses': ('ckpt_dir', 'Stable-diffusion'),
    'Wildcards': ('wildcards_dir', 'wildcards'),
    'Workflows': (None, 'workflows'),
    'VAE': ('vae_dir', 'VAE'),
    'MotionModule': (None, 'motion'),
    'Upscaler': ('esrgan_models_path', 'ESRGAN'),
    'Other': ('ckpt_dir', 'Stable-diffusion'),
}

# unmapped types already logged; one warning each per session
warned_types: set[str] = set()

# CivitAI has no type for a standalone transformer, so DiT finetunes ship as
# 'Checkpoint' like full models. Bases listed here are full checkpoints and stay
# in Stable-diffusion; any other base is transformer-only in practice and routes
# to UNET. Exact names: 'Pony' is SDXL but 'Pony V7' is AuraFlow.
FULL_CHECKPOINT_BASES = {'Pony', 'Illustrious', 'NoobAI', 'SVD', 'SVD XT', 'Kolors', 'Stable Cascade'}
FULL_CHECKPOINT_PREFIXES = ('SD 1', 'SD 2', 'SDXL')


def is_full_checkpoint_base(base_model: str) -> bool:
    return base_model in FULL_CHECKPOINT_BASES or base_model.startswith(FULL_CHECKPOINT_PREFIXES)


def get_type_folder(model_type: str, base_model: str = '') -> Path:
    from modules import shared, paths
    # Check for user-configured type folder overrides
    custom_json = getattr(shared.opts, 'civitai_save_type_folders', '') or ''
    if custom_json.strip():
        try:
            import json
            custom = json.loads(custom_json)
            if model_type in custom:
                p = Path(custom[model_type])
                if p.is_absolute():
                    return p
                return Path(paths.models_path) / custom[model_type]
        except Exception as e:
            log.warning(f'CivitAI type folder override parse error: {e}')
    if model_type not in TYPE_MAP and model_type not in warned_types:
        warned_types.add(model_type)
        log.warning(f'CivitAI type unmapped: type="{model_type}" folder="Stable-diffusion"')
    opt_attr, fallback_dir = TYPE_MAP.get(model_type, ('ckpt_dir', 'Stable-diffusion'))
    if model_type == 'Checkpoint' and base_model and not is_full_checkpoint_base(base_model):
        opt_attr, fallback_dir = 'unet_dir', 'UNET'
    if opt_attr:
        configured = getattr(shared.opts, opt_attr, '') or ''
        if configured:
            return Path(configured)
    return Path(paths.models_path) / fallback_dir


def iter_type_roots() -> set[Path]:
    """Every root folder downloads can resolve into, for maintenance sweeps."""
    from modules import shared, paths
    roots = set()
    custom_json = getattr(shared.opts, 'civitai_save_type_folders', '') or ''
    if custom_json.strip():
        try:
            import json
            for folder in json.loads(custom_json).values():
                p = Path(folder)
                roots.add(p if p.is_absolute() else Path(paths.models_path) / folder)
        except Exception:
            pass
    for opt_attr, fallback_dir in set(TYPE_MAP.values()) | {('unet_dir', 'UNET')}:
        configured = (getattr(shared.opts, opt_attr, '') or '') if opt_attr else ''
        roots.add(Path(configured) if configured else Path(paths.models_path) / fallback_dir)
    return {r for r in roots if r.is_dir()}


def path_under(filename: str, root: str | None) -> bool:
    if not root:
        return False
    root = os.path.normcase(os.path.abspath(root)).rstrip(os.sep) + os.sep
    return os.path.normcase(os.path.abspath(filename)).startswith(root)


def loader_kind(filename: str) -> str | None:
    """Model loader that lists this file, judged by the folder it is in."""
    from modules import shared, paths
    ckpt_roots = (getattr(shared.opts, 'ckpt_dir', ''), os.path.join(paths.models_path, 'Stable-diffusion'))
    if path_under(filename, getattr(shared.opts, 'vae_dir', '')) or path_under(filename, os.path.join(paths.models_path, 'VAE')):
        return 'vae'
    if filename.endswith('.vae.safetensors') and any(path_under(filename, root) for root in ckpt_roots):
        return 'vae'
    if path_under(filename, getattr(shared.opts, 'unet_dir', '')):
        return 'unet'
    if path_under(filename, getattr(shared.cmd_opts, 'lora_dir', '')):
        return 'lora'
    if any(path_under(filename, root) for root in ckpt_roots):
        return 'checkpoint'
    return None


def register_download(filename: str):
    """Add a finished download to its loader's list: one file for lora, a folder scan for unet, vae and checkpoint."""
    kind = loader_kind(filename)
    if kind == 'lora':
        from modules.lora.lora_load import add_network
        add_network(filename)
    elif kind == 'unet':
        from modules.sd_unet import refresh_unet_list
        refresh_unet_list()
    elif kind == 'vae':
        from modules.sd_vae import refresh_vae_list
        refresh_vae_list()
    elif kind == 'checkpoint':
        from modules.sd_models import list_models
        list_models()


def hash_cache_title(kind: str | None, filename: str, name: str | None = None) -> str | None:
    """Hash cache key the loader of kind reads for filename, or None when it keeps none."""
    from modules import shared, paths
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]
    if kind == 'lora': # lora_load registers the basename with dots replaced
        return 'lora/' + stem.replace('.', '_')
    if kind == 'unet': # sd_unet keeps the extension on anything but safetensors
        return f"unet/{name or (stem if '.safetensors' in basename else basename)}"
    if kind == 'vae':
        return f'vae/{os.path.abspath(filename)}'
    if kind == 'checkpoint':
        if name is None: # CheckpointInfo matches the folder by string prefix, then drops the extension
            relname = filename
            ckpt_dir = getattr(shared.opts, 'ckpt_dir', '') or ''
            model_path = os.path.abspath(os.path.join(paths.models_path, 'Stable-diffusion'))
            if ckpt_dir and relname.startswith(ckpt_dir):
                relname = os.path.relpath(filename, ckpt_dir)
            elif relname.startswith(model_path):
                relname = os.path.relpath(filename, model_path)
            name = os.path.splitext(relname)[0]
        return f'checkpoint/{name}'
    return None


hash_cache_pruned = False


def loader_root(kind: str) -> str | None:
    """Folder the loader of kind lists, or None for a kind no loader owns."""
    from modules import shared, paths
    if kind == 'vae':
        return getattr(shared.opts, 'vae_dir', '') or os.path.join(paths.models_path, 'VAE')
    if kind == 'unet':
        return getattr(shared.opts, 'unet_dir', '')
    if kind == 'lora':
        return getattr(shared.cmd_opts, 'lora_dir', '')
    if kind == 'checkpoint':
        return getattr(shared.opts, 'ckpt_dir', '') or os.path.join(paths.models_path, 'Stable-diffusion')
    return None


def loader_registry(kind: str) -> dict[str, str] | None:
    """Name to path map of the loader that reads the kind's hash cache keys, or None for a kind no loader owns."""
    if kind == 'vae':
        from modules.sd_vae import vae_dict
        return vae_dict
    if kind == 'unet':
        from modules.sd_unet import unet_dict
        return unet_dict
    if kind == 'lora':
        from modules.lora.lora_load import available_networks
        return {name: entry.filename for name, entry in available_networks.items()}
    if kind == 'checkpoint':
        from modules.sd_checkpoint import checkpoints_list
        return {entry.name: entry.filename for entry in checkpoints_list.values()}
    return None


def hash_cache_path(title: str) -> str | None:
    """Path the loader registry holds for a hash cache key, or None when no loaded registry names it."""
    kind, _, name = title.partition('/')
    if kind == 'vae' and os.path.isabs(name):
        return name
    registry = loader_registry(kind)
    return registry.get(name) if registry else None


def hash_cache_stale(title: str) -> bool:
    """True when the key's loader folder is reachable but the registry no longer lists the file, or lists a path that is gone."""
    kind, _, name = title.partition('/')
    if kind == 'vae' and os.path.isabs(name):
        return os.path.isdir(os.path.dirname(name)) and not os.path.exists(name)
    root = loader_root(kind)
    if not root or not os.path.isdir(root):
        return False
    path = (loader_registry(kind) or {}).get(name)
    return path is None or not os.path.exists(path)


def prune_hash_cache():
    """Drop hash cache entries for files that are gone, once per process."""
    global hash_cache_pruned # pylint: disable=global-statement
    if hash_cache_pruned:
        return
    hash_cache_pruned = True
    from modules import hashes
    gone = [title for title in list(hashes.cache()) if hash_cache_stale(title)]
    for title in gone:
        hashes.cache().pop(title, None)
    if gone:
        hashes.save_cache()
        log.info(f'CivitAI hash cache: pruned={len(gone)} entries without files')


def resolve_save_path(model_type: str, model_name: str = "", base_model: str = "",
                      nsfw: bool = False, creator: str = "", model_id: int = 0,
                      version_id: int = 0, version_name: str = "") -> Path:
    from modules import shared
    base_folder = get_type_folder(model_type, base_model=base_model)
    if not getattr(shared.opts, 'civitai_save_subfolder_enabled', False):
        return base_folder
    template = getattr(shared.opts, 'civitai_save_subfolder', '{{BASEMODEL}}') or ''
    if not template:
        return base_folder
    # Template variable substitution
    replacements = {
        '{{BASEMODEL}}': sanitize_filename(base_model) if base_model else '_unknown',
        '{{MODELNAME}}': sanitize_filename(model_name) if model_name else '',
        '{{CREATOR}}': sanitize_filename(creator) if creator else '_unknown',
        '{{MODELID}}': str(model_id) if model_id else '0',
        '{{VERSIONID}}': str(version_id) if version_id else '0',
        '{{VERSIONNAME}}': sanitize_filename(version_name) if version_name else '',
        '{{NSFW}}': 'nsfw' if nsfw else 'sfw',
        '{{TYPE}}': sanitize_filename(model_type) if model_type else 'other',
    }
    subfolder = template
    for key, value in replacements.items():
        subfolder = subfolder.replace(key, value)
    # Clean up empty path segments
    subfolder = re.sub(r'[/\\]+', lambda _match: os.sep, subfolder) # callable repl: a string repl reads the Windows backslash as an escape
    subfolder = subfolder.strip(os.sep)
    return base_folder / subfolder


def check_exists(folder: Path, filename: str) -> bool:
    return (folder / filename).exists()


def sanitize_filename(name: str) -> str:
    if not name:
        return ''
    # Replace unsafe characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name)
    # Collapse multiple underscores/spaces
    name = re.sub(r'[_ ]{2,}', '_', name)
    name = name.strip(' _.')
    # Truncate to 200 chars (leaving room for extension and path)
    if len(name.encode('utf-8')) > 200:
        while len(name.encode('utf-8')) > 200:
            name = name[:-1]
        name = name.rstrip(' _.')
    return name
