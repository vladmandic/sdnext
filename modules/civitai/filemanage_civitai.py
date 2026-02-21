import os
import re
from pathlib import Path
from modules.logger import log


# Map CivitAI model types to shared.opts directory settings and fallback subfolder names
_TYPE_MAP = {
    'Checkpoint': ('ckpt_dir', 'Stable-diffusion'),
    'TextualInversion': ('embeddings_dir', 'embeddings'),
    'Hypernetwork': ('hypernetwork_dir', 'hypernetworks'),
    'AestheticGradient': ('ckpt_dir', 'Stable-diffusion'),
    'LORA': ('lora_dir', 'Lora'),
    'LoCon': ('lora_dir', 'Lora'),
    'DoRA': ('lora_dir', 'Lora'),
    'Controlnet': ('control_dir', 'control'),
    'Poses': ('ckpt_dir', 'Stable-diffusion'),
    'Wildcards': (None, 'wildcards'),
    'Workflows': (None, 'workflows'),
    'VAE': ('vae_dir', 'VAE'),
    'MotionModule': (None, 'motion'),
    'Upscaler': (None, 'ESRGAN'),
    'Other': ('ckpt_dir', 'Stable-diffusion'),
}


def get_type_folder(model_type: str) -> Path:
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
    opt_attr, fallback_dir = _TYPE_MAP.get(model_type, ('ckpt_dir', 'Stable-diffusion'))
    if opt_attr:
        configured = getattr(shared.opts, opt_attr, '') or ''
        if configured:
            return Path(configured)
    return Path(paths.models_path) / fallback_dir


def resolve_save_path(model_type: str, model_name: str = "", base_model: str = "",
                      nsfw: bool = False, creator: str = "", model_id: int = 0,
                      version_id: int = 0, version_name: str = "") -> Path:
    from modules import shared
    base_folder = get_type_folder(model_type)
    template = getattr(shared.opts, 'civitai_save_subfolder', '') or ''
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
    subfolder = re.sub(r'[/\\]+', os.sep, subfolder)
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
