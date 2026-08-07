import os
import time
from datetime import datetime
from pathlib import Path
from modules import paths
from modules.logger import log
from modules.shared import opts, max_workers
from modules.modelstats import stat


class Location:
    name: str
    folders: list[str]
    paths: list[Path]
    nfiles: int = 0
    nfolders: int = 0
    nsymlinks: int = 0
    nerrors: int = 0
    type: str = ''
    size: int = 0
    time: float = 0.0
    mtime: datetime = datetime.fromtimestamp(0)

    def __init__(self, name: str | None, folders: str | list[str], what: str = ''):
        self.type = what
        if isinstance(folders, str):
            folders = [folders]
        self.name = name if name is not None else ', '.join(folders)
        self.folders = folders
        self.paths = [Path(f).resolve(strict=False) for f in self.folders if f is not None and f != '']

    def __repr__(self):
        return f'Location(type={self.type} name="{self.name}" folders={self.folders} size={self.size/1024/1024:.3f} mtime="{self.mtime}" files={self.nfiles} folders={self.nfolders} symlinks={self.nsymlinks} errors={self.nerrors} time={self.time:.3f})'

    def dict(self):
        return {
            'name': self.name,
            'type': self.type,
            'folders': self.folders,
            'paths': [str(p) for p in self.paths],
            'size': self.size,
            'mtime': self.mtime.timestamp(),
            'nfiles': self.nfiles,
            'nfolders': self.nfolders,
            'nsymlinks': self.nsymlinks,
            'nerrors': self.nerrors,
            'time': self.time,
        }

def get_all_locations(types: list[str] | None = []) -> list[Location]:
    locations = []
    if types is None or 'All' in types or 'Models' in types:
        locations.append(Location('SD Models', opts.ckpt_dir, 'Models'))
        locations.append(Location('Diffusers Models', opts.diffusers_dir, 'Models'))
        locations.append(Location('Huggingface Modules', opts.hfcache_dir, 'Models'))
        locations.append(Location('VAE', [opts.vae_dir, os.path.join(paths.models_path, "TAESD")], 'Models'))
        locations.append(Location('UNet', opts.unet_dir, 'Models'))
        locations.append(Location('TextEncoder', opts.te_dir, 'Models'))
        locations.append(Location('LoRA', opts.lora_dir, 'Models'))
        locations.append(Location('ControlNets', opts.control_dir, 'Models'))
        locations.append(Location('Embeddings', opts.embeddings_dir, 'Models'))
        locations.append(Location('Detailers', [opts.yolo_dir, os.path.join(paths.models_path, 'Ultralytics')], 'Models'))
        locations.append(Location('Upscalers', [opts.esrgan_models_path, opts.bsrgan_models_path, opts.realesrgan_models_path, opts.scunet_models_path, opts.swinir_models_path, os.path.join(paths.models_path, 'chaiNNer'), os.path.join(paths.models_path, 'GFPGAN'), os.path.join(paths.models_path, 'Spandrel'), os.path.join(paths.models_path, 'SeedVR2')], 'Models')) # chainners extension has late opts init
        locations.append(Location('CLiP', opts.clip_models_path, 'Models'))
        locations.append(Location('Rembg', os.path.join(paths.models_path, 'Rembg'), 'Models'))
        locations.append(Location('RIFE', os.path.join(paths.models_path, 'RIFE'), 'Models'))
    if types is None or 'All' in types or 'Data' in types:
        locations.append(Location('Configs', ['data', paths.sd_configs_path], 'Data'))
        locations.append(Location('AutoComplete', opts.autocomplete_dir, 'Data'))
        locations.append(Location('Styles', opts.styles_dir, 'Data'))
        locations.append(Location('Wildcards', opts.wildcards_dir, 'Data'))
        locations.append(Location('Reference', paths.reference_path, 'Data'))
        locations.append(Location('LUTs', os.path.join(paths.models_path, 'LUTs'), 'Data'))
        locations.append(Location('Wiki', 'wiki', 'Data'))
    if types is None or 'All' in types or 'Cache' in types:
        locations.append(Location('Temp', opts.temp_dir, 'Cache'))
        locations.append(Location('XET', opts.xetcache_dir, 'Cache'))
        locations.append(Location('OpenVINO', opts.openvino_cache_path, 'Cache'))
        locations.append(Location('ONNX', opts.onnx_cached_models_path, 'Cache'))
        locations.append(Location('VENV', 'venv', 'Cache'))
        locations.append(Location('Torch', [opts.tunable_dir, os.getenv("TORCHINDUCTOR_CACHE_DIR", None), os.getenv("TRITON_CACHE_DIR", None)], 'Cache'))
    if types is None or 'All' in types or 'Code' in types:
        locations.append(Location('Modules', 'modules', 'Code'))
        locations.append(Location('Pipelines', 'pipelines', 'Code'))
        locations.append(Location('Scripts', 'scripts', 'Code'))
        locations.append(Location('UI', 'ui', 'Code'))
        locations.append(Location('Builtin', paths.extensions_builtin_dir, 'Code'))
        locations.append(Location('Extensions', paths.extensions_dir, 'Code'))
    if types is None or 'All' in types or 'Images' in types:
        locations.append(Location('Text', [opts.outdir_txt2img_samples], 'Images'))
        locations.append(Location('Image', [opts.outdir_img2img_samples], 'Images'))
        locations.append(Location('Control', [opts.outdir_control_samples], 'Images'))
        locations.append(Location('Extras', [opts.outdir_extras_samples], 'Images'))
        locations.append(Location('Save', [opts.outdir_save], 'Images'))
        locations.append(Location('Grids', [opts.outdir_txt2img_grids, opts.outdir_img2img_grids, opts.outdir_control_grids], 'Images'))
    if types is None or 'All' in types or 'Videos' in types:
        locations.append(Location('Video', [opts.outdir_video], 'Videos'))
    return locations


def get_other_locations(locations: list[Location], name: str, folder: str, what: str = 'Other') -> list[Location]:
    # get list of first level subfolders in `folder` check each if its already in `locations` by comparing resolved paths if not, add each to the list as a new Location with type `what`
    existing_paths = set()
    for location in locations:
        for path in location.paths:
            existing_paths.add(path.resolve(strict=False))
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    path = Path(entry.path).resolve(strict=False)
                    if path not in existing_paths:
                        locations.append(Location(name, entry.path, what))
    except (FileNotFoundError, PermissionError):
        pass
    return locations


def print_summary(locations: list[Location]):
    summary = {}
    for location in locations:
        if location.type not in summary:
            summary[location.type] = {
                'size': 0,
                'mtime': datetime.fromtimestamp(0),
                'nfiles': 0,
                'nfolders': 0,
                'nerrors': 0,
            }
        summary[location.type]['size'] += location.size
        if location.mtime > summary[location.type]['mtime']:
            summary[location.type]['mtime'] = location.mtime
        summary[location.type]['nfiles'] += location.nfiles
        summary[location.type]['nfolders'] += location.nfolders
        summary[location.type]['nerrors'] += location.nerrors
    for k, v in summary.items():
        log.debug(f'Storage: type={k} size={v["size"]/1024/1024:.3f} files={v["nfiles"]} folders={v["nfolders"]}')


def check_storage(folders: str | list[str] | None = None, types: list[str] | None = None, silent: bool = False) -> list[Location]:
    if isinstance(folders, str):
        folders = [folders]
    if folders is not None and len(folders) > 0:
        locations = [Location(None, folder) for _i, folder in enumerate(folders)]
    else:
        locations = get_all_locations(types)
        if types is None or 'Other' in types or 'All' in types:
            locations = get_other_locations(locations, 'Other', paths.models_path)
    log.debug(f'Storage: locations={len(locations)} workers={max_workers} types={types} folders={folders} start')

    def update_stats(location: Location) -> Location:
        t0 = time.time()
        for f in location.paths:
            size, mtime, files, folders, symlinks, errors = stat(f, extended=True, exclude=['__pycache__', '.'])
            location.size += size
            if mtime > location.mtime:
                location.mtime = mtime
            location.nfiles += files
            location.nfolders += folders
            location.nsymlinks += symlinks
            location.nerrors += errors
        location.time = time.time() - t0
        return location

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_items = {executor.submit(update_stats, location): location for location in locations}
        for future in as_completed(future_items):
            location = future.result()
            if location.size > 0 and not silent:
                log.debug(location)
    t1 = time.time()
    print_summary(locations)
    log.debug(f'Storage: time={t1-t0:.3f} end')
    return locations
