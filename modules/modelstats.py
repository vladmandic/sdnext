import os
from datetime import datetime
import torch
from modules import shared, sd_models
from modules.logger import log


def walk(folder: str):
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def stat(folder: str, follow: bool = False, extended: bool = False, exclude: list[str] = []):
    _files = 0
    _folders = 0
    _symlinks = 0
    _errors = 0
    _size = 0
    _mtime = 0.0

    def recurse(folder: str):
        nonlocal _size, _mtime, _files, _folders, _symlinks, _errors
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if any(part == ex for part in entry.path.split(os.sep) for ex in exclude):
                        continue
                    if entry.is_file(follow_symlinks=follow):
                        try:
                            _stat = entry.stat(follow_symlinks=follow)
                        except Exception:
                            _stat = os.stat(entry.path, follow_symlinks=follow)
                        _size += _stat.st_size
                        _files += 1
                        if _stat.st_mtime > _mtime:
                            _mtime = _stat.st_mtime
                    elif entry.is_symlink():
                        _symlinks += 1
                    elif entry.is_dir(follow_symlinks=follow):
                        _folders += 1
                        recurse(entry.path)
                except (FileNotFoundError, PermissionError):
                    _errors += 1
                    continue

    try:
        s_folder = str(folder)
        if any(s_folder in ex for ex in exclude):
            return _size, datetime.fromtimestamp(_mtime).replace(microsecond=0), _files, _folders, _symlinks, _errors
        elif os.path.isfile(folder):
            _stat = os.stat(folder, follow_symlinks=follow)
            _size = _stat.st_size
            _mtime = _stat.st_mtime
            _files = 1
        elif os.path.isdir(folder):
            _folders = 1
            recurse(folder)
        else:
            pass
    except (FileNotFoundError, PermissionError):
        _errors += 1
    try:
        _datetime = datetime.fromtimestamp(_mtime).replace(microsecond=0)
    except (OSError, ValueError):
        _datetime = datetime.fromtimestamp(0)
    if extended:
        return _size, _datetime, _files, _folders, _symlinks, _errors
    return _size, _datetime


class Module:
    name: str = ''
    cls: str = None
    device: str = None
    dtype: str = None
    params: int = 0
    modules: int = 0
    quant: str = None
    config: dict = None

    def __init__(self, name, module):
        self.name = name
        self.cls = module.__class__.__name__
        if isinstance(module, tuple):
            self.cls = module[1]
        if hasattr(module, 'config'):
            self.config = module.config
        if isinstance(module, torch.nn.Module):
            self.device = getattr(module, 'device', None)
            self.dtype = getattr(module, 'dtype', None)
            self.params = sum(p.numel() for p in module.parameters(recurse=True))
            self.modules = len(list(module.modules()))
            self.quant = getattr(module, 'quantization_method', None)

    def __repr__(self):
        s = f'name="{self.name}" cls={self.cls} config={self.config is not None}'
        if self.device or self.dtype:
            s += f' device={self.device} dtype={self.dtype}'
        if self.params or self.modules:
            s += f' params={self.params} modules={self.modules}'
        return s


class Model:
    name: str = ''
    fn: str = ''
    type: str = ''
    cls: str = ''
    hash: str = ''
    meta: dict = {}
    size: int = 0
    mtime: datetime = None
    info: sd_models.CheckpointInfo = None
    modules: list[Module] = []

    def __init__(self, name):
        self.name = name
        if not shared.sd_loaded:
            return
        self.cls = shared.sd_model.__class__.__name__
        self.type = shared.sd_model_type
        self.info = sd_models.get_closest_checkpoint_match(name)
        if self.info is not None:
            self.name = self.info.name or self.name
            self.hash = self.info.shorthash or ''
            self.meta = self.info.metadata or {}
            self.size, self.mtime = stat(self.info.filename)

    def __repr__(self):
        return f'model="{self.name}" type={self.type} class={self.cls} size={self.size} mtime="{self.mtime}" modules={self.modules}'


def analyze():
    if not shared.sd_loaded:
        return None
    model = Model(shared.opts.sd_model_checkpoint)
    if model.cls == '':
        return model
    if hasattr(shared.sd_model, '_internal_dict'):
        keys = shared.sd_model._internal_dict.keys() # pylint: disable=protected-access
    else:
        keys = sd_models.get_signature(shared.sd_model).keys()
    model.modules.clear()
    for k in keys: # pylint: disable=protected-access
        if k.startswith('_'):
            continue
        component = getattr(shared.sd_model, k, None)
        module = Module(k, component)
        model.modules.append(module)
    log.debug(f'Analyzed: {model}')
    return model
