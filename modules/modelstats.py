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


def stat(fn: str):
    if fn is None or len(fn) == 0 or not os.path.exists(fn):
        return 0, datetime.fromtimestamp(0)
    fs_stat = os.stat(fn, follow_symlinks=False)
    mtime = datetime.fromtimestamp(fs_stat.st_mtime).replace(microsecond=0)
    if os.path.islink(fn):
        size = 0
    elif os.path.isfile(fn):
        size = round(fs_stat.st_size)
    elif os.path.isdir(fn):
        size = round(sum(stat(fn)[0] for fn in walk(fn)))
    else:
        size = 0
    return size, mtime


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
