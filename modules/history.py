"""
TODO: apply metadata, preview, load/save
"""

import sys
import datetime
from collections import deque
import torch
from modules import shared, devices
from modules.logger import log


class Item:
    latent: torch.Tensor | None = None
    size: int = 0
    preview: str | None = None
    info: str | None = None
    ops: list = []
    images: list | None = None

    def __init__(self, latent, preview=None, info=None, ops=None, images=None):
        if ops is None:
            ops = []
        self.ts = datetime.datetime.now().replace(microsecond=0)
        self.name = self.ts.strftime('%Y-%m-%d %H:%M:%S')
        if torch.is_tensor(latent):
            self.latent = latent.detach().clone().to(devices.cpu)
            self.size = sys.getsizeof(self.latent.storage())
        self.preview = preview
        self.info = info
        self.ops = ops.copy()
        if ['video'] in ops:
            self.images = None
        else:
            self.images = images

    def __str__(self):
        if self.latent is not None:
            return f'Item(ts="{self.name}" ops={self.ops} latent={self.latent.shape} size={self.size})'
        elif self.images is not None:
            return f'Item(ts="{self.name}" ops={self.ops} images={len(self.images) if isinstance(self.images, list) else self.images})'
        else:
            return f'Item(ts="{self.name}" ops={self.ops} unknown content)'


class History:
    def __init__(self):
        self.index = -1
        self.latents = deque(maxlen=1024)

    @property
    def count(self):
        return len(self.latents)

    @property
    def size(self):
        s = 0
        for item in self.latents:
            s += item.size
        return s

    @property
    def list(self):
        log.info(f'History: items={self.count}/{shared.opts.latent_history} size={self.size}')
        return [item.name for item in self.latents if item.latent is not None]

    @property
    def selected(self):
        if self.index >= 0 and self.index < self.count:
            current_index = self.index
            self.index = -1
        else:
            current_index = 0
        item = self.latents[current_index]
        if item.latent is None:
            return None
        log.debug(f'History get: index={current_index} time={item.ts} shape={list(item.latent.shape)} dtype={item.latent.dtype} count={self.count}')
        return item.latent.to(devices.device), current_index

    @property
    def last_item(self):
        return self.latents[0] if self.count > 0 else None

    @property
    def last_image(self):
        if self.count == 0:
            return None
        for item in self.latents:
            if item.images is not None:
                return item.images
        return None

    @property
    def last_latent(self):
        if self.count == 0:
            return None
        for item in self.latents:
            if item.latent is not None:
                return item.latent
        return None

    def find(self, name):
        for i, item in enumerate(self.latents):
            if item.name == name:
                return i
        return -1

    def add(self, latent, preview=None, info=None, ops=None, images=None):
        if ops is None:
            ops = []
        shared.state.latent_history += 1
        if shared.opts.latent_history == 0:
            return
        item = Item(latent, preview, info, ops, images)
        self.latents.appendleft(item)
        if self.count >= shared.opts.latent_history:
            self.latents.pop()
        log.debug(f'History: len={self.count} add={item}')

    def clear(self):
        self.latents.clear()

    def load(self):
        pass

    def save(self):
        pass
