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
    def __init__(self, latent, preview=None, info=None, ops=None):
        if ops is None:
            ops = []
        self.ts = datetime.datetime.now().replace(microsecond=0)
        self.name = self.ts.strftime('%Y-%m-%d %H:%M:%S')
        self.latent = latent.detach().clone().to(devices.cpu)
        self.preview = preview
        self.info = info
        self.ops = ops.copy()
        self.size = sys.getsizeof(self.latent.storage())


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
        return [item.name for item in self.latents]

    @property
    def selected(self):
        if self.index >= 0 and self.index < self.count:
            current_index = self.index
            self.index = -1
        else:
            current_index = 0
        item = self.latents[current_index]
        log.debug(f'History get: index={current_index} time={item.ts} shape={list(item.latent.shape)} dtype={item.latent.dtype} count={self.count}')
        return item.latent.to(devices.device), current_index

    def find(self, name):
        for i, item in enumerate(self.latents):
            if item.name == name:
                return i
        return -1

    def add(self, latent, preview=None, info=None, ops=None):
        if ops is None:
            ops = []
        shared.state.latent_history += 1
        if shared.opts.latent_history == 0:
            return
        if torch.is_tensor(latent):
            item = Item(latent, preview, info, ops)
            self.latents.appendleft(item)
            if self.count >= shared.opts.latent_history:
                self.latents.pop()

    def clear(self):
        self.latents.clear()
        # log.debug(f'History clear: count={self.count}')

    def load(self):
        pass

    def save(self):
        pass
