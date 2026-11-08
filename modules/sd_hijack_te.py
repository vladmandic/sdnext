import os
import time
import contextlib
from functools import wraps
import torch
from modules import shared, timer, sd_models
from modules.logger import log
from modules.attention import context as attention_context


def freeze(value):
    """Hashable form of a plain encode argument; tensors, images and other objects raise TypeError."""
    if value is None or isinstance(value, (str, int, float, bool, torch.device, torch.dtype)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, freeze(item)) for key, item in sorted(value.items()))
    raise TypeError(type(value).__name__)


class PromptCache:
    def __init__(self):
        self.cache = {}
        self.id = None
        self.max = 16

    @staticmethod
    def key(args, kwargs):
        """Key over every argument of the encode call; None when one is not a plain value, which leaves the call uncached."""
        try:
            return freeze(args), freeze(kwargs)
        except TypeError:
            return None

    def get(self, key, prompt=None):
        current_id = id(shared.sd_model.sd_checkpoint_info) if hasattr(shared.sd_model, 'sd_checkpoint_info') else id(shared.sd_model)
        if self.id != current_id:
            self.cache.clear()
            self.id = current_id
            log.debug(f'Encode: prompt cache activate id={self.id} depth={len(self.cache)}')
        cached = self.cache.get(key, None) if key is not None else None
        if cached is not None:
            if isinstance(prompt, list):
                length = sum(len(p) for p in prompt if isinstance(p, str))
                log.debug(f'Encode: length={length} cache={len(self.cache)} prompt={prompt}')
            else:
                length = len(prompt) if isinstance(prompt, str) else 0
                log.debug(f'Encode: length={length} cache={len(self.cache)} prompt="{prompt}"')
        return cached

    def set(self, key, encoded):
        if key is None:
            return
        if len(self.cache) >= self.max:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = encoded


prompt_cache = PromptCache()


def compile_stance(target='TE'):
    # SDNQ compiles static-shape helpers, so variable prompt lengths and image sizes can cause repeated recompiles without a warmup speedup
    if not shared.opts.torch_skip_compile_te or target in shared.opts.cuda_compile or not hasattr(torch.compiler, 'set_stance'):
        return contextlib.nullcontext()
    return torch.compiler.set_stance('force_eager')


def with_compile_stance(target='TE'):
    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with compile_stance(target):
                return func(*args, **kwargs)
        return wrapped
    return decorate


def hijack_encode_prompt(*args, **kwargs):
    jobid = shared.state.begin('TE Encode')
    t0 = time.time()
    if 'max_sequence_length' in kwargs and kwargs['max_sequence_length'] is not None:
        kwargs['max_sequence_length'] = max(kwargs['max_sequence_length'], int(os.environ.get('MAX_SEQUENCE_LENGTH', 256)))
    res = None
    try:
        args_copy = list(args)
        patch_prompt = False
        prompt = kwargs.get('prompt', None)
        if prompt is None and len(args_copy) > 0:
            prompt = args_copy[0]
            patch_prompt = True
        prompt = [p.strip(", \n") if isinstance(p, str) else p for p in prompt] if isinstance(prompt, list) else prompt

        if hasattr(shared.sd_model, 'before_prompt_encode'):
            log.debug('Encode: op=before')
            prompt = shared.sd_model.before_prompt_encode(prompt)
            if patch_prompt:
                args_copy[0] = prompt

        key = prompt_cache.key(args_copy, kwargs) # pipelines pass size, image and guidance arguments that change the result
        cached = prompt_cache.get(key, prompt)
        if cached is not None:
            res = cached
        else:
            if isinstance(prompt, list):
                length = sum(len(p) for p in prompt if isinstance(p, str))
                log.debug(f'Encode: length={length} prompt={prompt} hijack=True')
            else:
                length = len(prompt) if isinstance(prompt, str) else 0
                log.debug(f'Encode: length={length} prompt="{prompt}" hijack=True')
            with attention_context.role('te'), compile_stance():
                if hasattr(shared.sd_model, 'orig_encode_prompt'):
                    res = shared.sd_model.orig_encode_prompt(*args_copy, **kwargs)
                else:
                    res = shared.sd_model.encode_prompt(*args_copy, **kwargs)
            prompt_cache.set(key, res)

        if hasattr(shared.sd_model, 'after_prompt_encode'):
            log.debug('Encode: op=after')
            res = shared.sd_model.after_prompt_encode(res)
    finally: # an encode failure propagates: the pipeline has no embeddings to continue with
        t1 = time.time()
        timer.process.add('te', t1-t0)
        if t1 - t0 > 10:
            log.warning(f'Encode: time={t1-t0:.3f} long encode prompt')
        shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
        shared.state.end(jobid)
    # from modules import memstats
    # log.debug(f'Encode: memory={memstats.memory_stats()}')
    return res


def init_hijack(pipe):
    if (pipe is not None) and not hasattr(pipe, 'orig_encode_prompt') and hasattr(pipe, 'encode_prompt'):
        pipe.orig_encode_prompt = pipe.encode_prompt
        pipe.encode_prompt = hijack_encode_prompt
