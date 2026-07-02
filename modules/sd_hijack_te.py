import os
import time
from modules import shared, errors, timer, sd_models
from modules.logger import log


class PromptCache:
    def __init__(self):
        self.cache = {}
        self.id = None
        self.max = 16

    def get(self, prompt):
        if self.id != id(shared.sd_model):
            self.cache.clear()
            self.id = id(shared.sd_model)
            log.debug(f'Encode: prompt cache activate id={self.id} depth={len(self.cache)}')
        if (isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str)):
            cached = self.cache.get(prompt[0], None)
        elif isinstance(prompt, str):
            cached = self.cache.get(prompt, None)
        else:
            cached = None
        if cached:
            log.debug(f'Encode: prompt="{prompt}" cache={len(self.cache)} hit')
        return cached

    def set(self, prompt, encoded):
        if len(self.cache) >= self.max:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        if (isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str)):
            self.cache[prompt[0]] = encoded
        elif isinstance(prompt, str):
            self.cache[prompt] = encoded


prompt_cache = PromptCache()


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
        res = prompt

        if hasattr(shared.sd_model, 'before_prompt_encode'):
            log.debug(f'Encode: prompt="{prompt}" op=before')
            res = shared.sd_model.before_prompt_encode(prompt)
            if patch_prompt:
                args_copy[0] = res

        cached = prompt_cache.get(prompt)
        if cached is not None:
            res = cached
        else:
            log.debug(f'Encode: prompt="{prompt}" hijack=True')
            if hasattr(shared.sd_model, 'orig_encode_prompt'):
                res = shared.sd_model.orig_encode_prompt(*args_copy, **kwargs)
            else:
                res = shared.sd_model.encode_prompt(*args_copy, **kwargs)
            prompt_cache.set(prompt, res)

        if hasattr(shared.sd_model, 'after_prompt_encode'):
            log.debug(f'Encode: prompt="{prompt}" op=after')
            res = shared.sd_model.after_prompt_encode(res)

    except Exception as e:
        log.error(f'Encode prompt: {e}')
        errors.display(e, 'Encode prompt')
    t1 = time.time()
    timer.process.add('te', t1-t0)
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    shared.state.end(jobid)
    # from modules import memstats
    # log.debug(f'Encode: memory={memstats.memory_stats()}')
    return res


def init_hijack(pipe):
    if pipe is not None and not hasattr(pipe, 'orig_encode_prompt') and hasattr(pipe, 'encode_prompt'):
        pipe.orig_encode_prompt = pipe.encode_prompt
        pipe.encode_prompt = hijack_encode_prompt
