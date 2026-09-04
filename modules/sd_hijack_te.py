import os
import time
from modules import shared, errors, timer, sd_models
from modules.logger import log
from modules.attention import context as attention_context


class PromptCache:
    def __init__(self):
        self.cache = {}
        self.id = None
        self.max = 16

    @staticmethod
    def _hashable(val):
        if isinstance(val, list):
            return tuple(val)
        return val

    def get(self, prompt, negative_prompt=None, cfg_enabled=None):
        if self.id != id(shared.sd_model):
            self.cache.clear()
            self.id = id(shared.sd_model)
            log.debug(f'Encode: prompt cache activate id={self.id} depth={len(self.cache)}')
        prompt = self._hashable(prompt)
        negative_prompt = self._hashable(negative_prompt)
        if (isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str)):
            cached = self.cache.get((prompt[0], negative_prompt, cfg_enabled), None)
        elif isinstance(prompt, str):
            cached = self.cache.get((prompt, negative_prompt, cfg_enabled), None)
        else:
            cached = None
        if cached:
            log.debug(f'Encode: prompt="{prompt}" cache={len(self.cache)} hit')
        return cached

    def set(self, prompt, encoded, negative_prompt=None, cfg_enabled=None):
        if len(self.cache) >= self.max:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        prompt = self._hashable(prompt)
        negative_prompt = self._hashable(negative_prompt)
        if (isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str)):
            self.cache[(prompt[0], negative_prompt, cfg_enabled)] = encoded
        elif isinstance(prompt, str):
            self.cache[(prompt, negative_prompt, cfg_enabled)] = encoded


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

        # cache key must include cfg-affecting kwargs since encode_prompt output (e.g. negative_prompt_embeds) depends on them
        negative_prompt = kwargs.get('negative_prompt', None)
        cfg_enabled = kwargs.get('do_classifier_free_guidance', None)
        cached = prompt_cache.get(prompt, negative_prompt, cfg_enabled)
        if cached is not None:
            res = cached
        else:
            log.debug(f'Encode: prompt="{prompt}" hijack=True')
            with attention_context.role('te'):
                if hasattr(shared.sd_model, 'orig_encode_prompt'):
                    res = shared.sd_model.orig_encode_prompt(*args_copy, **kwargs)
                else:
                    res = shared.sd_model.encode_prompt(*args_copy, **kwargs)
            prompt_cache.set(prompt, res, negative_prompt, cfg_enabled)

        if hasattr(shared.sd_model, 'after_prompt_encode'):
            log.debug(f'Encode: prompt="{prompt}" op=after')
            res = shared.sd_model.after_prompt_encode(res)

    except Exception as e:
        log.error(f'Encode prompt: {e}')
        errors.display(e, 'Encode prompt')
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
