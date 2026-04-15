import os
import time
from modules import shared, errors, timer, sd_models
from modules.logger import log


debug_output = os.environ.get('SD_PROMPT_DEBUG', None)
debug = log.trace if debug_output is not None else lambda *args, **kwargs: None


def hijack_encode_prompt(*args, **kwargs):
    jobid = shared.state.begin('TE Encode')
    t0 = time.time()
    if 'max_sequence_length' in kwargs and kwargs['max_sequence_length'] is not None:
        kwargs['max_sequence_length'] = max(kwargs['max_sequence_length'], os.environ.get('MAX_SEQUENCE_LENGTH', 256))
    res = None
    try:
        args_copy = list(args)
        patch_prompt = False
        prompt = kwargs.get('prompt', None)
        if prompt is None and len(args_copy) > 0:
            prompt = args_copy[0]
            patch_prompt = True
        res = prompt
        debug(f'Encode: prompt="{prompt}" hijack=True')

        if hasattr(shared.sd_model, 'before_prompt_encode'):
            log.debug(f'Encode: prompt="{prompt}" op=before')
            res = shared.sd_model.before_prompt_encode(prompt)
            if patch_prompt:
                args_copy[0] = res

        if hasattr(shared.sd_model, 'orig_encode_prompt'):
            res = shared.sd_model.orig_encode_prompt(*args_copy, **kwargs)

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
    return res


def init_hijack(pipe):
    if pipe is not None and not hasattr(pipe, 'orig_encode_prompt') and hasattr(pipe, 'encode_prompt'):
        pipe.orig_encode_prompt = pipe.encode_prompt
        pipe.encode_prompt = hijack_encode_prompt
