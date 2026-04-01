import os
import time
from modules import shared, errors, timer, sd_models
from modules.logger import log


def hijack_encode_prompt(*args, **kwargs):
    jobid = shared.state.begin('TE Encode')
    t0 = time.time()
    if 'max_sequence_length' in kwargs and kwargs['max_sequence_length'] is not None:
        kwargs['max_sequence_length'] = max(kwargs['max_sequence_length'], os.environ.get('HIDREAM_MAX_SEQUENCE_LENGTH', 256))
    try:
        prompt = kwargs.get('prompt', None) or (args[0] if len(args) > 0 else None)
        if prompt is not None:
            log.debug(f'Encode: prompt="{prompt}" hijack=True')
        res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    except Exception as e:
        log.error(f'Encode prompt: {e}')
        errors.display(e, 'Encode prompt')
        res = None
    t1 = time.time()
    timer.process.add('te', t1-t0)
    # if hasattr(shared.sd_model, "maybe_free_model_hooks"):
    #     shared.sd_model.maybe_free_model_hooks()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    shared.state.end(jobid)
    return res


def init_hijack(pipe):
    if pipe is not None and not hasattr(pipe, 'orig_encode_prompt') and hasattr(pipe, 'encode_prompt'):
        pipe.orig_encode_prompt = pipe.encode_prompt
        pipe.encode_prompt = hijack_encode_prompt
