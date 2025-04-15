import os
import time
from modules import shared, errors, timer, sd_models


def hijack_encode_prompt(*args, **kwargs):
    t0 = time.time()
    if 'max_sequence_length' in kwargs:
        kwargs['max_sequence_length'] = max(kwargs['max_sequence_length'], os.environ.get('HIDREAM_MAX_SEQUENCE_LENGTH', 256))
    try:
        res = shared.sd_model.orig_encode_prompt(*args, **kwargs)
    except Exception as e:
        shared.log.error(f'Eencode prompt: {e}')
        errors.display(e, 'Video encode prompt')
        res = None
    t1 = time.time()
    timer.process.add('te', t1-t0)
    if hasattr(shared.sd_model, "maybe_free_model_hooks"):
        shared.sd_model.maybe_free_model_hooks()
    shared.sd_model = sd_models.apply_balanced_offload(shared.sd_model)
    return res


def init_hijack(pipe):
    if shared.opts.te_hijack and pipe is not None and not hasattr(pipe, 'orig_encode_prompt') and hasattr(pipe, 'encode_prompt'):
        # shared.log.debug(f'Model: cls={pipe.__class__.__name__} hijack encode')
        pipe.orig_encode_prompt = pipe.encode_prompt
        pipe.encode_prompt = hijack_encode_prompt
