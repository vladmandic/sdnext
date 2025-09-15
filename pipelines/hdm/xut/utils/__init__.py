import torch
from .. import env


def isiterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def compile_wrapper(func, **kwargs):
    kwargs.update(env.COMPILE_ARGS)
    compiled = torch.compile(func, **kwargs)

    def runner(*args, **kwargs):
        if env.TORCH_COMPILE:
            return compiled(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return runner
