from modules import errors
from modules.logger import log
from installer import install, torch_info


def set_diffusers_attention(pipe, quiet = False):
    from modules import shared, devices

    log.quiet(quiet, f'Setting model: attention="{shared.opts.cross_attention_optimization}"')
    if shared.opts.cross_attention_optimization == "Disabled":
        torch_info.set(attention="disabled")
    elif shared.opts.cross_attention_optimization == "Scaled-Dot-Product":  # The default set by Diffusers
        devices.set_sdpa_params()
    elif shared.opts.cross_attention_optimization == "xFormers":
        if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
            torch_info.set(attention="xformers")
            pipe.enable_xformers_memory_efficient_attention()
        else:
            log.warning(f"Attention: xFormers is not compatible with {pipe.__class__.__name__}")
    else:
        log.warning(f'Torch attention: method="{shared.opts.cross_attention_optimization}" unknown, pipe={pipe.__class__.__name__} keeps its own attention processor')

    if shared.opts.attention_slicing != "Default" and hasattr(pipe, "enable_attention_slicing") and hasattr(pipe, "disable_attention_slicing"):
        if shared.opts.attention_slicing == "Enabled":
            pipe.enable_attention_slicing()
        else:
            pipe.disable_attention_slicing()
        log.debug(f"Torch attention: slicing={shared.opts.attention_slicing}")

    pipe.current_attn_name = shared.opts.cross_attention_optimization
    pipe.current_attn_overrides = list(shared.opts.sdp_overrides)


orig_get_kernel = None
def get_kernel_hijack(repo_id, revision=None, version=None, backend=None, user_agent=None, trust_remote_code: bool | list[str] = False): # pylint: disable=unused-argument
    log.debug(f'Attention dispatcher hub: repo="{repo_id}" revision={revision} version={version} backend={backend}')
    user_agent = 'kernels/0.16.0'
    module = None
    try:
        module = orig_get_kernel(repo_id, revision=revision, version=version, backend=backend, user_agent=user_agent, trust_remote_code=True)
    except Exception as e:
        log.error(f'Attention dispatcher hub: {e}')
        errors.display(e, 'kernels')
    return module


def get_hf_api_hijack(user_agent = None): # pylint: disable=unused-argument
    from huggingface_hub import HfApi
    return HfApi(library_name="kernels", user_agent="donottrack")


def hijack_kernels():
    global orig_get_kernel # pylint: disable=global-statement
    try:
        install('kernels==0.16.0')
        import kernels
        import kernels.utils
        log.debug(f'Attention dispatcher: kernels={kernels.__version__}')
        if orig_get_kernel is None:
            orig_get_kernel = kernels.get_kernel
            kernels.get_kernel = get_kernel_hijack
            kernels.utils._get_hf_api = get_hf_api_hijack # pylint: disable=protected-access
        from diffusers.utils import import_utils
        import_utils._kernels_available = True # pylint: disable=protected-access
        import_utils._kernels_version = kernels.__version__ # pylint: disable=protected-access
    except Exception as e:
        log.error(f'Attention dispatcher kernels: {e}')
        return


def set_attention_dispatcher(pipe):
    from modules import shared
    attn = shared.opts.hf_attention.strip().lower()
    if pipe is None or not hasattr(pipe, 'transformer') or not hasattr(pipe.transformer, 'set_attention_backend'):
        return

    from diffusers.models import attention_dispatch as a
    backends = [b.value for b in a._AttentionBackendRegistry.list_backends()] # pylint: disable=protected-access
    # https://huggingface.co/docs/kernels/index
    # https://huggingface.co/docs/diffusers/optimization/attention_backends#available-backends

    if 'hub' in attn:
        hijack_kernels()

    prev = a._AttentionBackendRegistry.get_active_backend() # pylint: disable=protected-access
    if attn in backends:
        try:
            pipe.transformer.set_attention_backend(attn)
        except Exception as e:
            log.error(f'Attention dispatcher: target={attn} {e}')
        current = a._AttentionBackendRegistry.get_active_backend() # pylint: disable=protected-access
        log.debug(f'Attention dispatcher: target={attn} previous={prev[0].value} active={current[0]} list={backends}')
    elif len(attn) > 0:
        log.warning(f'Attention dispatcher: active={prev[0].value} list={backends} target={attn} not found')
    else:
        log.debug(f'Attention dispatcher: active={prev[0].value} list={backends}')


def list_dispatcher_backends() -> list:
    """The kernels diffusers can dispatch attention to, for anything that offers hf_attention as a choice."""
    try:
        from diffusers.models import attention_dispatch as a
        return sorted(b.value for b in a._AttentionBackendRegistry.list_backends()) # pylint: disable=protected-access
    except Exception as e:
        log.error(f'Attention dispatcher: {e}')
        return []
