import time
import torch
from modules import shared, errors, devices, sd_hijack_modular
from modules.logger import log


def apply_progress_bar_config(block):
    kwargs = {
        "ncols": 120,
        "colour": "#327fba",
        "bar_format": "Progress {rate_fmt}{postfix} {bar:15} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} {desc}",
    }
    if hasattr(block, "set_progress_bar_config"):
        block.set_progress_bar_config(**kwargs)
    for child in getattr(block, "sub_blocks", {}).values():
        apply_progress_bar_config(child)


def trace_modules(pipe):
    from modules.sd_offload_utils import get_module_names
    for module_name in get_module_names(pipe):
        module = getattr(pipe, module_name, None)
        if isinstance(module, torch.nn.Module):
            log.trace(f'Module: name={module_name} cls={module.__class__.__name__} device={next(module.parameters()).device} dtype={next(module.parameters()).dtype}')


def is_modular(obj) -> bool:
    if obj is None:
        return False
    cls = obj if isinstance(obj, type) else obj.__class__
    try:
        import diffusers
        modular_cls = getattr(diffusers, 'ModularPipeline', None)
        if isinstance(modular_cls, type) and issubclass(cls, modular_cls):
            return True
    except Exception:
        pass
    return 'Modular' in cls.__name__


def preload_components(pipe, workflow: str | None, load_config: dict | None = None) -> dict:
    """Load the denoiser and text encoder through the shared loaders rather than the pipeline's own.

    `load_components` fetches every component into the pipeline's cache directory with no
    single-file override, no shared text encoder and no per-component quantization control.
    The shared loaders do all three, and everything they need is already on the spec: repo,
    subfolder and class. Components differ per architecture, so each is recognized by the
    class its spec declares rather than by name.

    Only what the loaded workflow asks for is fetched, so an unused checkpoint partition is
    never pulled. `load_components` afterwards loads whatever is still unset, which is the
    tokenizer, processors, schedulers and VAEs.
    """
    from pipelines import generic
    specs = getattr(pipe, '_component_specs', {}) # pylint: disable=protected-access
    loaded = {}
    for name in missing_components(pipe, workflow):
        spec = specs.get(name)
        if spec is None or getattr(spec, 'default_creation_method', None) != 'from_pretrained':
            continue
        repo = getattr(spec, 'pretrained_model_name_or_path', None)
        cls = getattr(spec, 'type_hint', None)
        if not repo or cls is None:
            continue
        origin = getattr(cls, '__module__', '') or ''
        cls_name = getattr(cls, '__name__', '') or '' # TODO preload: components with remote code resolve to cls none
        subfolder = getattr(spec, 'subfolder', None) or name
        component = None
        if origin.startswith('diffusers') and ('Transformer' in cls_name or 'UNet' in cls_name):
            component = generic.load_transformer(repo, cls_name=cls, load_config=load_config, subfolder=subfolder, trust_remote_code=True)
        elif origin.startswith('transformers') and 'text_encoder' in name:
            # shared substitution is on: the map matches class plus a substring of the repo name, so its entries have to run narrow before broad
            component = generic.load_text_encoder(repo, cls_name=cls, load_config=load_config, subfolder=subfolder)
        if component is not None:
            loaded[name] = component
    return loaded


def missing_components(pipe, workflow: str | None) -> list:
    """Components the loaded workflow declares that did not materialize.

    A partition the workflow does not use is absent by design, so the comparison is
    against the workflow's own expected components rather than every declared spec.
    """
    blocks = getattr(pipe, '_blocks', None) # pylint: disable=protected-access
    if blocks is None:
        return []
    try:
        expected = blocks.get_workflow(workflow) if workflow else blocks
        names = [spec.name for spec in expected.expected_components]
    except Exception:
        names = list(getattr(pipe, '_component_specs', {})) # pylint: disable=protected-access
    return [name for name in names if getattr(pipe, name, None) is None]


def load_modular_pipe(repo_cls, repo: str, workflow: str | None = None, revision: str | None = None, offline_args: dict | None = None, base: bool = False, load_config: dict | None = None):
    if repo_cls is None or isinstance(repo_cls, str):
        log.error(f'Load modular: repo="{repo}" cls="{repo_cls}" pipeline class not found: diffusers too old')
        return None
    offline_args = offline_args or {}
    cache_dir = shared.opts.diffusers_dir if base else shared.opts.hfcache_dir # base models live in the diffusers folder so the model scan lists them; video-only models stay out of the dropdown
    try:
        t0 = time.time()
        log.debug(f'Load modular: repo="{repo}" cls={repo_cls.__name__} workflow={workflow} base={base}')
        pipe = repo_cls.from_pretrained(
            repo,
            revision=revision,
            cache_dir=cache_dir,
            **offline_args,
        )
        # the workflow restricts the component fetch only: passing it to from_pretrained instead would prune the blocks tree to one task and disable runtime dispatch between them
        preloaded = preload_components(pipe, workflow, load_config=load_config)
        if preloaded:
            pipe.update_components(**preloaded) # registered before the rest, which load_components then skips
            log.debug(f'Load modular: cls={pipe.__class__.__name__} preloaded={list(preloaded)}')
        pipe.load_components(
            workflow=workflow,
            dtype=devices.dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
            **offline_args,
        )
        loaded = [name for name, component in pipe.components.items() if component is not None]
        empty = [name for name, component in pipe.components.items() if component is None]
        missing = missing_components(pipe, workflow)
        pipe.sdnext_missing_components = missing # a caller that can recover a component clears its own entry
        pipe.sdnext_video_workflow = workflow # the workflow this pipe was loaded for, which is what the reference-workflow guard reads; the executed task is chosen per request
        log.info(f'Load modular: cls={pipe.__class__.__name__} workflow={workflow} components={loaded} empty={empty} time={time.time()-t0:.2f}')
        if missing:
            # load_components builds each component in its own try/except and reports a failure as a warning on the
            # diffusers logger, so the reason is in the log above this line rather than in the exception path
            log.error(f'Load modular: cls={pipe.__class__.__name__} workflow={workflow} missing={missing} components the workflow requires did not load')

        sd_hijack_modular.install_state_hook(pipe)
        sd_hijack_modular.register_callbacks(pipe)

        apply_progress_bar_config(pipe._blocks) # pylint: disable=protected-access
        return pipe
    except Exception as e:
        log.error(f'Load modular: repo="{repo}" workflow={workflow} {e}')
        errors.display(e, 'video')
        return None
