import time
import logging
import torch
from modules import shared, errors, devices, sd_offload
from modules.logger import log


class InterruptLogFilter(logging.Filter):
    """Drops the per-block error dumps the modular runner logs when an interrupt raises through it."""
    def filter(self, record):
        return 'Interrupted...' not in record.msg


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


def install_state_hook(pipe):
    runner_log = logging.getLogger('diffusers.modular_pipelines.modular_pipeline')
    if not any(isinstance(f, InterruptLogFilter) for f in runner_log.filters):
        runner_log.addFilter(InterruptLogFilter())

    def set_phase(phase: str, module: torch.nn.Module | None = None):
        # every stage runs inside one pipeline call, so the forward hooks are the only place the current stage is visible
        if getattr(pipe, 'sdnext_phase', None) != phase:
            pipe.sdnext_phase = phase
            jobid = getattr(pipe, 'sdnext_phaseid', None) # previous jobid if any
            shared.state.end(jobid) # clear the previous job if exists
            pipe.sdnext_phaseid = shared.state.begin(phase) # start a new job for the current phase
            log.debug(f'Pipeline: phase={phase.replace(" ", "")} cls={pipe.__class__.__name__} module={module.__class__.__name__ if module is not None else None}')
            return True
        return False

    def _pre_transformer_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Generate', module)
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['transformer', 'transformer_ref'], reason='generate')
        if shared.state.sampling_steps == 0 and getattr(pipe, 'num_timesteps', 0) > 0:
            shared.state.sampling_steps = pipe.num_timesteps
        if shared.state.paused:
            log.debug('Sampling paused')
            while shared.state.paused:
                if shared.state.interrupted or shared.state.skipped:
                    raise AssertionError('Interrupted...')
                time.sleep(0.1)
        shared.state.step()
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    def _pre_text_encode_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Text Encode', module)
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['text_encoder'], reason='text encode')
        if shared.state.interrupted or shared.state.skipped:
            raise AssertionError('Interrupted...')

    def _pre_vae_decode_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Decode', module)
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['vae', 'audio_vae'], reason='vae decode')
        if shared.state.interrupted or shared.state.skipped: # fires per tile, so tiled decodes abort promptly
            raise AssertionError('Interrupted...')

    def _pre_vae_encode_hook(module, args): # pylint: disable=unused-argument
        new_phase = set_phase('Encode', module)
        if new_phase:
            sd_offload.offload_ondemand(pipe, exclude=['vae', 'audio_vae'], reason='vae encode')
        if shared.state.interrupted or shared.state.skipped: # fires per tile, so tiled encodes abort promptly
            raise AssertionError('Interrupted...')

    for name in ('transformer', 'transformer_ref'):
        module = getattr(pipe, name, None)
        if module is not None:
            target = getattr(module, 'model', module) # conditioning calls the inner model directly
            if isinstance(target, torch.nn.Module) and getattr(target, 'sdnext_state_hook', None) is None:
                target.sdnext_state_hook = target.register_forward_pre_hook(_pre_transformer_hook)

    for name in ('text_encoder', 'text_encoder_2'):
        module = getattr(pipe, name, None)
        if module is not None:
            target = getattr(module, 'model', module) # conditioning calls the inner model directly
            if isinstance(target, torch.nn.Module) and getattr(target, 'sdnext_state_hook', None) is None:
                target.sdnext_state_hook = target.register_forward_pre_hook(_pre_text_encode_hook)

    for name in ('vae', 'audio_vae'):
        decoder = getattr(getattr(pipe, name, None), 'decoder', None) # decode entry points bypass forward, the inner decoder does not
        if isinstance(decoder, torch.nn.Module) and getattr(decoder, 'sdnext_state_hook', None) is None:
            decoder.sdnext_state_hook = decoder.register_forward_pre_hook(_pre_vae_decode_hook)
        encoder = getattr(getattr(pipe, name, None), 'encoder', None) # decode entry points bypass forward, the inner encoder does not
        if isinstance(encoder, torch.nn.Module) and getattr(encoder, 'sdnext_state_hook', None) is None:
            encoder.sdnext_state_hook = encoder.register_forward_pre_hook(_pre_vae_encode_hook)


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
        cls_name = getattr(cls, '__name__', '') or ''
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

        install_state_hook(pipe)
        apply_progress_bar_config(pipe._blocks) # pylint: disable=protected-access
        return pipe
    except Exception as e:
        log.error(f'Load modular: repo="{repo}" workflow={workflow} {e}')
        errors.display(e, 'video')
        return None
