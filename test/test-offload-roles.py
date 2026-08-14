#!/usr/bin/env python
"""
Offline unit tests for group offload placement in modules.sd_offload.

Every component takes exactly one role, derived from the component itself:

- ``resident``: named by the never-offload list or the model-type exclusion, or carrying
  an encode/decode entry point with no decorator for hooks to ride.
- ``ondemand``: an encode/decode entry bridge, or ``_supports_group_offloading = False``.
  Group hooks are forward scoped, so these take a whole-module hook instead.
- ``main``: a component name in ``group_offload_main``, entered once per denoising step.
- ``aux``: everything else, entered once per generation.

Covers:

- ``group_offload_role`` over the component names sdnext loads, including the precedence
  cases where an entry bridge or an upstream opt-out overrides a denoiser slot name
- ``apply_group_offload`` dispatching one arm per component, and the hooks landing on the
  wrapper rather than the inner model for wrapper-shaped text encoders
- ``offload_ondemand`` force sweeps moving only stamped components
- a settings change re-placing a previously resident component
- ``apply_group_offload_ondemand`` return contract and idempotency
- ``get_module_names`` on both pipeline kinds
- the upstream markers the roles read

No running server required. Nothing is moved to the accelerator.

Usage:
    python test/test-offload-roles.py
"""

import os
import sys
import types

import torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from diffusers.utils.accelerate_utils import apply_forward_hook  # pylint: disable=wrong-import-position
from modules.errors import log                                   # pylint: disable=wrong-import-position
from modules import shared                                       # pylint: disable=wrong-import-position,unused-import
from modules import sd_offload                                    # pylint: disable=wrong-import-position


# ============================================================
# Test infrastructure
# ============================================================

results: dict[str, dict] = {}


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'tests': []}
    return name


def record(cat: str, passed: bool, name: str, detail: str = ''):
    status = 'PASS' if passed else 'FAIL'
    results[cat]['passed' if passed else 'failed'] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if passed:
        log.info(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if ok is False:
            record(cat, False, name)
        else:
            record(cat, True, name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


# ============================================================
# Stub components
# ============================================================

class PlainModule(torch.nn.Module):
    """Denoiser shape: entered through its own forward, no upstream opt-out."""
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.proj(x)


class BridgeModule(torch.nn.Module):
    """Autoencoder shape: the pipeline enters through encode/decode, never forward."""
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    @apply_forward_hook
    def encode(self, x):
        return self.proj(x)

    @apply_forward_hook
    def decode(self, x):
        return self.proj(x)


class UnsupportedModule(torch.nn.Module):
    """Reads submodule weights outside those submodules' forward, so upstream opts out."""
    _supports_group_offloading = False

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class WrapperModule(torch.nn.Module):
    """Text encoder shape: an inner model plus a head the inner model does not carry."""
    def __init__(self):
        super().__init__()
        self.model = PlainModule()
        self.lm_head = torch.nn.Linear(4, 4)


class NoBridgeModule(torch.nn.Module):
    """Autoencoder shape without the diffusers entry decorator: no hook can see its entry points."""
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def encode(self, x):
        return self.proj(x)

    def decode(self, x):
        return self.proj(x)


class SweepModule(PlainModule):
    """Reports a non-cpu parameter so the sweep's device gate opens, and records moves instead of running them."""
    def __init__(self):
        super().__init__()
        self.moved = []

    def parameters(self, recurse=True):
        yield types.SimpleNamespace(device=torch.device('meta'))

    def to(self, *args, **kwargs):
        self.moved.append(args)
        return self


class FakePipe:
    """Classic pipeline shape for get_module_names: components plus the config dict."""
    def __init__(self, loaded):
        for name, module in loaded.items():
            setattr(self, name, module)
        self._internal_dict = dict.fromkeys(loaded)


class FakeModularPipe:
    """Modular pipeline shape: components come from the specs, not the config dict."""
    def __init__(self, loaded, spec_only=None):
        for name, module in {**loaded, **(spec_only or {})}.items():
            setattr(self, name, module)
        self._component_specs = dict.fromkeys(list(loaded) + list(spec_only or {}))
        self._internal_dict = dict.fromkeys(list(loaded) + ['canvas_short_edge'])

    @property
    def components(self):
        return {name: getattr(self, name, None) for name in self._component_specs}


# ============================================================
# group_offload_role
# ============================================================

ROLE_CASES = [
    # denoiser slots run once per step
    ('unet', PlainModule, 'main'),
    ('transformer', PlainModule, 'main'),
    ('transformer_2', PlainModule, 'main'),
    ('transformer_ref', PlainModule, 'main'),
    ('unconditional_transformer', PlainModule, 'main'),
    ('prior', PlainModule, 'main'),
    ('prior_prior', PlainModule, 'main'),
    ('decoder', PlainModule, 'main'),
    ('dit_model', PlainModule, 'main'),
    ('model', PlainModule, 'main'),
    ('controlnet', PlainModule, 'main'),
    # everything else runs once per generation
    ('text_encoder', WrapperModule, 'aux'),
    ('text_encoder_2', PlainModule, 'aux'),
    ('text_encoder_3', PlainModule, 'aux'),
    ('prior_text_encoder', PlainModule, 'aux'),
    ('prior_image_encoder', PlainModule, 'aux'),
    ('image_encoder', PlainModule, 'aux'),
    ('safety_checker', PlainModule, 'aux'),
    ('mllm', WrapperModule, 'aux'),
    ('llm_adapter', PlainModule, 'aux'),
    ('connectors', PlainModule, 'aux'),
    ('vocoder', PlainModule, 'aux'),
    ('duration_head', PlainModule, 'aux'),
    ('prompt_enhancer', WrapperModule, 'aux'),
    ('prompt_enhancer_head', PlainModule, 'aux'),
    ('latent_upsampler', PlainModule, 'aux'),
    ('motion_adapter', PlainModule, 'aux'),
    # an entry bridge means whole-module onload whatever the component is called
    ('vae', BridgeModule, 'ondemand'),
    ('audio_vae', BridgeModule, 'ondemand'),
    ('vqvae', BridgeModule, 'ondemand'),
    ('movq', BridgeModule, 'ondemand'),
    ('vqgan', BridgeModule, 'ondemand'),
    # an entry point with no decorator gives hooks nothing to fire on
    ('vae', NoBridgeModule, 'resident'),
]


def test_role_table():
    wrong = []
    for module_name, cls, expected in ROLE_CASES:
        role = sd_offload.group_offload_role(module_name, cls())
        if role != expected:
            wrong.append(f'{module_name}/{cls.__name__}: {role} != {expected}')
    assert not wrong, '; '.join(wrong)


def test_role_bridge_overrides_denoiser_slot_name():
    assert sd_offload.group_offload_role('decoder', BridgeModule()) == 'ondemand'


def test_role_upstream_optout_overrides_denoiser_slot_name():
    assert sd_offload.group_offload_role('transformer', UnsupportedModule()) == 'ondemand'


def test_role_undecorated_entry_points_stay_resident():
    # neither group hooks nor the on-demand hook fire for a plain method call, so residency is the only safe placement
    assert sd_offload.group_offload_role('vae', NoBridgeModule()) == 'resident'


def test_role_unknown_component_is_aux():
    # aux is the direction that stays correct when the guess is wrong
    assert sd_offload.group_offload_role('some_future_head', PlainModule()) == 'aux'


def role_with_opts(module_name, module, **opts):
    saved = {key: getattr(shared.opts, key) for key in opts}
    for key, value in opts.items():
        setattr(shared.opts, key, value)
    try:
        return sd_offload.group_offload_role(module_name, module)
    finally:
        for key, value in saved.items():
            setattr(shared.opts, key, value)


def test_role_never_offload_list_matches_a_class_name():
    role = role_with_opts('vae', BridgeModule(), diffusers_offload_never='BridgeModule')
    assert role == 'resident', role


def test_role_never_offload_list_matches_a_component_name():
    role = role_with_opts('text_encoder', PlainModule(), diffusers_offload_never='text_encoder')
    assert role == 'resident', role


def test_role_excluded_model_type_stays_resident():
    # offline the live model type is the 'none' sentinel; the parser and matcher still round-trip it
    role = role_with_opts('transformer', PlainModule(), models_not_to_offload=shared.sd_model_type)
    assert role == 'resident', role


def test_role_empty_exclusions_match_nothing():
    role = role_with_opts('transformer', PlainModule(), diffusers_offload_never='', models_not_to_offload='')
    assert role == 'main', role


def test_role_main_list_has_no_encoder_names():
    encoders = [n for n in sd_offload.group_offload_main if 'encoder' in n or 'vae' in n]
    assert not encoders, f'encoder-shaped names in the per-step list: {encoders}'


# ============================================================
# apply_group_offload dispatch
# ============================================================

def dispatch_calls(pipe):
    """Run one pass with every apply arm replaced by a recorder; return name -> [roles] and the module each arm received."""
    calls: dict[str, list] = {}
    seen: dict[str, object] = {}

    def name_of(module):
        return next((n for n in sd_offload.get_module_names(pipe) if getattr(pipe, n, None) is module), module.__class__.__name__)

    def record(name, role, module):
        calls.setdefault(name, []).append(role)
        seen[name] = module
        return True

    orig_component = sd_offload.apply_group_offload_component
    orig_ondemand = sd_offload.apply_group_offload_ondemand
    orig_resident = sd_offload.set_group_resident
    orig_stats = sd_offload.report_group_stats
    sd_offload.apply_group_offload_component = lambda module, module_name, main: record(module_name, 'main' if main else 'aux', module)
    sd_offload.apply_group_offload_ondemand = lambda module: record(name_of(module), 'ondemand', module)
    sd_offload.set_group_resident = lambda module: record(name_of(module), 'resident', module)
    sd_offload.report_group_stats = lambda sd_model, module_names: None
    try:
        sd_offload.apply_group_offload(pipe)
    finally:
        sd_offload.apply_group_offload_component = orig_component
        sd_offload.apply_group_offload_ondemand = orig_ondemand
        sd_offload.set_group_resident = orig_resident
        sd_offload.report_group_stats = orig_stats
    return calls, seen


def test_dispatch_is_one_arm_per_component():
    pipe = FakePipe({
        'transformer': PlainModule(),
        'text_encoder': WrapperModule(),
        'vae': BridgeModule(),
        'scheduler': object(),
    })
    calls, seen = dispatch_calls(pipe)
    assert calls.get('transformer') == ['main'], f'transformer took {calls.get("transformer")}'
    assert calls.get('text_encoder') == ['aux'], f'text_encoder took {calls.get("text_encoder")}'
    assert calls.get('vae') == ['ondemand'], f'vae took {calls.get("vae")}'
    assert all(len(roles) == 1 for roles in calls.values()), f'a component took more than one arm: {calls}'
    assert 'scheduler' not in calls, 'non-module components must not be dispatched'
    assert seen['text_encoder'] is pipe.text_encoder, 'hooks land on the wrapper, not the inner model'


def test_dispatch_skips_non_modules():
    pipe = FakePipe({'transformer': PlainModule(), 'tokenizer': object(), 'scheduler': object()})
    calls, _seen = dispatch_calls(pipe)
    assert list(calls) == ['transformer'], f'dispatched {list(calls)}'


def test_force_sweep_moves_only_stamped_components():
    stamped = SweepModule()
    stamped.sdnext_ondemand = True
    unstamped = SweepModule()
    pipe = FakePipe({'vae': stamped, 'transformer': unstamped})
    sd_offload.offload_ondemand(pipe, reason='test', force=True)
    assert stamped.moved, 'the stamped component must be swept to cpu'
    assert not unstamped.moved, 'a component with no onload path must not be swept'


def test_reapply_after_clearing_the_never_list_restores_hooks():
    module = PlainModule()
    pipe = FakePipe({'text_encoder': module})
    saved_never = shared.opts.diffusers_offload_never
    orig_device = sd_offload.devices.device
    orig_stats = sd_offload.report_group_stats
    sd_offload.devices.device = torch.device('cpu') # residency moves to the accelerator, so pin the target to cpu
    sd_offload.report_group_stats = lambda sd_model, module_names: None
    try:
        shared.opts.diffusers_offload_never = 'text_encoder'
        sd_offload.apply_group_offload(pipe)
        assert getattr(module, 'sdnext_group_offload_sig', None) is None, 'a resident component must carry no group signature'
        shared.opts.diffusers_offload_never = ''
        sd_offload.apply_group_offload(pipe)
        assert getattr(module, 'sdnext_group_offload_sig', None) not in (None, 'partial'), 'clearing the exclusion must re-place the component'
    finally:
        shared.opts.diffusers_offload_never = saved_never
        sd_offload.devices.device = orig_device
        sd_offload.report_group_stats = orig_stats


def test_ondemand_list_tracks_the_stamps():
    pipe = FakePipe({'transformer': PlainModule(), 'vae': BridgeModule()})
    orig_component = sd_offload.apply_group_offload_component
    orig_stats = sd_offload.report_group_stats
    sd_offload.apply_group_offload_component = lambda module, module_name, main: True
    sd_offload.report_group_stats = lambda sd_model, module_names: None
    try:
        sd_offload.apply_group_offload(pipe)
    finally:
        sd_offload.apply_group_offload_component = orig_component
        sd_offload.report_group_stats = orig_stats
    assert pipe.sdnext_ondemand_modules == ['vae'], f'on-demand list is {pipe.sdnext_ondemand_modules}'
    assert getattr(pipe.vae, 'sdnext_ondemand', False), 'the vae must carry the on-demand stamp'


# ============================================================
# apply_group_offload_ondemand contract
# ============================================================

def test_ondemand_apply_returns_bool_and_is_idempotent():
    module = BridgeModule()
    first = sd_offload.apply_group_offload_ondemand(module)
    second = sd_offload.apply_group_offload_ondemand(module)
    assert isinstance(first, bool) and isinstance(second, bool), 'placement must report a bool'
    assert first is True, 'the first placement changes the component'
    assert second is False, 'an unchanged component must report no change'
    assert getattr(module, 'sdnext_ondemand', False), 'stamp must survive the second pass'


def test_ondemand_apply_leaves_weights_on_cpu():
    module = BridgeModule()
    sd_offload.apply_group_offload_ondemand(module)
    assert next(module.parameters()).device.type == 'cpu', 'on-demand components rest on cpu'


def test_resident_placement_clears_the_ondemand_stamp():
    module = BridgeModule()
    sd_offload.apply_group_offload_ondemand(module)
    orig_device = sd_offload.devices.device
    sd_offload.devices.device = torch.device('cpu') # residency moves to the accelerator, so pin the target to cpu
    try:
        changed = sd_offload.set_group_resident(module)
    finally:
        sd_offload.devices.device = orig_device
    assert isinstance(changed, bool) and changed is True, 'moving off the on-demand hook is a change'
    assert not getattr(module, 'sdnext_ondemand', False), 'the on-demand stamp must not survive'
    assert not hasattr(module, '_hf_hook'), 'the on-demand hook must be removed'


# ============================================================
# get_module_names
# ============================================================

def test_module_names_reads_specs_on_modular_pipelines():
    # transformer_2 exists only in the specs, so only the specs branch can find it
    pipe = FakeModularPipe({'transformer': PlainModule(), 'vae': BridgeModule(), 'scheduler': object()}, spec_only={'transformer_2': PlainModule()})
    names = sd_offload.get_module_names(pipe)
    assert names == ['transformer', 'transformer_2', 'vae'], f'got {names}'
    assert 'canvas_short_edge' not in names, 'config scalars must not be enumerated'


def test_module_names_ignores_the_component_registry_on_classic_pipelines():
    # DiffusionPipeline.components raises when its config and signature disagree
    class RaisingPipe(FakePipe):
        @property
        def components(self):
            raise ValueError('config and signature disagree')

    pipe = RaisingPipe({'transformer': PlainModule(), 'vae': BridgeModule()})
    names = sd_offload.get_module_names(pipe)
    assert names == ['transformer', 'vae'], f'got {names}'


# ============================================================
# upstream markers the roles depend on
# ============================================================

def test_autoencoders_carry_the_entry_bridge():
    from diffusers import AutoencoderKL, VQModel
    missing = [cls.__name__ for cls in (AutoencoderKL, VQModel) if not sd_offload.has_entry_bridge(cls)]
    assert not missing, f'no entry bridge detected on {missing}'


def test_denoisers_do_not_carry_the_entry_bridge():
    from diffusers import SD3Transformer2DModel, UNet2DConditionModel
    bridged = [cls.__name__ for cls in (UNet2DConditionModel, SD3Transformer2DModel) if sd_offload.has_entry_bridge(cls)]
    assert not bridged, f'entry bridge detected on denoisers {bridged}'


def test_upstream_still_opts_hunyuandit_out_of_group_offload():
    from diffusers import HunyuanDiT2DModel
    assert hasattr(HunyuanDiT2DModel, '_supports_group_offloading'), 'upstream renamed the opt-out attribute the roles read'
    assert HunyuanDiT2DModel._supports_group_offloading is False, 'upstream now supports HunyuanDiT group offload, so its ondemand route can go' # pylint: disable=protected-access


def test_mageflow_vae_carries_the_entry_bridge():
    from pipelines.mageflow.autoencoder_mage_vae import AutoencoderMageVAE
    assert sd_offload.has_entry_bridge(AutoencoderMageVAE), 'the mageflow vae lost its entry decorators'


# ============================================================
# Runner
# ============================================================

def run_all():
    log.warning('=== group_offload_role ===')
    cat = category('role')
    for fn in [
        test_role_table,
        test_role_bridge_overrides_denoiser_slot_name,
        test_role_upstream_optout_overrides_denoiser_slot_name,
        test_role_undecorated_entry_points_stay_resident,
        test_role_unknown_component_is_aux,
        test_role_never_offload_list_matches_a_class_name,
        test_role_never_offload_list_matches_a_component_name,
        test_role_excluded_model_type_stays_resident,
        test_role_empty_exclusions_match_nothing,
        test_role_main_list_has_no_encoder_names,
    ]:
        run_test(cat, fn)

    log.warning('=== dispatch ===')
    cat = category('dispatch')
    for fn in [
        test_dispatch_is_one_arm_per_component,
        test_dispatch_skips_non_modules,
        test_ondemand_list_tracks_the_stamps,
        test_force_sweep_moves_only_stamped_components,
        test_reapply_after_clearing_the_never_list_restores_hooks,
    ]:
        run_test(cat, fn)

    log.warning('=== on-demand placement ===')
    cat = category('ondemand')
    for fn in [
        test_ondemand_apply_returns_bool_and_is_idempotent,
        test_ondemand_apply_leaves_weights_on_cpu,
        test_resident_placement_clears_the_ondemand_stamp,
    ]:
        run_test(cat, fn)

    log.warning('=== enumeration ===')
    cat = category('enumeration')
    for fn in [
        test_module_names_reads_specs_on_modular_pipelines,
        test_module_names_ignores_the_component_registry_on_classic_pipelines,
    ]:
        run_test(cat, fn)

    log.warning('=== upstream markers ===')
    cat = category('upstream')
    for fn in [
        test_autoencoders_carry_the_entry_bridge,
        test_denoisers_do_not_carry_the_entry_bridge,
        test_upstream_still_opts_hunyuandit_out_of_group_offload,
        test_mageflow_vae_carries_the_entry_bridge,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = 0
    total_failed = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        status = 'PASS' if ok else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed [{status}]")
        total_passed += info['passed']
        total_failed += info['failed']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = run_all()
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
