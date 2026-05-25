#!/usr/bin/env python
"""
Offline unit tests for pipelines.native_transformer.

Covers the pure helpers that own per-arch knob handling:

- ``strip_prefix`` for single/multi prefix detection and mixed-prefix rejection
- ``partition_siblings`` for inline-sibling key partitioning
- ``check_forbidden_markers`` for structural-mismatch rejection
- ``is_noop_converter`` for diffusers no-op lambda detection
- ``validate_state_dict_load`` for unexpected / missing key handling
- ``register`` / ``lookup`` registry behavior and default spec synthesis
- ``auto_pickup_converter`` for diffusers ``SINGLE_FILE_LOADABLE_CLASSES`` integration
- ``TransformerSpec`` / ``SiblingSpec`` defaults

Plus one end-to-end ``load`` test against a tiny mock module that exercises
the read -> strip -> convert -> from_config -> load_state_dict -> validate
pipeline without needing a real diffusers transformer or hf_hub_download.

No running server required.

Usage:
    python test/test-native-transformer.py
"""

import os
import sys
import tempfile

import torch
import safetensors.torch

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)

os.environ['SD_INSTALL_QUIET'] = '1'

# Bootstrap cmd_args before any module that pulls in shared.py.
import modules.cmd_args  # pylint: disable=wrong-import-position
import installer  # pylint: disable=wrong-import-position
_orig_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    modules.cmd_args.parse_args()
finally:
    sys.argv = _orig_argv
installer.add_args(modules.cmd_args.parser)
modules.cmd_args.parsed, _ = modules.cmd_args.parser.parse_known_args([])

from modules.errors import log                          # pylint: disable=wrong-import-position
from pipelines import native_transformer as nt          # pylint: disable=wrong-import-position


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
# strip_prefix
# ============================================================

def test_strip_prefix_bare_keys_pass_through():
    sd = {'layers.0.weight': 1, 'layers.0.bias': 2}
    out = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert out == sd, 'bare keys must pass through unchanged'


def test_strip_prefix_dominant_single_variant():
    sd = {f'model.diffusion_model.layers.{i}.weight': i for i in range(10)}
    out = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert all(k.startswith('layers.') for k in out)
    assert len(out) == 10


def test_strip_prefix_picks_longest_match_first():
    """``model.diffusion_model.`` must beat ``diffusion_model.`` when both match."""
    sd = {
        'model.diffusion_model.layers.0.weight': 1,
        'model.diffusion_model.layers.1.weight': 2,
    }
    out = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    # If shorter prefix matched, keys would start with 'model.'
    assert 'layers.0.weight' in out
    assert 'layers.1.weight' in out
    assert not any(k.startswith('model.') for k in out)


def test_strip_prefix_mixed_prefixes_raises():
    sd = {
        'model.diffusion_model.layers.0.weight': 1,
        'net.layers.0.weight': 2,
    }
    try:
        nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'mixed prefixes' in str(e)


def test_strip_prefix_net_variant():
    sd = {'net.layers.0.weight': 1, 'net.layers.1.bias': 2}
    out = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert set(out.keys()) == {'layers.0.weight', 'layers.1.bias'}


def test_strip_prefix_diffusion_model_variant():
    sd = {'diffusion_model.layers.0.weight': 1, 'diffusion_model.layers.1.bias': 2}
    out = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert set(out.keys()) == {'layers.0.weight', 'layers.1.bias'}


def test_strip_prefix_custom_prefix_set():
    sd = {'lora_unet_blocks_0.weight': 1, 'lora_unet_blocks_1.weight': 2}
    out = nt.strip_prefix(sd, ('lora_unet_',), 'Test')
    assert set(out.keys()) == {'blocks_0.weight', 'blocks_1.weight'}


# ============================================================
# partition_siblings
# ============================================================

def test_partition_siblings_empty_spec_returns_state_dict_unchanged():
    sd = {'a': 1, 'b': 2}
    transformer_sd, siblings = nt.partition_siblings(sd, {})
    assert transformer_sd == sd
    assert siblings == {}


def test_partition_siblings_no_matches_keeps_all_in_transformer():
    sd = {'layers.0.weight': 1, 'layers.1.weight': 2}
    siblings_spec = {'llm_adapter': nt.SiblingSpec(subfolder='llm_adapter', inline_prefix='llm_adapter.')}
    transformer_sd, siblings = nt.partition_siblings(sd, siblings_spec)
    assert transformer_sd == sd
    assert siblings == {'llm_adapter': {}}


def test_partition_siblings_single_sibling_split():
    sd = {
        'layers.0.weight': 'tx0',
        'layers.1.weight': 'tx1',
        'llm_adapter.input_proj.weight': 'ad0',
        'llm_adapter.output_proj.weight': 'ad1',
    }
    siblings_spec = {'llm_adapter': nt.SiblingSpec(subfolder='llm_adapter', inline_prefix='llm_adapter.')}
    transformer_sd, siblings = nt.partition_siblings(sd, siblings_spec)
    assert set(transformer_sd.keys()) == {'layers.0.weight', 'layers.1.weight'}
    assert set(siblings['llm_adapter'].keys()) == {'input_proj.weight', 'output_proj.weight'}
    assert siblings['llm_adapter']['input_proj.weight'] == 'ad0'


def test_partition_siblings_multiple_siblings():
    sd = {
        'layers.0.weight': 'tx',
        'sibling_a.x.weight': 'a0',
        'sibling_b.y.weight': 'b0',
        'sibling_b.z.weight': 'b1',
    }
    siblings_spec = {
        'sibling_a': nt.SiblingSpec(subfolder='a', inline_prefix='sibling_a.'),
        'sibling_b': nt.SiblingSpec(subfolder='b', inline_prefix='sibling_b.'),
    }
    transformer_sd, siblings = nt.partition_siblings(sd, siblings_spec)
    assert list(transformer_sd.keys()) == ['layers.0.weight']
    assert set(siblings['sibling_a'].keys()) == {'x.weight'}
    assert set(siblings['sibling_b'].keys()) == {'y.weight', 'z.weight'}


# ============================================================
# check_forbidden_markers
# ============================================================

def test_forbidden_markers_passes_when_absent():
    sd = {'layers.0.weight': 1}
    markers = (('legacy.marker.weight', 'old format'),)
    nt.check_forbidden_markers(sd, markers, 'Test', '/tmp/x.safetensors')
    # no exception = pass


def test_forbidden_markers_raises_when_present():
    sd = {'layers.0.weight': 1, 'legacy.marker.weight': 2}
    markers = (('legacy.marker.weight', 'old Cosmos 1.0 structure'),)
    try:
        nt.check_forbidden_markers(sd, markers, 'Test', '/tmp/x.safetensors')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        msg = str(e)
        assert 'old Cosmos 1.0 structure' in msg
        assert 'legacy.marker.weight' in msg


def test_forbidden_markers_empty_tuple_no_op():
    sd = {'layers.0.weight': 1}
    nt.check_forbidden_markers(sd, (), 'Test', '/tmp/x.safetensors')


# ============================================================
# is_noop_converter
# ============================================================

def test_noop_converter_identity_lambda():
    fn = lambda checkpoint, **kwargs: checkpoint  # pylint: disable=unnecessary-lambda-assignment
    assert nt.is_noop_converter(fn) is True


def test_noop_converter_real_function():
    def real(checkpoint, **kwargs):  # pylint: disable=unused-argument
        return {k.replace('a.', 'b.'): v for k, v in checkpoint.items()}
    assert nt.is_noop_converter(real) is False


def test_noop_converter_lambda_with_modification():
    fn = lambda checkpoint, **kwargs: {k: v.float() for k, v in checkpoint.items()}  # pylint: disable=unnecessary-lambda-assignment
    assert nt.is_noop_converter(fn) is False


# ============================================================
# validate_state_dict_load
# ============================================================

def test_validate_accepts_buffer_only_missing():
    nt.validate_state_dict_load(
        component_name='transformer',
        missing=['rope.freqs', 'pos_embedder.pos'],
        unexpected=[],
        acceptable_missing=('rope.', 'pos_embedder.'),
    )


def test_validate_rejects_unexpected():
    try:
        nt.validate_state_dict_load(
            component_name='transformer',
            missing=[],
            unexpected=['some.junk.weight'],
            acceptable_missing=(),
        )
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'unexpected' in str(e)
        assert 'some.junk.weight' in str(e)


def test_validate_rejects_hard_missing():
    try:
        nt.validate_state_dict_load(
            component_name='transformer',
            missing=['layers.0.weight', 'rope.freqs'],
            unexpected=[],
            acceptable_missing=('rope.',),
        )
        raise AssertionError('expected ValueError')
    except ValueError as e:
        msg = str(e)
        assert 'missing' in msg
        assert 'layers.0.weight' in msg
        # Buffer-only missing must not show up in the hard-missing list
        assert msg.count('rope.freqs') == 0


def test_validate_empty_passes():
    nt.validate_state_dict_load(
        component_name='transformer',
        missing=[],
        unexpected=[],
        acceptable_missing=(),
    )


# ============================================================
# register / lookup
# ============================================================

class FakeTransformer:
    """Minimal stand-in for a diffusers transformer class."""


class FakeTransformer2:
    """Second stand-in for register/lookup tests."""


def test_register_with_explicit_spec():
    nt.REGISTRY.clear()
    spec = nt.TransformerSpec(cls=FakeTransformer, subfolder='custom_sub')
    nt.register(FakeTransformer, spec)
    assert nt.lookup(FakeTransformer) is spec


def test_register_with_default_spec():
    nt.REGISTRY.clear()
    nt.register(FakeTransformer)
    spec = nt.lookup(FakeTransformer)
    assert spec.cls is FakeTransformer
    assert spec.subfolder == 'transformer'
    assert spec.prefixes == nt.DEFAULT_PREFIXES
    assert spec.converter is None
    assert spec.siblings == {}


def test_register_idempotent_replaces():
    nt.REGISTRY.clear()
    spec1 = nt.TransformerSpec(cls=FakeTransformer, subfolder='sub_one')
    spec2 = nt.TransformerSpec(cls=FakeTransformer, subfolder='sub_two')
    nt.register(FakeTransformer, spec1)
    nt.register(FakeTransformer, spec2)
    assert nt.lookup(FakeTransformer) is spec2


def test_register_rejects_mismatched_cls():
    try:
        nt.register(FakeTransformer, nt.TransformerSpec(cls=FakeTransformer2))
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'does not match' in str(e)


def test_lookup_synthesizes_default_for_unregistered():
    nt.REGISTRY.clear()
    spec = nt.lookup(FakeTransformer)
    assert spec.cls is FakeTransformer
    assert spec.subfolder == 'transformer'  # default
    assert spec.converter is None  # FakeTransformer has no diffusers entry


# ============================================================
# auto_pickup_converter
# ============================================================

def test_auto_pickup_returns_none_for_unknown_class():
    assert nt.auto_pickup_converter(FakeTransformer) is None


def test_auto_pickup_returns_real_diffusers_converter():
    import diffusers
    fn = nt.auto_pickup_converter(diffusers.FluxTransformer2DModel)
    assert fn is not None
    assert callable(fn)
    assert fn.__name__ == 'convert_flux_transformer_checkpoint_to_diffusers'


def test_auto_pickup_skips_noop_converter_qwen():
    """QwenImageTransformer2DModel registers a no-op lambda in diffusers;
    auto_pickup_converter must return None so the spec falls back to no
    converter (the user-registered spec can override with a real converter)."""
    import diffusers
    assert nt.auto_pickup_converter(diffusers.QwenImageTransformer2DModel) is None


# ============================================================
# TransformerSpec / SiblingSpec defaults
# ============================================================

def test_transformer_spec_defaults():
    spec = nt.TransformerSpec(cls=FakeTransformer)
    assert spec.subfolder == 'transformer'
    assert spec.prefixes == ('model.diffusion_model.', 'diffusion_model.', 'net.')
    assert spec.converter is None
    assert spec.siblings == {}
    assert spec.acceptable_missing == ('rope.', 'pos_embedder.', 'learnable_pos_embed.')
    assert spec.forbidden_markers == ()


def test_sibling_spec_defaults():
    spec = nt.SiblingSpec(subfolder='llm_adapter', inline_prefix='llm_adapter.')
    assert spec.subfolder == 'llm_adapter'
    assert spec.inline_prefix == 'llm_adapter.'
    assert spec.acceptable_missing == ()


def test_transformer_spec_is_frozen():
    spec = nt.TransformerSpec(cls=FakeTransformer)
    try:
        spec.subfolder = 'changed'  # type: ignore[misc]
    except Exception as e:  # pylint: disable=broad-except
        assert 'FrozenInstanceError' in type(e).__name__ or 'frozen' in str(e).lower()
        return
    raise AssertionError('expected FrozenInstanceError')


# ============================================================
# Integration: end-to-end load() with a tiny mock module
# ============================================================
# We sidestep diffusers + hf_hub_download by patching:
#   - ``fetch_component_config`` to return a hand-rolled config dict
#   - ``model_quant.get_dit_args`` / ``model_quant.get_quant_type`` / quant
#     application to no-ops (we only want to test the load path itself).
# The mock cls is a torch.nn.Module subclass whose ``from_config`` constructs
# a fresh module of the expected shape; ``load_state_dict`` is the standard
# PyTorch method.

class MockMiniTransformer(torch.nn.Module):
    """Tiny stand-in: linear in -> linear out, plus a nested rope sub-module
    holding a buffer the trainer state dict won't carry. Nested mirrors how
    real DiTs structure rope / pos_embedder buffers."""

    @classmethod
    def from_config(cls, config: dict) -> 'MockMiniTransformer':
        return cls(dim=config['dim'])

    def __init__(self, dim: int):
        super().__init__()
        self.in_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)
        self.rope = torch.nn.Module()
        self.rope.register_buffer('freqs', torch.zeros(dim))


def write_fixture(state_dict_keys: dict, fd: int, path: str) -> str:
    os.close(fd)
    safetensors.torch.save_file(state_dict_keys, path)
    return path


def test_load_end_to_end_with_bfl_prefix_no_converter():
    """Exercise the full load pipeline: read .safetensors, strip prefix,
    no converter, instantiate via from_config, load weights, validate.
    """
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        # Save with model.diffusion_model. prefix; in_proj.* and out_proj.*
        # are the real weights the mock cls expects after the strip.
        raw = {
            'model.diffusion_model.in_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.in_proj.bias': torch.zeros(dim),
            'model.diffusion_model.out_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.out_proj.bias': torch.zeros(dim),
        }
        write_fixture(raw, fd, path)

        # Patch fetch_component_config to return our hand-rolled config.
        orig_fetch = nt.fetch_component_config
        nt.fetch_component_config = lambda repo, sub: {'dim': dim}

        # Patch quant helpers (we only care about the load path).
        from modules import model_quant
        orig_get_dit = model_quant.get_dit_args
        orig_get_qtype = model_quant.get_quant_type
        orig_do_post = model_quant.do_post_load_quant
        model_quant.get_dit_args = lambda *a, **k: ({}, {})
        model_quant.get_quant_type = lambda *a, **k: None
        model_quant.do_post_load_quant = lambda *a, **k: None

        try:
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, siblings = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
            )
        finally:
            nt.fetch_component_config = orig_fetch
            model_quant.get_dit_args = orig_get_dit
            model_quant.get_quant_type = orig_get_qtype
            model_quant.do_post_load_quant = orig_do_post

        assert isinstance(transformer, MockMiniTransformer)
        assert transformer.in_proj.weight.shape == (dim, dim)
        # Weights from the fixture should match what was loaded.
        loaded_in_w = transformer.in_proj.weight.detach().cpu()
        fixture_in_w = raw['model.diffusion_model.in_proj.weight'].to(loaded_in_w.dtype)
        assert torch.allclose(loaded_in_w, fixture_in_w)
        assert siblings == {}
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_end_to_end_with_sibling_partition():
    """Bundled-sibling case: file carries both transformer and sibling weights,
    sibling_classes supplies the runtime sibling class, partition routes each
    half into its target."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        sibling_dim = 4
        raw = {
            # Transformer half (after strip).
            'model.diffusion_model.in_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.in_proj.bias': torch.zeros(dim),
            'model.diffusion_model.out_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.out_proj.bias': torch.zeros(dim),
            # Sibling half (after strip + sibling partition).
            'model.diffusion_model.sibling.in_proj.weight': torch.randn(sibling_dim, sibling_dim),
            'model.diffusion_model.sibling.in_proj.bias': torch.zeros(sibling_dim),
            'model.diffusion_model.sibling.out_proj.weight': torch.randn(sibling_dim, sibling_dim),
            'model.diffusion_model.sibling.out_proj.bias': torch.zeros(sibling_dim),
        }
        write_fixture(raw, fd, path)

        orig_fetch = nt.fetch_component_config

        def patched_fetch(_repo, sub):
            return {'dim': dim if sub == 'transformer' else sibling_dim}

        nt.fetch_component_config = patched_fetch

        from modules import model_quant
        orig_get_dit = model_quant.get_dit_args
        orig_get_qtype = model_quant.get_quant_type
        orig_do_post = model_quant.do_post_load_quant
        model_quant.get_dit_args = lambda *a, **k: ({}, {})
        model_quant.get_quant_type = lambda *a, **k: None
        model_quant.do_post_load_quant = lambda *a, **k: None

        try:
            spec = nt.TransformerSpec(
                cls=MockMiniTransformer,
                siblings={
                    'sibling': nt.SiblingSpec(
                        subfolder='sibling',
                        inline_prefix='sibling.',
                        acceptable_missing=('rope.',),
                    ),
                },
            )
            transformer, siblings = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                sibling_classes={'sibling': MockMiniTransformer},
            )
        finally:
            nt.fetch_component_config = orig_fetch
            model_quant.get_dit_args = orig_get_dit
            model_quant.get_quant_type = orig_get_qtype
            model_quant.do_post_load_quant = orig_do_post

        assert isinstance(transformer, MockMiniTransformer)
        assert transformer.in_proj.weight.shape == (dim, dim)
        assert 'sibling' in siblings
        assert isinstance(siblings['sibling'], MockMiniTransformer)
        assert siblings['sibling'].in_proj.weight.shape == (sibling_dim, sibling_dim)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_raises_on_missing_sibling_class():
    """Sibling keys present in file but caller forgot to supply the class."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        sibling_dim = 4
        raw = {
            'model.diffusion_model.in_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.in_proj.bias': torch.zeros(dim),
            'model.diffusion_model.out_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.out_proj.bias': torch.zeros(dim),
            'model.diffusion_model.sibling.in_proj.weight': torch.randn(sibling_dim, sibling_dim),
        }
        write_fixture(raw, fd, path)

        orig_fetch = nt.fetch_component_config
        nt.fetch_component_config = lambda repo, sub: {'dim': dim}

        from modules import model_quant
        orig_get_dit = model_quant.get_dit_args
        orig_get_qtype = model_quant.get_quant_type
        orig_do_post = model_quant.do_post_load_quant
        model_quant.get_dit_args = lambda *a, **k: ({}, {})
        model_quant.get_quant_type = lambda *a, **k: None
        model_quant.do_post_load_quant = lambda *a, **k: None

        try:
            spec = nt.TransformerSpec(
                cls=MockMiniTransformer,
                siblings={'sibling': nt.SiblingSpec(subfolder='s', inline_prefix='sibling.')},
            )
            raised = False
            try:
                nt.load(
                    local_file=path,
                    repo_id='fake/repo',
                    spec=spec,
                    diffusers_cfg={},
                    sibling_classes={},  # missing!
                )
            except ValueError as e:
                raised = True
                assert "'sibling'" in str(e)
                assert 'sibling_classes' in str(e)
            assert raised, 'expected ValueError'
        finally:
            nt.fetch_component_config = orig_fetch
            model_quant.get_dit_args = orig_get_dit
            model_quant.get_quant_type = orig_get_qtype
            model_quant.do_post_load_quant = orig_do_post
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_rejects_non_safetensors():
    spec = nt.TransformerSpec(cls=MockMiniTransformer)
    try:
        nt.load(
            local_file='/tmp/some.gguf',
            repo_id='fake/repo',
            spec=spec,
            diffusers_cfg={},
        )
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert '.safetensors' in str(e)


# ============================================================
# Run
# ============================================================

def run_all():
    log.warning('=== strip_prefix ===')
    cat = category('strip')
    for fn in [
        test_strip_prefix_bare_keys_pass_through,
        test_strip_prefix_dominant_single_variant,
        test_strip_prefix_picks_longest_match_first,
        test_strip_prefix_mixed_prefixes_raises,
        test_strip_prefix_net_variant,
        test_strip_prefix_diffusion_model_variant,
        test_strip_prefix_custom_prefix_set,
    ]:
        run_test(cat, fn)

    log.warning('=== partition_siblings ===')
    cat = category('partition')
    for fn in [
        test_partition_siblings_empty_spec_returns_state_dict_unchanged,
        test_partition_siblings_no_matches_keeps_all_in_transformer,
        test_partition_siblings_single_sibling_split,
        test_partition_siblings_multiple_siblings,
    ]:
        run_test(cat, fn)

    log.warning('=== forbidden_markers ===')
    cat = category('forbidden')
    for fn in [
        test_forbidden_markers_passes_when_absent,
        test_forbidden_markers_raises_when_present,
        test_forbidden_markers_empty_tuple_no_op,
    ]:
        run_test(cat, fn)

    log.warning('=== noop_converter detection ===')
    cat = category('noop')
    for fn in [
        test_noop_converter_identity_lambda,
        test_noop_converter_real_function,
        test_noop_converter_lambda_with_modification,
    ]:
        run_test(cat, fn)

    log.warning('=== validate_state_dict_load ===')
    cat = category('validate')
    for fn in [
        test_validate_accepts_buffer_only_missing,
        test_validate_rejects_unexpected,
        test_validate_rejects_hard_missing,
        test_validate_empty_passes,
    ]:
        run_test(cat, fn)

    log.warning('=== register / lookup ===')
    cat = category('registry')
    for fn in [
        test_register_with_explicit_spec,
        test_register_with_default_spec,
        test_register_idempotent_replaces,
        test_register_rejects_mismatched_cls,
        test_lookup_synthesizes_default_for_unregistered,
    ]:
        run_test(cat, fn)

    log.warning('=== auto_pickup_converter ===')
    cat = category('autopickup')
    for fn in [
        test_auto_pickup_returns_none_for_unknown_class,
        test_auto_pickup_returns_real_diffusers_converter,
        test_auto_pickup_skips_noop_converter_qwen,
    ]:
        run_test(cat, fn)

    log.warning('=== TransformerSpec / SiblingSpec ===')
    cat = category('specs')
    for fn in [
        test_transformer_spec_defaults,
        test_sibling_spec_defaults,
        test_transformer_spec_is_frozen,
    ]:
        run_test(cat, fn)

    log.warning('=== end-to-end load ===')
    cat = category('load')
    for fn in [
        test_load_end_to_end_with_bfl_prefix_no_converter,
        test_load_end_to_end_with_sibling_partition,
        test_load_raises_on_missing_sibling_class,
        test_load_rejects_non_safetensors,
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
