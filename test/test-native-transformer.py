#!/usr/bin/env python
"""
Offline unit tests for pipelines.native_transformer.

Covers the pure helpers that own per-arch knob handling:

- ``drop_companion_keys`` for filtering bundled TE/VAE families out of all-in-one files
- ``strip_prefix`` for single/multi prefix detection and mixed-prefix rejection
- ``partition_siblings`` for inline-sibling key partitioning
- ``check_forbidden_markers`` for structural-mismatch rejection
- ``detect_comfy_quant`` for comfy_quant marker detection and format gating
- ``remap_comfy_quant`` for comfy_quant -> SDNQ key translation
- ``is_noop_converter`` for diffusers no-op lambda detection
- ``validate_state_dict_load`` for unexpected / missing key handling
- ``make_default_spec`` default-spec synthesis with diffusers converter pickup
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
# drop_companion_keys
# ============================================================

def test_drop_companion_keys_no_companions_pass_through():
    sd = {'net.blocks.0.weight': 1, 'net.blocks.1.weight': 2}
    out = nt.drop_companion_keys(sd, nt.DEFAULT_IGNORED_PREFIXES, 'Test')
    assert out == sd, 'files without companion keys must pass through unchanged'


def test_drop_companion_keys_filters_all_in_one_layout():
    sd = {
        'net.blocks.0.weight': 1,
        'net.llm_adapter.proj.weight': 2,
        'cond_stage_model.qwen3_06b.transformer.model.embed_tokens.weight': 3,
        'first_stage_model.decoder.conv1.weight': 4,
        'vae.decoder.conv_in.weight': 5,
        'text_encoders.clip_l.weight': 6,
    }
    out = nt.drop_companion_keys(sd, nt.DEFAULT_IGNORED_PREFIXES, 'Test')
    assert set(out.keys()) == {'net.blocks.0.weight', 'net.llm_adapter.proj.weight'}


def test_drop_companion_keys_raises_when_nothing_left():
    sd = {
        'cond_stage_model.te.weight': 1,
        'first_stage_model.decoder.weight': 2,
    }
    try:
        nt.drop_companion_keys(sd, nt.DEFAULT_IGNORED_PREFIXES, 'Test')
        raise AssertionError('expected ValueError')
    except ValueError as e:
        assert 'no transformer keys' in str(e)


def test_drop_companion_keys_empty_prefix_tuple_no_op():
    sd = {'cond_stage_model.te.weight': 1}
    out = nt.drop_companion_keys(sd, (), 'Test')
    assert out == sd, 'empty ignored_prefixes must disable filtering'


def test_drop_then_strip_all_in_one_layout():
    """Companion filtering must clear the way for normal prefix detection."""
    sd = {
        'net.blocks.0.weight': 1,
        'net.final_layer.weight': 2,
        'cond_stage_model.te.weight': 3,
        'first_stage_model.decoder.weight': 4,
    }
    out = nt.drop_companion_keys(sd, nt.DEFAULT_IGNORED_PREFIXES, 'Test')
    out, prefix = nt.strip_prefix(out, nt.DEFAULT_PREFIXES, 'Test')
    assert prefix == 'net.'
    assert set(out.keys()) == {'blocks.0.weight', 'final_layer.weight'}


# ============================================================
# strip_prefix
# ============================================================

def test_strip_prefix_bare_keys_pass_through():
    sd = {'layers.0.weight': 1, 'layers.0.bias': 2}
    out, prefix = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert out == sd, 'bare keys must pass through unchanged'
    assert prefix == '', 'bare keys must report an empty prefix'


def test_strip_prefix_dominant_single_variant():
    sd = {f'model.diffusion_model.layers.{i}.weight': i for i in range(10)}
    out, prefix = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert all(k.startswith('layers.') for k in out)
    assert len(out) == 10
    assert prefix == 'model.diffusion_model.'


def test_strip_prefix_picks_longest_match_first():
    """``model.diffusion_model.`` must beat ``diffusion_model.`` when both match."""
    sd = {
        'model.diffusion_model.layers.0.weight': 1,
        'model.diffusion_model.layers.1.weight': 2,
    }
    out, prefix = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    # If shorter prefix matched, keys would start with 'model.'
    assert 'layers.0.weight' in out
    assert 'layers.1.weight' in out
    assert not any(k.startswith('model.') for k in out)
    assert prefix == 'model.diffusion_model.'


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
    out, prefix = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert set(out.keys()) == {'layers.0.weight', 'layers.1.bias'}
    assert prefix == 'net.'


def test_strip_prefix_diffusion_model_variant():
    sd = {'diffusion_model.layers.0.weight': 1, 'diffusion_model.layers.1.bias': 2}
    out, prefix = nt.strip_prefix(sd, nt.DEFAULT_PREFIXES, 'Test')
    assert set(out.keys()) == {'layers.0.weight', 'layers.1.bias'}
    assert prefix == 'diffusion_model.'


def test_strip_prefix_custom_prefix_set():
    sd = {'lora_unet_blocks_0.weight': 1, 'lora_unet_blocks_1.weight': 2}
    out, prefix = nt.strip_prefix(sd, ('lora_unet_',), 'Test')
    assert set(out.keys()) == {'blocks_0.weight', 'blocks_1.weight'}
    assert prefix == 'lora_unet_'


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
# detect_comfy_quant / remap_comfy_quant
# ============================================================

def comfy_marker(fmt: str) -> torch.Tensor:
    import json
    return torch.tensor(list(json.dumps({'format': fmt}).encode()), dtype=torch.uint8)


def test_detect_comfy_no_markers_returns_none():
    sd = {'blocks.0.attn.wq.weight': torch.zeros(2), 'blocks.0.attn.wq.bias': torch.zeros(2)}
    assert nt.detect_comfy_quant(sd, 'Test') is None


def test_detect_comfy_valid_markers():
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.tensor(0.5),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('int8_tensorwise'),
        'blocks.0.mlp.up.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.mlp.up.weight_scale': torch.tensor(0.25),
        'blocks.0.mlp.up.comfy_quant': comfy_marker('int8_tensorwise'),
        'norm.weight': torch.zeros(4),
    }
    detected = nt.detect_comfy_quant(sd, 'Test')
    assert detected is not None
    marked, fmt = detected
    assert set(marked) == {'blocks.0.attn.wq', 'blocks.0.mlp.up'}
    assert marked['blocks.0.attn.wq'] == {'format': 'int8_tensorwise'}
    assert fmt == 'int8_tensorwise'


def test_detect_comfy_unsupported_format_raises():
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4)),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('float8_e4m3fn_scaled'),
    }
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'float8_e4m3fn_scaled' in str(e)


def test_detect_comfy_malformed_marker_raises():
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4)),
        'blocks.0.attn.wq.comfy_quant': torch.tensor(list(b'not json'), dtype=torch.uint8),
    }
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'malformed' in str(e)


def test_detect_comfy_convrot_accepted():
    """ConvRot markers map onto SDNQ's Hadamard support; detection carries the
    per-layer fields through instead of rejecting."""
    import json
    payload = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 256, 'per_row': True})
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.comfy_quant': torch.tensor(list(payload.encode()), dtype=torch.uint8),
        'blocks.0.mlp.up.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.mlp.up.comfy_quant': comfy_marker('int8_tensorwise'),
    }
    marked, fmt = nt.detect_comfy_quant(sd, 'Test')
    assert fmt == 'int8_tensorwise'
    assert marked['blocks.0.attn.wq'].get('convrot') is True
    assert marked['blocks.0.attn.wq'].get('convrot_groupsize') == 256
    assert not marked['blocks.0.mlp.up'].get('convrot')


def test_detect_comfy_convrot_bad_groupsize_raises():
    """Regular Hadamards only exist for power-of-4 sizes; any other convrot
    group size cannot be a compatible rotation."""
    import json
    payload = json.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 8})
    sd = {
        'blocks.0.attn.wq.comfy_quant': torch.tensor(list(payload.encode()), dtype=torch.uint8),
    }
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'power of 4' in str(e)


def test_detect_comfy_marker_missing_format_field_raises():
    import json
    sd = {
        'blocks.0.attn.wq.comfy_quant': torch.tensor(list(json.dumps({'fmt': 'x'}).encode()), dtype=torch.uint8),
    }
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'malformed' in str(e)


def test_remap_comfy_renames_and_reshapes_scale():
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.tensor(0.5),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('int8_tensorwise'),
    }
    out = nt.remap_comfy_quant(sd, {'blocks.0.attn.wq'})
    assert set(out.keys()) == {'blocks.0.attn.wq.weight', 'blocks.0.attn.wq.scale'}
    assert out['blocks.0.attn.wq.scale'].shape == (1, 1)
    assert out['blocks.0.attn.wq.scale'].item() == 0.5
    assert out['blocks.0.attn.wq.weight'].dtype == torch.int8


def test_remap_comfy_passes_unmarked_keys_verbatim():
    unrelated_scale = torch.tensor(2.0)
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.tensor(0.5),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('int8_tensorwise'),
        'norm.weight': torch.ones(4),
        'other.weight_scale': unrelated_scale,
    }
    out = nt.remap_comfy_quant(sd, {'blocks.0.attn.wq'})
    assert 'norm.weight' in out
    assert out['other.weight_scale'] is unrelated_scale, 'weight_scale outside marked names must pass through untouched'


def test_remap_comfy_does_not_mutate_input():
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.tensor(0.5),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('int8_tensorwise'),
    }
    keys_before = set(sd.keys())
    nt.remap_comfy_quant(sd, {'blocks.0.attn.wq'})
    assert set(sd.keys()) == keys_before, 'input dict must not be mutated (read_state_dict caches it)'


def test_remap_comfy_drops_input_scale():
    """Optional activation-calibration sidecars are dropped for marked layers
    (SDNQ derives activation scales dynamically); unmarked ones pass through."""
    unrelated = torch.tensor(1.0)
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.tensor(0.5),
        'blocks.0.attn.wq.input_scale': torch.tensor(0.125),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('int8_tensorwise'),
        'other.input_scale': unrelated,
    }
    out = nt.remap_comfy_quant(sd, {'blocks.0.attn.wq'})
    assert 'blocks.0.attn.wq.input_scale' not in out
    assert out['other.input_scale'] is unrelated


# ============================================================
# header _quantization_metadata
# ============================================================

def quant_metadata(layers: dict) -> dict:
    import json
    return {'_quantization_metadata': json.dumps({'format_version': '1.0', 'layers': layers})}


def test_read_quant_metadata_absent_returns_none():
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        write_fixture({'a.weight': torch.zeros(2)}, fd, path)
        assert nt.read_quantization_metadata(path) is None
    finally:
        os.unlink(path)


def test_read_quant_metadata_parses_layers():
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        layers = {'blocks.0.attn.wq': {'format': 'int8_tensorwise'}}
        write_fixture({'blocks.0.attn.wq.weight': torch.zeros(2)}, fd, path, metadata=quant_metadata(layers))
        assert nt.read_quantization_metadata(path) == layers
    finally:
        os.unlink(path)


def test_read_quant_metadata_malformed_raises():
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        write_fixture({'a.weight': torch.zeros(2)}, fd, path, metadata={'_quantization_metadata': 'not json'})
        try:
            nt.read_quantization_metadata(path)
            raise AssertionError('expected OverrideArchMismatch')
        except nt.OverrideArchMismatch as e:
            assert 'malformed' in str(e)
    finally:
        os.unlink(path)


def test_read_quant_metadata_non_dict_layers_raises():
    import json
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        payload = json.dumps({'layers': {'blocks.0.attn.wq': 'int8_tensorwise'}})
        write_fixture({'a.weight': torch.zeros(2)}, fd, path, metadata={'_quantization_metadata': payload})
        try:
            nt.read_quantization_metadata(path)
            raise AssertionError('expected OverrideArchMismatch')
        except nt.OverrideArchMismatch as e:
            assert 'malformed' in str(e)
    finally:
        os.unlink(path)


def test_transcode_quant_metadata_synthesizes_markers():
    """Header entries become marker tensors for layers whose weight is present;
    entries without a matching weight (companion components) are skipped."""
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.weight_scale': torch.tensor(0.5),
    }
    layers = {
        'blocks.0.attn.wq': {'format': 'int8_tensorwise'},
        'text_encoder.mlp.up': {'format': 'int8_tensorwise'},
    }
    out, source = nt.transcode_quant_metadata(sd, layers)
    assert source == 'header'
    assert 'blocks.0.attn.wq.comfy_quant' in out
    assert 'text_encoder.mlp.up.comfy_quant' not in out
    assert 'blocks.0.attn.wq.comfy_quant' not in sd, 'input dict must not be mutated'
    marked, fmt = nt.detect_comfy_quant(out, 'Test')
    assert fmt == 'int8_tensorwise'
    assert marked['blocks.0.attn.wq'] == {'format': 'int8_tensorwise'}


def test_transcode_quant_metadata_header_wins():
    """When both forms are present the header entry overwrites the marker,
    matching the reference loader's precedence."""
    sd = {
        'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8),
        'blocks.0.attn.wq.comfy_quant': comfy_marker('float8_e4m3fn'),
    }
    layers = {'blocks.0.attn.wq': {'format': 'int8_tensorwise', 'full_precision_matrix_mult': True}}
    out, source = nt.transcode_quant_metadata(sd, layers)
    assert source == 'both'
    marked, fmt = nt.detect_comfy_quant(out, 'Test')
    assert fmt == 'int8_tensorwise'
    assert marked['blocks.0.attn.wq'].get('full_precision_matrix_mult') is True


def test_transcode_quant_metadata_no_matches_is_noop():
    sd = {'blocks.0.attn.wq.weight': torch.zeros((4, 4), dtype=torch.int8)}
    out, source = nt.transcode_quant_metadata(sd, {'unrelated.layer': {'format': 'int8_tensorwise'}})
    assert source == 'markers'
    assert out is sd


# ============================================================
# nvfp4: sdnq codec, scale unswizzle, detection
# ============================================================

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def to_blocked_reference(matrix: torch.Tensor) -> torch.Tensor:
    """cuBLAS block-scaling tile layout as written into nvfp4 containers;
    the loader's unswizzle must invert this exactly."""
    rows, cols = matrix.shape
    row_blocks = -(rows // -128)
    col_blocks = -(cols // -4)
    padded = torch.zeros((row_blocks * 128, col_blocks * 4), dtype=matrix.dtype)
    padded[:rows, :cols] = matrix
    blocks = padded.view(row_blocks, 128, col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(row_blocks * 128, col_blocks * 4)


def test_unswizzle_block_scales_roundtrip():
    for rows, groups in ((128, 4), (8, 3), (200, 10)):
        mat = torch.arange(rows * groups, dtype=torch.float32).reshape(rows, groups)
        back = nt.unswizzle_block_scales(to_blocked_reference(mat), rows, groups)
        assert torch.equal(back, mat), f'unswizzle mismatch for {(rows, groups)}'


def test_nvfp4_codec_ocp_table():
    """SDNQ's float4_e2m1fn decodes OCP FP4 E2M1 exactly, including the
    subnormal codes 1/9 as +/-0.5; nvfp4 containers adopt it directly."""
    from modules.sdnq.packed_float import unpack_float
    packed = torch.tensor([(2 * j) | (((2 * j) + 1) << 4) for j in range(8)], dtype=torch.uint8)
    dec = unpack_float(packed, 'float4_e2m1fn', torch.Size([16]))
    for code in range(16):
        assert float(dec[code]) == E2M1_VALUES[code], f'code {code}: {float(dec[code])} != {E2M1_VALUES[code]}'


def test_nvfp4_pack_ocp_grid_roundtrip():
    """Every OCP grid value survives a pack/unpack round trip exactly
    (subnormals included), and off-grid values land inside the value set.
    Exact nearest-rounding near the grid midpoints is not asserted: the
    packer's staged rounding may resolve boundary values to either side."""
    from modules.sdnq.packed_float import pack_float, unpack_float
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0]
    vals = torch.tensor(grid, dtype=torch.float32)
    out = unpack_float(pack_float(vals, 'float4_e2m1fn'), 'float4_e2m1fn', vals.shape)
    assert torch.equal(out, vals), f'grid values must roundtrip exactly: {out.tolist()}'
    off_grid = torch.tensor([0.2, 0.6, 0.9, 1.2, 2.4, 5.5, -0.4, -0.9, -1.7, -3.4, -5.9, 0.05, 0.99, -0.99], dtype=torch.float32)
    out = unpack_float(pack_float(off_grid, 'float4_e2m1fn'), 'float4_e2m1fn', off_grid.shape)
    for v, o in zip(off_grid.tolist(), out.tolist()):
        assert abs(o) in {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, f'{v} -> {o} outside the OCP set'


def test_detect_comfy_nvfp4_marker_accepted():
    import json
    payload = json.dumps({'format': 'nvfp4', 'group_size': 16, 'orig_dtype': 'torch.float16', 'orig_shape': [32, 32]})
    sd = {
        'in_proj.weight': torch.zeros((32, 16), dtype=torch.uint8),
        'in_proj.comfy_quant': torch.tensor(list(payload.encode()), dtype=torch.uint8),
    }
    marked, fmt = nt.detect_comfy_quant(sd, 'Test')
    assert fmt == 'nvfp4'
    assert marked['in_proj']['orig_shape'] == [32, 32]


def test_detect_comfy_nvfp4_bad_group_size_raises():
    import json
    payload = json.dumps({'format': 'nvfp4', 'group_size': 32})
    sd = {'a.comfy_quant': torch.tensor(list(payload.encode()), dtype=torch.uint8)}
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'group_size' in str(e)


def test_detect_comfy_nvfp4_convrot_raises():
    import json
    payload = json.dumps({'format': 'nvfp4', 'convrot': True, 'convrot_groupsize': 16})
    sd = {'a.comfy_quant': torch.tensor(list(payload.encode()), dtype=torch.uint8)}
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'convrot' in str(e)


# ============================================================
# model_probe quant detection (header-only, mirrors the loader's coverage)
# ============================================================

def test_probe_detects_header_quant_metadata():
    import json
    from modules import model_probe
    header = {
        '__metadata__': {'_quantization_metadata': json.dumps({'layers': {'blocks.0.wq': {'format': 'float8_e4m3fn'}}})},
        'blocks.0.wq.weight': {'dtype': 'F8_E4M3', 'shape': [4, 4]},
        'blocks.0.wq.weight_scale': {'dtype': 'F32', 'shape': []},
    }
    quant = model_probe.analyze_header(header)['quant']
    assert quant == {'scheme': 'comfy_quant', 'format': 'float8_e4m3fn', 'marked_layers': 1, 'source': 'header'}


def test_probe_resolves_nvfp4_from_weight_dtype():
    from modules import model_probe
    header = {
        'model.diffusion_model.blocks.0.wq.weight': {'dtype': 'U8', 'shape': [32, 16]},
        'model.diffusion_model.blocks.0.wq.weight_scale': {'dtype': 'F8_E4M3', 'shape': [128, 4]},
        'model.diffusion_model.blocks.0.wq.weight_scale_2': {'dtype': 'F32', 'shape': []},
        'model.diffusion_model.blocks.0.wq.comfy_quant': {'dtype': 'U8', 'shape': [60]},
    }
    quant = model_probe.analyze_header(header)['quant']
    assert quant['format'] == 'nvfp4'
    assert quant['source'] == 'weight-dtype'


def test_probe_fp8_markers_despite_u8_noise():
    """The marked layer's own weight dtype decides the format; unrelated U8
    tensors must not outvote it."""
    from modules import model_probe
    header = {
        'blocks.0.wq.weight': {'dtype': 'F8_E4M3', 'shape': [8, 8]},
        'blocks.0.wq.weight_scale': {'dtype': 'F32', 'shape': []},
        'blocks.0.wq.comfy_quant': {'dtype': 'U8', 'shape': [27]},
        'extra.blob1': {'dtype': 'U8', 'shape': [64]},
        'extra.blob2': {'dtype': 'U8', 'shape': [64]},
    }
    quant = model_probe.analyze_header(header)['quant']
    assert quant['format'] == 'float8_e4m3fn'


def test_probe_int8_markers():
    from modules import model_probe
    header = {
        'blocks.0.wq.weight': {'dtype': 'I8', 'shape': [8, 8]},
        'blocks.0.wq.weight_scale': {'dtype': 'F32', 'shape': []},
        'blocks.0.wq.comfy_quant': {'dtype': 'U8', 'shape': [29]},
    }
    quant = model_probe.analyze_header(header)['quant']
    assert quant['format'] == 'int8_tensorwise'
    assert quant['marked_layers'] == 1


def test_probe_plain_file_no_quant():
    from modules import model_probe
    header = {'blocks.0.wq.weight': {'dtype': 'BF16', 'shape': [8, 8]}}
    assert model_probe.analyze_header(header)['quant']['scheme'] is None


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
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
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
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
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
# make_default_spec
# ============================================================

class FakeTransformer:
    """Minimal stand-in for a diffusers transformer class."""


def test_make_default_spec_for_unknown_class():
    spec = nt.make_default_spec(FakeTransformer)
    assert spec.cls is FakeTransformer
    assert spec.subfolder == 'transformer'
    assert spec.prefixes == nt.DEFAULT_PREFIXES
    assert spec.converter is None  # no diffusers entry for FakeTransformer
    assert spec.siblings == {}
    assert spec.forbidden_markers == ()


def test_make_default_spec_picks_up_real_diffusers_converter():
    import diffusers
    spec = nt.make_default_spec(diffusers.FluxTransformer2DModel)
    assert spec.converter is not None
    assert spec.converter.__name__ == 'convert_flux_transformer_checkpoint_to_diffusers'


def test_make_default_spec_skips_qwen_image_noop():
    """QwenImageTransformer2DModel's diffusers entry is a no-op lambda; the
    default spec must NOT pick it up, leaving converter=None so the caller
    sees only their own (potentially absent) override."""
    import diffusers
    spec = nt.make_default_spec(diffusers.QwenImageTransformer2DModel)
    assert spec.converter is None


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
    assert spec.ignored_prefixes == ('cond_stage_model.', 'conditioner.', 'first_stage_model.', 'text_encoders.', 'vae.')
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


class MockKwargsTransformer(MockMiniTransformer):
    """Records the kwargs from_config received, so a test can assert the native
    path forwards caller kwargs to construction. Mirrors diffusers from_config,
    which accepts **kwargs."""

    last_kwargs: dict = {}

    @classmethod
    def from_config(cls, config: dict, **kwargs) -> 'MockKwargsTransformer':
        cls.last_kwargs = dict(kwargs)
        return cls(dim=config['dim'])


def write_fixture(state_dict_keys: dict, fd: int, path: str, metadata: dict | None = None) -> str:
    os.close(fd)
    safetensors.torch.save_file(state_dict_keys, path, metadata=metadata)
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


def test_load_forwards_kwargs_to_from_config():
    """Caller **kwargs reach cls.from_config through the native load path
    rather than being dropped."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = {
            'model.diffusion_model.in_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.in_proj.bias': torch.zeros(dim),
            'model.diffusion_model.out_proj.weight': torch.randn(dim, dim),
            'model.diffusion_model.out_proj.bias': torch.zeros(dim),
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

        MockKwargsTransformer.last_kwargs = {}
        try:
            spec = nt.TransformerSpec(cls=MockKwargsTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                low_cpu_mem_usage=True,
            )
        finally:
            nt.fetch_component_config = orig_fetch
            model_quant.get_dit_args = orig_get_dit
            model_quant.get_quant_type = orig_get_qtype
            model_quant.do_post_load_quant = orig_do_post

        assert MockKwargsTransformer.last_kwargs == {'low_cpu_mem_usage': True}
        assert isinstance(transformer, MockKwargsTransformer)
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


def crashing_converter(sd):
    """diffusers-style layer count that blows up when the block family is
    absent, mirroring convert_chroma_..._to_diffusers on a wrong-arch file."""
    return list(set(int(k.split('.')[1]) for k in sd if 'double_blocks.' in k))[-1]


def test_build_component_converter_crash_raises_mismatch():
    """A converter that crashes on wrong-arch keys is wrapped as
    OverrideArchMismatch (chaining the original), not the raw IndexError."""
    try:
        nt.build_component(
            component_name='transformer',
            state_dict={'blocks.0.self_attn.weight': torch.zeros(2)},
            config={'dim': 8},
            cls=MockMiniTransformer,
            converter=crashing_converter,
            acceptable_missing=(),
            quant_args={},
            quant_type=None,
        )
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'MockMiniTransformer' in str(e)
        assert isinstance(e.__cause__, IndexError), 'original error must be chained'


def test_build_component_shape_mismatch_is_hard_error():
    """A tensor shape mismatch on otherwise-matching keys stays a hard
    RuntimeError (the native size-mismatch message), it is NOT converted to
    OverrideArchMismatch and so does not silently fall back to base."""
    # MockMiniTransformer(dim=8) expects (8, 8) projections; feed (4, 4).
    sd = {
        'in_proj.weight': torch.randn(4, 4), 'in_proj.bias': torch.zeros(4),
        'out_proj.weight': torch.randn(4, 4), 'out_proj.bias': torch.zeros(4),
    }
    orig_display = nt.errors.display
    nt.errors.display = lambda *a, **k: None  # silence the expected traceback dump
    try:
        nt.build_component(
            component_name='transformer', state_dict=sd, config={'dim': 8},
            cls=MockMiniTransformer, converter=None, acceptable_missing=(),
            quant_args={}, quant_type=None,
        )
        raise AssertionError('expected RuntimeError')
    except nt.OverrideArchMismatch:
        raise AssertionError('shape mismatch must not be OverrideArchMismatch') from None
    except RuntimeError as e:
        assert 'size mismatch' in str(e).lower()
    finally:
        nt.errors.display = orig_display


def test_load_converter_crash_raises_mismatch():
    """End-to-end: a crashing converter surfaces from load() as
    OverrideArchMismatch so load_transformer can drop the override."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        raw = {'model.diffusion_model.blocks.0.self_attn.weight': torch.zeros(8, 8)}
        write_fixture(raw, fd, path)
        orig_fetch = nt.fetch_component_config
        nt.fetch_component_config = lambda repo, sub: {'dim': 8}
        from modules import model_quant
        orig_get_dit = model_quant.get_dit_args
        orig_get_qtype = model_quant.get_quant_type
        model_quant.get_dit_args = lambda *a, **k: ({}, {})
        model_quant.get_quant_type = lambda *a, **k: None
        try:
            spec = nt.TransformerSpec(cls=MockMiniTransformer, converter=crashing_converter)
            raised = False
            try:
                nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={})
            except nt.OverrideArchMismatch as e:
                raised = True
                assert 'MockMiniTransformer' in str(e)
            assert raised, 'expected OverrideArchMismatch'
        finally:
            nt.fetch_component_config = orig_fetch
            model_quant.get_dit_args = orig_get_dit
            model_quant.get_quant_type = orig_get_qtype
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ============================================================
# Ideogram 4 converter (fused community layout -> diffusers layout)
# ============================================================

def ideogram_converter():
    from pipelines.ideogram import convert_ideogram4_transformer_checkpoint
    return convert_ideogram4_transformer_checkpoint


def test_ideogram4_converter_splits_fused_qkv_weight():
    convert = ideogram_converter()
    fused = torch.arange(48, dtype=torch.float32).reshape(12, 4)
    out = convert({'layers.0.attention.qkv.weight': fused})
    assert set(out.keys()) == {f'layers.0.attention.{n}.weight' for n in ('to_q', 'to_k', 'to_v')}
    assert torch.equal(out['layers.0.attention.to_q.weight'], fused[0:4])
    assert torch.equal(out['layers.0.attention.to_k.weight'], fused[4:8])
    assert torch.equal(out['layers.0.attention.to_v.weight'], fused[8:12])


def test_ideogram4_converter_splits_rowwise_scale():
    convert = ideogram_converter()
    fused = torch.zeros((12, 4), dtype=torch.int8)
    scale = torch.arange(12, dtype=torch.float32).reshape(12, 1)
    out = convert({'layers.3.attention.qkv.weight': fused, 'layers.3.attention.qkv.weight_scale': scale})
    assert torch.equal(out['layers.3.attention.to_q.weight_scale'], scale[0:4])
    assert torch.equal(out['layers.3.attention.to_v.weight_scale'], scale[8:12])
    assert out['layers.3.attention.to_k.weight_scale'].shape == (4, 1)


def test_ideogram4_converter_scalar_scale_copied_not_sliced():
    convert = ideogram_converter()
    fused = torch.zeros((12, 4), dtype=torch.int8)
    scale = torch.tensor(0.5)
    out = convert({'layers.0.attention.qkv.weight': fused, 'layers.0.attention.qkv.weight_scale': scale})
    for name in ('to_q', 'to_k', 'to_v'):
        assert out[f'layers.0.attention.{name}.weight_scale'] is scale


def test_ideogram4_converter_duplicates_marker():
    convert = ideogram_converter()
    fused = torch.zeros((12, 4), dtype=torch.int8)
    marker = comfy_marker('int8_tensorwise')
    out = convert({'layers.0.attention.qkv.weight': fused, 'layers.0.attention.qkv.comfy_quant': marker})
    for name in ('to_q', 'to_k', 'to_v'):
        assert out[f'layers.0.attention.{name}.comfy_quant'] is marker


def test_ideogram4_converter_renames_o_with_sidecars():
    convert = ideogram_converter()
    sd = {
        'layers.5.attention.o.weight': torch.zeros((4, 4), dtype=torch.int8),
        'layers.5.attention.o.weight_scale': torch.tensor(1.0),
        'layers.5.attention.o.comfy_quant': comfy_marker('int8_tensorwise'),
    }
    out = convert(sd)
    assert set(out.keys()) == {
        'layers.5.attention.to_out.0.weight',
        'layers.5.attention.to_out.0.weight_scale',
        'layers.5.attention.to_out.0.comfy_quant',
    }


def test_ideogram4_converter_passthrough():
    convert = ideogram_converter()
    sd = {
        'input_proj.weight': torch.zeros(4),
        'llm_cond_proj.weight': torch.zeros(4),
        'layers.0.feed_forward.w1.weight': torch.zeros(4),
        'layers.0.attention.norm_q.weight': torch.zeros(4),
        'layers.0.adaln_modulation.bias': torch.zeros(4),
        'final_layer.linear.weight': torch.zeros(4),
    }
    out = convert(sd)
    assert set(out.keys()) == set(sd.keys())
    for k, v in sd.items():
        assert out[k] is v


def test_ideogram4_converter_defensive_bias_split():
    convert = ideogram_converter()
    fused = torch.zeros((12, 4), dtype=torch.float32)
    bias = torch.arange(12, dtype=torch.float32)
    out = convert({'layers.0.attention.qkv.weight': fused, 'layers.0.attention.qkv.bias': bias})
    assert torch.equal(out['layers.0.attention.to_k.bias'], bias[4:8])


def test_ideogram4_converter_does_not_mutate_input():
    convert = ideogram_converter()
    sd = {
        'layers.0.attention.qkv.weight': torch.zeros((12, 4)),
        'layers.0.attention.o.weight': torch.zeros((4, 4)),
        'input_proj.weight': torch.zeros(4),
    }
    keys_before = set(sd.keys())
    convert(sd)
    assert set(sd.keys()) == keys_before


# ============================================================
# Integration: comfy_quant pre-quantized load
# ============================================================
# Same patching strategy as the plain load tests, plus pinned SDNQ opts so the
# builder's settings reads are deterministic. dtype=float32 makes the dequant
# math bit-exact against a manual int8 * scale reference.

def comfy_fixture(dim: int, fmt: str = 'int8_tensorwise') -> dict:
    """comfy_quant export shape: in_proj carries a quantized weight + scalar
    fp32 scale + marker; out_proj and biases stay plain f16."""
    if fmt == 'float8_e4m3fn':
        weight = torch.randn(dim, dim).to(torch.float8_e4m3fn)
    else:
        weight = torch.randint(-128, 127, (dim, dim), dtype=torch.int8)
    return {
        'model.diffusion_model.in_proj.weight': weight,
        'model.diffusion_model.in_proj.weight_scale': torch.tensor(0.03125, dtype=torch.float32),
        'model.diffusion_model.in_proj.comfy_quant': comfy_marker(fmt),
        'model.diffusion_model.in_proj.bias': torch.randn(dim, dtype=torch.float16),
        'model.diffusion_model.out_proj.weight': torch.randn(dim, dim, dtype=torch.float16),
        'model.diffusion_model.out_proj.bias': torch.zeros(dim, dtype=torch.float16),
    }


class ComfyTestEnv:
    """Patches quant helpers + config fetch and pins the SDNQ opts the
    prequantized builder reads, restoring everything on exit.
    ``fp8_compile_supported`` pins the fp8 storage-remap decision so tests
    behave identically regardless of the host GPU and compile settings."""

    def __init__(self, dim: int, fp8_compile_supported: bool = True):
        self.dim = dim
        self.fp8_compile_supported = fp8_compile_supported

    def __enter__(self):
        from modules import model_quant, shared
        from modules.sdnq import common as sdnq_common
        self.shared = shared
        self.sdnq_common = sdnq_common
        self.orig_fetch = nt.fetch_component_config
        self.orig_get_dit = model_quant.get_dit_args
        self.orig_get_qtype = model_quant.get_quant_type
        self.orig_do_post = model_quant.do_post_load_quant
        self.orig_fp8_supported = sdnq_common.is_fp8_compile_supported
        self.orig_check_compile = sdnq_common.check_torch_compile
        self.model_quant = model_quant
        nt.fetch_component_config = lambda repo, sub: {'dim': self.dim}
        model_quant.get_dit_args = lambda *a, **k: ({}, {})
        model_quant.get_quant_type = lambda *a, **k: None
        model_quant.do_post_load_quant = lambda *a, **k: None
        sdnq_common.is_fp8_compile_supported = self.fp8_compile_supported
        sdnq_common.check_torch_compile = lambda: not self.fp8_compile_supported
        self.orig_opts = {
            'sdnq_use_quantized_matmul': shared.opts.sdnq_use_quantized_matmul,
            'sdnq_dequantize_fp32': shared.opts.sdnq_dequantize_fp32,
            'diffusers_offload_mode': shared.opts.diffusers_offload_mode,
        }
        shared.opts.data['sdnq_use_quantized_matmul'] = False
        shared.opts.data['sdnq_dequantize_fp32'] = True
        shared.opts.data['diffusers_offload_mode'] = 'model'  # force CPU placement
        return self

    def __exit__(self, *exc):
        nt.fetch_component_config = self.orig_fetch
        self.model_quant.get_dit_args = self.orig_get_dit
        self.model_quant.get_quant_type = self.orig_get_qtype
        self.model_quant.do_post_load_quant = self.orig_do_post
        self.sdnq_common.is_fp8_compile_supported = self.orig_fp8_supported
        self.sdnq_common.check_torch_compile = self.orig_check_compile
        for key, value in self.orig_opts.items():
            self.shared.opts.data[key] = value
        return False


def test_load_comfy_int8_end_to_end():
    """Full pipeline on a comfy_quant fixture: the marked linear becomes an
    SDNQ int8 layer holding the file's exact tensors, the unmarked linear
    loads plain, and the forward pass matches a dequantized reference."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim)
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, siblings = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        assert siblings == {}
        in_proj = transformer.in_proj
        assert in_proj.__class__.__name__ == 'SDNQLinear', f'marked layer is {in_proj.__class__.__name__}'
        dq = in_proj.sdnq_dequantizer
        assert dq.weights_dtype == 'int8'
        assert dq.group_size == -1
        assert dq.use_quantized_matmul is False
        assert dq.original_shape == (dim, dim)

        # File tensors adopted bit-exact: int8 codes and fp32 scalar scale.
        assert in_proj.weight.dtype == torch.int8
        assert torch.equal(in_proj.weight.detach().cpu(), raw['model.diffusion_model.in_proj.weight'])
        assert in_proj.scale.dtype == torch.float32
        assert tuple(in_proj.scale.shape) == (1, 1)
        assert in_proj.scale.item() == raw['model.diffusion_model.in_proj.weight_scale'].item()
        assert in_proj.zero_point is None

        # Dequant matches comfy semantics exactly: fp = int8.float() * scale.
        expected = raw['model.diffusion_model.in_proj.weight'].float() * raw['model.diffusion_model.in_proj.weight_scale']
        dequantized = dq(in_proj.weight, in_proj.scale, zero_point=None, svd_up=None, svd_down=None)
        assert torch.equal(dequantized.detach().cpu(), expected)

        # Forward parity against a reference built from the dequantized weight.
        x = torch.randn(2, dim)
        out = in_proj(x)
        ref = torch.nn.functional.linear(x, expected, in_proj.bias.detach().cpu())
        assert torch.allclose(out.detach().cpu(), ref, atol=1e-6)

        # Unmarked layer loads as a plain Linear cast to the target dtype.
        assert transformer.out_proj.__class__ is torch.nn.Linear
        assert transformer.out_proj.weight.dtype == torch.float32
        assert torch.allclose(
            transformer.out_proj.weight.detach().cpu(),
            raw['model.diffusion_model.out_proj.weight'].float(),
        )

        # Marked as SDNQ-quantized so downstream never re-quantizes.
        assert getattr(transformer, 'quantization_config', None) is not None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_fp8_end_to_end():
    """float8_e4m3fn variant: same container, fp8 storage. The marked linear
    must keep fp8 codes and dequantize as weight.float() * scale."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim, fmt='float8_e4m3fn')
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        in_proj = transformer.in_proj
        assert in_proj.__class__.__name__ == 'SDNQLinear'
        assert in_proj.sdnq_dequantizer.weights_dtype == 'float8_e4m3fn'
        assert in_proj.weight.dtype == torch.float8_e4m3fn
        assert in_proj.scale.dtype == torch.float32

        expected = raw['model.diffusion_model.in_proj.weight'].float() * raw['model.diffusion_model.in_proj.weight_scale']
        dequantized = in_proj.sdnq_dequantizer(in_proj.weight, in_proj.scale, zero_point=None, svd_up=None, svd_down=None)
        assert torch.equal(dequantized.detach().cpu(), expected)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_fp8_remaps_storage_when_compile_unsupported():
    """When compiled dequant is on but the hardware cannot compile e4m3,
    fp8 weights are adopted through the uint8-backed codec: identical
    decoded values from storage the compiler can always touch."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim, fmt='float8_e4m3fn')
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim, fp8_compile_supported=False):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        in_proj = transformer.in_proj
        assert in_proj.__class__.__name__ == 'SDNQLinear'
        assert in_proj.sdnq_dequantizer.weights_dtype == 'float8_e4m3fn_sdnq'
        assert in_proj.weight.dtype == torch.uint8
        expected = raw['model.diffusion_model.in_proj.weight'].float() * raw['model.diffusion_model.in_proj.weight_scale']
        dequantized = in_proj.sdnq_dequantizer(in_proj.weight, in_proj.scale, zero_point=None, svd_up=None, svd_down=None)
        assert torch.equal(dequantized.detach().cpu(), expected)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_metadata_end_to_end():
    """Marker-less container: quant info only in the header
    _quantization_metadata, layer names carrying the file's tensor prefix.
    The load path must re-key through the prefix strip and land on the same
    prequantized build as the marker form."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim)
        del raw['model.diffusion_model.in_proj.comfy_quant']
        layers = {'model.diffusion_model.in_proj': {'format': 'int8_tensorwise'}}
        write_fixture(raw, fd, path, metadata=quant_metadata(layers))

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        in_proj = transformer.in_proj
        assert in_proj.__class__.__name__ == 'SDNQLinear'
        assert in_proj.weight.dtype == torch.int8
        assert torch.equal(in_proj.weight.detach().cpu(), raw['model.diffusion_model.in_proj.weight'])
        assert transformer.out_proj.__class__ is torch.nn.Linear
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_metadata_bare_names_end_to_end():
    """Marker-less container without a tensor prefix (official Comfy-Org
    exports): metadata names match the bare keys and need no re-keying."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = {key[len('model.diffusion_model.'):]: value for key, value in comfy_fixture(dim).items()}
        del raw['in_proj.comfy_quant']
        raw['in_proj.input_scale'] = torch.tensor(0.125, dtype=torch.float32)
        layers = {'in_proj': {'format': 'int8_tensorwise'}}
        write_fixture(raw, fd, path, metadata=quant_metadata(layers))

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        assert transformer.in_proj.__class__.__name__ == 'SDNQLinear'
        assert torch.equal(transformer.in_proj.weight.detach().cpu(), raw['in_proj.weight'])
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_full_precision_mm_excluded_from_matmul():
    """Layers flagged full_precision_matrix_mult land on the config's
    per-layer matmul exclusion list, which apply_sdnq_options_to_model
    honors when the user enables quantized matmul."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        import json
        raw = comfy_fixture(dim)
        payload = {'format': 'int8_tensorwise', 'full_precision_matrix_mult': True}
        raw['model.diffusion_model.in_proj.comfy_quant'] = torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        assert 'in_proj.weight' in transformer.quantization_config.modules_to_not_use_matmul
        assert transformer.in_proj.sdnq_dequantizer.use_quantized_matmul is False
    finally:
        if os.path.exists(path):
            os.unlink(path)


def nvfp4_fixture(dim: int, with_marker: bool = True) -> tuple[dict, torch.Tensor]:
    """A valid nvfp4 container for one marked linear, plus the reference
    dequantized weight per the format definition: E2M1 decode * unswizzled
    e4m3 block scale * fp32 global scale."""
    import json
    groups = dim // 16
    gen = torch.Generator().manual_seed(7)
    codes = torch.randint(0, 16, (dim, dim), generator=gen, dtype=torch.uint8)
    block_scales = (torch.rand(dim, groups, generator=gen) * 2 + 0.5).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.0125, dtype=torch.float32)
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32)
    reference = lut[codes.long()].reshape(dim, groups, 16) * (block_scales.float() * global_scale).unsqueeze(-1)
    reference = reference.reshape(dim, dim)
    packed = torch.bitwise_or(torch.bitwise_left_shift(codes[:, 0::2], 4), codes[:, 1::2])  # even element in the high nibble
    sd = {
        'model.diffusion_model.in_proj.weight': packed,
        'model.diffusion_model.in_proj.weight_scale': to_blocked_reference(block_scales),
        'model.diffusion_model.in_proj.weight_scale_2': global_scale,
        'model.diffusion_model.in_proj.bias': torch.randn(dim, generator=gen).to(torch.float16),
        'model.diffusion_model.out_proj.weight': torch.randn(dim, dim, generator=gen).to(torch.float16),
        'model.diffusion_model.out_proj.bias': torch.zeros(dim, dtype=torch.float16),
    }
    if with_marker:
        marker = json.dumps({'format': 'nvfp4', 'group_size': 16, 'orig_dtype': 'torch.float16', 'orig_shape': [dim, dim]})
        sd['model.diffusion_model.in_proj.comfy_quant'] = torch.tensor(list(marker.encode()), dtype=torch.uint8)
    return sd, reference


def test_load_comfy_nvfp4_end_to_end():
    """Full pipeline on an nvfp4 container: packed uint8 weights survive with
    swapped nibble order, block scales unswizzle and fold with the global
    scale into fp32 grouped scales, and the dequantized weight matches the
    format reference exactly."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 32
        raw, reference = nvfp4_fixture(dim)
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        in_proj = transformer.in_proj
        assert in_proj.__class__.__name__ == 'SDNQLinear'
        dq = in_proj.sdnq_dequantizer
        assert dq.weights_dtype == 'float4_e2m1fn'
        assert dq.group_size == 16
        assert tuple(dq.quantized_weight_shape) == (dim, dim // 16, 16)
        assert dq.re_quantize_for_matmul is True
        assert in_proj.weight.dtype == torch.uint8
        assert tuple(in_proj.weight.shape) == (dim, dim // 2)
        assert in_proj.scale.dtype == torch.float32
        assert tuple(in_proj.scale.shape) == (dim, dim // 16, 1)

        dequantized = dq(in_proj.weight, in_proj.scale, zero_point=None, svd_up=None, svd_down=None)
        assert torch.equal(dequantized.detach().cpu(), reference)

        x = torch.randn(2, dim)
        out = in_proj(x)
        ref = torch.nn.functional.linear(x, reference, in_proj.bias.detach().cpu())
        assert torch.allclose(out.detach().cpu(), ref, atol=1e-5)

        assert transformer.out_proj.__class__ is torch.nn.Linear
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_nvfp4_metadata_form():
    """Marker-less nvfp4 (header metadata only, no orig_shape/group_size):
    the format constant and module dimensions fill the gaps."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 32
        raw, reference = nvfp4_fixture(dim, with_marker=False)
        layers = {'model.diffusion_model.in_proj': {'format': 'nvfp4'}}
        write_fixture(raw, fd, path, metadata=quant_metadata(layers))

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(
                local_file=path,
                repo_id='fake/repo',
                spec=spec,
                diffusers_cfg={},
                dtype=torch.float32,
            )

        in_proj = transformer.in_proj
        assert in_proj.__class__.__name__ == 'SDNQLinear'
        assert in_proj.sdnq_dequantizer.weights_dtype == 'float4_e2m1fn'
        dequantized = in_proj.sdnq_dequantizer(in_proj.weight, in_proj.scale, zero_point=None, svd_up=None, svd_down=None)
        assert torch.equal(dequantized.detach().cpu(), reference)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_nvfp4_orig_shape_mismatch_raises():
    """A marker whose orig_shape disagrees with the model config is a wrong
    file for the class; reject so the base-repo fallback engages."""
    import json
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 32
        raw, _ = nvfp4_fixture(dim)
        marker = json.dumps({'format': 'nvfp4', 'group_size': 16, 'orig_shape': [dim, dim * 2]})
        raw['model.diffusion_model.in_proj.comfy_quant'] = torch.tensor(list(marker.encode()), dtype=torch.uint8)
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            try:
                nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={}, dtype=torch.float32)
                raise AssertionError('expected OverrideArchMismatch')
            except nt.OverrideArchMismatch as e:
                assert 'orig_shape' in str(e)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_detect_comfy_mixed_formats_raises():
    """One file mixing int8 and fp8 layers has no single SDNQ mapping; reject."""
    sd = {
        'a.weight': torch.zeros((4, 4), dtype=torch.int8),
        'a.comfy_quant': comfy_marker('int8_tensorwise'),
        'b.weight': torch.zeros((4, 4), dtype=torch.float8_e4m3fn),
        'b.comfy_quant': comfy_marker('float8_e4m3fn'),
    }
    try:
        nt.detect_comfy_quant(sd, 'Test')
        raise AssertionError('expected OverrideArchMismatch')
    except nt.OverrideArchMismatch as e:
        assert 'mixes formats' in str(e)


def test_load_comfy_marker_dtype_mismatch_raises():
    """A marker whose declared format does not match the stored weight dtype
    (mislabeled container) must be rejected, not silently misinterpreted."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim)
        raw['model.diffusion_model.in_proj.weight'] = torch.randn(dim, dim, dtype=torch.float16)  # marker says int8
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            raised = False
            try:
                nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={})
            except nt.OverrideArchMismatch as e:
                raised = True
                assert 'torch.int8' in str(e) and 'torch.float16' in str(e)
            assert raised, 'expected OverrideArchMismatch'
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_unsupported_format_raises_mismatch():
    """A comfy_quant file in a format sdnext cannot adopt must surface as
    OverrideArchMismatch so load_transformer falls back to the base repo."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim)
        raw['model.diffusion_model.in_proj.comfy_quant'] = comfy_marker('float8_e4m3fn_scaled')
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            raised = False
            try:
                nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={})
            except nt.OverrideArchMismatch as e:
                raised = True
                assert 'float8_e4m3fn_scaled' in str(e)
            assert raised, 'expected OverrideArchMismatch'
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_marker_for_unknown_module_raises_mismatch():
    """A marker naming a module the target class does not have means the file
    belongs to a different arch; must fall back, not crash."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim)
        raw['model.diffusion_model.ghost.weight'] = torch.zeros((dim, dim), dtype=torch.int8)
        raw['model.diffusion_model.ghost.weight_scale'] = torch.tensor(1.0)
        raw['model.diffusion_model.ghost.comfy_quant'] = comfy_marker('int8_tensorwise')
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            raised = False
            try:
                nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={})
            except nt.OverrideArchMismatch as e:
                raised = True
                assert 'ghost' in str(e)
            assert raised, 'expected OverrideArchMismatch'
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_transformer_syncs_loaded_unet():
    """A full model load that consumes the UNET dropdown override must mark it
    as loaded, so the queued sd_unet onchange callback does not trigger a
    second, redundant full model reload."""
    from modules import sd_unet, shared
    from pipelines import generic_transformer as gt

    fd, path = tempfile.mkstemp(suffix='.safetensors')
    dim = 8
    raw = {
        'model.diffusion_model.in_proj.weight': torch.randn(dim, dim),
        'model.diffusion_model.in_proj.bias': torch.zeros(dim),
        'model.diffusion_model.out_proj.weight': torch.randn(dim, dim),
        'model.diffusion_model.out_proj.bias': torch.zeros(dim),
    }
    write_fixture(raw, fd, path)

    orig_unet_opt = shared.opts.sd_unet
    orig_loaded = sd_unet.loaded_unet
    sd_unet.unet_dict['mock-unet'] = path
    shared.opts.data['sd_unet'] = 'mock-unet'
    sd_unet.loaded_unet = None
    try:
        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer = gt.load_transformer('fake/repo', cls_name=MockMiniTransformer, native_spec=spec)
        assert transformer is not None
        assert sd_unet.loaded_unet == 'mock-unet', f'override consumed but loaded_unet={sd_unet.loaded_unet}'
    finally:
        sd_unet.unet_dict.pop('mock-unet', None)
        shared.opts.data['sd_unet'] = orig_unet_opt
        sd_unet.loaded_unet = orig_loaded
        if os.path.exists(path):
            os.unlink(path)


class MockFusedAttention(torch.nn.Module):
    """Split-attention module tree matching Ideogram4Transformer2DModel's
    naming (to_q/to_k/to_v + to_out ModuleList), fed by fused checkpoints."""

    def __init__(self, dim: int):
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim, bias=False), torch.nn.Dropout(0.0)])


class MockFusedBlock(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = MockFusedAttention(dim)


class MockFusedTransformer(torch.nn.Module):
    @classmethod
    def from_config(cls, config: dict, **kwargs) -> 'MockFusedTransformer':
        return cls(dim=config['dim'])

    def __init__(self, dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([MockFusedBlock(dim)])
        self.input_proj = torch.nn.Linear(dim, dim)


def fused_spec():
    from pipelines.ideogram import convert_ideogram4_transformer_checkpoint
    return nt.TransformerSpec(
        cls=MockFusedTransformer,
        converter=convert_ideogram4_transformer_checkpoint,
        converter_handles_quant=True,
    )


def fused_fixture(dim: int, quantized: bool = True) -> dict:
    if quantized:
        qkv = torch.randint(-128, 127, (3 * dim, dim), dtype=torch.int8)
    else:
        qkv = torch.randn(3 * dim, dim, dtype=torch.float16)
    raw = {
        'model.diffusion_model.layers.0.attention.qkv.weight': qkv,
        'model.diffusion_model.layers.0.attention.o.weight': torch.randn(dim, dim, dtype=torch.float16),
        'model.diffusion_model.input_proj.weight': torch.randn(dim, dim, dtype=torch.float16),
        'model.diffusion_model.input_proj.bias': torch.zeros(dim, dtype=torch.float16),
    }
    if quantized:
        raw['model.diffusion_model.layers.0.attention.qkv.weight_scale'] = torch.rand(3 * dim, 1, dtype=torch.float32)
        raw['model.diffusion_model.layers.0.attention.qkv.comfy_quant'] = comfy_marker('int8_tensorwise')
    return raw


def test_load_fused_comfy_end_to_end():
    """Quant-aware converter + comfy adoption: fused int8 qkv with row-wise
    scales lands as three SDNQ linears holding the exact row slices."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = fused_fixture(dim, quantized=True)
        write_fixture(raw, fd, path)
        with ComfyTestEnv(dim):
            transformer, _ = nt.load(local_file=path, repo_id='fake/repo', spec=fused_spec(), diffusers_cfg={}, dtype=torch.float32)

        attn = transformer.layers[0].attention
        fused_w = raw['model.diffusion_model.layers.0.attention.qkv.weight']
        fused_s = raw['model.diffusion_model.layers.0.attention.qkv.weight_scale']
        for i, name in enumerate(('to_q', 'to_k', 'to_v')):
            layer = getattr(attn, name)
            assert layer.__class__.__name__ == 'SDNQLinear', f'{name} is {layer.__class__.__name__}'
            assert layer.weight.dtype == torch.int8
            assert torch.equal(layer.weight.detach().cpu(), fused_w[i * dim:(i + 1) * dim])
            assert tuple(layer.scale.shape) == (dim, 1), f'{name} scale shape {layer.scale.shape}'
            assert torch.equal(layer.scale.detach().cpu(), fused_s[i * dim:(i + 1) * dim])
            expected = fused_w[i * dim:(i + 1) * dim].float() * fused_s[i * dim:(i + 1) * dim]
            dequantized = layer.sdnq_dequantizer(layer.weight, layer.scale, zero_point=None, svd_up=None, svd_down=None)
            assert torch.equal(dequantized.detach().cpu(), expected)
        # unmarked o.weight passes through the rename as a plain Linear
        assert attn.to_out[0].__class__ is torch.nn.Linear
        assert torch.allclose(attn.to_out[0].weight.detach().cpu(), raw['model.diffusion_model.layers.0.attention.o.weight'].float())
        assert transformer.input_proj.__class__ is torch.nn.Linear
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_fused_bf16_end_to_end():
    """Converter-first on a plain float fused file: no markers, standard load
    path, split weights land on the class-native linears."""
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = fused_fixture(dim, quantized=False)
        write_fixture(raw, fd, path)
        with ComfyTestEnv(dim):
            transformer, _ = nt.load(local_file=path, repo_id='fake/repo', spec=fused_spec(), diffusers_cfg={}, dtype=torch.float32)

        attn = transformer.layers[0].attention
        fused_w = raw['model.diffusion_model.layers.0.attention.qkv.weight']
        for i, name in enumerate(('to_q', 'to_k', 'to_v')):
            layer = getattr(attn, name)
            assert layer.__class__ is torch.nn.Linear
            assert torch.allclose(layer.weight.detach().cpu(), fused_w[i * dim:(i + 1) * dim].float())
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_fused_convrot_end_to_end():
    """Fused qkv with a convrot marker: the converter duplicates the marker to
    the three split layers, each of which loads with hadamard geometry. Row
    slicing commutes with the in-features rotation, so slices stay lossless."""
    import json as json_mod
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim, group_size = 8, 4
        raw = fused_fixture(dim, quantized=True)
        payload = json_mod.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': group_size, 'per_row': True})
        raw['model.diffusion_model.layers.0.attention.qkv.comfy_quant'] = torch.tensor(list(payload.encode()), dtype=torch.uint8)
        raw['model.diffusion_model.layers.0.attention.qkv.weight_scale'] = torch.rand(3 * dim, 1, dtype=torch.float32)
        write_fixture(raw, fd, path)
        with ComfyTestEnv(dim):
            transformer, _ = nt.load(local_file=path, repo_id='fake/repo', spec=fused_spec(), diffusers_cfg={}, dtype=torch.float32)
        attn = transformer.layers[0].attention
        for name in ('to_q', 'to_k', 'to_v'):
            dq = getattr(attn, name).sdnq_dequantizer
            assert dq.use_hadamard is True, f'{name} lost the convrot flag'
            assert dq.hadamard_group_size == group_size
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_build_component_comfy_skips_converter_when_not_quant_aware():
    """With converter_handles_quant=False (the default for every other arch),
    a comfy state dict must reach the prequantized path without the converter
    ever running; a crashing converter proves it was not invoked."""
    dim = 8
    raw = comfy_fixture(dim)
    sd = {k[len('model.diffusion_model.'):]: v for k, v in raw.items()}
    with ComfyTestEnv(dim):
        component = nt.build_component(
            component_name='transformer',
            state_dict=sd,
            config={'dim': dim},
            cls=MockMiniTransformer,
            converter=crashing_converter,
            acceptable_missing=('rope.',),
            quant_args={},
            quant_type='SDNQConfig',
            dtype=torch.float32,
        )
    assert component.in_proj.__class__.__name__ == 'SDNQLinear'


def test_load_transformer_secondary_slot_syncs_tracker():
    """The secondary slot consumes sd_unet_secondary and syncs its own
    tracker without touching the primary slot."""
    from modules import sd_unet, shared
    from pipelines import generic_transformer as gt

    fd, path = tempfile.mkstemp(suffix='.safetensors')
    dim = 8
    raw = {
        'model.diffusion_model.in_proj.weight': torch.randn(dim, dim),
        'model.diffusion_model.in_proj.bias': torch.zeros(dim),
        'model.diffusion_model.out_proj.weight': torch.randn(dim, dim),
        'model.diffusion_model.out_proj.bias': torch.zeros(dim),
    }
    write_fixture(raw, fd, path)

    orig_primary_opt = shared.opts.sd_unet
    orig_secondary_opt = shared.opts.sd_unet_secondary
    orig_primary = sd_unet.loaded_unet
    orig_secondary = sd_unet.loaded_unet_secondary
    sd_unet.unet_dict['mock-unet-2'] = path
    shared.opts.data['sd_unet'] = 'Default'
    shared.opts.data['sd_unet_secondary'] = 'mock-unet-2'
    sd_unet.loaded_unet = None
    sd_unet.loaded_unet_secondary = None
    try:
        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer = gt.load_transformer('fake/repo', cls_name=MockMiniTransformer, native_spec=spec, override_slot='secondary')
        assert transformer is not None
        assert sd_unet.loaded_unet_secondary == 'mock-unet-2'
        assert sd_unet.loaded_unet is None, 'primary tracker must stay untouched'
    finally:
        sd_unet.unet_dict.pop('mock-unet-2', None)
        shared.opts.data['sd_unet'] = orig_primary_opt
        shared.opts.data['sd_unet_secondary'] = orig_secondary_opt
        sd_unet.loaded_unet = orig_primary
        sd_unet.loaded_unet_secondary = orig_secondary
        if os.path.exists(path):
            os.unlink(path)


def regular_hadamard(n: int) -> torch.Tensor:
    """Regular Hadamard construction the ConvRot format and SDNQ share:
    4x4 base, Kronecker recursion, 1/sqrt(n) normalization. Symmetric and
    involutory, so rotation and inverse are the same matrix."""
    h4 = torch.tensor([[1., 1., 1., -1.], [1., 1., -1., 1.], [1., -1., 1., 1.], [-1., 1., 1., 1.]])
    h = h4
    size = 4
    while size < n:
        h = torch.kron(h, h4)
        size *= 4
    return h / (n ** 0.5)


def comfy_convrot_quantize(weight: torch.Tensor, group_size: int):
    """Reference ConvRot int8 quantizer (mirrors the ComfyUI runtime): rotate
    grouped in-features by the regular Hadamard, then row-wise symmetric int8."""
    h = regular_hadamard(group_size)
    out_f, in_f = weight.shape
    rotated = (weight.reshape(out_f, -1, group_size) @ h.T).reshape(out_f, in_f)
    scale = rotated.abs().amax(dim=-1, keepdim=True) / 127
    q = rotated.div(scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def test_load_comfy_convrot_end_to_end():
    """ConvRot parity: a layer quantized with the reference ConvRot math
    loads as an SDNQ hadamard layer whose dequant matches the reference dequant
    and recovers the original weight within int8 quantization error. The plain
    second layer proves per-layer mixing within one file."""
    import json as json_mod
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim, group_size = 8, 4
        original = torch.randn(dim, dim)
        q, scale = comfy_convrot_quantize(original, group_size)
        marker = json_mod.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': group_size, 'per_row': True})
        raw = {
            'model.diffusion_model.in_proj.weight': q,
            'model.diffusion_model.in_proj.weight_scale': scale,
            'model.diffusion_model.in_proj.comfy_quant': torch.tensor(list(marker.encode()), dtype=torch.uint8),
            'model.diffusion_model.in_proj.bias': torch.zeros(dim, dtype=torch.float16),
            'model.diffusion_model.out_proj.weight': torch.randint(-128, 127, (dim, dim), dtype=torch.int8),
            'model.diffusion_model.out_proj.weight_scale': torch.tensor(0.03125, dtype=torch.float32),
            'model.diffusion_model.out_proj.comfy_quant': comfy_marker('int8_tensorwise'),
            'model.diffusion_model.out_proj.bias': torch.zeros(dim, dtype=torch.float16),
        }
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            transformer, _ = nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={}, dtype=torch.float32)

        in_proj = transformer.in_proj
        dq = in_proj.sdnq_dequantizer
        assert dq.use_hadamard is True
        assert dq.hadamard_group_size == group_size
        assert tuple(in_proj.scale.shape) == (dim, 1)

        dequantized = dq(in_proj.weight, in_proj.scale, zero_point=None, svd_up=None, svd_down=None).detach().cpu()
        h = regular_hadamard(group_size)
        reference = ((q.float() * scale).reshape(dim, -1, group_size) @ h.T).reshape(dim, dim)
        assert torch.allclose(dequantized, reference, atol=1e-5), 'SDNQ hadamard dequant diverges from the ConvRot reference'
        assert torch.allclose(dequantized, original, atol=0.15), 'dequant does not recover the original weight'

        # plain layer in the same file stays rotation-free
        assert transformer.out_proj.sdnq_dequantizer.use_hadamard is False
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_load_comfy_convrot_nondivisible_falls_back():
    """A convrot group size that does not divide in_features cannot be undone;
    the file must fall back to the base repo."""
    import json as json_mod
    fd, path = tempfile.mkstemp(suffix='.safetensors')
    try:
        dim = 8
        raw = comfy_fixture(dim)
        marker = json_mod.dumps({'format': 'int8_tensorwise', 'convrot': True, 'convrot_groupsize': 16})
        raw['model.diffusion_model.in_proj.comfy_quant'] = torch.tensor(list(marker.encode()), dtype=torch.uint8)
        write_fixture(raw, fd, path)

        with ComfyTestEnv(dim):
            spec = nt.TransformerSpec(cls=MockMiniTransformer)
            raised = False
            try:
                nt.load(local_file=path, repo_id='fake/repo', spec=spec, diffusers_cfg={})
            except nt.OverrideArchMismatch as e:
                raised = True
                assert 'does not divide' in str(e)
            assert raised, 'expected OverrideArchMismatch'
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_build_component_comfy_preempts_sdnq_fresh_quant():
    """When SDNQ on-load quant settings are active (quant_type=SDNQConfig), a
    comfy_quant file must still take the pre-quantized path: fresh quant of
    int8 data would corrupt it. quant_args={} proves the comfy branch never
    reaches build_component_quantized (which requires a quantization_config)."""
    dim = 8
    raw = comfy_fixture(dim)
    sd = {k[len('model.diffusion_model.'):]: v for k, v in raw.items()}
    with ComfyTestEnv(dim):
        component = nt.build_component(
            component_name='transformer',
            state_dict=sd,
            config={'dim': dim},
            cls=MockMiniTransformer,
            converter=None,
            acceptable_missing=('rope.',),
            quant_args={},
            quant_type='SDNQConfig',
            dtype=torch.float32,
        )
    assert component.in_proj.__class__.__name__ == 'SDNQLinear'
    assert component.in_proj.weight.dtype == torch.int8


# ============================================================
# Run
# ============================================================

def run_all():
    log.warning('=== drop_companion_keys ===')
    cat = category('companion')
    for fn in [
        test_drop_companion_keys_no_companions_pass_through,
        test_drop_companion_keys_filters_all_in_one_layout,
        test_drop_companion_keys_raises_when_nothing_left,
        test_drop_companion_keys_empty_prefix_tuple_no_op,
        test_drop_then_strip_all_in_one_layout,
    ]:
        run_test(cat, fn)

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

    log.warning('=== comfy_quant detection / remap ===')
    cat = category('comfy')
    for fn in [
        test_detect_comfy_no_markers_returns_none,
        test_detect_comfy_valid_markers,
        test_detect_comfy_unsupported_format_raises,
        test_detect_comfy_malformed_marker_raises,
        test_detect_comfy_convrot_accepted,
        test_detect_comfy_convrot_bad_groupsize_raises,
        test_detect_comfy_marker_missing_format_field_raises,
        test_remap_comfy_renames_and_reshapes_scale,
        test_remap_comfy_passes_unmarked_keys_verbatim,
        test_remap_comfy_does_not_mutate_input,
        test_remap_comfy_drops_input_scale,
        test_read_quant_metadata_absent_returns_none,
        test_read_quant_metadata_parses_layers,
        test_read_quant_metadata_malformed_raises,
        test_read_quant_metadata_non_dict_layers_raises,
        test_transcode_quant_metadata_synthesizes_markers,
        test_transcode_quant_metadata_header_wins,
        test_transcode_quant_metadata_no_matches_is_noop,
    ]:
        run_test(cat, fn)

    log.warning('=== nvfp4 codec / unswizzle / load ===')
    cat = category('nvfp4')
    for fn in [
        test_unswizzle_block_scales_roundtrip,
        test_nvfp4_codec_ocp_table,
        test_nvfp4_pack_ocp_grid_roundtrip,
        test_detect_comfy_nvfp4_marker_accepted,
        test_detect_comfy_nvfp4_bad_group_size_raises,
        test_detect_comfy_nvfp4_convrot_raises,
        test_load_comfy_nvfp4_end_to_end,
        test_load_comfy_nvfp4_metadata_form,
        test_load_comfy_nvfp4_orig_shape_mismatch_raises,
    ]:
        run_test(cat, fn)

    log.warning('=== model_probe quant detection ===')
    cat = category('probe')
    for fn in [
        test_probe_detects_header_quant_metadata,
        test_probe_resolves_nvfp4_from_weight_dtype,
        test_probe_fp8_markers_despite_u8_noise,
        test_probe_int8_markers,
        test_probe_plain_file_no_quant,
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

    log.warning('=== make_default_spec ===')
    cat = category('default_spec')
    for fn in [
        test_make_default_spec_for_unknown_class,
        test_make_default_spec_picks_up_real_diffusers_converter,
        test_make_default_spec_skips_qwen_image_noop,
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
        test_load_forwards_kwargs_to_from_config,
        test_load_end_to_end_with_sibling_partition,
        test_load_raises_on_missing_sibling_class,
        test_load_rejects_non_safetensors,
        test_build_component_converter_crash_raises_mismatch,
        test_build_component_shape_mismatch_is_hard_error,
        test_load_converter_crash_raises_mismatch,
    ]:
        run_test(cat, fn)

    log.warning('=== comfy_quant load ===')
    cat = category('comfy_load')
    for fn in [
        test_load_comfy_int8_end_to_end,
        test_load_comfy_fp8_end_to_end,
        test_load_comfy_fp8_remaps_storage_when_compile_unsupported,
        test_load_comfy_metadata_end_to_end,
        test_load_comfy_metadata_bare_names_end_to_end,
        test_load_comfy_full_precision_mm_excluded_from_matmul,
        test_detect_comfy_mixed_formats_raises,
        test_load_comfy_marker_dtype_mismatch_raises,
        test_load_comfy_unsupported_format_raises_mismatch,
        test_load_comfy_marker_for_unknown_module_raises_mismatch,
        test_load_comfy_convrot_end_to_end,
        test_load_comfy_convrot_nondivisible_falls_back,
        test_load_transformer_syncs_loaded_unet,
        test_load_transformer_secondary_slot_syncs_tracker,
        test_build_component_comfy_preempts_sdnq_fresh_quant,
        test_build_component_comfy_skips_converter_when_not_quant_aware,
    ]:
        run_test(cat, fn)

    log.warning('=== ideogram4 converter / fused load ===')
    cat = category('ideogram4')
    for fn in [
        test_ideogram4_converter_splits_fused_qkv_weight,
        test_ideogram4_converter_splits_rowwise_scale,
        test_ideogram4_converter_scalar_scale_copied_not_sliced,
        test_ideogram4_converter_duplicates_marker,
        test_ideogram4_converter_renames_o_with_sidecars,
        test_ideogram4_converter_passthrough,
        test_ideogram4_converter_defensive_bias_split,
        test_ideogram4_converter_does_not_mutate_input,
        test_load_fused_comfy_end_to_end,
        test_load_fused_bf16_end_to_end,
        test_load_fused_convrot_end_to_end,
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
