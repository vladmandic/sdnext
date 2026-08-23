#!/usr/bin/env python
"""
Offline tests for the block mask input of the sdnq Triton attention kernel.

Covers:

- the kernel contract through the raw triton op, which takes the per query block count and
  ascending index list of kept kv blocks that get_block_mask_input builds from the int8 mask: a
  block mask is within one ulp of the same kernel fed the token-expanded dense mask, over a ragged
  tail with sub tiles past the end of the sequence, GQA, batch and head broadcasts, the contiguous padded list contract
  and an empty block row; bitwise against no mask with every block kept, and bitwise against the
  token mask when both arms carry one. The token-mask path has its own -inf-safe softmax
  update, which the compiler rounds differently by up to one ulp depending on tile and dtype; the
  block path shares the dense arithmetic, which the all-ones row proves
- the launcher's validation of the block mask
- the nesting filter: prune_configs keeps only tiles that sit inside one mask block and raises,
  naming the env vars, when none do
- the public entry: sdnq_triton_atten(block_mask=...) on the quantized paths, under the same one
  ulp and all-ones rules, bool and 3d masks normalized, and the backward entry refusing a block
  mask; these rows skip until the entry accepts block_mask
- the flex consumer and the kernel fed one BlockSelection both sit within tolerance of fp32 sdpa
  on the same tiles

Bitwise rows need the autotuner pinned to one config: the two arms have different autotune keys,
and a different tile changes the accumulation order. A single run pins 64x32 unless the
SDNQ_TRITON_ATTEN_*_LIST env says otherwise, and the last row asserts the pin left the autotuner
exactly one config. --tiles runs the file once per tile pair in a subprocess, which is also what
proves every candidate tile nests the 128x64 block: a tile that did not would fail the kernel's
static_assert at compile time.

Usage:
    python test/test-attention-sdnq-sparse.py
    python test/test-attention-sdnq-sparse.py --tiles
"""

import os
import sys
import inspect

TILES = [(32, 16), (32, 32), (32, 64), (64, 16), (64, 32), (64, 64), (128, 16), (128, 32), (128, 64)]
PIN = {
    'SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST': '64',
    'SDNQ_TRITON_ATTEN_BLOCK_SIZE_N_LIST': '32',
    'SDNQ_TRITON_ATTEN_NUM_WARPS_LIST': '4',
    'SDNQ_TRITON_ATTEN_NUM_STAGES_LIST': '1',
}


def run_tiles() -> bool:
    import subprocess
    ok = True
    for block_m, block_n in TILES:
        env = dict(os.environ, **PIN)
        env['SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST'] = str(block_m)
        env['SDNQ_TRITON_ATTEN_BLOCK_SIZE_N_LIST'] = str(block_n)
        print(f'=== tile {block_m}x{block_n} ===', flush=True)
        code = subprocess.call([sys.executable, os.path.abspath(__file__)], env=env)
        print(f'=== tile {block_m}x{block_n}: {"PASS" if code == 0 else "FAIL"} ===', flush=True)
        ok = ok and code == 0
    return ok


if __name__ == '__main__' and '--tiles' in sys.argv:
    sys.exit(0 if run_tiles() else 1)

for pin_name, pin_value in PIN.items():
    os.environ.setdefault(pin_name, pin_value)


def env_list(name: str) -> list[int]:
    return [int(x) for x in os.environ[name].replace(' ', '').split(',')]


pinned = {name: env_list(name) for name in PIN}
if any(len(values) != 1 for values in pinned.values()):
    print('this test needs the attention autotuner pinned to one config: set each SDNQ_TRITON_ATTEN_*_LIST to a single value', flush=True)
    sys.exit(2)
block_size_m = pinned['SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST'][0]
block_size_n = pinned['SDNQ_TRITON_ATTEN_BLOCK_SIZE_N_LIST'][0]

import torch  # pylint: disable=wrong-import-position

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

stock_sdpa = torch.nn.functional.scaled_dot_product_attention # captured before shared installs the configured hijacks

from modules.errors import log                                # pylint: disable=wrong-import-position
from modules import shared                                    # pylint: disable=wrong-import-position,unused-import
from modules.attention import sparse                          # pylint: disable=wrong-import-position
from modules.attention.sparse import flex as sparse_flex      # pylint: disable=wrong-import-position
from sdnq.kernels import triton_atten as atten_module         # pylint: disable=wrong-import-position
from sdnq.kernels import triton_atten_backward as backward_module # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
kernel_available = device.type == 'cuda'
int8_ok = block_size_n >= 32 # the int8 paths prune tiles narrower than 32, so a narrower pin cannot exercise them
BLOCK_M, BLOCK_N = 128, 64


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, passed, name: str, detail: str = ''):
    status = 'SKIP' if passed is None else ('PASS' if passed else 'FAIL')
    key = {'SKIP': 'skipped', 'PASS': 'passed', 'FAIL': 'failed'}[status]
    results[cat][key] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    (log.info if status != 'FAIL' else log.error)(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        outcome = fn()
        record(cat, None if outcome is None else bool(outcome), name)
    except AssertionError as e:
        record(cat, False, name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, False, name, f'exception: {e}')
        import traceback
        traceback.print_exc()


generator = torch.Generator(device=device).manual_seed(4321)


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=generator, device=device, dtype=torch.float32).to(dtype).contiguous()


def qkv(batch=2, heads=4, kv_heads=None, seq=1000, dim=64, dtype=torch.bfloat16):
    kv_heads = kv_heads or heads
    return randn(batch, heads, seq, dim, dtype=dtype), randn(batch, kv_heads, seq, dim, dtype=dtype), randn(batch, kv_heads, seq, dim, dtype=dtype)


def blocks(seq_q: int, seq_kv: int):
    return sparse.selector.block_count(seq_q, BLOCK_M), sparse.selector.block_count(seq_kv, BLOCK_N)


def random_keep(batch: int, heads: int, seq_q: int, seq_kv: int, density: float = 0.4):
    """A random block mask with the diagonal kept, so no row is empty unless a test empties it."""
    nq, nk = blocks(seq_q, seq_kv)
    keep = torch.rand(batch, heads, nq, nk, generator=generator, device=device) < density
    keep |= sparse.selector.diagonal_blocks(nq, nk, BLOCK_M, BLOCK_N, device)
    return keep.to(torch.int8)


def expand(keep: torch.Tensor, seq_q: int, seq_kv: int):
    """The token-granular int8 mask a block mask stands for."""
    return keep.repeat_interleave(BLOCK_M, dim=-2).repeat_interleave(BLOCK_N, dim=-1)[..., :seq_q, :seq_kv].contiguous()


def ordered(keep: torch.Tensor, batch: int, heads: int):
    """The count and ascending index list per query block that the kernel walks, built the way the entry builds them: size-1 batch and head dims kept, padded to the descriptor chunks."""
    return atten_module.get_block_mask_input(keep, batch, heads)


def raw(q, k, v, attn_mask=None, block_mask=None, block_count=None, block_index=None):
    """The triton op without the input prep: nothing is quantized, so the mask path is the only difference between arms."""
    if block_mask is not None:
        block_count, block_index = ordered(block_mask, q.shape[0], q.shape[1])
    sparse = block_count is not None
    return atten_module.sdnq_triton_atten_fwd(
        q, k, v, None, None, None,
        attn_mask=attn_mask, is_causal=False, sm_scale=q.shape[-1] ** -0.5, use_fp16_accum=False, out_dtype=q.dtype,
        block_count=block_count, block_index=block_index, block_mask_m=BLOCK_M if sparse else 0, block_mask_n=BLOCK_N if sparse else 0,
    )


def entry_accepts_block_mask() -> bool:
    return 'block_mask' in inspect.signature(inspect.unwrap(atten_module.sdnq_triton_atten)).parameters


def entry(q, k, v, **kwargs):
    return atten_module.sdnq_triton_atten(q, k, v, **kwargs)


def configs_for(sizes_m, sizes_n):
    import triton
    return [triton.Config({'BLOCK_SIZE_M': m, 'BLOCK_SIZE_N': n}, num_warps=4, num_stages=1) for m in sizes_m for n in sizes_n]


def nests(conf) -> bool:
    return BLOCK_M % conf.kwargs['BLOCK_SIZE_M'] == 0 and BLOCK_N % conf.kwargs['BLOCK_SIZE_N'] == 0


def assert_one_ulp(got: torch.Tensor, want: torch.Tensor, label: str = ''):
    """Within one unit in the last place at each element, plus an absolute floor for the drift between the two rounding sequences (a straddled rounding of one p before the pv dot, amplified on rows with few keys), two orders below what a mis-skipped tile moves."""
    if torch.equal(got, want):
        return
    eps = torch.finfo(got.dtype).eps
    atol = want.float().abs().max().item() * 2**-10
    if not torch.allclose(got.float(), want.float(), rtol=eps, atol=atol):
        gap = (got.float() - want.float()).abs()
        raise AssertionError(f'{label} max diff {gap.max().item():.3e} over {int((got != want).sum().item())} elements, bound one ulp plus {atol:.1e}')


# ============================================================
# Kernel contract
# ============================================================

def test_block_mask_matches_the_expanded_token_mask():
    if not kernel_available:
        return None
    for seq in (1000, 970): # 970 leaves the last block with sub tiles past the end of the sequence under the narrower tiles
        q, k, v = qkv(seq=seq)
        keep = random_keep(2, 4, seq, seq)
        got = raw(q, k, v, block_mask=keep)
        assert_one_ulp(got, raw(q, k, v, attn_mask=expand(keep, seq, seq)), f'seq {seq}')
        assert torch.isfinite(got).all()
        assert not torch.equal(got, raw(q, k, v)), 'a 40 percent block mask left the output identical to dense: the mask is not applied'
    return True


def test_gqa_block_mask_indexes_query_heads():
    if not kernel_available:
        return None
    q, k, v = qkv(heads=8, kv_heads=2)
    for heads in (8, 1):
        keep = random_keep(2, heads, 1000, 1000)
        assert_one_ulp(raw(q, k, v, block_mask=keep), raw(q, k, v, attn_mask=expand(keep, 1000, 1000)), f'mask heads {heads}')
    return True


def test_block_mask_broadcasts_over_batch_and_heads():
    if not kernel_available:
        return None
    q, k, v = qkv(batch=3, heads=4)
    for batch, heads in ((1, 1), (1, 4), (3, 1)):
        keep = random_keep(batch, heads, 1000, 1000)
        assert_one_ulp(raw(q, k, v, block_mask=keep), raw(q, k, v, attn_mask=expand(keep, 1000, 1000)), f'mask shape {tuple(keep.shape)}')
    return True


def test_block_lists_must_be_contiguous_and_padded():
    """The kernel reads the lists through descriptors from shapes alone, so the launcher takes only what the entry builds."""
    if not kernel_available:
        return None
    q, k, v = qkv(seq=1100) # 9 query blocks and 18 kv blocks, so both lists get padded (to 12 and 32)
    keep = random_keep(2, 4, 1100, 1100)
    count, index = ordered(keep, 2, 4)
    nq, nk = blocks(1100, 1100)
    assert count.is_contiguous() and index.is_contiguous()
    assert count.shape[-1] % atten_module.block_count_chunk == 0 and index.shape[-1] % atten_module.block_index_chunk == 0
    assert count.shape[-1] > nq and index.shape[-2] == nq and index.shape[-1] > nk, 'this geometry should pad both lists'
    assert torch.equal(count[..., :nq], keep.sum(-1, dtype=torch.int32)), 'padding changed the counts'
    reference = raw(q, k, v, block_count=count, block_index=index)
    count_view = count.transpose(-1, -2).contiguous().transpose(-1, -2)
    index_view = index.transpose(-1, -2).contiguous().transpose(-1, -2)
    unpadded_count = count[..., :nq].contiguous()
    unpadded_index = index[..., :nk].contiguous()
    rejected = (
        ('a strided count', {'block_count': count_view, 'block_index': index}),
        ('a strided index', {'block_count': count, 'block_index': index_view}),
        ('an unpadded count', {'block_count': unpadded_count, 'block_index': index}),
        ('an unpadded index', {'block_count': count, 'block_index': unpadded_index}),
    )
    for label, lists in rejected:
        try:
            raw(q, k, v, **lists)
        except ValueError:
            continue
        raise AssertionError(f'{label} was accepted')
    assert torch.equal(raw(q, k, v, block_mask=keep), reference)
    shared_keep = keep[:1, :1]
    shared_count, shared_index = ordered(shared_keep, 2, 4)
    assert tuple(shared_count.shape[:2]) == (1, 1) and tuple(shared_index.shape[:2]) == (1, 1), 'size-1 batch and head dims stay as given, the kernel broadcasts by shape'
    assert torch.equal(raw(q, k, v, block_count=shared_count, block_index=shared_index), raw(q, k, v, block_mask=shared_keep.expand(2, 4, -1, -1).contiguous()))
    return True


def test_empty_block_row_gives_zeros_without_nan():
    if not kernel_available:
        return None
    q, k, v = qkv()
    keep = random_keep(2, 4, 1000, 1000)
    keep[..., 2, :] = 0
    got = raw(q, k, v, block_mask=keep)
    assert_one_ulp(got, raw(q, k, v, attn_mask=expand(keep, 1000, 1000)))
    assert torch.isfinite(got).all()
    assert got[..., 256:384, :].abs().max().item() == 0.0, 'an empty block row should give zeros'
    assert got[..., :256, :].abs().max().item() > 0.0
    return True


def test_block_mask_composes_with_a_token_mask():
    if not kernel_available:
        return None
    q, k, v = qkv()
    keep = random_keep(2, 4, 1000, 1000)
    padding = torch.ones(2, 1, 1000, 1000, dtype=torch.int8, device=device)
    padding[..., -37:] = 0 # the last keys are padding, as a packed sequence with a tail would have
    got = raw(q, k, v, attn_mask=padding, block_mask=keep)
    want = raw(q, k, v, attn_mask=(padding.bool() & expand(keep, 1000, 1000).bool()).to(torch.int8))
    assert torch.equal(got, want), 'both arms carry a token mask, so this one is bitwise' # pylint: disable=line-too-long
    assert not torch.equal(got, raw(q, k, v, block_mask=keep)), 'the token mask was ignored beside the block mask'
    return True


def same_arithmetic(got: torch.Tensor, want: torch.Tensor, label: str):
    """Every block kept against no mask: the same tiles in the same order, so any gap is compiler scheduling of a loop with a runtime trip count; report whether it was bitwise and bound it either way."""
    bitwise = torch.equal(got, want)
    if not bitwise:
        gap = (got.float() - want.float()).abs()
        log.info(f'    {label}: all-ones differs from dense by {gap.max().item():.3e} over {int((got != want).sum().item())} elements (compiler scheduling, not tiles)')
    assert_one_ulp(got, want, label)
    return bitwise


def test_all_ones_block_mask_matches_no_mask():
    if not kernel_available:
        return None
    q, k, v = qkv()
    nq, nk = blocks(1000, 1000)
    ones = torch.ones(1, 1, nq, nk, dtype=torch.int8, device=device)
    same_arithmetic(raw(q, k, v, block_mask=ones), raw(q, k, v), 'raw bf16')
    return True


def test_sub_tile_past_the_sequence_is_masked():
    """992 divides every kernel tile but not the 64-wide mask block, so the last block's trailing sub tile sits entirely past the keys.

    Without a tail rule that knows the block geometry it loads zeros, scores them as zero and inflates the softmax denominator.
    """
    if not kernel_available:
        return None
    seq = 992
    assert seq % block_size_n == 0 or seq % BLOCK_N != 0, 'this row needs a sequence the tile divides and the mask block does not'
    q, k, v = qkv(seq=seq)
    nq, nk = blocks(seq, seq)
    assert nk * BLOCK_N > seq, 'the last mask block should run past the sequence'
    ones = torch.ones(1, 1, nq, nk, dtype=torch.int8, device=device)
    assert_one_ulp(raw(q, k, v, block_mask=ones), raw(q, k, v), f'seq {seq}, every block kept')
    keep = random_keep(2, 4, seq, seq)
    assert_one_ulp(raw(q, k, v, block_mask=keep), raw(q, k, v, attn_mask=expand(keep, seq, seq)), f'seq {seq}, 40 percent kept')
    return True


def test_launcher_validates_the_block_lists():
    if not kernel_available:
        return None
    q, k, v = qkv()
    ones = torch.ones(1, 1, *blocks(1000, 1000), dtype=torch.int8, device=device)
    count, index = ordered(ones, 2, 4)
    wide_count, wide_index = ordered(ones.expand(5, 4, -1, -1).contiguous(), 5, 4)
    bad = (
        ('a count without an index', {'block_count': count}),
        ('an index without a count', {'block_index': index}),
        ('an int8 index', {'block_count': count, 'block_index': index.to(torch.int8)}),
        ('a 3d index', {'block_count': count, 'block_index': index[0]}),
        ('a short index', {'block_count': count, 'block_index': index[..., :-1].contiguous()}),
        ('a wrong batch', {'block_count': wide_count, 'block_index': wide_index}),
    )
    for label, lists in bad:
        try:
            raw(q, k, v, **lists)
        except ValueError:
            continue
        raise AssertionError(f'{label} was accepted')
    try:
        atten_module.sdnq_triton_atten_fwd(q, k, v, None, None, None, sm_scale=0.125, out_dtype=q.dtype, block_count=count, block_index=index, block_mask_m=0, block_mask_n=0)
    except ValueError:
        return True
    raise AssertionError('block lists without block sizes were accepted')


# ============================================================
# Autotune nesting
# ============================================================

def test_nesting_filter_keeps_only_nesting_tiles():
    configs = configs_for((32, 64, 128, 256), (16, 32, 64, 128))
    args = {'do_block_mask': 1, 'BLOCK_MASK_M': BLOCK_M, 'BLOCK_MASK_N': BLOCK_N}
    kept = atten_module.nest_block_mask_configs(configs, args)
    assert {(c.kwargs['BLOCK_SIZE_M'], c.kwargs['BLOCK_SIZE_N']) for c in kept} == {(m, n) for m in (32, 64, 128) for n in (16, 32, 64)}
    assert atten_module.nest_block_mask_configs(configs, {'do_block_mask': 0}) is configs
    assert atten_module.nest_block_mask_configs(configs, {}) is configs
    try:
        atten_module.nest_block_mask_configs(configs_for((256,), (128,)), args)
    except ValueError as e:
        assert 'SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST' in str(e), e
        return True
    raise AssertionError('a tile list with nothing nesting did not raise')


def test_prune_configs_applies_the_filter():
    if not kernel_available:
        return None
    q, k, v = qkv()
    args = {
        'q_ptr': q, 'k_ptr': k, 'v_ptr': v, 'out_ptr': q,
        'QN': 1000, 'KN': 1000, 'QHD': 64, 'KHD': 64, 'VHD': 64,
        'is_causal': 0, 'do_block_mask': 1, 'BLOCK_MASK_M': BLOCK_M, 'BLOCK_MASK_N': BLOCK_N,
    }
    with_mask = atten_module.prune_configs(configs_for((32, 64, 128, 256), (16, 32, 64, 128)), args)
    without = atten_module.prune_configs(configs_for((32, 64, 128, 256), (16, 32, 64, 128)), dict(args, do_block_mask=0))
    assert with_mask, 'nothing survived'
    assert [c.kwargs for c in with_mask] == [c.kwargs for c in without if nests(c)], ([c.kwargs for c in with_mask], [c.kwargs for c in without])
    return True


# ============================================================
# Public entry
# ============================================================

def entry_rows_ready() -> bool:
    return kernel_available and entry_accepts_block_mask()


def entry_rows(q, k, v, keep, **kwargs) -> bool:
    """The block arm within one ulp of the token mask, applied at all, and bitwise the dense path with every block kept."""
    seq_q, seq_kv = q.shape[-2], k.shape[-2]
    got = entry(q, k, v, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, **kwargs)
    assert_one_ulp(got, entry(q, k, v, attn_mask=expand(keep, seq_q, seq_kv).bool(), **kwargs))
    dense = entry(q, k, v, **kwargs)
    assert not torch.equal(got, dense), 'the block mask is not applied'
    nq, nk = blocks(seq_q, seq_kv)
    ones = torch.ones(1, 1, nq, nk, dtype=torch.int8, device=device)
    same_arithmetic(entry(q, k, v, block_mask=ones, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, **kwargs), dense, f'entry {kwargs.get("matmul_dtype", "int8")}')
    return True


def test_entry_int8_block_mask():
    if not entry_rows_ready() or not int8_ok:
        return None
    q, k, v = qkv()
    return entry_rows(q, k, v, random_keep(2, 4, 1000, 1000), matmul_dtype='int8')


def test_entry_int8_pv_block_mask():
    if not entry_rows_ready() or not int8_ok:
        return None
    q, k, v = qkv()
    return entry_rows(q, k, v, random_keep(2, 4, 1000, 1000), matmul_dtype='int8', pv_matmul_dtype='int8')


def test_entry_fp16_accum_block_mask():
    if not entry_rows_ready():
        return None
    q, k, v = qkv(dtype=torch.float16)
    return entry_rows(q, k, v, random_keep(2, 4, 1000, 1000), matmul_dtype='float16', pv_matmul_dtype='float16', use_fp16_accum=True)


def test_entry_normalizes_bool_and_3d_block_masks():
    if not entry_rows_ready():
        return None
    q, k, v = qkv(batch=1)
    keep = random_keep(1, 4, 1000, 1000)
    want = entry(q, k, v, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False)
    assert torch.equal(entry(q, k, v, block_mask=keep.bool(), block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False), want), 'bool'
    assert torch.equal(entry(q, k, v, block_mask=keep[0], block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False), want), '3d'
    return True


# ============================================================
# Backward
# ============================================================

def backward_available() -> bool:
    return kernel_available and 'block_mask' in inspect.signature(backward_module.sdnq_triton_atten_with_backward).parameters


def upstream_grad(shape, seed=17):
    """The same upstream gradient for every arm: two arms fed different ones are not comparable at all."""
    return torch.randn(shape, generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=torch.float32)


def grads(q, k, v, upstream=None, **kwargs):
    """dq, dk and dv for one attention call through the kernel's autograd function."""
    q, k, v = q.detach().clone().requires_grad_(True), k.detach().clone().requires_grad_(True), v.detach().clone().requires_grad_(True)
    out = backward_module.sdnq_triton_atten_with_backward(q, k, v, **kwargs)
    if upstream is None:
        upstream = upstream_grad(out.shape)
    out.backward(upstream.to(out.dtype))
    return q.grad, k.grad, v.grad


def test_backward_matches_the_expanded_token_mask():
    """The load-bearing row: the same selection through the block lists and through a token mask has to give the same gradients."""
    if not backward_available():
        return None
    seq = 1000
    q, k, v = qkv(seq=seq)
    keep = random_keep(2, 4, seq, seq)
    block = grads(q, k, v, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False)
    token = grads(q, k, v, attn_mask=expand(keep, seq, seq).bool(), do_quantize=False)
    dense = grads(q, k, v, do_quantize=False)
    for name, got, want, other in zip(('dq', 'dk', 'dv'), block, token, dense):
        assert got is not None and torch.isfinite(got).all(), f'{name} is not finite'
        assert_one_ulp(got, want, name)
        assert not torch.equal(got, other), f'{name} matches the dense gradient, so the mask never reached the backward'
    return True


def test_backward_gqa_and_int8():
    """GQA sends every query head's selection into one kv head's gradient, and the quantized path is the one that ships."""
    if not backward_available():
        return None
    seq = 1000
    q, k, v = qkv(heads=4, kv_heads=2, seq=seq)
    keep = random_keep(2, 4, seq, seq)
    for label, kwargs in (('bf16', {'do_quantize': False}), ('int8', {'matmul_dtype': 'int8'})):
        if label == 'int8' and not int8_ok:
            continue
        block = grads(q, k, v, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, **kwargs)
        token = grads(q, k, v, attn_mask=expand(keep, seq, seq).bool(), **kwargs)
        for name, got, want in zip(('dq', 'dk', 'dv'), block, token):
            assert_one_ulp(got, want, f'{label} {name}')
    return True


def test_backward_zeroes_dropped_blocks():
    """A query block that keeps nothing gets no gradient, and a kv block no query block kept gets none either."""
    if not backward_available():
        return None
    seq = 1000
    q, k, v = qkv(seq=seq)
    keep = random_keep(2, 4, seq, seq)
    keep[..., 2, :] = 0 # query block 2 attends to nothing
    keep[..., :, 5] = 0 # kv block 5 is attended by nothing
    dq, dk, dv = grads(q, k, v, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False)
    assert torch.isfinite(dq).all() and torch.isfinite(dk).all() and torch.isfinite(dv).all()
    assert dq[..., 2 * BLOCK_M:3 * BLOCK_M, :].abs().max().item() == 0.0, 'an empty query block still got a gradient'
    assert dq[..., :BLOCK_M, :].abs().max().item() > 0.0
    assert dk[..., 5 * BLOCK_N:6 * BLOCK_N, :].abs().max().item() == 0.0, 'a dropped kv block still got a key gradient'
    assert dv[..., 5 * BLOCK_N:6 * BLOCK_N, :].abs().max().item() == 0.0, 'a dropped kv block still got a value gradient'
    assert dk[..., :BLOCK_N, :].abs().max().item() > 0.0
    return True


def test_backward_matches_fp32_autograd():
    """Against truth rather than against the other masked path: torch's own gradients through fp32 sdpa on the same selection.

    Run in fp32, where the kernel sits within 0.1 percent of autograd; in bf16 the dense backward is already 1.8 percent
    out, so a bf16 arm could only carry a bound too loose to catch a mis-walked block. Head dim 32 keeps fp32 inside the
    shared memory the widest pinned tile has, which fp32 at head dim 64 exceeds on the dense backward too.
    """
    if not backward_available():
        return None
    seq, dim = 1000, 32
    q, k, v = (t.float().contiguous() for t in qkv(batch=1, heads=4, seq=seq, dim=dim))
    keep = random_keep(1, 4, seq, seq)
    upstream = upstream_grad((1, 4, seq, dim))
    reference = [t.detach().clone().requires_grad_(True) for t in (q, k, v)]
    stock_sdpa(*reference, attn_mask=expand(keep, seq, seq).bool()).backward(upstream)
    got = grads(q, k, v, upstream=upstream, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False)
    for name, mine, want in zip(('dq', 'dk', 'dv'), got, reference):
        scale = want.grad.abs().max().item()
        gap = (mine.float() - want.grad).abs().max().item()
        log.info(f'    {name}: {100 * gap / scale:.3f} percent of the gradient scale against fp32 autograd')
        assert gap < 5e-3 * scale, f'{name} is {gap:.3e} off fp32 autograd, more than 0.5 percent of {scale:.3e}'
    return True


def test_backward_ragged_sub_tile():
    """The transposed lists have the same ragged tail as the forward's: 992 divides the tile and not the mask block."""
    if not backward_available():
        return None
    seq = 992
    q, k, v = qkv(seq=seq)
    keep = random_keep(2, 4, seq, seq)
    block = grads(q, k, v, block_mask=keep, block_mask_m=BLOCK_M, block_mask_n=BLOCK_N, do_quantize=False)
    token = grads(q, k, v, attn_mask=expand(keep, seq, seq).bool(), do_quantize=False)
    for name, got, want in zip(('dq', 'dk', 'dv'), block, token):
        assert torch.isfinite(got).all(), f'{name} is not finite'
        assert_one_ulp(got, want, f'seq {seq} {name}')
    return True


# ============================================================
# Consumers
# ============================================================

def test_kernel_and_flex_agree_on_one_selection():
    if not entry_rows_ready():
        return None
    seq = 2048
    q, k, v = qkv(batch=1, heads=4, seq=seq)
    selection = sparse.select_blocks(q, k, sparse.SparseSpec(budget=0.25))
    assert selection is not None
    reference = stock_sdpa(q.float(), k.float(), v.float(), attn_mask=expand(selection.keep, seq, seq).bool())
    flex_gap = (sparse_flex.attend(q, k, v, selection).float() - reference).abs().max().item()
    kernel_gap = (entry(q, k, v, block_mask=selection.keep, block_mask_m=selection.block_q, block_mask_n=selection.block_kv, do_quantize=False).float() - reference).abs().max().item()
    log.info(f'    gaps against fp32 sdpa on the same tiles: flex {flex_gap:.5f} kernel {kernel_gap:.5f}')
    assert flex_gap < 1e-2 and kernel_gap < 1e-2, (flex_gap, kernel_gap)
    return True


def test_the_pin_left_the_autotuner_one_config():
    # with one config the autotuner benchmarks nothing and prunes nothing, so both arms of every
    # bitwise row compiled the same tile; the in-kernel static_assert is the nesting guard there
    configs = [conf.kwargs for conf in atten_module.autotune_configs]
    assert configs == [{'BLOCK_SIZE_M': block_size_m, 'BLOCK_SIZE_N': block_size_n}], f'the pin did not take: {configs}'
    tuner_configs = [conf.kwargs for conf in getattr(atten_module.sdnq_attn_kernel, 'configs', [])]
    assert tuner_configs == configs, f'the autotuner holds {tuner_configs}'
    return True


def run_all():
    log.warning(f'Running sdnq block mask tests on {device}, tile pinned to {block_size_m}x{block_size_n}')

    log.warning('=== kernel contract ===')
    cat = category('contract')
    for fn in [
        test_block_mask_matches_the_expanded_token_mask,
        test_gqa_block_mask_indexes_query_heads,
        test_block_mask_broadcasts_over_batch_and_heads,
        test_block_lists_must_be_contiguous_and_padded,
        test_empty_block_row_gives_zeros_without_nan,
        test_block_mask_composes_with_a_token_mask,
        test_all_ones_block_mask_matches_no_mask,
        test_sub_tile_past_the_sequence_is_masked,
        test_launcher_validates_the_block_lists,
    ]:
        run_test(cat, fn)

    log.warning('=== autotune nesting ===')
    cat = category('autotune')
    for fn in [
        test_nesting_filter_keeps_only_nesting_tiles,
        test_prune_configs_applies_the_filter,
    ]:
        run_test(cat, fn)

    log.warning('=== public entry ===')
    cat = category('entry')
    for fn in [
        test_entry_int8_block_mask,
        test_entry_int8_pv_block_mask,
        test_entry_fp16_accum_block_mask,
        test_entry_normalizes_bool_and_3d_block_masks,
    ]:
        run_test(cat, fn)

    log.warning('=== backward ===')
    cat = category('backward')
    for fn in [
        test_backward_matches_the_expanded_token_mask,
        test_backward_gqa_and_int8,
        test_backward_zeroes_dropped_blocks,
        test_backward_matches_fp32_autograd,
        test_backward_ragged_sub_tile,
    ]:
        run_test(cat, fn)

    log.warning('=== consumers ===')
    cat = category('consumers')
    for fn in [
        test_kernel_and_flex_agree_on_one_selection,
        test_the_pin_left_the_autotuner_one_config,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total_passed = total_failed = total_skipped = 0
    for cat_name, info in results.items():
        ok = info['failed'] == 0
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{'PASS' if ok else 'FAIL'}]")
        total_passed += info['passed']
        total_failed += info['failed']
        total_skipped += info['skipped']
    log.warning(f'Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped')
    return total_failed == 0


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = run_all()
    torch.nn.functional.scaled_dot_product_attention = stock_sdpa
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
