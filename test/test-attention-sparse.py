#!/usr/bin/env python
"""
Offline unit tests for block-sparse attention in modules.attention.sparse.

Covers:

- block pooling, including the ragged tail, against a per-block reference
- the diagonal invariant: every query tile keeps the key tiles its tokens overlap
- budget semantics: density tracks the budget over the candidates, pins survive, drops never do
- the dense short circuit, and the force flag that suppresses it for tests
- determinism of the selection for identical inputs
- layout reading: the *_indices form a pipeline passes by name, with a non-final video run
  relabelled as conditioning, and the segment form a transformer knows at its packing site
- pins and drops derived from a layout: pinned columns, dropped padding, pinned boundary tiles
- the flex consumer: a full-keep selection through flex_attention reproduces dense sdpa, and a
  selection with dropped tiles reproduces sdpa given the same tiles masked out
- the density matched radial control and the step schedule
- the router stage: the gates it applies (component role, mask, causal, cross attention, minimum
  sequence), the per step budget schedule, and layout resolution with and without a publisher

The flex rows need a cuda device and compile the flex kernel; they skip on cpu.

Usage:
    python test/test-attention-sparse.py
"""

import os
import sys

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

stock_sdpa = torch.nn.functional.scaled_dot_product_attention # captured before shared installs the configured hijacks

from modules.errors import log                                # pylint: disable=wrong-import-position
from modules import shared                                    # pylint: disable=wrong-import-position,unused-import
from modules.attention import sparse                          # pylint: disable=wrong-import-position
from modules.attention.sparse import flex as sparse_flex      # pylint: disable=wrong-import-position


results: dict[str, dict] = {}
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


generator = torch.Generator(device=device).manual_seed(1234)


def randn(*shape, dtype=torch.float32):
    return torch.randn(*shape, generator=generator, device=device, dtype=dtype)


def qkv(heads=4, seq=1024, dim=64):
    return randn(1, heads, seq, dim), randn(1, heads, seq, dim), randn(1, heads, seq, dim)


# ============================================================
# Selector
# ============================================================

def test_pooling_matches_a_per_block_reference():
    x = randn(1, 2, 300, 8)
    pooled = sparse.selector.pool_blocks(x, 128)
    assert pooled.shape == (1, 2, 3, 8), pooled.shape
    for index, (start, end) in enumerate([(0, 128), (128, 256), (256, 300)]):
        expected = x[..., start:end, :].to(torch.float32).mean(dim=-2)
        assert torch.allclose(pooled[..., index, :], expected, atol=1e-5), index
    return True


def test_diagonal_covers_every_overlapping_tile():
    nq, nk, bq, bk = 4, 8, 128, 64
    diagonal = sparse.selector.diagonal_blocks(nq, nk, bq, bk, device)
    for i in range(nq):
        for j in range(nk):
            overlaps = (i * bq < (j + 1) * bk) and (j * bk < (i + 1) * bq)
            assert bool(diagonal[i, j]) == overlaps, (i, j)
    assert int(diagonal.sum().item()) == nq * (bq // bk) # two kv tiles per query tile at 128 over 64
    return True


def test_budget_sets_density_over_the_candidates():
    q, k = randn(1, 4, 1024, 32), randn(1, 4, 1024, 32)
    for budget in (0.15, 0.30, 0.50):
        spec = sparse.SparseSpec(budget=budget)
        selection = sparse.select_blocks(q, k, spec)
        assert selection is not None, budget
        keep = selection.keep
        diagonal = sparse.selector.diagonal_blocks(keep.shape[-2], keep.shape[-1], spec.block_q, spec.block_kv, device)
        candidates = int((~diagonal).sum().item())
        chosen = int((keep.bool() & ~diagonal).sum().item()) / keep.shape[1]
        expected = candidates * budget
        assert abs(chosen - expected) <= keep.shape[-2], f'budget={budget} chose {chosen} of {candidates}, expected about {expected}'
        assert bool((keep.bool() | ~diagonal).all()), 'a diagonal tile was dropped'
    return True


def test_pins_survive_and_drops_never_appear():
    q, k = randn(1, 2, 512, 32), randn(1, 2, 512, 32)
    spec = sparse.SparseSpec(budget=0.10)
    nq = sparse.block_count(512, spec.block_q)
    nk = sparse.block_count(512, spec.block_kv)
    pins = torch.zeros(1, 1, nq, nk, dtype=torch.bool, device=device)
    drops = torch.zeros_like(pins)
    pins[..., 0] = True # a pinned column, as a text prefix produces
    drops[..., -1] = True # a padding column
    selection = sparse.select_blocks(q, k, spec, pins=pins, drops=drops)
    assert selection is not None
    assert bool(selection.keep[..., 0].all()), 'pinned column not kept'
    assert not bool(selection.keep[..., -1].any()), 'dropped column kept'
    return True


def test_dense_short_circuit_and_force():
    q, k = randn(1, 2, 512, 32), randn(1, 2, 512, 32)
    assert sparse.select_blocks(q, k, sparse.SparseSpec(budget=1.0)) is None, 'full budget must report dense'
    forced = sparse.select_blocks(q, k, sparse.SparseSpec(budget=1.0, force=True))
    assert forced is not None and bool(forced.keep.all()), 'forced full budget must keep every tile'
    return True


def test_selection_is_deterministic():
    q, k = randn(1, 4, 1024, 32), randn(1, 4, 1024, 32)
    spec = sparse.SparseSpec(budget=0.25)
    first = sparse.select_blocks(q, k, spec)
    second = sparse.select_blocks(q, k, spec)
    assert torch.equal(first.keep, second.keep)
    return True


def test_head_shared_collapses_the_head_dimension():
    q, k = randn(1, 8, 1024, 32), randn(1, 8, 1024, 32)
    selection = sparse.select_blocks(q, k, sparse.SparseSpec(budget=0.25, head_shared=True))
    assert selection.keep.shape[1] == 1, selection.keep.shape
    return True


def test_gqa_scores_on_query_heads():
    q, k = randn(1, 8, 1024, 32), randn(1, 2, 1024, 32)
    selection = sparse.select_blocks(q, k, sparse.SparseSpec(budget=0.25))
    assert selection.keep.shape[1] == 8, selection.keep.shape # both consumers need the mask head dim to be Hq or 1
    return True


# ============================================================
# Layout
# ============================================================

def test_layout_from_index_kwargs_relabels_the_conditioning_video_run():
    kwargs = { # the shape MiniMax H3 passes its transformer: text, a keyframe video run, audio, then the generated video
        'text_indices': torch.arange(0, 8, device=device),
        'video_indices': torch.cat([torch.arange(8, 12, device=device), torch.arange(20, 40, device=device)]),
        'audio_indices': torch.arange(12, 20, device=device),
        'hidden_states': torch.zeros(1, device=device), # not an index tensor, must be ignored
    }
    layout = sparse.layout_from_index_kwargs(kwargs, length=40)
    kinds = [(s.kind, s.start, s.end) for s in layout.spans]
    assert kinds == [('text', 0, 8), ('cond', 8, 12), ('audio', 12, 20), ('video', 20, 40)], kinds
    assert layout.sparsifiable_tokens() == 20
    return True


def test_layout_from_index_kwargs_returns_none_without_indices():
    assert sparse.layout_from_index_kwargs({'hidden_states': torch.zeros(4, device=device)}, length=4) is None
    return True


def test_layout_from_segments_and_prefix():
    layout = sparse.layout_from_segments([('text', 128), ('image', 4096), ('pad', 128)])
    assert layout.length == 4352 and layout.sparsifiable_tokens() == 4096
    prefix = sparse.layout_from_prefix(1024, 64)
    assert prefix.sparsifiable_tokens() == 960 and prefix.source == 'prefix'
    return True


def test_block_pins_pin_conditioning_and_drop_padding():
    block_q, block_kv = 128, 64
    layout = sparse.layout_from_segments([('text', 128), ('video', 1024), ('pad', 128)])
    pins, drops = sparse.block_pins(layout, 1280, 1280, block_q, block_kv, device)
    assert pins.shape == (1, 1, 10, 20) and drops.shape == pins.shape, (pins.shape, drops.shape)
    assert bool(pins[0, 0, :, 0:2].all()), 'the text columns must be pinned'
    assert bool(drops[0, 0, :, 18:20].all()), 'the padding columns must be dropped'
    assert not bool(drops[0, 0, :, 0:18].any()), 'only padding may be dropped'
    assert bool(pins[0, 0, 0, 0:18].all()), 'the query tile holding text must stay dense over every column that is not padding'
    assert not bool(pins[0, 0, :, 18:20].any()), 'a dropped column is skipped, never pinned'
    assert not bool(pins[0, 0, 1:9, 2:18].any()), 'video against video must remain sparsifiable'
    return True


def test_block_pins_pin_a_boundary_tile():
    layout = sparse.layout_from_segments([('text', 100), ('video', 1180)]) # the boundary falls inside the first tile
    pins, drops = sparse.block_pins(layout, 1280, 1280, 128, 64, device)
    assert not bool(drops.any()), 'nothing is padding here'
    assert bool(pins[0, 0, 0, :].all()), 'a query tile straddling a boundary must stay dense'
    assert bool(pins[0, 0, :, 0:2].all()), 'a key tile straddling a boundary must stay dense'
    return True


def test_segments_from_live_splits_interior_padding():
    from modules.attention.sparse import layout as layout_mod
    live = torch.zeros(512, dtype=torch.bool, device=device)
    live[:40] = True
    live[-8:] = True
    segments = layout_mod.segments_from_live(live, 'text')
    assert segments == [('text', 40), ('pad', 464), ('text', 8)], segments
    token_layout = layout_mod.layout_from_segments(segments + [('image', 1024)])
    assert token_layout.length == 1536, token_layout.length
    _, drops = layout_mod.block_pins(token_layout, 1536, 1536, 128, 64, device)
    # only a key block that is padding all the way through is dropped, so the two straddling blocks survive
    assert drops[..., 1:7].all(), 'whole padded key blocks are dropped'
    assert not drops[..., 0].any() and not drops[..., 7].any(), 'a straddling block keeps its live tokens'
    return True


def test_layout_from_stream_ids_reads_the_joint_convention():
    from modules.attention.sparse import layout as layout_mod
    flat = (torch.zeros(512, 3, device=device), torch.zeros(4096, 3, device=device)) # flux1 passes 2d ids
    batched = (torch.zeros(1, 512, 4, device=device), torch.zeros(1, 4096, 4, device=device)) # flux2 passes 3d
    for text, image in (flat, batched):
        token_layout = layout_mod.layout_from_stream_ids({'txt_ids': text, 'img_ids': image}, 'FluxTransformer2DModel')
        assert token_layout is not None and token_layout.length == 4608, token_layout
        assert [(s.kind, s.start, s.end) for s in token_layout.spans] == [('text', 0, 512), ('image', 512, 4608)], token_layout.spans
    # an architecture whose packing order is not verified publishes nothing rather than pinning the wrong half dense
    assert layout_mod.layout_from_stream_ids({'txt_ids': flat[0], 'img_ids': flat[1]}, 'HiDreamImageTransformer2DModel') is None
    assert layout_mod.layout_from_stream_ids({'txt_ids': flat[0], 'img_ids': flat[1]}, None) is None
    assert layout_mod.layout_from_stream_ids({}, 'FluxTransformer2DModel') is None
    indices = {'video_indices': torch.arange(0, 64, device=device), 'txt_ids': flat[0], 'img_ids': flat[1]}
    assert layout_mod.layout_from_kwargs(indices, 'FluxTransformer2DModel').source == 'indices', 'the index form wins when both are present'
    return True


def test_layout_hook_publishes_from_the_denoiser_kwargs():
    from modules.attention import context as ctx

    class FluxTransformer2DModel(torch.nn.Module): # the reader keys on the class name, so the fake carries a real one
        def forward(self, hidden_states=None, txt_ids=None, img_ids=None): # pylint: disable=unused-argument
            return hidden_states

    class Pipe:
        def __init__(self, transformer):
            self.transformer = transformer

    denoiser = FluxTransformer2DModel()
    pipe = Pipe(denoiser)
    previous = getattr(shared.opts, 'sparse_attention_enabled', False)
    try:
        shared.opts.data['sparse_attention_enabled'] = False
        ctx.install_layout_hook(pipe)
        assert getattr(denoiser, 'sdnext_layout_hook', None) is None, 'nothing is hooked while the feature is off'
        shared.opts.data['sparse_attention_enabled'] = True
        ctx.install_layout_hook(pipe)
        ctx.install_layout_hook(pipe)
        assert getattr(denoiser, 'sdnext_layout_hook', None) is not None, 'the denoiser is hooked once'
        ctx.set_layout(None)
        denoiser(hidden_states=torch.zeros(1, 4096, 4, device=device), txt_ids=torch.zeros(512, 3, device=device), img_ids=torch.zeros(4096, 3, device=device))
        published = ctx.current.layout
        assert published is not None and published.length == 4608 and published.source == 'stream-ids', published
    finally:
        shared.opts.data['sparse_attention_enabled'] = previous
        ctx.set_layout(None)
    return True


def test_block_pins_are_cached_per_geometry():
    layout = sparse.layout_from_segments([('text', 128), ('video', 1024)])
    first = sparse.block_pins(layout, 1152, 1152, 128, 64, device)
    second = sparse.block_pins(layout, 1152, 1152, 128, 64, device)
    assert first[0] is second[0] and first[1] is second[1], 'identical geometry should hit the cache'
    return True


# ============================================================
# Consumers and controls
# ============================================================

def test_radial_control_matches_the_requested_density():
    spec = sparse.SparseSpec()
    for density in (0.15, 0.30):
        control = sparse.radial_blocks(4096, 4096, density, spec, device)
        assert abs(control.density() - density) < 0.05, f'requested {density}, got {control.density()}'
    return True


def test_schedule_has_at_most_two_budgets():
    flat = sparse.schedule(20, 0.3)
    assert set(flat) == {0.3} and len(flat) == 20
    bumped = sparse.schedule(20, 0.3, bump=0.3, bump_steps=2)
    assert len(set(bumped)) == 2, set(bumped)
    assert bumped[0] == bumped[1] == 0.6 and bumped[-1] == bumped[-2] == 0.6 and bumped[10] == 0.3
    return True


def flex_available():
    return device.type == 'cuda'


def kernel_floor(q, k, v):
    """How far the flex kernel sits from sdpa on the same dense problem, which bounds what any sparse row can prove."""
    full = sparse.select_blocks(q, k, sparse.SparseSpec(budget=1.0, force=True))
    return (sparse_flex.attend(q, k, v, full) - stock_sdpa(q, k, v)).abs().max().item()


def test_flex_full_selection_reproduces_dense_sdpa():
    if not flex_available():
        return None
    q, k, v = qkv()
    floor = kernel_floor(q, k, v)
    assert floor < 5e-3, f'a full selection should reproduce dense sdpa, differs by {floor}'
    log.info(f'    flex kernel floor vs sdpa: {floor:.6f}')
    return True


def test_flex_sparse_selection_matches_the_same_tiles_under_sdpa():
    if not flex_available():
        return None
    q, k, v = qkv()
    spec = sparse.SparseSpec(budget=0.25)
    selection = sparse.select_blocks(q, k, spec)
    got = sparse_flex.attend(q, k, v, selection)
    # expand the tile selection to tokens and hand sdpa the same thing
    token_mask = selection.keep.bool().repeat_interleave(spec.block_q, dim=-2).repeat_interleave(spec.block_kv, dim=-1)
    expected = stock_sdpa(q, k, v, attn_mask=token_mask[..., :q.shape[-2], :k.shape[-2]])
    delta = (got - expected).abs().max().item()
    floor = kernel_floor(q, k, v)
    assert delta <= max(4 * floor, 2e-3), f'sparse selection differs from the same tiles under sdpa by {delta}, floor {floor}'
    return True


def test_flex_applies_the_selection_at_all():
    if not flex_available():
        return None
    # flex reads the block lists only when compiled; eager evaluates mask_mod instead, so a
    # block only mask silently attends densely. this row fails if the consumer stops compiling.
    q, k, v = qkv()
    selection = sparse.select_blocks(q, k, sparse.SparseSpec(budget=0.25))
    delta = (sparse_flex.attend(q, k, v, selection) - stock_sdpa(q, k, v)).abs().max().item()
    floor = kernel_floor(q, k, v)
    assert delta > 20 * max(floor, 1e-6), f'a 25 percent selection changed the output by only {delta}, floor {floor}: the mask is not being applied'
    return True


def test_flex_handles_a_ragged_tail():
    if not flex_available():
        return None
    seq = 1000 # neither block size divides this
    q, k, v = qkv(heads=2, seq=seq)
    selection = sparse.select_blocks(q, k, sparse.SparseSpec(budget=1.0, force=True))
    delta = (sparse_flex.attend(q, k, v, selection) - stock_sdpa(q, k, v)).abs().max().item()
    assert delta < 5e-3, f'ragged tail differs by {delta}'
    return True


# ============================================================
# Router stage
# ============================================================

def stage_options(**kwargs):
    from modules.attention.sparse import stage as stage_mod
    base = dict(enabled=True, budget=0.30, min_tokens=1024)
    base.update(kwargs)
    return stage_mod.StageOptions(**base)


def with_context(fn):
    from modules.attention import context as ctx
    ctx.begin(None, steps=10)
    try:
        return fn()
    finally:
        ctx.end()


def test_stage_is_none_when_disabled_or_at_full_budget():
    from modules.attention.sparse import stage as stage_mod
    assert stage_mod.make_stage(stage_options(enabled=False)) is None
    assert stage_mod.make_stage(stage_options(budget=1.0)) is None
    assert stage_mod.make_stage(stage_options()) is not None
    return True


def test_stage_gates():
    from modules.attention.sparse import stage as stage_mod
    stage = stage_mod.make_stage(stage_options())
    q, k, v = qkv(heads=2, seq=2048)
    short_q, short_k, short_v = qkv(heads=2, seq=512)
    cross_k, cross_v = randn(1, 2, 77, 64), randn(1, 2, 77, 64)

    def checks():
        from modules.attention import context as ctx
        assert stage(q, k, v, None, False) is not None, 'an eligible call must be selected'
        mask = torch.zeros(1, 1, 2048, 2048, dtype=torch.bool, device=device)
        assert stage(q, k, v, mask, False) is None, 'a masked call needs a backend that composes the two'
        assert stage(q, k, v, mask, False, frozenset({'masked_block'})) is not None, 'a composing backend takes the masked call'
        assert stage(q, k, v, None, True) is None, 'a causal call is not eligible'
        assert stage(q, cross_k, cross_v, None, False) is None, 'cross attention is not eligible'
        assert stage(short_q, short_k, short_v, None, False) is None, 'below the gate attention stays dense'
        assert stage.last_skip == 'below the minimum sequence', stage.last_skip
        ctx.set_role('vae')
        assert stage(q, k, v, None, False) is None, 'only the denoiser is sparsified'
        ctx.set_role('transformer')
        return True
    return with_context(checks)


def test_published_segments_reach_the_stage():
    from modules.attention.sparse import layout as layout_mod
    from modules.attention.sparse import stage as stage_mod
    from modules.attention import context as ctx
    stage = stage_mod.make_stage(stage_options(budget=0.30))
    q, k, v = qkv(heads=2, seq=2048)

    def checks():
        layout_mod.publish_segments((('text', 256), ('image', 1536), ('pad', 256)), source='test')
        published = ctx.current.layout
        assert published.length == 2048 and published.source == 'test', published
        assert published.kinds() == ('text', 'image', 'pad'), published.kinds()
        selection = stage(q, k, v, None, False)
        keep, block_kv = selection.keep, selection.block_kv
        assert keep[..., :256 // block_kv].all(), 'conditioning key tiles stay dense'
        assert not keep[..., 1792 // block_kv:].any(), 'padding key tiles are dropped'
        return True
    return with_context(checks)


def test_stage_follows_the_step_schedule():
    from modules.attention.sparse import stage as stage_mod
    from modules.attention import context as ctx
    stage = stage_mod.make_stage(stage_options(budget=0.30, schedule_steps=2, schedule_bump=0.40))
    q, k, v = qkv(heads=2, seq=2048)

    def checks():
        densities = []
        for step in range(10):
            ctx.set_step(step)
            selection = stage(q, k, v, None, False)
            densities.append(selection.budget)
        assert densities[0] == densities[1] > densities[5], densities
        assert densities[-1] == densities[-2] > densities[5], densities
        assert len(set(densities)) == 2, set(densities)
        return True
    return with_context(checks)


def test_stage_uses_a_published_layout_and_falls_back_without_one():
    from modules.attention.sparse import stage as stage_mod
    from modules.attention import context as ctx
    stage = stage_mod.make_stage(stage_options(budget=0.20))
    q, k, v = qkv(heads=2, seq=2048)

    def checks():
        loose = stage(q, k, v, None, False)
        ctx.set_layout(sparse.layout_from_segments([('text', 256), ('video', 1792)]))
        pinned = stage(q, k, v, None, False)
        assert pinned.density() > loose.density(), f'pinning conditioning must keep more tiles: {pinned.density()} vs {loose.density()}'
        assert bool(pinned.keep[..., 0:4].all()), 'the pinned text columns must survive'
        return True
    return with_context(checks)


def run_all():
    log.warning(f'=== selector (device={device}) ===')
    cat = category('selector')
    for fn in [
        test_pooling_matches_a_per_block_reference,
        test_diagonal_covers_every_overlapping_tile,
        test_budget_sets_density_over_the_candidates,
        test_pins_survive_and_drops_never_appear,
        test_dense_short_circuit_and_force,
        test_selection_is_deterministic,
        test_head_shared_collapses_the_head_dimension,
        test_gqa_scores_on_query_heads,
    ]:
        run_test(cat, fn)

    log.warning('=== layout ===')
    cat = category('layout')
    for fn in [
        test_layout_from_index_kwargs_relabels_the_conditioning_video_run,
        test_layout_from_index_kwargs_returns_none_without_indices,
        test_layout_from_segments_and_prefix,
        test_block_pins_pin_conditioning_and_drop_padding,
        test_block_pins_pin_a_boundary_tile,
        test_block_pins_are_cached_per_geometry,
        test_segments_from_live_splits_interior_padding,
        test_layout_from_stream_ids_reads_the_joint_convention,
        test_layout_hook_publishes_from_the_denoiser_kwargs,
    ]:
        run_test(cat, fn)

    log.warning('=== consumers ===')
    cat = category('consumers')
    for fn in [
        test_radial_control_matches_the_requested_density,
        test_schedule_has_at_most_two_budgets,
        test_flex_full_selection_reproduces_dense_sdpa,
        test_flex_sparse_selection_matches_the_same_tiles_under_sdpa,
        test_flex_applies_the_selection_at_all,
        test_flex_handles_a_ragged_tail,
    ]:
        run_test(cat, fn)

    log.warning('=== stage ===')
    cat = category('stage')
    for fn in [
        test_stage_is_none_when_disabled_or_at_full_budget,
        test_stage_gates,
        test_stage_follows_the_step_schedule,
        test_published_segments_reach_the_stage,
        test_stage_uses_a_published_layout_and_falls_back_without_one,
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
