from modules.logger import log
from .teacache_flux import teacache_flux_forward
from .teacache_hidream import teacache_hidream_forward
from .teacache_lumina2 import teacache_lumina2_forward
from .teacache_ltx import teacache_ltx_forward
from .teacache_mochi import teacache_mochi_forward
from .teacache_cogvideox import teacache_cog_forward
from .teacache_chroma import teacache_chroma_forward


supported_models = ['Flux', 'Chroma', 'CogVideoX', 'Mochi', 'LTX', 'HiDream', 'Lumina2']


def apply_teacache(p):
    from modules import shared
    if not shared.opts.teacache_enabled:
        return
    if not any(shared.sd_model.__class__.__name__.startswith(x) for x in supported_models):
        return
    if not hasattr(shared.sd_model, 'transformer'):
        return
    shared.sd_model.transformer.__class__.enable_teacache = shared.opts.teacache_thresh > 0
    shared.sd_model.transformer.__class__.cnt = 0
    shared.sd_model.transformer.__class__.num_steps = p.steps
    shared.sd_model.transformer.__class__.rel_l1_thresh = shared.opts.teacache_thresh # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    shared.sd_model.transformer.__class__.accumulated_rel_l1_distance = 0
    shared.sd_model.transformer.__class__.previous_modulated_input = None
    shared.sd_model.transformer.__class__.previous_residual = None
    if shared.sd_model.__class__.__name__.startswith('HiDream'):
        shared.sd_model.transformer.__class__.ret_steps = p.steps * 0.1
    if shared.sd_model.__class__.__name__.startswith('Lumina2'):
        shared.sd_model.transformer.__class__.cache = {}
        shared.sd_model.transformer.__class__.uncond_seq_len = None
    log.info(f'Transformers cache: type=teacache cls={shared.sd_model.__class__.__name__} thresh={shared.opts.teacache_thresh}')
