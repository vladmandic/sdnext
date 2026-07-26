"""Per-block LoRA strength: <lora:name:1.0:lbw=VALUE>.

Each targeted layer maps to one slot of a per-architecture weight vector and
the network's multiplier is scaled by that slot. Slot 0 is BASE: on unet
architectures it covers the text encoder and the unet layers outside the
block chain, on transformer architectures the layers outside the block
chain(s). The remaining slots follow the merge block-weight layout on unet
architectures (26 on sd, 20 on sdxl: input blocks, mid, output blocks) and
the transformer chain(s) in depth order elsewhere, with chain lengths
scanned from the live network_layer_mapping rather than hardcoded.

VALUE is a preset name (case-insensitive), a single number broadcast to
every slot, or a comma list with one number per slot. Named presets force
BASE to 1.0, since the merge tables carry 0 there with merge semantics, and
stretch onto the block count of the current model; classic segment names
(INS, OUTALL, ...) generate from ranges, so they also work on transformer
chains via thirds, and DOUBLE/SINGLE mute one chain on two-chain
architectures. Explicit vectors are taken verbatim at the slot count, with
the a1111 17-slot (sd) and 12-slot (sdxl) layouts accepted and expanded,
omitted slots neutral. A value that fits nothing is ignored with a warning
and the network applies at its plain strength.
"""

import re

from modules import shared
from modules.logger import log
from modules.lora import lora_common as l


UNET_ARCHES = ('sd', 'sdxl')
CHAINS = { # arch -> anchored tail prefixes, one per chain, in depth order
    'sd3': ('transformer_blocks_',),
    'anima': ('transformer_blocks_',),
    'f1': ('transformer_blocks_', 'single_transformer_blocks_'),
    'f2': ('transformer_blocks_', 'single_transformer_blocks_'),
    'chroma': ('transformer_blocks_', 'single_transformer_blocks_'),
    'zimage': ('layers_',),
    'ernieimage': ('layers_',),
    'krea2': ('blocks_',),
}
CLASSIC = ('ALL', 'NONE', 'INALL', 'INS', 'IND', 'MIDD', 'OUTALL', 'OUTD', 'OUTS')
CHAIN_NAMES = ('DOUBLE', 'SINGLE')
SD1_17 = (0, 2, 3, 5, 6, 8, 9, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25) # BASE, IN01, IN02, IN04, IN05, IN07, IN08, MID, OUT03..OUT11
SDXL_12 = (0, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16) # BASE, IN04, IN05, IN07, IN08, MID, OUT00..OUT05
VECTOR_MEMO_CAP = 64
MISS = object()

re_down = re.compile(r'^down_blocks_(\d+)_(resnets|attentions|downsamplers)_(\d+)')
re_up = re.compile(r'^up_blocks_(\d+)_(resnets|attentions|upsamplers)_(\d+)')
re_chain_index = re.compile(r'^(\d+)')

state: dict = {'stamp': None, 'layout': None, 'index': {}, 'vectors': {}}
warned: set = set()


def warn_once(key, message):
    if key not in warned:
        warned.add(key)
        log.warning(message)


def build_unet_layout(arch, mapping):
    down, up = -1, -1
    for key in mapping:
        if not key.startswith('lora_unet_'):
            continue
        tail = key[len('lora_unet_'):]
        m = re_down.match(tail)
        if m is not None:
            down = max(down, int(m.group(1)))
            continue
        m = re_up.match(tail)
        if m is not None:
            up = max(up, int(m.group(1)))
    if down < 0 or up < 0:
        return None
    n_in = 3 * (down + 1) # conv_in plus two pairs and a sampler slot per group: the compvis input_blocks count
    n_out = 3 * (up + 1)
    n = 2 + n_in + n_out
    return {
        'arch': arch, 'kind': 'unet', 'n': n, 'n_in': n_in,
        'ins': list(range(1, 1 + n_in)),
        'mids': [1 + n_in],
        'outs': list(range(2 + n_in, n)),
    }


def build_dit_layout(arch, mapping):
    prefixes = CHAINS.get(arch)
    if prefixes is None:
        return None
    counts = [0 for _ in prefixes]
    for key in mapping:
        if not key.startswith('lora_transformer_'):
            continue
        tail = key[len('lora_transformer_'):]
        for i, prefix in enumerate(prefixes):
            if tail.startswith(prefix):
                m = re_chain_index.match(tail[len(prefix):])
                if m is not None:
                    counts[i] = max(counts[i], int(m.group(1)) + 1)
                break
    total = sum(counts)
    if total == 0:
        return None
    chains = []
    offset = 0
    for prefix, count in zip(prefixes, counts, strict=False):
        chains.append((prefix, count, offset))
        offset += count
    n = 1 + total
    blocks = list(range(1, n))
    return {
        'arch': arch, 'kind': 'dit', 'n': n, 'chains': chains,
        'ins': [s for i, s in enumerate(blocks) if i * 3 // total == 0],
        'mids': [s for i, s in enumerate(blocks) if i * 3 // total == 1],
        'outs': [s for i, s in enumerate(blocks) if i * 3 // total == 2],
    }


def layout():
    sd_model = getattr(shared, 'sd_model', None)
    mapping = getattr(sd_model, 'network_layer_mapping', None) if sd_model is not None else None
    if not mapping:
        return None
    arch = shared.sd_model_type
    stamp = (arch, id(mapping))
    if state['stamp'] == stamp:
        return state['layout']
    state['stamp'] = stamp
    state['layout'] = build_unet_layout(arch, mapping) if arch in UNET_ARCHES else build_dit_layout(arch, mapping)
    state['index'].clear()
    state['vectors'].clear()
    return state['layout']


def classify(sd_key, lay):
    if sd_key.startswith('lora_te'):
        return 0 if lay['kind'] == 'unet' else None # BASE covers the TE on unet arches; transformer vectors do not model the TE
    if sd_key.startswith('lora_llm_adapter_'):
        return None
    if lay['kind'] == 'unet':
        if not sd_key.startswith('lora_unet_'):
            return None
        tail = sd_key[len('lora_unet_'):]
        m = re_down.match(tail)
        if m is not None:
            slot = 1 + 3 * int(m.group(1)) + (2 if m.group(2) == 'downsamplers' else int(m.group(3)))
            return 1 + slot
        m = re_up.match(tail)
        if m is not None:
            slot = 3 * int(m.group(1)) + (2 if m.group(2) == 'upsamplers' else int(m.group(3)))
            return 2 + lay['n_in'] + slot
        if tail.startswith('mid_block'):
            return 1 + lay['n_in']
        if tail.startswith('conv_in'):
            return 1 # IN00
        if tail.startswith('conv_out') or tail.startswith('conv_norm_out'):
            return lay['n'] - 1 # the compvis out group belongs to the last output block
        return 0 # time_embedding, add_embedding and other non-block leaves
    if not sd_key.startswith('lora_transformer_'):
        return None
    tail = sd_key[len('lora_transformer_'):]
    for prefix, count, offset in lay['chains']:
        if tail.startswith(prefix):
            m = re_chain_index.match(tail[len(prefix):])
            if m is not None and int(m.group(1)) < count:
                return 1 + offset + int(m.group(1))
            return 0
    return 0 # embedders, projections, refiners and other non-chain layers


def block_index(sd_key):
    lay = layout()
    if lay is None:
        return None
    cached = state['index'].get(sd_key, MISS)
    if cached is not MISS:
        return cached
    idx = classify(sd_key, lay)
    state['index'][sd_key] = idx
    return idx


def fill_band(vec, slots, lo, hi):
    k = len(slots)
    for i, s in enumerate(slots):
        if lo * k <= i < hi * k:
            vec[s] = 1.0


def classic_vector(name, lay):
    if name == 'ALL':
        return [1.0] * lay['n']
    vec = [0.0] * lay['n']
    if name == 'NONE':
        return vec
    vec[0] = 1.0
    if name == 'INALL':
        fill_band(vec, lay['ins'], 0.0, 1.0)
    elif name == 'INS': # shallow half of the input side
        fill_band(vec, lay['ins'], 0.0, 0.5)
    elif name == 'IND': # deep half of the input side
        fill_band(vec, lay['ins'], 0.5, 1.0)
    elif name == 'MIDD': # the middle of the network: deep input half, mid, deep output half
        fill_band(vec, lay['ins'], 0.5, 1.0)
        fill_band(vec, lay['mids'], 0.0, 1.0)
        fill_band(vec, lay['outs'], 0.0, 0.5)
    elif name == 'OUTALL':
        fill_band(vec, lay['outs'], 0.0, 1.0)
    elif name == 'OUTD': # deep half of the output side, nearest the mid
        fill_band(vec, lay['outs'], 0.0, 0.5)
    elif name == 'OUTS': # shallow half of the output side, nearest the image
        fill_band(vec, lay['outs'], 0.5, 1.0)
    return vec


def chain_vector(name, lay):
    chains = lay.get('chains') or []
    if len(chains) != 2:
        return None
    vec = [1.0] * lay['n']
    keep = 0 if name == 'DOUBLE' else 1
    for i, (_prefix, count, offset) in enumerate(chains):
        val = 1.0 if i == keep else 0.0
        for s in range(1 + offset, 1 + offset + count):
            vec[s] = val
    return vec


def stretch(src, k):
    if k == len(src):
        return [float(v) for v in src]
    out = []
    for i in range(k):
        x = i * (len(src) - 1) / (k - 1) if k > 1 else 0.0
        lo = int(x)
        hi = min(lo + 1, len(src) - 1)
        f = x - lo
        out.append(float(src[lo]) * (1.0 - f) + float(src[hi]) * f)
    return out


def preset_vector(name, lay):
    from modules.merging.merge_presets import BLOCK_WEIGHTS_PRESETS, SDXL_BLOCK_WEIGHTS_PRESETS
    if name in CHAIN_NAMES:
        return chain_vector(name, lay)
    if name in CLASSIC:
        return classic_vector(name, lay)
    if lay['arch'] == 'sdxl':
        src = SDXL_BLOCK_WEIGHTS_PRESETS.get(name) or SDXL_BLOCK_WEIGHTS_PRESETS.get('SDXL_' + name)
        if src is not None:
            return [1.0] + [float(v) for v in src[1:]] # merge tables carry 0 in the BASE slot; a preset must leave the TE alone
    if name.startswith('SDXL_'):
        return None # explicitly arch-tagged, not reinterpreted elsewhere
    src = BLOCK_WEIGHTS_PRESETS.get(name)
    if src is None:
        return None
    if lay['arch'] == 'sd':
        return [1.0] + [float(v) for v in src[1:]]
    return [1.0] + stretch(src[1:], lay['n'] - 1)


def parse_vector(parts, lay):
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return None
    n = lay['n']
    if len(vals) == n:
        return vals
    if len(vals) == n - 1:
        return [1.0] + vals
    legacy = SD1_17 if lay['arch'] == 'sd' else (SDXL_12 if lay['arch'] == 'sdxl' else None)
    if legacy is not None and len(vals) == len(legacy):
        vec = [1.0] * n # slots the a1111 layouts omit stay neutral
        for slot, v in zip(legacy, vals, strict=False):
            vec[slot] = v
        return vec
    return None


def resolve(spec):
    """Resolve a raw lbw value into a slot vector for the current model, or None when it fits nothing."""
    lay = layout()
    if lay is None:
        return None
    raw = str(spec).strip()
    key = raw.lower()
    if key in state['vectors']:
        return state['vectors'][key]
    if len(state['vectors']) > VECTOR_MEMO_CAP:
        state['vectors'].clear()
    vec = None
    if ',' in raw:
        vec = parse_vector([x.strip() for x in raw.split(',')], lay)
        if vec is None:
            warn_once(f'lbw-vector:{key}:{lay["arch"]}', f'Network blocks: value="{raw}" arch={lay["arch"]} expected={lay["n"]} fallback=none')
    else:
        try:
            vec = [float(raw)] * lay['n']
        except ValueError:
            vec = preset_vector(raw.upper(), lay)
            if vec is None:
                warn_once(f'lbw-name:{key}:{lay["arch"]}', f'Network blocks: preset="{raw}" arch={lay["arch"]} fallback=none')
    if vec is not None:
        log.info(f'Network blocks: value="{raw}" arch={lay["arch"]} slots={lay["n"]} range={min(vec):.2f}-{max(vec):.2f}')
    state['vectors'][key] = vec
    return vec


def factor(sd_key, net):
    """Per-layer scale from a network's block vector; 1.0 whenever the vector does not apply."""
    try:
        spec = getattr(net, 'block_spec', None)
        if not spec:
            return 1.0
        vec = resolve(spec)
        if vec is None:
            return 1.0
        idx = block_index(sd_key)
        if idx is None:
            return 1.0
        return float(vec[idx])
    except Exception as e:
        warn_once('lbw-error', f'Network blocks: {e} fallback=none')
        return 1.0


def net_signature(net):
    """Normalized spec of one network, or None; joins content identities such as the factor cache signature."""
    spec = getattr(net, 'block_spec', None)
    if not spec:
        return None
    return str(spec).strip().lower()


def active():
    return any(getattr(net, 'block_spec', None) for net in l.loaded_networks)


def signature():
    """Identity suffix for the per-module apply stamp; empty while no loaded network carries block weights."""
    specs = [f'{net.name}:{net_signature(net)}' for net in l.loaded_networks if getattr(net, 'block_spec', None)]
    if len(specs) == 0:
        return ''
    return '|lbw=' + ','.join(specs)
