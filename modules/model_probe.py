"""Header-only model analysis: architecture fingerprinting, precision/quant
detection, and embedded metadata extraction from safetensors/gguf files
without reading tensor data. Consumed by the civitai remote peek and the
local model audit; analyze_header is pure so both paths share one analyzer.
"""
import os
import re
import json
import threading
from collections import Counter
from dataclasses import dataclass

from modules.logger import log


SCHEMA_VERSION = 4
MAX_HEADER_BYTES = 16 * 1024 * 1024
STRIP_PREFIXES = ('model.diffusion_model.', 'diffusion_model.', 'net.')
# companion families bundled alongside the diffusion core in full checkpoints
COMPANION_PREFIXES = ('text_encoders.', 'cond_stage_model.', 'conditioner.', 'first_stage_model.', 'vae.', 'vocoder.', 'audio_vae.')
LORA_SUFFIXES = ('.lora_down.weight', '.lora_up.weight', '.lora_A.weight', '.lora_B.weight', '.hada_w1_a', '.lokr_w1', '.lokr_w2', '.dora_scale', '.diff', '.diff_b')


@dataclass(frozen=True)
class ArchFingerprint:
    family: str
    display: str
    required: tuple  # regex fragments, all must match at least one inner key
    forbidden: tuple = ()
    weight: int = 1


# Markers verified against real headers (local library + civitai ranged peeks).
# Families not listed report 'unknown' rather than guessing.
FINGERPRINTS = (
    ArchFingerprint('sdxl', 'Stable Diffusion XL', (r'^input_blocks\.', r'^middle_block\.', r'^label_emb\.')),
    ArchFingerprint('sd', 'Stable Diffusion 1.x/2.x', (r'^input_blocks\.', r'^middle_block\.'), forbidden=(r'^label_emb\.',)),
    ArchFingerprint('sd3', 'Stable Diffusion 3', (r'^joint_blocks\.', r'^context_embedder\.')),
    ArchFingerprint('f1', 'FLUX.1', (r'^double_blocks\.\d+\.img_attn', r'^vector_in\.'), forbidden=(r'^distilled_guidance_layer\.', r'stream_modulation')),
    ArchFingerprint('chroma', 'Chroma', (r'^double_blocks\.\d+\.img_attn', r'^distilled_guidance_layer\.')),
    ArchFingerprint('f2', 'FLUX.2', (r'^double_blocks\.\d+\.img_attn', r'stream_modulation'), forbidden=(r'^vector_in\.',)),
    ArchFingerprint('qwen', 'Qwen Image', (r'^transformer_blocks\.\d+\.img_mlp', r'^time_text_embed\.'), forbidden=(r'audio_attn',)),
    ArchFingerprint('krea2', 'Krea 2', (r'^blocks\.\d+\.attn', r'^txtfusion\.')),
    ArchFingerprint('wanai-animate', 'Wan 2.2 Animate', (r'^blocks\.\d+\.cross_attn', r'^patch_embedding\.', r'^face_adapter\.|^motion_encoder\.|^pose_patch_embedding\.')),
    ArchFingerprint('wanai', 'Wan DiT', (r'^blocks\.\d+\.cross_attn', r'^blocks\.\d+\.self_attn', r'^patch_embedding\.', r'^time_projection\.'), forbidden=(r'^face_adapter\.', r'^motion_encoder\.', r'^pose_patch_embedding\.', r'^llm_adapter\.')),
    ArchFingerprint('anima', 'Anima', (r'^blocks\.\d+\.cross_attn', r'^llm_adapter\.')),
    ArchFingerprint('ernieimage', 'Ernie Image', (r'^layers\.\d+\.self_attention', r'^adaLN_sa_ln|^final_norm\.')),
    ArchFingerprint('zimage', 'Z-Image', (r'^layers\.\d+\.attention', r'^noise_refiner\.', r'^cap_embedder\.')),
    ArchFingerprint('ltxvideo', 'LTX Video', (r'^transformer_blocks\.\d+\.attn1', r'^adaln_single\.|^vocoder\.')),
    ArchFingerprint('ideogram4', 'Ideogram 4', (r'^layers\.\d+\.attention\.(qkv|o)\.', r'^adaln_proj\.|^t_embedding\.')),
)

# name/text to family hints, ordered specific-first; matched against lowercased
# text with spaces removed. Used for lora trainer metadata and for names that
# imply a base (folder names, base tags).
FAMILY_NAME_HINTS = (
    ('stable-diffusion-xl', 'sdxl'), ('sdxl', 'sdxl'), ('sd_xl', 'sdxl'),
    ('animagine', 'sdxl'), ('pony', 'sdxl'), ('illustrious', 'sdxl'), ('noobai', 'sdxl'),
    ('stable-diffusion-3', 'sd3'), ('sd3', 'sd3'),
    ('stable-diffusion-v1', 'sd'), ('sd-v1', 'sd'), ('sd_v1', 'sd'), ('sd15', 'sd'), ('sd1.5', 'sd'), ('sd2', 'sd'),
    ('flux.2', 'f2'), ('flux2', 'f2'), ('klein', 'f2'), ('chroma', 'chroma'), ('flux', 'f1'),
    ('krea', 'krea2'), ('qwen', 'qwen'), ('wan', 'wanai'), ('anima', 'anima'),
    ('ideogram', 'ideogram4'), ('z-image', 'zimage'), ('zimage', 'zimage'),
    ('ernie', 'ernieimage'), ('ltx', 'ltxvideo'),
)


def family_from_text(text: str) -> str | None:
    compact = (text or '').lower().replace(' ', '')
    for hint, family in FAMILY_NAME_HINTS:
        if hint in compact:
            return family
    return None
LORA_KEY_HINTS = (
    ('txtfusion', 'krea2'),
    ('distilled_guidance', 'chroma'),
    ('stream_modulation', 'f2'),
    ('double_blocks', 'f1'),
    ('joint_blocks', 'sd3'),
    ('llm_adapter', 'anima'),
    ('cross_attn', 'wanai'),
    ('img_mlp', 'qwen'),
    ('noise_refiner', 'zimage'),
    ('context_refiner', 'zimage'),
    # diffusers flux layout; f1 vs f2 not separable from lora keys
    ('single_transformer_blocks', 'f1'),
    ('transformer_blocks', 'qwen'),
)
# checked in order; trainers sometimes write a bogus modelspec.architecture
# while ss_base_model_version names the real base
LORA_META_KEYS = ('ss_base_model_version', 'modelspec.architecture', 'ss_sd_model_name')
FAMILY_DISPLAY = {fp.family: fp.display for fp in FINGERPRINTS}

# expert-role vocabulary for dual-transformer archs, matched against
# __metadata__ values; wan uses phrases because a bare 'high' appears in
# unrelated metadata (e.g. 'high quality')
ROLE_WORDS = {
    'wanai': (('high-noise', ('high noise',)), ('low-noise', ('low noise',))),
    'ideogram4': (('uncond', ('uncond', 'unconditional')),),
}


def strip_key(key: str) -> str:
    for prefix in STRIP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def split_words(text: str) -> str:
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text or '')
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()


def detect_kind(inner_keys: list) -> str:
    """Container-level classification before arch matching."""
    if any(k.endswith(LORA_SUFFIXES) or '.lora_down.' in k or '.lora_up.' in k or '.lora.down.' in k or '.lora.up.' in k or '.lora_A.' in k or '.lora_B.' in k for k in inner_keys):
        return 'lora'
    top = Counter(k.split('.')[0] for k in inner_keys)
    total = sum(top.values())
    vae_keys = top.get('encoder', 0) + top.get('decoder', 0) + top.get('quant_conv', 0) + top.get('post_quant_conv', 0)
    if total > 0 and top.get('decoder', 0) > 0 and vae_keys > total * 0.9:
        return 'vae'
    diffusion_core = any(k.startswith(('input_blocks', 'double_blocks', 'blocks.', 'layers.', 'transformer_blocks', 'joint_blocks')) for k in inner_keys)
    llm_markers = any(('.mlp.gate_proj' in k or 'embed_tokens' in k or '.DenseReluDense.' in k) for k in inner_keys) or top.get('shared', 0) > 0
    if not diffusion_core and llm_markers:
        return 'text-encoder'
    return 'model'


def lora_context_dim(shapes: dict | None) -> int | None:
    """Cross-attention context width from a lora down/A tensor: 768 for sd1.x,
    1024 for sd2.x, 2048 for sdxl. Arbitrates when trainer metadata lies."""
    for k, shape in (shapes or {}).items():
        if 'attn2' in k and 'to_k' in k and ('lora_down' in k or '.lora.down.' in k or 'lora_A' in k):
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                return int(shape[1])
    return None


def match_lora_base(core_keys: list, metadata: dict | None, shapes: dict | None = None):
    """Loras train a module subset, so full fingerprints cannot apply. Trainer
    metadata names the base; keyless PEFT loras resolve via distinctive
    substrings. Returns (family, confidence, marker) or None."""
    def unet_split(family, confidence, marker):
        # sd-vs-sdxl from metadata or key heuristics is unreliable; shapes win
        dim = lora_context_dim(shapes)
        if dim is not None:
            return ('sdxl' if dim >= 2048 else 'sd'), 0.8, f'shape:context-dim={dim}'
        return family, confidence, marker

    for key in LORA_META_KEYS:
        family = family_from_text(str((metadata or {}).get(key, '')))
        if family is not None:
            if family in ('sd', 'sdxl'):
                return unet_split(family, 0.9, f'metadata:{key}')
            return family, 0.9, f'metadata:{key}'
    joined = '\n'.join(core_keys)
    for hint, family in LORA_KEY_HINTS:
        if hint in joined:
            return family, 0.5, f'key:{hint}'
    # bare blocks.N.attn without self/cross variants is the krea2 DiT layout
    if re.search(r'blocks\.\d+\.attn\.', joined) and 'self_attn' not in joined:
        return 'krea2', 0.5, 'key:blocks.attn'
    if 'input_blocks' in joined or 'down_blocks' in joined:
        return unet_split('sdxl' if 'te2' in joined else 'sd', 0.5, 'key:unet-blocks')
    return None


def match_arch(keys: list, metadata: dict | None, shapes: dict | None = None) -> dict:
    inner = [strip_key(k) for k in keys]
    core = [k for k in inner if not k.startswith(COMPANION_PREFIXES)]
    prefixes = {k[: len(k) - len(s)] for k, s in ((k, strip_key(k)) for k in keys) if k != s}
    kind = detect_kind(core)
    matched_family = None
    matched_markers = []
    candidates = []
    confidence = 0.0
    if kind == 'model':
        for fp in FINGERPRINTS:
            hits = [pat for pat in fp.required if any(re.search(pat, k) for k in core)]
            if len(hits) < len(fp.required):
                continue
            if any(any(re.search(pat, k) for k in core) for pat in fp.forbidden):
                continue
            candidates.append(fp)
            if matched_family is None:
                matched_family = fp.family
                matched_markers = hits
        confidence = 1.0 if matched_family and len(candidates) == 1 else (0.7 if matched_family else 0.0)
    elif kind == 'lora':
        resolved = match_lora_base(core, metadata, shapes)
        if resolved is not None:
            matched_family, confidence, marker = resolved
            matched_markers = [marker]
    variant = None
    if matched_family is not None and metadata:
        meta_text = split_words(' '.join(str(v) for v in metadata.values() if isinstance(v, str)))
        for suffix, words in ROLE_WORDS.get(matched_family, ()):
            if any(re.search(rf'\b{w}\b', meta_text) for w in words):
                variant = suffix
                break
    return {
        'kind': kind,
        'family': matched_family or 'unknown',
        'display': FAMILY_DISPLAY.get(matched_family, 'Unknown') if matched_family else 'Unknown',
        'confidence': confidence,
        'variant': variant,
        'detected_prefix': sorted(prefixes)[0] if prefixes else '',
        'matched_markers': matched_markers,
        'candidates': [{'family': fp.family, 'display': fp.display} for fp in candidates],
    }


def detect_quant(keys: list, dtypes: Counter, container: str) -> dict:
    if container == 'gguf':
        quantized = {d: n for d, n in dtypes.items() if d not in ('F32', 'F16', 'BF16')}
        dominant = max(quantized, key=quantized.get) if quantized else None
        return {'scheme': 'gguf' if dominant else None, 'format': dominant, 'marked_layers': None, 'source': 'gguf-qtype'}
    comfy = [k for k in keys if k.endswith('.comfy_quant')]
    if comfy:
        if dtypes.get('U8', 0) > dtypes.get('F8_E4M3', 0):
            fmt = 'int8_tensorwise'
        elif dtypes.get('F8_E4M3', 0):
            fmt = 'float8_e4m3fn'
        else:
            fmt = None
        return {'scheme': 'comfy_quant', 'format': fmt, 'marked_layers': len(comfy), 'source': 'dtype-inferred'}
    has_fp8 = dtypes.get('F8_E4M3', 0) + dtypes.get('F8_E5M2', 0) > 0
    has_scales = any(k.endswith(('scaled_fp8', '.scale_weight', '.scale_input', '.weight_scale')) for k in keys)
    if has_fp8 and has_scales:
        return {'scheme': 'scaled_fp8', 'format': 'float8_e4m3fn' if dtypes.get('F8_E4M3') else 'float8_e5m2', 'marked_layers': None, 'source': 'marker'}
    return {'scheme': None, 'format': None, 'marked_layers': None, 'source': None}


def analyze_header(header: dict, container: str = 'safetensors', arch_metadata: dict | None = None) -> dict:
    metadata = header.get('__metadata__') or {}
    if arch_metadata:
        metadata = {**metadata, **arch_metadata}
    entries = {k: v for k, v in header.items() if k != '__metadata__' and isinstance(v, dict)}
    keys = list(entries)
    dtypes = Counter(v.get('dtype') for v in entries.values() if v.get('dtype'))
    params = 0
    # dominant dtype is element-weighted over core tensors only: bundled
    # companions and scale scalars must not outvote the diffusion weights
    core_elements = Counter()
    for k, v in entries.items():
        shape = v.get('shape')
        if not isinstance(shape, (list, tuple)) or k.endswith('.comfy_quant'):
            continue
        n = 1
        for dim in shape:
            n *= int(dim)
        params += n
        if not strip_key(k).startswith(COMPANION_PREFIXES) and v.get('dtype'):
            core_elements[v['dtype']] += n
    inner_keys = [strip_key(k) for k in keys]
    shapes = {k: v.get('shape') for k, v in entries.items()}
    arch = match_arch(keys, metadata, shapes)
    quant = detect_quant(inner_keys, dtypes, container)
    flags = []
    if not metadata:
        flags.append('no_metadata')
    if any(k.startswith(COMPANION_PREFIXES) for k in inner_keys):
        flags.append('companion_bundled')
    if quant['scheme'] == 'comfy_quant':
        flags.append('comfy_marker')
    if arch['family'] == 'wanai' and arch['variant'] is None:
        flags.append('role_ambiguous')
    if arch['kind'] == 'lora' and 0 < arch['confidence'] < 0.9:
        flags.append('lora_base_inferred')
    return {
        'schema': SCHEMA_VERSION,
        'ok': True,
        'error': None,
        'container': container,
        'tensors': len(keys),
        'params': params,
        'dtypes': dict(dtypes),
        'dominant_dtype': core_elements.most_common(1)[0][0] if core_elements else (dtypes.most_common(1)[0][0] if dtypes else None),
        'arch': arch,
        'quant': quant,
        'metadata': metadata,
        'metadata_present': bool(metadata),
        'flags': flags,
    }


def error_result(container: str, error: str, flag: str = 'unreadable') -> dict:
    return {
        'schema': SCHEMA_VERSION, 'ok': False, 'error': error, 'container': container,
        'tensors': 0, 'params': 0, 'dtypes': {}, 'dominant_dtype': None,
        'arch': {'kind': 'unknown', 'family': 'unknown', 'display': 'Unknown', 'confidence': 0.0, 'variant': None, 'detected_prefix': '', 'matched_markers': [], 'candidates': []},
        'quant': {'scheme': None, 'format': None, 'marked_layers': None, 'source': None},
        'metadata': {}, 'metadata_present': False, 'flags': [flag],
    }


def read_safetensors_header(path: str) -> dict:
    """Raw header read: 8-byte length prefix + JSON. Does not touch the global
    sd_metadata cache and keeps ss_tag_frequency at full fidelity."""
    with open(path, 'rb') as f:
        header_len = int.from_bytes(f.read(8), 'little')
        if header_len <= 0 or header_len > MAX_HEADER_BYTES:
            raise ValueError(f'implausible header length: {header_len}')
        return json.loads(f.read(header_len).decode('utf-8'))


def comfy_marker_format(path: str) -> str | None:
    """Exact comfy_quant format string from the first marker tensor's bytes;
    only possible for local files, remote peeks stay dtype-inferred."""
    try:
        with open(path, 'rb') as f:
            header_len = int.from_bytes(f.read(8), 'little')
            if header_len <= 0 or header_len > MAX_HEADER_BYTES:
                return None
            header = json.loads(f.read(header_len).decode('utf-8'))
            for k, v in header.items():
                if k == '__metadata__' or not k.endswith('.comfy_quant') or not isinstance(v, dict):
                    continue
                start, end = v.get('data_offsets', (0, 0))
                if end <= start or end - start > 4096:
                    return None
                f.seek(8 + header_len + start)
                fmt = json.loads(f.read(end - start).decode('utf-8')).get('format')
                return str(fmt) if fmt else None
    except Exception:
        return None
    return None


# quant format / dtype to the precision token used in filenames; fp8 stays
# variant-specific because e4m3fn and e5m2 differ in kernel support
QUANT_PRECISION_TOKENS = {'int8_tensorwise': 'int8', 'float8_e4m3fn': 'fp8_e4m3fn', 'float8_e5m2': 'fp8_e5m2', 'nvfp4': 'nvfp4', 'mxfp8': 'mxfp8'}
DTYPE_PRECISION_TOKENS = {'F32': 'fp32', 'F16': 'fp16', 'BF16': 'bf16', 'F8_E4M3': 'fp8_e4m3fn', 'F8_E5M2': 'fp8_e5m2'}


def precision_token(probe: dict) -> str | None:
    """Filename token for a probe's true precision; None when the container
    encodes it already (gguf) or nothing is known."""
    quant = probe.get('quant') or {}
    scheme = quant.get('scheme')
    if scheme == 'gguf':
        return None
    if scheme in ('comfy_quant', 'scaled_fp8'):
        fmt = quant.get('format') or ''
        return QUANT_PRECISION_TOKENS.get(fmt, re.sub(r'[^a-z0-9]', '', fmt.lower()) or None)
    return DTYPE_PRECISION_TOKENS.get(probe.get('dominant_dtype') or '')


def probe_safetensors_file(path: str) -> dict:
    try:
        header = read_safetensors_header(path)
    except Exception as e:
        return error_result('safetensors', str(e), 'corrupt_header')
    result = analyze_header(header)
    if result['quant']['scheme'] == 'comfy_quant':
        fmt = comfy_marker_format(path)
        if fmt:
            result['quant']['format'] = fmt
            result['quant']['source'] = 'marker'
    return result


def probe_gguf_file(path: str) -> dict:
    try:
        from modules.ggml import install_gguf
        install_gguf()
        import gguf
        reader = gguf.GGUFReader(path)
        header = {}
        for tensor in reader.tensors:
            header[str(tensor.name)] = {'dtype': str(tensor.tensor_type.name), 'shape': [int(d) for d in reversed(tensor.shape)]}
        arch_metadata = {}
        for key in ('general.architecture', 'general.name'):
            fld = reader.fields.get(key)
            if fld is not None:
                try:
                    arch_metadata[key] = str(bytes(fld.parts[fld.data[0]]).decode('utf-8'))
                except Exception:
                    pass
        return analyze_header(header, container='gguf', arch_metadata=arch_metadata)
    except Exception as e:
        return error_result('gguf', str(e))


probe_cache = None
probe_cache_lock = threading.Lock()


def probe_cache_file() -> str:
    from modules import paths
    return os.path.join(paths.data_path, 'data', 'model_probe.json')


def probe_file(path: str, use_cache: bool = True) -> dict:
    """Dispatch by extension with an mtime-validated persistent cache."""
    global probe_cache  # pylint: disable=global-statement
    ext = os.path.splitext(path)[1].lower()
    if ext not in ('.safetensors', '.gguf'):
        return error_result('unknown', f'unsupported extension: {ext}', 'unsupported')
    try:
        stat = os.stat(path)
    except OSError as e:
        return error_result('unknown', str(e))
    with probe_cache_lock:
        if probe_cache is None:
            from modules.json_helpers import readfile
            fn = probe_cache_file()
            probe_cache = readfile(fn, silent=True, lock=True, as_type='dict') if os.path.isfile(fn) else {}
        entry = probe_cache.get(path) if use_cache else None
    if entry and entry.get('mtime') == stat.st_mtime and entry.get('size') == stat.st_size and entry.get('schema') == SCHEMA_VERSION:
        return entry['probe']
    probe = probe_safetensors_file(path) if ext == '.safetensors' else probe_gguf_file(path)
    with probe_cache_lock:
        probe_cache[path] = {'mtime': stat.st_mtime, 'size': stat.st_size, 'schema': SCHEMA_VERSION, 'probe': probe}
    return probe


def save_probe_cache():
    """Flush the probe cache to disk; call once per scan, not per file."""
    with probe_cache_lock:
        if probe_cache is None:
            return
        from modules.json_helpers import writefile
        try:
            writefile(probe_cache, probe_cache_file(), silent=True, atomic=True)
        except Exception as e:
            log.warning(f'Model probe cache save error: {e}')
