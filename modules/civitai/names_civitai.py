"""Save names for CivitAI version files.

CivitAI serves one canonical name for every variant of a version, so the
variant (fp8, bf16, Q8_0) is suffixed whenever the name does not already
carry it, and the name stays stable when a creator adds variants later.
Dual-transformer expert roles join it, and same-name collisions cascade to
the size class, then the file id. With civitai_save_precision off the
variant is added only on collision.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse
from modules.civitai.models_civitai import CivitFile, CivitVersion


COMPANION_TYPES = ('VAE', 'Text Encoder')

# Dual-transformer bases publish each expert as its own version with the role
# named only in the version title; there is no structured field for it.
DUAL_TRANSFORMER_ROLES = [
    {
        'base': re.compile(r'^Wan Video 2\.2 .*A14B', re.IGNORECASE),
        'roles': [('high-noise', ('high',)), ('low-noise', ('low',))],
    },
    {
        # The conditional transformer is the primary and stays unsuffixed.
        # name_hint covers uploads tagged base "Other", which predate the
        # Ideogram 4.0 base tag on CivitAI.
        'base': re.compile(r'^Ideogram 4', re.IGNORECASE),
        'name_hint': re.compile(r'ideogram\s*4', re.IGNORECASE),
        'roles': [('uncond', ('uncond', 'unconditional'))],
    },
]


@dataclass
class NameContext:
    name: str = ''  # version name
    base_model: str = ''
    model_name: str = ''
    roles: dict[int, str | None] = field(default_factory=dict)  # header-derived, keyed by file id; wins over name-derived roles
    variants: dict[int, str] = field(default_factory=dict)  # header-upgraded tokens (fp8 -> fp8_e4m3fn), keyed by file id


def insert_name_suffix(name: str, suffix: str) -> str:
    stem, dot, ext = name.rpartition('.')
    return f'{stem}-{suffix}.{ext}' if dot and stem else f'{name}-{suffix}'


def url_quant_type(url: str) -> str | None:
    values = parse_qs(urlparse(url or '').query).get('quantType')
    return values[0] if values else None


def is_gguf(f: CivitFile) -> bool:
    return f.metadata.format == 'GGUF' or f.name.lower().endswith('.gguf')


def file_variant(f: CivitFile) -> str | None:
    """Precision for safetensors is metadata.fp. A GGUF's variant is its quant:
    metadata.quantType (model endpoint only) or the quantType query param of
    the download URL; its metadata.fp is the dtype it was quantized from, so it
    never names the file."""
    quant = f.metadata.quant_type or url_quant_type(f.download_url)
    if is_gguf(f):
        return quant
    return f.metadata.fp or quant


def split_words(text: str) -> str:
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return re.sub(r'[^a-zA-Z0-9]+', ' ', text).lower()


def has_word(text: str, word: str) -> bool:
    return re.search(rf'\b{re.escape(word)}\b', text) is not None


def name_carries(name: str, token: str) -> bool:
    """True when the file name already spells the token out (x_fp8 and fp8, x-Q8_0 and Q8_0)."""
    return has_word(split_words(name), split_words(token).strip())


def find_arch(context: NameContext) -> dict | None:
    for arch in DUAL_TRANSFORMER_ROLES:
        if arch['base'].search(context.base_model or ''):
            return arch
        hint = arch.get('name_hint')
        if hint and hint.search(f'{context.model_name or ""} {context.name or ""}'):
            return arch
    return None


def match_role(arch: dict, text: str) -> str | None:
    for suffix, words in arch['roles']:
        if any(has_word(text, w) for w in words):
            return suffix
    return None


def file_role(f: CivitFile, context: NameContext) -> str | None:
    """Expert role for a file: from its header, else the version title; never
    for companions or when the file name already carries a role word."""
    arch = find_arch(context)
    if arch is None or f.type in COMPANION_TYPES or match_role(arch, split_words(f.name)):
        return None
    return context.roles.get(f.id) or match_role(arch, split_words(context.name or ''))


def role_from_metadata(metadata: dict | None, context: NameContext) -> str | None:
    """Header __metadata__ keys are tool-specific (model_type, modelspec.*), so scan every string value for role words."""
    arch = find_arch(context)
    if arch is None or not metadata:
        return None
    return match_role(arch, split_words(' '.join(v for v in metadata.values() if isinstance(v, str))))


def peek_targets(files: list[CivitFile]) -> list[CivitFile]:
    return [f for f in files if f.name.lower().endswith(('.safetensors', '.gguf'))]


def precision_from_dtype(dtype: str | None) -> str | None:
    from modules.model_probe import DTYPE_PRECISION_TOKENS
    return DTYPE_PRECISION_TOKENS.get(dtype) if dtype else None


def precision_claim_satisfied(claimed: str, actual: str) -> bool:
    """A generic claim (fp8) is satisfied by any of its variants (fp8_e4m3fn)."""
    return claimed == actual or actual.startswith(claimed)


def apply_peeks(context: NameContext, files: list[CivitFile], peeks: dict[int, dict]) -> NameContext:
    """Fold header probes into the context: roles from __metadata__, a GGUF's
    quant from its header, and a generic metadata variant upgraded to the
    dtype-exact token."""
    for f in files:
        data = peeks.get(f.id) or {}
        context.roles[f.id] = role_from_metadata(data.get('metadata'), context)
        if data.get('quant'):
            context.variants[f.id] = data['quant']
            continue
        probe = data.get('probe')
        if not probe:
            continue
        claimed = file_variant(f)
        exact = precision_from_dtype(probe.get('dominant_dtype')) if probe.get('ok') else None
        if claimed and exact and exact != claimed and precision_claim_satisfied(claimed, exact):
            context.variants[f.id] = exact
    return context


def version_context(version: CivitVersion, peek: bool = True) -> NameContext:
    """Naming context for a version; peek=True probes every safetensors file header."""
    context = NameContext(name=version.name, base_model=version.base_model, model_name=version.model.name if version.model else '')
    if peek:
        from modules.civitai.peek_civitai import peek_headers
        targets = peek_targets(version.files)
        apply_peeks(context, targets, peek_headers(targets))
    return context


def precision_enabled() -> bool:
    from modules import shared
    return bool(getattr(shared.opts, 'civitai_save_precision', True))


def save_name(f: CivitFile, siblings: list[CivitFile], context: NameContext | None = None, precision: bool = True) -> str:
    context = context or NameContext()

    def with_variant(name, x):
        variant = context.variants.get(x.id) or file_variant(x)
        return insert_name_suffix(name, variant) if variant and not name_carries(x.name, variant) else name

    def with_role(name, x):
        role = file_role(x, context)
        return insert_name_suffix(name, role) if role else name

    def with_size(name, x):
        return insert_name_suffix(name, x.metadata.size) if x.metadata.size else name

    def tier(x, steps):
        name = x.name
        for step in steps:
            name = step(name, x)
        return name

    if precision:
        tiers = [[with_variant, with_role], [with_variant, with_role, with_size]]
    else:
        tiers = [[with_role], [with_role, with_variant], [with_role, with_variant, with_size]]
    others = [s for s in siblings if s.id != f.id]
    for steps in tiers:
        name = tier(f, steps)
        if not any(tier(s, steps) == name for s in others):
            return name
    return insert_name_suffix(tier(f, tiers[0]), str(f.id))


def route_type(f: CivitFile, model_type: str) -> str:
    """Companion files route to their own type's folder, not the model's."""
    return f.type if f.type in COMPANION_TYPES else model_type


def version_names(version: CivitVersion, context: NameContext, precision: bool = True) -> list[dict]:
    model_type = version.model.type if version.model else 'Checkpoint'
    return [{
        'id': f.id,
        'name': f.name,
        'save_name': save_name(f, version.files, context, precision),
        'type': route_type(f, model_type),
        'variant': context.variants.get(f.id) or file_variant(f),
        'role': file_role(f, context),
        'size': f.metadata.size,
        'sha256': f.hashes.sha256,
    } for f in version.files]


def fetch_version(version_id: int, token: str | None = None) -> tuple[CivitVersion | None, str, int]:
    """Version whose files come from /models, the payload the UIs list from:
    /model-versions drops GGUF quantType and can still name a fresh file by
    its upload key. The version's own files stand in when /models fails."""
    from modules.civitai.client_civitai import client
    version, error, status = client.fetch_version(version_id, token=token)
    if version is None:
        return None, error, status
    model = client.get_model(version.model_id, token=token)
    match = next((v for v in model.versions if v.id == version.id), None) if model else None
    if match is not None:
        version.files = match.files
    return version, '', 200


def resolve_file(version_id: int, file_id: int, token: str | None = None) -> tuple[dict | None, str, int]:
    """Download parameters for one file of a version: url, save name, hash and route type."""
    version, error, status = fetch_version(version_id, token=token)
    if version is None:
        return None, f'version {version_id} fetch failed: {error}', status
    f = next((x for x in version.files if x.id == file_id), None)
    if f is None:
        return None, f'file {file_id} not in version {version_id}', 404
    context = version_context(version)
    model = version.model
    return {
        'url': f.download_url,
        'filename': save_name(f, version.files, context, precision_enabled()),
        'expected_hash': (f.hashes.sha256 or '').lower(),
        'model_type': route_type(f, model.type if model else 'Checkpoint'),
        'model_name': model.name if model else '',
        'base_model': version.base_model,
        'model_id': version.model_id,
        'version_id': version.id,
        'version_name': version.name,
        'nsfw': bool(model.nsfw) if model else False,
    }, '', 200
