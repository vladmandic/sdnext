import re
import time
from modules.logger import log


# Canonical base-model metadata lives in the civitai/civitai repo. The live
# /images validator is generated from it, so the file has metadata (group,
# ecosystem, engine, hidden) while the validator has the current name list.
# Callers merge both.
github_cache: list[dict] = []
github_cache_time: float = 0
GITHUB_TTL = 6 * 3600  # 6 hours
GITHUB_URL = 'https://raw.githubusercontent.com/civitai/civitai/main/src/shared/constants/base-model.constants.ts'


def parse_base_model_config(ts_source: str) -> list[dict]:
    """Parse the baseModelConfig array from base-model.constants.ts.

    Uses character-by-character bracket walking rather than regex because
    entries can span multiple lines. Returns list of dicts with keys
    name/type/group/hidden plus optional ecosystem/engine/family.
    """
    start_match = re.search(r'const\s+baseModelConfig\s*=\s*\[', ts_source)
    if not start_match:
        return []
    # Walk to the matching ] respecting string literals
    pos = start_match.end()
    depth = 1
    in_string: str | None = None
    end_pos = -1
    while pos < len(ts_source):
        ch = ts_source[pos]
        if in_string is not None:
            if ch == '\\':
                pos += 2
                continue
            if ch == in_string:
                in_string = None
        else:
            if ch in ("'", '"', '`'):
                in_string = ch
            elif ch == '[':
                depth += 1
            elif ch == ']':
                depth -= 1
                if depth == 0:
                    end_pos = pos
                    break
        pos += 1
    if end_pos < 0:
        return []
    array_body = ts_source[start_match.end():end_pos]
    # Extract top-level {...} entries, respecting strings and nested braces
    entries: list[str] = []
    brace_start = -1
    brace_depth = 0
    in_string = None
    i = 0
    while i < len(array_body):
        ch = array_body[i]
        if in_string is not None:
            if ch == '\\':
                i += 2
                continue
            if ch == in_string:
                in_string = None
        else:
            if ch in ("'", '"', '`'):
                in_string = ch
            elif ch == '{':
                if brace_depth == 0:
                    brace_start = i
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth == 0 and brace_start >= 0:
                    entries.append(array_body[brace_start:i + 1])
                    brace_start = -1
        i += 1
    # Per-entry field extraction (string + bool values only)
    field_re = re.compile(
        r"(\w+)\s*:\s*(?:'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"|(true|false))"
    )
    parsed: list[dict] = []
    for entry in entries:
        fields: dict = {}
        for m in field_re.finditer(entry):
            key = m.group(1)
            if m.group(2) is not None:
                fields[key] = m.group(2)
            elif m.group(3) is not None:
                fields[key] = m.group(3)
            elif m.group(4) is not None:
                fields[key] = m.group(4) == 'true'
        if 'name' in fields and 'type' in fields and 'group' in fields:
            item: dict = {
                'name': fields['name'],
                'type': fields['type'],
                'group': fields['group'],
                'hidden': bool(fields.get('hidden', False)),
            }
            for opt in ('ecosystem', 'engine', 'family'):
                if opt in fields:
                    item[opt] = fields[opt]
            parsed.append(item)
    return parsed


def fetch_github_base_models() -> list[dict]:
    """Fetch and parse civitai's base-model constants from GitHub.

    Returns list of metadata dicts (name, type, group, hidden, plus
    optional ecosystem/engine/family). Returns [] on any failure;
    callers fall back to the live /images probe. Cached with a longer
    TTL than discover_options since these constants change rarely.
    """
    global github_cache, github_cache_time  # pylint: disable=global-statement
    now = time.time()
    if github_cache and (now - github_cache_time) < GITHUB_TTL:
        return github_cache
    try:
        from modules import shared
        r = shared.req(GITHUB_URL)
        if r.status_code != 200:
            log.debug(f'CivitAI github constants: code={r.status_code}')
            return []
        parsed = parse_base_model_config(r.text)
        if parsed:
            github_cache = parsed
            github_cache_time = now
        return parsed
    except Exception as e:
        log.debug(f'CivitAI github constants fetch failed: {e}')
        return []
