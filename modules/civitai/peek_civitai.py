"""Remote safetensors header probe for CivitAI files, cached by file id."""

import hashlib
import json
import struct
from modules.logger import log


peek_cache = None


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]


def peek_cache_get(file_id: int, url: str):
    """Persistent probe cache keyed by civitai file id; content per id is
    immutable, so entries never expire. A url hash guards against id reuse."""
    global peek_cache  # pylint: disable=global-statement
    from modules import paths
    from modules.json_helpers import readfile
    if peek_cache is None:
        peek_cache = readfile(paths.civitai_probe_file, silent=True, lock=True, as_type='dict')
    entry = peek_cache.get(str(file_id))
    if entry and entry.get('url_hash') == url_hash(url):
        return entry.get('response')
    return None


def peek_cache_put(file_id: int, url: str, response: dict):
    from modules import paths
    from modules.json_helpers import writefile
    peek_cache[str(file_id)] = {
        'url_hash': url_hash(url),
        'response': response,
    }
    writefile(peek_cache, paths.civitai_probe_file, silent=True, atomic=True)


def normalize_url(url: str) -> str:
    # civitai.red serves the same download service and rewrites downloadUrl to
    # its own host; normalize so the guard, cache hash and fetch host agree.
    return url.replace('https://civitai.red/', 'https://civitai.com/', 1)


def peek_header(url: str, file_id: int = 0) -> dict:
    """Read a remote safetensors JSON header via ranged requests. Returns the
    __metadata__ block plus a full model_probe analysis (architecture,
    dtypes, quant scheme) without downloading the file. Failures come back
    as {"metadata": None, "error": reason} and are not cached."""
    from modules import shared
    url = normalize_url(url)
    if not url.startswith('https://civitai.com/'):
        raise ValueError('only civitai urls are allowed')
    if file_id:
        cached = peek_cache_get(file_id, url)
        if cached is not None:
            return cached
    base_headers = {}
    token = getattr(shared.opts, 'civitai_token', '') or ''
    if token:
        base_headers['Authorization'] = f'Bearer {token}'

    def read_range(start: int, end: int) -> bytes:
        r = shared.req(url, headers={**base_headers, 'Range': f'bytes={start}-{end}'}, stream=True)
        status = getattr(r, 'status_code', 500)
        if status not in (200, 206):
            raise RuntimeError(f'HTTP {status}')
        want = end - start + 1
        # A 200 reply means the server ignored the range and sends from byte 0.
        need = end + 1 if status == 200 else want
        buf = b''
        try:
            for chunk in r.iter_content(chunk_size=65536):
                buf += chunk
                if len(buf) >= need:
                    break
        finally:
            r.close()
        return buf[start:start + want] if status == 200 else buf[:want]

    try:
        prefix = read_range(0, 7)
        if len(prefix) < 8:
            return {"metadata": None, "error": "short read"}
        header_len = struct.unpack('<Q', prefix)[0]
        if header_len <= 0 or header_len > 16 * 1024 * 1024:
            return {"metadata": None, "error": f"implausible header length: {header_len}"}
        header = json.loads(read_range(8, 7 + header_len).decode('utf-8'))
    except Exception as e:
        return {"metadata": None, "error": str(e)}
    from modules import model_probe
    response = {
        "metadata": header.get('__metadata__'),
        "tensors": len([k for k in header if k != '__metadata__']),
        "probe": model_probe.analyze_header(header),
    }
    if file_id:
        peek_cache_put(file_id, url, response)
    return response


def peek_headers(files: list) -> dict[int, dict]:
    """Probe several CivitFile entries concurrently; failed probes log and yield an error dict."""
    from concurrent.futures import ThreadPoolExecutor
    if not files:
        return {}

    def probe(f):
        try:
            return peek_header(f.download_url, f.id)
        except Exception as e:
            return {"metadata": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:
        results = dict(zip([f.id for f in files], pool.map(probe, files)))
    for f in files:
        error = results[f.id].get('error')
        if error:
            log.warning(f'CivitAI peek: file={f.id} name="{f.name}" error="{error}"')
    return results
