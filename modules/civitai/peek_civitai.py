"""Remote safetensors and GGUF header probes for CivitAI files, cached by file id."""

import hashlib
import json
import struct
from modules.logger import log


peek_cache = None
GGUF_READ_LIMIT = 1024 * 1024  # KV block plus tensor infos of a diffusion GGUF fit well inside 1 MB


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
    url = url.replace('https://civitai.red/', 'https://civitai.com/', 1)
    if not url.startswith('https://civitai.com/'):
        raise ValueError('only civitai urls are allowed')
    return url


def read_range(url: str, start: int, end: int) -> bytes:
    """Bytes start..end of a remote file; a 200 reply means the server ignored the range and sends from byte 0."""
    from modules import shared
    headers = {'Range': f'bytes={start}-{end}'}
    token = getattr(shared.opts, 'civitai_token', '') or ''
    if token:
        headers['Authorization'] = f'Bearer {token}'
    r = shared.req(url, headers=headers, stream=True)
    status = getattr(r, 'status_code', 500)
    if status not in (200, 206):
        raise RuntimeError(f'HTTP {status}')
    want = end - start + 1
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


def peek_header(url: str, file_id: int = 0) -> dict:
    """Read a remote safetensors JSON header via ranged requests. Returns the
    __metadata__ block plus a full model_probe analysis (architecture,
    dtypes, quant scheme) without downloading the file. Failures come back
    as {"metadata": None, "error": reason} and are not cached."""
    url = normalize_url(url)
    if file_id:
        cached = peek_cache_get(file_id, url)
        if cached is not None:
            return cached
    try:
        prefix = read_range(url, 0, 7)
        if len(prefix) < 8:
            return {"metadata": None, "error": "short read"}
        header_len = struct.unpack('<Q', prefix)[0]
        if header_len <= 0 or header_len > 16 * 1024 * 1024:
            return {"metadata": None, "error": f"implausible header length: {header_len}"}
        header = json.loads(read_range(url, 8, 7 + header_len).decode('utf-8'))
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


class GGUFReader:
    """Minimal GGUF v2/v3 header parser over an in-memory prefix of the file."""
    SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, fmt: str):
        size = struct.calcsize('<' + fmt)
        if self.pos + size > len(self.buf):
            raise EOFError('header exceeds the read window')
        value = struct.unpack_from('<' + fmt, self.buf, self.pos)[0]
        self.pos += size
        return value

    def string(self) -> str:
        n = self.take('Q')
        if self.pos + n > len(self.buf):
            raise EOFError('header exceeds the read window')
        value = self.buf[self.pos:self.pos + n].decode('utf-8', 'replace')
        self.pos += n
        return value

    def value(self, kind: int):
        if kind == 8:
            return self.string()
        if kind == 9:
            element = self.take('I')
            count = self.take('Q')
            return [self.value(element) for _ in range(count)]
        return self.take(self.SCALARS[kind])

    def parse(self) -> tuple[dict, dict]:
        """(kv, tensors) where tensors maps name to dtype and shape; tensor
        infos past the read window are dropped, the dominant type survives."""
        if self.buf[:4] != b'GGUF':
            raise ValueError('not a gguf file')
        self.pos = 4
        version = self.take('I')
        if version not in (2, 3):
            raise ValueError(f'unsupported gguf version {version}')
        n_tensors = self.take('Q')
        n_kv = self.take('Q')
        kv = {}
        for _ in range(n_kv):
            key = self.string()
            kv[key] = self.value(self.take('I'))
        tensors = {}
        try:
            for _ in range(n_tensors):
                name = self.string()
                dims = [self.take('Q') for _ in range(self.take('I'))]
                ggml_type = self.take('I')
                self.take('Q')
                tensors[name] = {'dtype': ggml_type, 'shape': list(reversed(dims))}
        except EOFError:
            pass
        return kv, tensors


def gguf_quant_token(kv: dict, probe: dict) -> str | None:
    """general.file_type names the quant exactly (Q4_K_M); files without it fall back to the dominant tensor type (Q4_K)."""
    import gguf
    file_type = kv.get('general.file_type')
    if isinstance(file_type, int):
        try:
            return gguf.LlamaFileType(file_type).name.removeprefix('MOSTLY_').removeprefix('ALL_')
        except ValueError:
            pass
    return (probe.get('quant') or {}).get('format')


def peek_gguf(url: str, file_id: int = 0) -> dict:
    """GGUF counterpart of peek_header: the general.* metadata, a model_probe
    analysis over the tensor types and the quant token in "quant"."""
    url = normalize_url(url)
    if file_id:
        cached = peek_cache_get(file_id, url)
        if cached is not None:
            return cached
    try:
        import gguf
        kv, tensors = GGUFReader(read_range(url, 0, GGUF_READ_LIMIT - 1)).parse()
        header = {name: {'dtype': gguf.GGMLQuantizationType(t['dtype']).name, 'shape': t['shape']} for name, t in tensors.items()}
    except Exception as e:
        return {"metadata": None, "error": str(e)}
    from modules import model_probe
    metadata = {k: v for k, v in kv.items() if k.startswith('general.') and not isinstance(v, list)}
    probe = model_probe.analyze_header(header, container='gguf', arch_metadata={k: v for k, v in metadata.items() if isinstance(v, str)})
    response = {
        "metadata": metadata,
        "tensors": len(header),
        "probe": probe,
        "quant": gguf_quant_token(kv, probe),
    }
    if file_id:
        peek_cache_put(file_id, url, response)
    return response


def peek_file(name: str, url: str, file_id: int = 0) -> dict:
    return peek_gguf(url, file_id) if name.lower().endswith('.gguf') else peek_header(url, file_id)


def peek_headers(files: list) -> dict[int, dict]:
    """Probe several CivitFile entries concurrently; failed probes log and yield an error dict."""
    from concurrent.futures import ThreadPoolExecutor
    if not files:
        return {}

    def probe(f):
        try:
            return peek_file(f.name, f.download_url, f.id)
        except Exception as e:
            return {"metadata": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=min(4, len(files))) as pool:
        results = dict(zip([f.id for f in files], pool.map(probe, files)))
    for f in files:
        error = results[f.id].get('error')
        if error:
            log.warning(f'CivitAI peek: file={f.id} name="{f.name}" error="{error}"')
    return results
