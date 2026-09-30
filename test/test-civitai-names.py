#!/usr/bin/env python
"""
Offline tests for modules.civitai.names_civitai and peek_civitai.

Covers:

- save names for recorded CivitAI versions with the precision suffix on, matching the reference names in the fixture
- the collision cascade with the suffix off: role first, then variant, size class and the file id
- variant tokens from metadata.fp, metadata.quantType and the quantType query parameter of the download url, skipped when the name spells the token already
- expert roles from the version title or the safetensors header, never for companions or names that carry a role word
- header probes upgrading a generic fp8 claim to the exact dtype, and leaving it alone when the probe failed
- companion files routing to their own type folder
- a failed remote header read returning an error and never entering the probe cache
- a successful header read entering the probe cache
- a GGUF header read naming the quant from general.file_type, or from the dominant tensor type without it
- resolve_file reporting the version fetch reason and an unknown file id

No running server required.

Usage:
    python test/test-civitai-names.py
"""

import json
import os
import struct
import sys
import tempfile
import types

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)
os.chdir(script_dir)
sys.argv = [sys.argv[0]]

from modules.logger import log  # pylint: disable=wrong-import-position
from modules.civitai.models_civitai import CivitFile, CivitVersion  # pylint: disable=wrong-import-position
from modules.civitai import names_civitai as names  # pylint: disable=wrong-import-position

FIXTURE = os.path.join(script_dir, 'test', 'fixtures', 'civitai-names.json')
passed = 0
failed = 0


def check(label: str, got, want):
    global passed, failed  # pylint: disable=global-statement
    if got == want:
        passed += 1
    else:
        failed += 1
        log.error(f'{label}: got={got!r} want={want!r}')


def context_for(version: CivitVersion, peeks: dict) -> names.NameContext:
    context = names.NameContext(name=version.name, base_model=version.base_model, model_name=version.model.name if version.model else '')
    targets = names.peek_targets(version.files)
    return names.apply_peeks(context, targets, {f.id: peeks.get(str(f.id)) or {} for f in targets})


def make_file(file_id: int, name: str, file_type: str = 'Model', fp: str | None = None, size: str | None = None, quant: str | None = None, url: str = '') -> CivitFile:
    return CivitFile.parse_obj({'id': file_id, 'name': name, 'type': file_type, 'metadata': {'fp': fp, 'size': size, 'quantType': quant}, 'downloadUrl': url})


def test_fixture_names():
    fixture = json.load(open(FIXTURE, encoding='utf-8'))
    for raw in fixture['versions']:
        version = CivitVersion.parse_obj(raw)
        context = context_for(version, fixture['peeks'])
        for f in version.files:
            check(f'{version.id}/{f.id} on', names.save_name(f, version.files, context, precision=True), fixture['expected']['precision_on'][str(version.id)][str(f.id)])
            check(f'{version.id}/{f.id} off', names.save_name(f, version.files, context, precision=False), fixture['expected']['precision_off'][str(version.id)][str(f.id)])


def test_suffix_and_variant():
    check('suffix ext', names.insert_name_suffix('a.b.safetensors', 'fp8'), 'a.b-fp8.safetensors')
    check('suffix no ext', names.insert_name_suffix('model', 'fp8'), 'model-fp8')
    check('suffix dotfile', names.insert_name_suffix('.hidden', 'x'), '.hidden-x')
    check('variant fp', names.file_variant(make_file(1, 'a.safetensors', fp='bf16', quant='Q8_0')), 'bf16')
    check('variant quant', names.file_variant(make_file(1, 'a.gguf', quant='Q8_0')), 'Q8_0')
    check('variant url', names.file_variant(make_file(1, 'a.gguf', url='https://civitai.com/api/download/models/1?format=GGUF&quantType=Q4_K_M')), 'Q4_K_M')
    check('variant none', names.file_variant(make_file(1, 'a.gguf', url='https://civitai.com/api/download/models/1?fileId=1')), None)
    check('variant gguf ignores fp', names.file_variant(make_file(1, 'a.gguf', fp='fp16')), None)
    check('variant gguf format ignores fp', names.file_variant(CivitFile.parse_obj({'id': 1, 'name': 'a.bin', 'metadata': {'format': 'GGUF', 'fp': 'fp16', 'quantType': 'Q5_0'}})), 'Q5_0')
    check('variant safetensors quant fallback', names.file_variant(make_file(1, 'a.safetensors', quant='Q8_0')), 'Q8_0')
    check('carries fp8', names.name_carries('model_fp8.safetensors', 'fp8'), True)
    check('carries exact only', names.name_carries('model_fp8.safetensors', 'fp8_e4m3fn'), False)
    check('carries quant', names.name_carries('model-Q8_0.gguf', 'Q8_0'), True)
    check('carries not q50', names.name_carries('wan_highQ50.gguf', 'Q5_0'), False)
    check('carries camel', names.name_carries('ideogram4Nvfp4_nvfp4_nf4.safetensors', 'nf4'), True)
    f = make_file(1, 'model_fp8.safetensors', fp='fp8')
    check('name skips carried variant', names.save_name(f, [f]), 'model_fp8.safetensors')
    context = names.apply_peeks(names.NameContext(), [f], {1: {'metadata': None, 'probe': {'ok': True, 'dominant_dtype': 'F8_E4M3'}}})
    check('name keeps upgraded variant', names.save_name(f, [f], context), 'model_fp8-fp8_e4m3fn.safetensors')
    check('variant bad url', names.file_variant(make_file(1, 'a.gguf', url='')), None)


def test_roles():
    wan = names.NameContext(name='HIGH Q8', base_model='Wan Video 2.2 T2V-A14B')
    check('role title', names.file_role(make_file(1, 'wan_v10_1.safetensors'), wan), 'high-noise')
    check('role in file name', names.file_role(make_file(1, 'wan_highQ8.gguf'), wan), None)
    check('role camel case', names.file_role(make_file(1, 'wanLowNoise.gguf'), wan), None)
    check('role no arch', names.file_role(make_file(1, 'x.safetensors'), names.NameContext(name='high', base_model='SDXL 1.0')), None)
    ideo = names.NameContext(name='uncond', base_model='Other', model_name='Ideogram 4 mirror')
    check('role name hint', names.file_role(make_file(1, 'x.safetensors'), ideo), 'uncond')
    check('role metadata', names.role_from_metadata({'model_type': 'ideogram4_uncond'}, ideo), 'uncond')
    check('role metadata cond', names.role_from_metadata({'model_type': 'ideogram4_cond', 'n': 3}, ideo), None)
    check('role metadata none', names.role_from_metadata(None, ideo), None)
    header = names.NameContext(name='v1.0', base_model='Ideogram 4.0', roles={1: 'uncond', 2: 'uncond', 3: 'uncond'})
    check('role header', names.file_role(make_file(1, 'x.safetensors'), header), 'uncond')
    check('role header gated by name', names.file_role(make_file(2, 'x_uncond.safetensors'), header), None)
    check('role header not on companion', names.file_role(make_file(3, 'x_txt.safetensors', 'Text Encoder'), header), None)
    check('role title not on companion', names.file_role(make_file(4, 'x.vae.safetensors', 'VAE'), wan), None)


def test_peek_upgrade():
    f = make_file(7, 'm.safetensors', fp='fp8')
    context = names.apply_peeks(names.NameContext(), [f], {7: {'metadata': None, 'probe': {'ok': True, 'dominant_dtype': 'F8_E5M2'}}})
    check('upgrade fp8', context.variants.get(7), 'fp8_e5m2')
    context = names.apply_peeks(names.NameContext(), [f], {7: {'metadata': None, 'probe': {'ok': True, 'dominant_dtype': 'BF16'}}})
    check('upgrade unsatisfied', context.variants.get(7), None)
    context = names.apply_peeks(names.NameContext(), [f], {7: {'metadata': None, 'error': 'HTTP 401'}})
    check('upgrade failed peek', context.variants.get(7), None)
    check('failed peek name', names.save_name(f, [f], context), 'm-fp8.safetensors')
    context = names.apply_peeks(names.NameContext(), [f], {7: {'metadata': None, 'probe': {'ok': False, 'dominant_dtype': None}}})
    check('upgrade bad probe', context.variants.get(7), None)
    g = make_file(8, 'm.gguf', fp='fp8', quant='Q8_0')
    context = names.apply_peeks(names.NameContext(), [g], {8: {'metadata': {}, 'probe': {'ok': True}, 'quant': 'Q4_K_M'}})
    check('gguf header quant wins', names.save_name(g, [g], context), 'm-Q4_K_M.gguf')
    context = names.apply_peeks(names.NameContext(), [g], {8: {'metadata': None, 'error': 'HTTP 401'}})
    check('gguf failed peek uses metadata quant', names.save_name(g, [g], context), 'm-Q8_0.gguf')
    check('peek targets', [x.id for x in names.peek_targets([f, g, make_file(9, 'c.zip')])], [7, 8])


def test_route_type():
    check('route model', names.route_type(make_file(1, 'a', 'Model'), 'LORA'), 'LORA')
    check('route vae', names.route_type(make_file(1, 'a', 'VAE'), 'Checkpoint'), 'VAE')
    check('route te', names.route_type(make_file(1, 'a', 'Text Encoder'), 'Checkpoint'), 'Text Encoder')


class FakeResponse:
    def __init__(self, status: int, payload: bytes, headers: dict):
        self.status_code = status
        start, end = headers['Range'].removeprefix('bytes=').split('-')
        self.chunk = payload[int(start):int(end) + 1]

    def iter_content(self, chunk_size=0): # pylint: disable=unused-argument
        yield self.chunk

    def close(self):
        pass


def install_stubs(tmpdir: str, payload: bytes | None):
    """Stand-in shared and paths modules: req serves payload by range, or fails when payload is None."""
    def req(url, headers=None, **_kwargs): # pylint: disable=unused-argument
        if payload is None:
            raise ConnectionError('simulated network failure')
        return FakeResponse(206, payload, headers)
    sys.modules['modules.shared'] = types.SimpleNamespace(opts=types.SimpleNamespace(civitai_token=''), req=req)
    sys.modules['modules.paths'] = types.SimpleNamespace(civitai_probe_file=os.path.join(tmpdir, 'civitai.json'))


def test_peek_cache():
    from modules.civitai import peek_civitai
    header = json.dumps({'__metadata__': {'model_type': 'ideogram4_uncond'}, 'w': {'dtype': 'F8_E4M3', 'shape': [2, 2], 'data_offsets': [0, 4]}}).encode('utf-8')
    payload = struct.pack('<Q', len(header)) + header + b'\0' * 4
    url = 'https://civitai.red/api/download/models/1?fileId=9'
    with tempfile.TemporaryDirectory() as tmpdir:
        install_stubs(tmpdir, None)
        peek_civitai.peek_cache = {}
        result = peek_civitai.peek_header(url, 9)
        check('peek failure error', 'simulated network failure' in (result.get('error') or ''), True)
        check('peek failure metadata', result.get('metadata'), None)
        check('peek failure not cached', peek_civitai.peek_cache, {})
        check('peek failure no cache file', os.path.exists(os.path.join(tmpdir, 'civitai.json')), False)
        install_stubs(tmpdir, payload)
        result = peek_civitai.peek_header(url, 9)
        check('peek metadata', result.get('metadata'), {'model_type': 'ideogram4_uncond'})
        check('peek dtype', result['probe'].get('dominant_dtype'), 'F8_E4M3')
        check('peek cached', peek_civitai.peek_cache_get(9, 'https://civitai.com/api/download/models/1?fileId=9') is result, True)
        check('peek cache file', os.path.exists(os.path.join(tmpdir, 'civitai.json')), True)
        install_stubs(tmpdir, None)
        check('peek cache hit survives failure', peek_civitai.peek_header(url, 9), result)
        try:
            peek_civitai.peek_header('https://example.com/x', 1)
            check('peek host guard', 'no error', ValueError)
        except ValueError:
            passed_guard = True
            check('peek host guard', passed_guard, True)
        results = peek_civitai.peek_headers([make_file(9, 'a.safetensors', url=url), make_file(10, 'b.safetensors', url=url + '0')])
        check('peek_headers cached', results[9], result)
        check('peek_headers failed', 'simulated network failure' in results[10]['error'], True)
    sys.modules.pop('modules.shared', None)
    sys.modules.pop('modules.paths', None)


def gguf_bytes(kv: dict, tensors: list) -> bytes:
    def string(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b
    out = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(kv))
    for key, value in kv.items():
        out += string(key)
        if isinstance(value, str):
            out += struct.pack('<I', 8) + string(value)
        else:
            out += struct.pack('<II', 4, value)
    for name, dims, ggml_type in tensors:
        out += string(name) + struct.pack('<I', len(dims)) + b''.join(struct.pack('<Q', d) for d in dims) + struct.pack('<IQ', ggml_type, 0)
    return out


def test_peek_gguf():
    from modules.civitai import peek_civitai
    url = 'https://civitai.com/api/download/models/2?fileId=12'
    with tempfile.TemporaryDirectory() as tmpdir:
        peek_civitai.peek_cache = {}
        install_stubs(tmpdir, gguf_bytes({'general.architecture': 'wan', 'general.file_type': 15}, [('blocks.0.attn.q.weight', [5120, 5120], 12), ('blocks.0.norm.weight', [5120], 0)]))
        result = peek_civitai.peek_file('m.gguf', url, 12)
        check('gguf file_type quant', result.get('quant'), 'Q4_K_M')
        check('gguf metadata', result.get('metadata'), {'general.architecture': 'wan', 'general.file_type': 15})
        check('gguf probe format', result['probe']['quant']['format'], 'Q4_K')
        check('gguf tensors', result.get('tensors'), 2)
        check('gguf cached', peek_civitai.peek_cache_get(12, url), result)
        peek_civitai.peek_cache = {}
        install_stubs(tmpdir, gguf_bytes({}, [('w', [16, 16], 8), ('b', [16], 1)]))
        check('gguf tensor quant', peek_civitai.peek_file('m.gguf', url, 12).get('quant'), 'Q8_0')
        peek_civitai.peek_cache = {}
        install_stubs(tmpdir, gguf_bytes({'general.file_type': 999}, [('w', [16, 16], 2)]))
        check('gguf unknown file_type falls back', peek_civitai.peek_file('m.gguf', url, 12).get('quant'), 'Q4_0')
        peek_civitai.peek_cache = {}
        install_stubs(tmpdir, b'GGUF' + struct.pack('<IQQ', 3, 1, 1) + b'\x05')
        result = peek_civitai.peek_file('m.gguf', url, 12)
        check('gguf truncated error', 'read window' in (result.get('error') or ''), True)
        check('gguf truncated not cached', peek_civitai.peek_cache, {})
        install_stubs(tmpdir, b'\x08' + b'\0' * 40)
        check('gguf bad magic', peek_civitai.peek_file('m.gguf', url, 12).get('error'), 'not a gguf file')
    sys.modules.pop('modules.shared', None)
    sys.modules.pop('modules.paths', None)


def test_resolve_file():
    fixture = json.load(open(FIXTURE, encoding='utf-8'))
    raw = next(v for v in fixture['versions'] if v['id'] == 3357716)

    class FakeClient:
        def fetch_version(self, version_id, token=None): # pylint: disable=unused-argument
            if version_id == 3357716:
                return CivitVersion.parse_obj(raw), '', 200
            return None, 'HTTP 404 Model not found', 404

        def get_model(self, model_id, token=None): # pylint: disable=unused-argument
            return None

    sys.modules['modules.civitai.client_civitai'] = types.SimpleNamespace(client=FakeClient())
    sys.modules['modules.shared'] = types.SimpleNamespace(opts=types.SimpleNamespace(civitai_save_precision=True))
    names.version_context = lambda version, peek=True: context_for(version, fixture['peeks'])
    try:
        resolved, error, status = names.resolve_file(1, 1)
        check('resolve missing version', (resolved, status), (None, 404))
        check('resolve missing version reason', error, 'version 1 fetch failed: HTTP 404 Model not found')
        resolved, error, status = names.resolve_file(3357716, 42)
        check('resolve unknown file', (resolved, error, status), (None, 'file 42 not in version 3357716', 404))
        resolved, error, status = names.resolve_file(3357716, 3246005)
        check('resolve ok', (error, status), ('', 200))
        check('resolve name', resolved['filename'], 'krea2Anime_v20_3246005-fp8_e4m3fn.safetensors')
        check('resolve url', resolved['url'], 'https://civitai.com/api/download/models/3357716?fileId=3246005')
        check('resolve hash', resolved['expected_hash'], (raw['files'][0]['hashes']['SHA256'] or '').lower())
        check('resolve type', resolved['model_type'], 'Checkpoint')
        check('resolve base', (resolved['base_model'], resolved['model_name'], resolved['nsfw']), ('Krea 2', 'Krea2-Anime', True))
        sys.modules['modules.shared'].opts.civitai_save_precision = False
        resolved, _error, _status = names.resolve_file(3357716, 3246005)
        check('resolve name toggle off', resolved['filename'], 'krea2Anime_v20_3246005.safetensors')
    finally:
        sys.modules.pop('modules.civitai.client_civitai', None)
        sys.modules.pop('modules.shared', None)


if __name__ == '__main__':
    for test in [test_fixture_names, test_suffix_and_variant, test_roles, test_peek_upgrade, test_route_type, test_peek_cache, test_peek_gguf, test_resolve_file]:
        test()
    log.warning(f'Total: {passed} passed, {failed} failed')
    sys.exit(1 if failed else 0)
