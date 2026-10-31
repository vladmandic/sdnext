#!/usr/bin/env python
"""
Offline tests for the Hugging Face repo access check (``sd_models.hf_auth_check``).

The check sends the token that model downloads send. sdnext sets ``HF_HUB_DISABLE_IMPLICIT_TOKEN``, so the hub
client never adds a stored token on its own; downloads get it back through ``sd_hijack_hfhub``.

Covers:

- a token stored by ``hf auth login`` reaching the access check, with no ERROR logged
- ``HF_TOKEN`` taking precedence over the stored token, for the check and for downloads alike
- no token anywhere: the request carries none and the denial is logged

The hub request is answered by a stub session; the token file is a temp file. No network, no running server.

Usage:
    python test/test-hf-auth.py
"""

import os
import sys
import time
import shutil
import logging
import tempfile
import contextlib

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

import httpx2                                                               # pylint: disable=wrong-import-position
from huggingface_hub import constants, file_download, hf_api                # pylint: disable=wrong-import-position
from modules import shared, sd_models, modelloader, sd_hijack_hfhub         # pylint: disable=wrong-import-position
from modules.errors import log                                              # pylint: disable=wrong-import-position

results: dict[str, dict] = {}
work_dir = tempfile.mkdtemp(prefix='sdnext-hf-auth-', suffix='-work')
token_file = os.path.join(work_dir, 'token')
stored_token = 'hf_storedlogin0000000000000000000000'
env_token = 'hf_environment000000000000000000000'
repo_id = 'owner/gated-model'
token_vars = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_OIDC_RESOURCE')


def category(name: str):
    if name not in results:
        results[name] = {'passed': 0, 'failed': 0, 'skipped': 0, 'tests': []}
    return name


def record(cat: str, status: str, name: str, detail: str = ''):
    results[cat][{'PASS': 'passed', 'FAIL': 'failed', 'SKIP': 'skipped'}[status]] += 1
    results[cat]['tests'].append((status, name))
    msg = f'  {status}: {name}'
    if detail:
        msg += f' ({detail})'
    if status == 'PASS':
        log.info(msg)
    elif status == 'SKIP':
        log.warning(msg)
    else:
        log.error(msg)


def run_test(cat: str, fn):
    name = fn.__name__
    try:
        ok = fn()
        if isinstance(ok, str):
            record(cat, 'SKIP', name, ok)
        else:
            record(cat, 'FAIL' if ok is False else 'PASS', name)
    except AssertionError as e:
        record(cat, 'FAIL', name, str(e))
    except Exception as e:  # pylint: disable=broad-except
        record(cat, 'FAIL', name, f'exception: {type(e).__name__}: {e}')


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):  # pylint: disable=redefined-outer-name
        self.records.append(record)

    def messages(self, level: int = logging.DEBUG) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno >= level]


@contextlib.contextmanager
def capture_log():
    handler = LogCapture()
    logger = logging.getLogger('sd')
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


class StubSession:
    """Answers the access check: 200 with an Authorization header, 401 without, as the hub does for a gated repo."""

    def __init__(self):
        self.requests: list[tuple[str, dict]] = []

    def get(self, url, headers=None, **kwargs):  # pylint: disable=unused-argument
        headers = dict(headers or {})
        self.requests.append((url, headers))
        status = 200 if any(k.lower() == 'authorization' for k in headers) else 401
        return httpx2.Response(status, request=httpx2.Request('GET', url))


def sent_authorization(session: StubSession) -> str | None:
    assert len(session.requests) == 1, f'expected one hub request, got {len(session.requests)}'
    url, headers = session.requests[0]
    assert url.endswith(f'/api/models/{repo_id}/auth-check'), f'unexpected url {url}'
    return next((v for k, v in headers.items() if k.lower() == 'authorization'), None)


def download_authorization() -> str | None:
    headers = file_download.build_hf_headers()
    return next((v for k, v in headers.items() if k.lower() == 'authorization'), None)


@contextlib.contextmanager
def hub(stored: str | None = None, env: str | None = None):
    """Token sources as given, sdnext's implicit-token setting, a stub session for the access check."""
    saved_env = {k: os.environ.pop(k, None) for k in token_vars}
    saved_path = constants.HF_TOKEN_PATH
    saved_implicit = constants.HF_HUB_DISABLE_IMPLICIT_TOKEN
    saved_session = hf_api.get_session
    saved_install = modelloader.install
    saved_opts = {k: shared.opts.data.get(k, None) for k in ('huggingface_token', 'offline_mode')}
    session = StubSession()
    try:
        if os.path.exists(token_file):
            os.remove(token_file)
        if stored is not None:
            with open(token_file, 'w', encoding='utf8') as f:
                f.write(stored)
        if env is not None:
            os.environ['HF_TOKEN'] = env
        constants.HF_TOKEN_PATH = token_file
        constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = True
        hf_api.get_session = lambda: session
        modelloader.install = lambda *args, **kwargs: None
        shared.opts.data['huggingface_token'] = ''
        shared.opts.data['offline_mode'] = False
        yield session
    finally:
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        constants.HF_TOKEN_PATH = saved_path
        constants.HF_HUB_DISABLE_IMPLICIT_TOKEN = saved_implicit
        hf_api.get_session = saved_session
        modelloader.install = saved_install
        shared.opts.data.update(saved_opts)


def test_stored_login_reaches_auth_check():
    """Red before the fix: the check sent no token and logged ``Auth: ... 401`` while downloads authenticated."""
    with hub(stored=stored_token) as session, capture_log() as cap:
        log.warning('probe: capture works')
        ok = sd_models.hf_auth_check(repo_id, force=True)
        sent = sent_authorization(session)
        download = download_authorization()
    assert 'probe: capture works' in cap.messages(logging.WARNING), 'log capture saw nothing'
    assert download == f'Bearer {stored_token}', f'download header {download!r}'
    assert sent == download, f'auth check sent {sent!r}, downloads send {download!r}'
    assert ok is True, f'auth check returned {ok!r}'
    errors = cap.messages(logging.ERROR)
    assert not errors, f'errors logged: {errors}'
    return True


def test_env_token_takes_precedence():
    with hub(stored=stored_token, env=env_token) as session, capture_log() as cap:
        ok = sd_models.hf_auth_check(repo_id, force=True)
        sent = sent_authorization(session)
        download = download_authorization()
    assert download == f'Bearer {env_token}', f'download header {download!r}'
    assert sent == download, f'auth check sent {sent!r}, downloads send {download!r}'
    assert ok is True, f'auth check returned {ok!r}'
    errors = cap.messages(logging.ERROR)
    assert not errors, f'errors logged: {errors}'
    return True


def test_no_token_is_denied_and_logged():
    with hub() as session, capture_log() as cap:
        ok = sd_models.hf_auth_check(repo_id, force=True)
        sent = sent_authorization(session)
    assert sent is None, f'auth check sent {sent!r} with no token configured'
    assert ok is False, f'auth check returned {ok!r}'
    errors = [m for m in cap.messages(logging.ERROR) if m.startswith('Auth: repo=') and repo_id in m]
    assert len(errors) == 1, f'expected one Auth error, got {cap.messages(logging.ERROR)}'
    return True


def run_all():
    sd_hijack_hfhub.init_hijack()
    log.warning('=== HF auth check ===')
    cat = category('token')
    for fn in [
        test_stored_login_reaches_auth_check,
        test_env_token_takes_precedence,
        test_no_token_is_denied_and_logged,
    ]:
        run_test(cat, fn)

    log.warning('=== Results ===')
    total = {'passed': 0, 'failed': 0, 'skipped': 0}
    for cat_name, info in results.items():
        status = 'PASS' if info['failed'] == 0 else 'FAIL'
        log.info(f"  {cat_name}: {info['passed']} passed, {info['failed']} failed, {info['skipped']} skipped [{status}]")
        for key in total:
            total[key] += info[key]
    log.warning(f"Total: {total['passed']} passed, {total['failed']} failed, {total['skipped']} skipped")
    return total['failed'] == 0


if __name__ == '__main__':
    t0 = time.time()
    try:
        ok = run_all()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    log.warning(f'Total time: {time.time() - t0:.2f}s')
    sys.exit(0 if ok else 1)
