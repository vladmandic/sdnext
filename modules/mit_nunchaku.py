# MIT-Han-Lab Nunchaku: <https://github.com/mit-han-lab/nunchaku>

from installer import log, pip
from modules import devices


nunchaku_versions = {
    '2.5': '1.0.1',
    '2.6': '1.0.1',
    '2.7': '1.1.0',
    '2.8': '1.1.0',
    '2.9': '1.1.0',
    '2.10': '1.0.2',
    '2.11': '1.1.0',
}
ok = False


def _expected_ver():
    try:
        import torch
        torch_ver = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
        return nunchaku_versions.get(torch_ver)
    except Exception:
        return None


def check():
    global ok # pylint: disable=global-statement
    if ok:
        return True
    try:
        import nunchaku
        import nunchaku.utils
        from nunchaku import __version__
        expected = _expected_ver()
        log.info(f'Nunchaku: path={nunchaku.__path__} version={__version__.__version__} precision={nunchaku.utils.get_precision()}')
        if expected is not None and __version__.__version__ != expected:
            ok = False
            return False
        ok = True
        return True
    except Exception as e:
        log.error(f'Nunchaku: {e}')
        ok = False
        return False


def install_nunchaku():
    if devices.backend is None:
        return False # too early
    if not check():
        import os
        import sys
        import platform
        import importlib
        import pkg_resources
        import torch
        python_ver = f'{sys.version_info.major}{sys.version_info.minor}'
        if python_ver not in ['311', '312', '313']:
            log.error(f'Nunchaku: python={sys.version_info} unsupported')
            return False
        arch = platform.system().lower()
        if arch not in ['linux', 'windows']:
            log.error(f'Nunchaku: platform={arch} unsupported')
            return False
        if devices.backend not in ['cuda']:
            log.error(f'Nunchaku: backend={devices.backend} unsupported')
            return False
        torch_ver = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
        nunchaku_ver = nunchaku_versions.get(torch_ver)
        if nunchaku_ver is None:
            log.error(f'Nunchaku: torch={torch.__version__} unsupported')
            return False
        suffix = 'x86_64' if arch == 'linux' else 'win_amd64'
        url = os.environ.get('NUNCHAKU_COMMAND', None)
        if url is None:
            arch = f'{arch}_' if arch == 'linux' else ''
            url = f'https://huggingface.co/nunchaku-ai/nunchaku/resolve/main/nunchaku-{nunchaku_ver}'
            url += f'+torch{torch_ver}-cp{python_ver}-cp{python_ver}-{arch}{suffix}.whl'
        cmd = f'install --upgrade {url}'
        log.debug(f'Nunchaku: install="{url}"')
        pip(cmd, ignore=False, uv=False)
        importlib.reload(pkg_resources)
    if not check():
        log.error('Nunchaku: install failed')
        return False
    return True
