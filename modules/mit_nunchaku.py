# MIT-Han-Lab Nunchaku: <https://github.com/mit-han-lab/nunchaku>

from installer import log, pip
from modules import devices


ver = '0.3.2'
ok = False


def check():
    global ok # pylint: disable=global-statement
    if ok:
        return True
    try:
        import nunchaku
        import nunchaku.utils
        log.info(f'Nunchaku: path={nunchaku.__path__} precision={nunchaku.utils.get_precision()}')
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
        torch_ver = torch.__version__[:3]
        if torch_ver not in ['2.5', '2.6', '2.7', '2.8']:
            log.error(f'Nunchaku: torch={torch.__version__} unsupported')
        suffix = 'x86_64' if arch == 'linux' else 'win_amd64'
        url = os.environ.get('NUNCHAKU_COMMAND', None)
        if url is None:
            arch = f'{arch}_' if arch == 'linux' else ''
            url = f'https://huggingface.co/nunchaku-tech/nunchaku/resolve/main/nunchaku-{ver}'
            url += f'+torch{torch_ver}-cp{python_ver}-cp{python_ver}-{arch}{suffix}.whl'
        cmd = f'install --upgrade {url}'
        # pip install https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl
        log.debug(f'Nunchaku: install="{url}"')
        pip(cmd, ignore=False, uv=False)
        importlib.reload(pkg_resources)
    if not check():
        log.error('Nunchaku: install failed')
        return False
    return True
