# MIT-Han-Lab Nunchaku: <https://github.com/mit-han-lab/nunchaku>

from installer import pip
from modules.logger import log


ok = False


def check(force=False):
    global ok # pylint: disable=global-statement
    if force:
        return False
    if ok:
        return True
    try:
        import nunchaku
        import nunchaku.utils # pylint: disable=no-name-in-module, no-member
        from nunchaku import __version__  # pylint: disable=no-name-in-module
        log.info(f'Nunchaku: path={nunchaku.__path__} version={__version__.__version__} precision={nunchaku.utils.get_precision()}')
        ok = True
        return True
    except Exception as e:
        log.error(f'Nunchaku: {e}')
        ok = False
        return False


def install_nunchaku(force=False):
    if not force:
        from modules import devices, shared
        if devices.backend is None:
            return False # too early
        if shared.cmd_opts.reinstall:
            force = True
    if not check(force):
        import os
        import sys
        import platform
        python_ver = f'{sys.version_info.major}{sys.version_info.minor}'
        if python_ver not in ['311', '312', '313']:
            log.error(f'Nunchaku: python={sys.version_info} unsupported')
            return False
        arch = platform.system().lower()
        if arch not in ['linux', 'windows']:
            log.error(f'Nunchaku: platform={arch} unsupported')
            return False
        if not force and devices.backend not in ['cuda']:
            log.error(f'Nunchaku: backend={devices.backend} unsupported')
            return False

        url = os.environ.get('NUNCHAKU_COMMAND', None)
        if url is not None:
            cmd = f'install --upgrade {url}'
            log.debug(f'Nunchaku: install="{url}"')
            result, _output = pip(cmd, uv=False, ignore=not force, quiet=not force)
            return result.returncode == 0
        else:
            import torch
            torch_ver = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
            cuda_ver = torch.__version__.split('+')[1]if '+' in torch.__version__ else None
            cuda_ver = cuda_ver[:4] + '.' + cuda_ver[-1] if cuda_ver and len(cuda_ver) >= 4 else None
            suffix = 'linux_x86_64' if arch == 'linux' else 'win_amd64'
            if cuda_ver is None:
                log.error(f'Nunchaku: torch={torch.__version__} cuda="unknown"')
                return False
            if cuda_ver.startswith('cu13'):
                nunchaku_versions = ['1.2.1', '1.2.0', '1.1.0', '1.0.2', '1.0.1']
            else:
                nunchaku_versions = ['1.2.1', '1.0.2', '1.0.1'] # 1.2.0 and 1.1.0 imply cu13 but do not specify it
            for v in nunchaku_versions:
                url = f'https://github.com/nunchaku-ai/nunchaku/releases/download/v{v}/'
                fn = f'nunchaku-{v}+{cuda_ver}torch{torch_ver}-cp{python_ver}-cp{python_ver}-{suffix}.whl'
                result, _output = pip(f'install --upgrade {url+fn}', uv=False, ignore=True, quiet=True)
                if (result is None) or (_output == 'offline'):
                    log.error(f'Nunchaku: install url="{url+fn}" offline mode')
                    return False
                if force:
                    log.debug(f'Nunchaku: url="{fn}" code={result.returncode} stdout={result.stdout} stderr={result.stderr} output={_output}')
                if result.returncode == 0:
                    log.info(f'Nunchaku: install url="{url}"')
                    return True
                fn = f'nunchaku-{v}+torch{torch_ver}-cp{python_ver}-cp{python_ver}-{suffix}.whl'
                result, _output = pip(f'install --upgrade {url+fn}', uv=False, ignore=True, quiet=True)
                if (result is None) or (_output == 'offline'):
                    log.error(f'Nunchaku: install url="{url+fn}" offline mode')
                    return False
                if force:
                    log.debug(f'Nunchaku: url="{fn}" code={result.returncode} stdout={result.stdout} stderr={result.stderr} output={_output}')
                if result.returncode == 0:
                    log.info(f'Nunchaku: install version={v} url="{url+fn}"')
                    return True
        log.error(f'Nunchaku install failed: torch={torch.__version__} cuda={cuda_ver} python={python_ver} platform={arch}')
        return False

    if not check():
        log.error('Nunchaku: install failed')
        return False
    return True


if __name__ == '__main__':
    from modules.logger import setup_logging
    setup_logging()
    log.info('Nunchaku: manual install')
    install_nunchaku(force=True)
