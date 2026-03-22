import os
import installer
from modules.logger import log
from modules.json_helpers import readfile, writefile


CONFIG = os.path.join('data', 'rocm.json')
DATA = {}


def load():
    global DATA # pylint: disable=global-statement
    DATA = readfile(CONFIG, silent=True) or {}
    log.debug(f'ROCm load: config={CONFIG} items={len(DATA)}')


def reset():
    log.info(f"ROCm reset: config={CONFIG}")
    DATA.clear()
    writefile(DATA, CONFIG)


def apply(db_path):
    DATA["MIOPEN_SYSTEM_DB_PATH"] = db_path
    log.info(f'ROCm apply: config={CONFIG} items={len(DATA)}')
    writefile(DATA, CONFIG)


def info():
    for gpu in installer.gpu_info:
        gpu['db'] = DATA.get('MIOPEN_SYSTEM_DB_PATH', '')
    return installer.gpu_info
