import os
import sys
from typing import Dict, Mapping


class Interpreter:
    env_globals: Dict
    env_locals: Mapping

    def __init__(self, env_globals, env_locals):
        self.env_globals = env_globals
        self.env_locals = env_locals

    def execute(self, s: str):
        try:
            exec(s, self.env_globals, self.env_locals) # pylint: disable=exec-used
        except Exception as e:
            print(f'{e.__class__.__name__}: {e}')

    def from_file(self, path):
        with open(path, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                self.execute(line)


if __name__ == '__main__':
    sys.path.append(os.getcwd())

    from modules import zluda_installer
    zluda_installer.install()
    zluda_installer.load()

    import torch
    interpreter = Interpreter({
        'torch': torch,
    }, {})

    if len(sys.argv) > 1:
        interpreter.from_file(sys.argv[1])
    else:
        print(f'Python with ZLUDA {sys.version}')
        print('Type "help", "copyright", "credits" or "license" for more information.')

        while True:
            print('>>> ', end='')
            interpreter.execute(input())
