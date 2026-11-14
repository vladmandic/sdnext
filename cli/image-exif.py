#!/bin/env python
import os
import sys
import json
import importlib.util
from PIL import Image
from rich import print # pylint: disable=redefined-builtin


module_file = os.path.abspath(__file__)
module_dir = os.path.dirname(module_file)
infotext_spec = importlib.util.spec_from_file_location('infotext', os.path.join(module_dir, '..', 'modules', 'infotext.py'))
infotext = importlib.util.module_from_spec(infotext_spec)
infotext_spec.loader.exec_module(infotext)
metadata_spec = importlib.util.spec_from_file_location('metadata', os.path.join(module_dir, '..', 'modules', 'image', 'metadata.py'))
metadata = importlib.util.module_from_spec(metadata_spec)
metadata_spec.loader.exec_module(metadata)
extensions_spec = importlib.util.spec_from_file_location('extensions', os.path.join(module_dir, '..', 'modules', 'image', 'extensions.py'))
extensions = importlib.util.module_from_spec(extensions_spec)
extensions_spec.loader.exec_module(extensions)


def print_json(data):
    try:
        for k, v in data.items():
            print(f'json: k={k}', json.loads(v))
    except Exception:
        pass


def read_exif(filename: str):
    try:
        image = Image.open(filename)
        image.load()
        info, data = metadata.read_info_from_image(filename)
        print('image:', filename, 'format:', image.format)
        print('exif:', data)
        print('info:', infotext.parse(info))
        print_json(data)
    except Exception as e:
        print('metadata error reading:', filename, e)


if __name__ == '__main__':
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        print('metadata:', 'no files specified')
        sys.exit(0)
    extensions.register_extensions()
    for fn in sys.argv:
        if os.path.isfile(fn):
            read_exif(fn)
        elif os.path.isdir(fn):
            for root, _dirs, files in os.walk(fn):
                for file in files:
                    read_exif(os.path.join(root, file))
        else:
            print('file not found: ', fn)
