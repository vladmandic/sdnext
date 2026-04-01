#!/usr/bin/env python

# Remove the entries that no longer exist in locale from override.

import sys
import json
from rich import print # pylint: disable=redefined-builtin

if __name__ == "__main__":
    sys.argv.pop(0)
    if len(sys.argv) == 0:
        print('Invalid parameters.')
        sys.exit(1)
    filename = sys.argv[0]
    labels = []
    override = None
    try:
        with open('html/locale_en.json', 'r', encoding="utf-8") as f:
            locale = json.load(f)
        for v in locale.values():
            for item in v:
                labels.append(item['label'])
        with open(filename, 'r', encoding="utf-8") as f:
            override = json.load(f)
    except Exception:
        print('Invalid file format.')
        sys.exit(1)
    with open(filename, 'w', encoding="utf-8") as f:
        json.dump([item for item in override if item['label'] in labels], f, ensure_ascii=False)
