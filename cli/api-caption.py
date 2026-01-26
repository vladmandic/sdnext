#!/usr/bin/env python
"""
use clip to caption image(s)
"""

import io
import base64
import sys
import os
import asyncio
import filetype
from types import SimpleNamespace
from PIL import Image
from util import log, Map
import sdapi


stats = { 'captions': {}, 'keywords': {} }
exclude = ['a', 'in', 'on', 'out', 'at', 'the', 'and', 'with', 'next', 'to', 'it', 'for', 'of', 'into', 'that']


def decode(encoding):
    if encoding.startswith("data:image/"):
        encoding = encoding.split(";")[1].split(",")[1]
    return Image.open(io.BytesIO(base64.b64decode(encoding)))


def encode(f):
    image = Image.open(f)
    exif = image.getexif()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG', exif = exif)
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def print_summary():
    captions = dict(sorted(stats['captions'].items(), key=lambda x:x[1], reverse=True))
    log.info({ 'caption stats': captions })
    keywords = dict(sorted(stats['keywords'].items(), key=lambda x:x[1], reverse=True))
    log.info({ 'keyword stats': keywords })


async def caption(f):
    if not filetype.is_image(f):
        log.info({ 'caption skip': f })
        return
    json = Map({ 'image': encode(f) })
    log.info({ 'caption': f })
    # run clip
    json.model = 'clip'
    res = await sdapi.post('/sdapi/v1/caption', json)
    caption = ""
    style = ""
    if 'caption' in res:
        caption = res.caption
        log.info({ 'caption result': caption })
        if ', by' in caption:
            style = caption.split(', by')[1].strip()
            log.info({ 'caption style': style })
        for word in caption.split(' '):
            if word not in exclude:
                stats['captions'][word] = stats['captions'][word] + 1 if word in stats['captions'] else 1
    else:
        log.error({ 'caption clip error': res })
    # run tagger (DeepBooru)
    tagger_req = SimpleNamespace(image=json.image, model='deepbooru', show_scores=True)
    res = await sdapi.post('/sdapi/v1/tagger', tagger_req)
    keywords = {}
    if 'scores' in res and res.scores:
        keywords = dict(sorted(res.scores.items(), key=lambda x: x[1], reverse=True))
        for word in keywords:
            stats['keywords'][word] = stats['keywords'][word] + 1 if word in stats['keywords'] else 1
        log.info({'caption keywords': keywords})
    elif 'tags' in res:
        for tag in res.tags.split(', '):
            stats['keywords'][tag] = stats['keywords'][tag] + 1 if tag in stats['keywords'] else 1
        log.info({'caption tags': res.tags})
    else:
        log.error({'caption tagger error': res})
    return caption, keywords, style


async def main():
    sys.argv.pop(0)
    await sdapi.session()
    if len(sys.argv) == 0:
        log.error({ 'caption': 'no files specified' })
    for arg in sys.argv:
        if os.path.exists(arg):
            if os.path.isfile(arg):
                await caption(arg)
            elif os.path.isdir(arg):
                for root, _dirs, files in os.walk(arg):
                    for f in files:
                        _caption, _keywords, _style = await caption(os.path.join(root, f))
            else:
                log.error({ 'caption unknown file type': arg })
        else:
            log.error({ 'caption file missing': arg })
    await sdapi.close()
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
