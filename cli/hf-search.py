#!/usr/bin/env python

import sys
import huggingface_hub as hf
from rich import print # pylint: disable=redefined-builtin

if __name__ == "__main__":
    sys.argv.pop(0)
    keyword = sys.argv[0] if len(sys.argv) > 0 else ''
    hf.logging.set_verbosity_info()
    hf_api = hf.HfApi()
    res = hf_api.list_models(
        model_name=keyword,
        full=True,
        limit=100,
        sort="downloads",
        direction=-1,
    )
    res = sorted(res, key=lambda x: x.id)
    exact = [m for m in res if keyword.lower() in m.id.lower()]
    if len(exact) > 0:
        res = exact
    for m in res:
        meta = hf_api.model_info(m.id, files_metadata=True)
        m.files = [f.rfilename for f in meta.siblings if f.rfilename.endswith('.bin') or f.rfilename.endswith('.safetensors')]
        m.size = round(sum([f.size for f in meta.siblings]) / 1024 / 1024 / 1024, 2)
        print({ 'name': m.id, 'files': len(m.files), 'size': m.size, 'downloads': m.downloads, 'ctime': m.created_at.isoformat(), 'url': f'https://huggingface.co/{m.id}', 'pipeline': m.pipeline_tag })
