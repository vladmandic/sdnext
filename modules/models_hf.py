import os
import time
import gradio as gr
from installer import install
from modules.logger import log
from modules.shared import opts


# initialize huggingface environment
def hf_init():
    os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('HF_HUB_VERBOSITY', 'warning')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '10')
    os.environ.setdefault('HF_ENABLE_PARALLEL_LOADING', 'true' if opts.sd_parallel_load else 'false')
    os.environ.setdefault('HF_HUB_CACHE', opts.hfcache_dir)
    os.environ.setdefault('HF_XET_CACHE', opts.xetcache_dir)
    if opts.hf_transfer_mode == 'requests':
        os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', 'false')
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', 'false')
        os.environ.setdefault('HF_HUB_DISABLE_XET', 'true')
    elif opts.hf_transfer_mode == 'rust':
        install('hf_transfer')
        os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', 'false')
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', 'true')
        os.environ.setdefault('HF_HUB_DISABLE_XET', 'true')
    elif opts.hf_transfer_mode == 'xet':
        install('hf_xet')
        import huggingface_hub
        huggingface_hub.utils._runtime.is_xet_available = lambda: True  # pylint: disable=protected-access
        os.environ.setdefault('HF_XET_HIGH_PERFORMANCE', 'true')
        os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', 'true')
        os.environ.setdefault('HF_HUB_DISABLE_XET', 'false')

    obfuscated_token = None
    if len(opts.huggingface_token) > 0 and opts.huggingface_token.startswith('hf_'):
        obfuscated_token = 'hf_...' + opts.huggingface_token[-4:]
    log.info(f'Huggingface: transfer={opts.hf_transfer_mode} parallel={opts.sd_parallel_load} direct={opts.diffusers_to_gpu} token="{obfuscated_token}" cache="{opts.hfcache_dir}" init')


def hf_check_cache():
    from modules.modelstats import stat
    prev_default = os.environ.get("SD_HFCACHEDIR", None) or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
    if opts.hfcache_dir != prev_default:
        size, _mtime = stat(prev_default)
        if size//1024//1024 > 32:
            log.warning(f'Huggingface cache changed: type=huggingface unused="{prev_default}" size={size//1024//1024} MB')
    prev_default = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'xet')
    if opts.xetcache_dir != prev_default:
        size, _mtime = stat(prev_default)
        if size//1024//1024 > 32:
            log.warning(f'Huggingface cache changed: type=xet unused="{prev_default}" size={size//1024//1024} MB')

    hf_size, _mtime = stat(opts.hfcache_dir)
    xet_size, _mtime = stat(opts.xetcache_dir)
    log.debug(f'Huggingface: cache="{opts.hfcache_dir}" size={hf_size//1024//1024} MB xet="{opts.xetcache_dir}" size={xet_size//1024//1024} MB')


def hf_search(keyword):
    import huggingface_hub as hf
    t0 = time.time()
    hf_api = hf.HfApi()
    models = hf_api.list_models(model_name=keyword, full=True, filter="diffusers", limit=50, sort="downloads")
    data = []
    for model in models:
        tags = [t for t in model.tags if not t.startswith('diffusers') and not t.startswith('license') and not t.startswith('arxiv') and len(t) > 2]
        data.append([model.id, model.pipeline_tag, tags, model.downloads, model.lastModified, f'https://huggingface.co/{model.id}'])
    log.debug(f'Huggingface: search="{keyword}" results={len(data)} time={time.time()-t0:.2f}')
    return data


def hf_select(evt: gr.SelectData, df):
    row = list(df.iloc[evt.index[0]])
    log.debug(f'Huggingface: selected={row} index={evt.index}')
    return row[0] # repo_id only


def hf_download_model(hub_id: str, token, variant, revision, mirror, custom_pipeline):
    from modules.modelloader import download_diffusers_model
    download_diffusers_model(hub_id, cache_dir=opts.diffusers_dir, token=token, variant=variant, revision=revision, mirror=mirror, custom_pipeline=custom_pipeline)
    from modules.sd_models import list_models  # pylint: disable=W0621
    list_models()
    log.info(f'Huggingface: model="{hub_id}" downloaded')
    return f'Diffuser model downloaded: model="{hub_id}"'


def hf_update_token(token):
    log.debug('Huggingface: update token')
    opts.huggingface_token = token
    opts.save()
