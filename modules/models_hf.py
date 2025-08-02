import os
import gradio as gr
from modules.shared import log, opts


def hf_init():
    os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
    os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
    os.environ.setdefault('HUGGINGFACE_HUB_VERBOSITY', 'warning')


def hf_search(keyword):
    hf_init()
    import huggingface_hub as hf
    hf_api = hf.HfApi()
    models = hf_api.list_models(model_name=keyword, full=True, library="diffusers", limit=50, sort="downloads", direction=-1)
    data = []
    for model in models:
        tags = [t for t in model.tags if not t.startswith('diffusers') and not t.startswith('license') and not t.startswith('arxiv') and len(t) > 2]
        data.append([model.id, model.pipeline_tag, tags, model.downloads, model.lastModified, f'https://huggingface.co/{model.id}'])
    return data


def hf_select(evt: gr.SelectData, data):
    return data[evt.index[0]][0]


def hf_download_model(hub_id: str, token, variant, revision, mirror, custom_pipeline):
    hf_init()
    from modules.modelloader import download_diffusers_model
    download_diffusers_model(hub_id, cache_dir=opts.diffusers_dir, token=token, variant=variant, revision=revision, mirror=mirror, custom_pipeline=custom_pipeline)
    from modules.sd_models import list_models  # pylint: disable=W0621
    list_models()
    log.info(f'Diffuser model downloaded: model="{hub_id}"')
    return f'Diffuser model downloaded: model="{hub_id}"'


def hf_update_token(token):
    log.debug('Huggingface update token')
    opts.huggingface_token = token
    opts.save()
