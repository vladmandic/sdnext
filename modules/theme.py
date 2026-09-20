from functools import lru_cache
import json
import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypedDict

import gradio as gr

from modules.logger import log
from modules.json_helpers import readfile, writefile
from modules.shared_helpers import req


if TYPE_CHECKING:
    class FontParams(TypedDict):
        font: list[str]
        font_mono: list[str]


gradio_theme = gr.themes.Base()
dct: dict = readfile('config.json', as_type='dict')
opts = SimpleNamespace(**dct)
if 'theme_type' not in opts.__dict__:
    opts.theme_type = 'Modern'
if 'gradio_theme' not in opts.__dict__:
    opts.gradio_theme = 'Default'
if 'theme_style' not in opts.__dict__:
    opts.theme_style = 'Auto'
if 'ui_locale' not in opts.__dict__:
    opts.ui_locale = 'Auto'


def list_builtin_themes():
    from modules.paths import script_path
    folder = os.path.join(script_path, "ui", "css")
    exclude = ['base.css', 'sdnext.css', 'style.css', 'timesheet.css', 'swagger.css']
    files = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith('.css') and f not in exclude]
    return files


def refresh_themes(no_update=False):
    themes_file = os.path.join('data', 'themes.json')
    res = []
    if os.path.exists(themes_file):
        try:
            with open(themes_file, encoding='utf8') as f:
                res = json.load(f)
        except Exception:
            log.error('Exception loading UI themes')
    if not no_update:
        try:
            log.info('Refreshing UI themes')
            r = req('https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json')
            if r.status_code == 200:
                res = r.json()
                writefile(res, themes_file)
            else:
                log.error('Error refreshing UI themes')
        except Exception:
            log.error('Exception refreshing UI themes')
    return res


@lru_cache(maxsize=1024)
def list_locales():
    return ['Auto', 'en: English', 'hr: Croatian', 'de: German', 'es: Spanish', 'fr: French', 'it: Italian', 'pt: Portuguese', 'zh: Chinese', 'ja: Japanese', 'ko: Korean', 'ru: Russian']


@lru_cache(maxsize=1024)
def list_themes():
    if opts.theme_type.lower() == 'none':
        gradio = ["gradio/default", "gradio/base", "gradio/glass", "gradio/monochrome", "gradio/soft"]
        huggingface = refresh_themes(no_update=True)
        huggingface = {x['id'] for x in huggingface if x['status'] == 'RUNNING' and 'test' not in x['id'].lower()}
        huggingface = [f'huggingface/{x}' for x in huggingface]
        themes = sorted(gradio) + sorted(huggingface, key=str.casefold)
    elif opts.theme_type.lower() == 'standard':
        builtin = list_builtin_themes()
        themes = sorted(builtin)
    elif opts.theme_type.lower() == 'modern':
        # ext = next((e for e in modules.extensions.extensions if e.name == 'sdnext-modernui'), None)
        # folder = os.path.join(ext.path, 'themes')
        folder = os.path.join('extensions-builtin', 'sdnext-modernui', 'themes')
        themes = []
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith('.css'):
                    themes.append(os.path.splitext(f)[0])
        if len(themes) == 0:
            themes.append('modern/Default')
        themes = sorted(themes)
    else:
        log.error(f'UI themes: type={opts.theme_type} unknown')
        themes = []
    return themes


def reload_gradio_theme(_opts, _cmd_opts):
    global opts, gradio_theme # pylint: disable=global-statement
    if _opts is not None:
        opts = _opts
    if _cmd_opts is not None:
        if _cmd_opts.theme is not None:
            if '/' in _cmd_opts.theme:
                opts.theme_type, opts.gradio_theme = _cmd_opts.theme.split('/', 1)
            else:
                opts.theme_type = _cmd_opts.theme
    theme_name = opts.gradio_theme
    default_font_params: FontParams = {
        'font':['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        'font_mono':['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace']
    }
    gradio_theme = gr.themes.Base(**default_font_params)
    available_themes = list_themes()
    if theme_name not in available_themes:
        # log.error(f'UI theme invalid: type={opts.theme_type} theme="{theme_name}"')
        if opts.theme_type.lower() == 'standard':
            theme_name = 'black-teal'
        elif opts.theme_type.lower() == 'none':
            theme_name = 'gradio/default'
        else:
            opts.theme_type = 'modern'
            theme_name = 'Default'

    opts.gradio_theme = theme_name
    log.info(f'UI locale: name="{opts.ui_locale}"')

    if opts.theme_type.lower() == 'standard':
        gradio_theme = gr.themes.Base(**default_font_params)
        log.debug(f'UI theme: type={opts.theme_type} available={len(available_themes)}')
        log.warning('UI theme: please switch to ModernUI for best experience')
        return 'sdnext.css'

    if opts.theme_type.lower() == 'modern':
        gradio_theme = gr.themes.Base(**default_font_params)
        log.debug(f'UI theme: type={opts.theme_type} available={len(available_themes)}')
        return 'base.css'

    if opts.theme_type.lower() == 'none':
        if theme_name.startswith('gradio/'):
            log.warning('UI theme: using Gradio default theme which is not optimized for SD.Next')
            if theme_name == "gradio/default":
                gradio_theme = gr.themes.Default(**default_font_params)
            elif theme_name == "gradio/base":
                gradio_theme = gr.themes.Base(**default_font_params)
            elif theme_name == "gradio/glass":
                gradio_theme = gr.themes.Glass(**default_font_params)
            elif theme_name == "gradio/monochrome":
                gradio_theme = gr.themes.Monochrome(**default_font_params)
            elif theme_name == "gradio/soft":
                gradio_theme = gr.themes.Soft(**default_font_params)
            else:
                log.warning('UI theme: unknown Gradio theme')
                theme_name = "gradio/default"
                gradio_theme = gr.themes.Default(**default_font_params)
        elif theme_name.startswith('huggingface/'):
            log.warning('UI theme: using 3rd party theme which is not optimized for SD.Next')
            try:
                hf_theme_name = theme_name.replace('huggingface/', '')
                gradio_theme = gr.themes.ThemeClass.from_hub(hf_theme_name)
            except Exception as e:
                log.error(f"UI theme: download error accessing HuggingFace {e}")
                gradio_theme = gr.themes.Default(**default_font_params)
        log.debug(f'UI theme: type={opts.theme_type} style={opts.theme_style}')
        log.warning('UI theme: please switch to ModernUI for best experience')
        return 'base.css'

    log.error(f'UI theme: type={opts.theme_type} unknown')
    return None
