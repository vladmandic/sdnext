import os
import json
import gradio as gr
import modules.shared
import modules.extensions
from modules.logger import log
from modules.json_helpers import writefile


gradio_theme = gr.themes.Base()


def list_builtin_themes():
    files = [os.path.splitext(f)[0] for f in os.listdir('javascript') if f.endswith('.css') and f not in ['base.css', 'sdnext.css', 'style.css']]
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
            r = modules.shared.req('https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json')
            if r.status_code == 200:
                res = r.json()
                writefile(res, themes_file)
            else:
                log.error('Error refreshing UI themes')
        except Exception:
            log.error('Exception refreshing UI themes')
    return res


def list_locales():
    return ['Auto', 'en: English', 'hr: Croatian', 'de: German', 'es: Spanish', 'fr: French', 'it: Italian', 'pt: Portuguese', 'zh: Chinese', 'ja: Japanese', 'ko: Korean', 'ru: Russian']


def list_themes():
    extensions = [e.name for e in modules.extensions.extensions if e.enabled]
    if 'sd-webui-lobe-theme' in extensions and modules.shared.opts.gradio_theme == 'lobe':
        themes = ['lobe']
        modules.shared.opts.data['gradio_theme'] = themes[0]
        modules.shared.opts.data['theme_type'] = 'None'
        log.info('UI theme: extension="lobe"')
    elif 'Cozy-Nest' in extensions and modules.shared.opts.gradio_theme == 'cozy-nest':
        themes = ['cozy-nest']
        modules.shared.opts.data['gradio_theme'] = themes[0]
        modules.shared.opts.data['theme_type'] = 'None'
        log.info('UI theme: extension="cozy-nest"')
    elif modules.shared.opts.theme_type == 'None':
        gradio = ["gradio/default", "gradio/base", "gradio/glass", "gradio/monochrome", "gradio/soft"]
        huggingface = refresh_themes(no_update=True)
        huggingface = {x['id'] for x in huggingface if x['status'] == 'RUNNING' and 'test' not in x['id'].lower()}
        huggingface = [f'huggingface/{x}' for x in huggingface]
        themes = sorted(gradio) + sorted(huggingface, key=str.casefold)
    elif modules.shared.opts.theme_type == 'Standard':
        builtin = list_builtin_themes()
        themes = sorted(builtin)
    elif modules.shared.opts.theme_type == 'Modern':
        ext = next((e for e in modules.extensions.extensions if e.name == 'sdnext-modernui'), None)
        if ext is None:
            log.error('UI themes: ModernUI not found')
            builtin = list_builtin_themes()
            themes = sorted(builtin)
            modules.shared.opts.theme_type = 'Standard'
            return themes
        folder = os.path.join(ext.path, 'themes')
        themes = []
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith('.css'):
                    themes.append(os.path.splitext(f)[0])
        if len(themes) == 0:
            themes.append('modern/Default')
        themes = sorted(themes)
    else:
        log.error(f'UI themes: type={modules.shared.opts.theme_type} unknown')
        themes = []
    return themes


def reload_gradio_theme():
    global gradio_theme # pylint: disable=global-statement
    theme_name = modules.shared.opts.gradio_theme
    default_font_params = {
        'font':['Helvetica', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        'font_mono':['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace']
    }
    gradio_theme = gr.themes.Base(**default_font_params)
    available_themes = list_themes()
    if theme_name not in available_themes:
        # log.error(f'UI theme invalid: type={modules.shared.opts.theme_type} theme="{theme_name}"')
        if modules.shared.opts.theme_type == 'Standard':
            theme_name = 'black-teal'
        elif modules.shared.opts.theme_type == 'Modern':
            theme_name = 'Default'
        elif modules.shared.opts.theme_type == 'None':
            theme_name = 'gradio/default'
        else:
            modules.shared.opts.theme_type = 'Standard'
            theme_name = 'black-teal'

    modules.shared.opts.data['gradio_theme'] = theme_name
    log.info(f'UI locale: name="{modules.shared.opts.ui_locale}"')

    if theme_name.lower() in ['lobe', 'cozy-nest']:
        log.info(f'UI theme extension: name="{theme_name}"')
        return None
    elif modules.shared.opts.theme_type == 'Standard':
        gradio_theme = gr.themes.Base(**default_font_params)
        log.info(f'UI theme: type={modules.shared.opts.theme_type} name="{theme_name}" available={len(available_themes)}')
        return 'sdnext.css'
    elif modules.shared.opts.theme_type == 'Modern':
        gradio_theme = gr.themes.Base(**default_font_params)
        log.info(f'UI theme: type={modules.shared.opts.theme_type} name="{theme_name}" available={len(available_themes)}')
        return 'base.css'
    elif modules.shared.opts.theme_type == 'None':
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
        log.info(f'UI theme: type={modules.shared.opts.theme_type} name="{theme_name}" style={modules.shared.opts.theme_style}')
        return 'base.css'
    log.error(f'UI theme: type={modules.shared.opts.theme_type} unknown')
    return None
