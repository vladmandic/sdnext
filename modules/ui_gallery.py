import os
from urllib.parse import unquote
import gradio as gr
from PIL import Image
from modules import shared, ui_symbols, ui_common, images, video, modelstats
from modules.ui_components import ToolButton


def read_media(fn):
    fn = unquote(fn).replace('%3A', ':')
    if not os.path.isfile(fn):
        shared.log.error(f'Gallery not found: file="{fn}"')
        return [[], None, '', '', f'Media not found: {fn}']
    stat_size, stat_mtime = modelstats.stat(fn)
    if fn.lower().endswith('.mp4'):
        frames, fps, duration, w, h, codec, _frame = video.get_video_params(fn)
        geninfo = ''
        log = f'''
            <p>Video <b>{w} x {h}</b>
            | Codec <b>{codec}</b>
            | Frames <b>{frames:,}</b>
            | FPS <b>{fps:.2f}</b>
            | Duration <b>{duration:.2f}</b>
            | Size <b>{stat_size:,}</b>
            | Modified <b>{stat_mtime}</b></p><br>
            '''
        return [gr.update(visible=False, value=[]), gr.update(visible=True, value=fn), geninfo, geninfo, log]
    else:
        image = Image.open(fn)
        image.already_saved_as = fn
        geninfo, _items = images.read_info_from_image(image)
        log = f'''
            <p>Image <b>{image.width} x {image.height}</b>
            | Format <b>{image.format}</b>
            | Mode <b>{image.mode}</b>
            | Size <b>{stat_size:,}</b>
            | Modified <b>{stat_mtime}</b></p><br>
            '''
        return [gr.update(visible=True, value=[image]), gr.update(visible=False), geninfo, geninfo, log]


def create_ui():
    with gr.Blocks() as tab:
        with gr.Row(elem_id='tab-gallery-sort-buttons'):
            sort_buttons = []
            sort_buttons.append(ToolButton(value=ui_symbols.sort_alpha_asc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_alpha_dsc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_size_asc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_size_dsc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_num_asc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_num_dsc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_time_asc, elem_classes=['gallery-sort']))
            sort_buttons.append(ToolButton(value=ui_symbols.sort_time_dsc, elem_classes=['gallery-sort']))
            gr.Textbox(show_label=False, placeholder='Search', elem_id='tab-gallery-search')
            gr.HTML('', elem_id='tab-gallery-status')
            for btn in sort_buttons:
                btn.click(fn=None, _js='gallerySort', inputs=[btn], outputs=[])
        with gr.Row():
            with gr.Column():
                gr.HTML('', elem_id='tab-gallery-folders')
            with gr.Column():
                gr.HTML('', elem_id='tab-gallery-files')
            with gr.Column():
                btn_gallery_image = gr.Button('', elem_id='tab-gallery-send-image', visible=False, interactive=True)
                gallery_video = gr.Video(None, elem_id='tab-gallery-video', show_label=False, visible=False)
                gallery_images, gen_info, html_info, _html_info_formatted, html_log = ui_common.create_output_panel("gallery")
                btn_gallery_image.click(fn=read_media, _js='gallerySendImage', inputs=[html_info], outputs=[gallery_images, gallery_video, html_info, gen_info, html_log])
    return [(tab, 'Gallery', 'tab-gallery')]
