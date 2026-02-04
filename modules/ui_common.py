import json
import html
import os
import shutil
import platform
import subprocess
import gradio as gr
from modules import paths, call_queue, shared, errors, ui_sections, ui_symbols, ui_components, generation_parameters_copypaste, images, scripts_manager, script_callbacks, infotext, processing


folder_symbol = ui_symbols.folder
debug = shared.log.trace if os.environ.get('SD_PASTE_DEBUG', None) is not None else lambda *args, **kwargs: None
debug('Trace: PASTE')


def gr_show(visible=True):
    return {"visible": visible, "__type__": "update"}


def update_generation_info(generation_info, html_info, img_index):
    try:
        if len(generation_info) == 0 and processing.processed is not None:
            generation_info = processing.processed.js() or {}
        if len(generation_info) == 0:
            return html_info, html_info
        generation_json = json.loads(generation_info)
        if len(generation_json.get("infotexts", [])) == 0:
            return html_info, 'no infotexts found'
        if img_index == -1:
            img_index = 0
        if img_index >= len(generation_json["infotexts"]):
            return html_info, 'error fetching infotext'
        info = generation_json["infotexts"][img_index]
        html_info_formatted = infotext_to_html(info)
        return html_info, html_info_formatted
    except Exception as e:
        shared.log.trace(f'Update info: info="{generation_info}" {e}')
    return html_info, html_info


def plaintext_to_html(text, elem_classes=[]):
    res = f'<p class="plaintext {" ".join(elem_classes)}">' + '<br>\n'.join([f"{html.escape(x)}" for x in text.split('\n')]) + '</p>'
    return res


def infotext_to_html(text):
    res = infotext.parse(text)
    prompt = res.get('Prompt', '')
    negative = res.get('Negative prompt', '')
    res.pop('Prompt', None)
    res.pop('Negative prompt', None)
    params = [f'{k}: {v}' for k, v in res.items() if v is not None and not k.endswith('-1') and not k.endswith('-2')]
    params = '| '.join(params) if len(params) > 0 else ''
    code = ''
    if len(prompt) > 0:
        code += f'<p><b>Prompt:</b> {html.escape(prompt)}</p>'
    if len(negative) > 0:
        code += f'<p><b>Negative:</b> {html.escape(negative)}</p>'
    if len(params) > 0:
        code += f'<p><b>Parameters:</b> {html.escape(params)}</p>'
    return code


def delete_files(js_data, files, all_files, index):
    try:
        data = json.loads(js_data)
    except Exception:
        data = { 'index_of_first_image': 0 }
    start_index = 0
    first_index = data['index_of_first_image']
    if (index > -1) and shared.opts.save_selected_only and (index >= first_index):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only # pylint: disable=no-member
        if index < len(files):
            files = [files[index]]
            start_index = index
        else:
            shared.log.error(f'Delete: index={index} first={first_index} files={len(files)} out of range')
            files = []
    deleted = []
    all_files = [f.split('/file=')[1] if 'file=' in f else f for f in all_files] if isinstance(all_files, list) else []
    all_files = [os.path.normpath(f) for f in all_files]
    reference_dir = os.path.join('models', 'Reference')
    for _image_index, filedata in enumerate(files, start_index):
        try:
            fn = os.path.normpath(filedata['name'])
            if reference_dir in fn:
                shared.log.warning(f'Delete: file="{fn}" not allowed')
                continue
            if os.path.exists(fn) and os.path.isfile(fn):
                deleted.append(fn)
                os.remove(fn)
                if fn in all_files:
                    all_files.remove(fn)
                    shared.log.info(f'Delete: image="{fn}"')
                else:
                    shared.log.warning(f'Delete: image="{fn}" ui mismatch')
            base, _ext = os.path.splitext(fn)
            desc = f'{base}.txt'
            if os.path.exists(desc) and os.path.isfile(desc):
                os.remove(desc)
                shared.log.info(f'Delete: text="{fn}"')
        except Exception as e:
            shared.log.error(f'Delete: file="{fn}" {e}')
    deleted = ', '.join(deleted) if len(deleted) > 0 else 'none'
    return all_files, plaintext_to_html(f"Deleted: {deleted}", ['performance'])


def save_files(js_data, files, html_info, index):
    os.makedirs(paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save), exist_ok=True)

    class PObject: # pylint: disable=too-few-public-methods
        def __init__(self, d=None):
            if d is not None:
                for k, v in d.items():
                    setattr(self, k, v)
            self.prompt = getattr(self, 'prompt', None) or getattr(self, 'Prompt', None) or ''
            self.negative_prompt = getattr(self, 'negative_prompt', None) or getattr(self, 'Negative_prompt', None) or ''
            self.sampler = getattr(self, 'sampler', None) or getattr(self, 'Sampler', None) or ''
            self.sampler_name = self.sampler
            self.seed = getattr(self, 'seed', None) or getattr(self, 'Seed', None) or 0
            self.steps = getattr(self, 'steps', None) or getattr(self, 'Steps', None) or 0
            self.width = getattr(self, 'width', None) or getattr(self, 'Width', None) or getattr(self, 'Size-1', None) or 0
            self.height = getattr(self, 'height', None) or getattr(self, 'Height', None) or getattr(self, 'Size-2', None) or 0
            self.cfg_scale = getattr(self, 'cfg_scale', None) or getattr(self, 'CFG scale', None) or 0
            self.clip_skip = getattr(self, 'clip_skip', None) or getattr(self, 'Clip skip', None) or 1
            self.denoising_strength = getattr(self, 'denoising_strength', None) or getattr(self, 'Denoising', None) or 0
            self.index_of_first_image = getattr(self, 'index_of_first_image', 0)
            self.subseed = getattr(self, 'subseed', None) or getattr(self, 'Subseed', None)
            self.styles = getattr(self, 'styles', None) or getattr(self, 'Styles', None) or []
            self.styles = [s.strip() for s in self.styles.split(',')] if isinstance(self.styles, str) else self.styles

            self.outpath_grids = paths.resolve_output_path(shared.opts.outdir_grids, shared.opts.outdir_txt2img_grids)
            self.infotexts = getattr(self, 'infotexts', [html_info])
            self.infotext = self.infotexts[0] if len(self.infotexts) > 0 else html_info
            self.all_negative_prompt = getattr(self, 'all_negative_prompts', [self.negative_prompt])
            self.all_prompts = getattr(self, 'all_prompts', [self.prompt])
            self.all_seeds = getattr(self, 'all_seeds', [self.seed])
            self.all_subseeds = getattr(self, 'all_subseeds', [self.subseed])

            self.n_iter = 1
            self.batch_size = 1
    try:
        data = json.loads(js_data)
    except Exception:
        data = {}
    p = PObject(data)
    start_index = 0
    if (index > -1) and shared.opts.save_selected_only and (index >= p.index_of_first_image):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only # pylint: disable=no-member
        if index < len(files):
            files = [files[index]]
            start_index = index
        else:
            shared.log.error(f'Save: index={index} first={p.index_of_first_image} files={len(files)} out of range')
            files = []
    filenames = []
    fullfns = []
    for image_index, filedata in enumerate(files, start_index):
        is_grid = image_index < p.index_of_first_image # pylint: disable=no-member
        i = 0 if is_grid else (image_index - p.index_of_first_image) # pylint: disable=no-member
        while len(p.all_seeds) <= i:
            p.all_seeds.append(p.seed)
        while len(p.all_prompts) <= i:
            p.all_prompts.append(p.prompt)
        while len(p.infotexts) <= i:
            p.infotexts.append(p.infotext)
        if 'name' in filedata and (paths.temp_dir not in filedata['name']) and os.path.isfile(filedata['name']):
            fullfn = filedata['name']
            fullfns.append(fullfn)
            destination = paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save)
            namegen = images.FilenameGenerator(p, seed=p.all_seeds[i], prompt=p.all_prompts[i], image=None)  # pylint: disable=no-member
            dirname = namegen.apply(shared.opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
            destination = os.path.join(destination, dirname)
            destination = namegen.sanitize(destination)
            os.makedirs(destination, exist_ok = True)
            tgt_filename = os.path.join(destination, os.path.basename(fullfn))
            relfn = os.path.relpath(tgt_filename, paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save))
            filenames.append(relfn)
            if not os.path.exists(tgt_filename):
                try:
                    shutil.copy(fullfn, destination)
                    shared.log.info(f'Copying image: file="{fullfn}" folder="{destination}"')
                except Exception as e:
                    shared.log.error(f'Copying image: {fullfn} {e}')
            if shared.opts.save_txt:
                try:
                    from PIL import Image
                    image = Image.open(fullfn)
                    info, _ = images.read_info_from_image(image)
                    filename_txt = f"{os.path.splitext(tgt_filename)[0]}.txt"
                    with open(filename_txt, "w", encoding="utf8") as file:
                        file.write(f"{info}\n")
                    shared.log.debug(f'Save: text="{filename_txt}"')
                except Exception as e:
                    shared.log.warning(f'Image description save failed: {filename_txt} {e}')
            script_callbacks.image_save_btn_callback(tgt_filename)
        else:
            image = generation_parameters_copypaste.image_from_url_text(filedata)
            info = p.infotexts[i + 1] if len(p.infotexts) > len(p.all_seeds) else p.infotexts[i] # infotexts may be offset by 1 because the first image is the grid
            if len(info) == 0:
                info = None
            if (js_data is None or len(js_data) == 0) and image is not None and image.info is not None:
                info, _items = images.read_info_from_image(image)
                items = infotext.parse(info)
                p = PObject(items)
            try:
                seed = p.all_seeds[i] if i < len(p.all_seeds) else p.seed
                prompt = p.all_prompts[i] if i < len(p.all_prompts) else p.prompt
                fullfn, txt_fullfn, _exif = images.save_image(image, paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save), "", seed=seed, prompt=prompt, info=info, extension=shared.opts.samples_format, grid=is_grid, p=p)
            except Exception as e:
                fullfn, txt_fullfn = None, None
                shared.log.error(f'Save: image={image} i={i} seeds={p.all_seeds} prompts={p.all_prompts}')
                errors.display(e, 'save')
            if fullfn is None:
                continue
            filename = os.path.relpath(fullfn, paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save))
            filenames.append(filename)
            fullfns.append(fullfn)
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
                # fullfns.append(txt_fullfn)
            script_callbacks.image_save_btn_callback(filename)
    if shared.opts.samples_save_zip and len(fullfns) > 1:
        zip_filepath = os.path.join(paths.resolve_output_path(shared.opts.outdir_samples, shared.opts.outdir_save), "images.zip")
        from zipfile import ZipFile
        with ZipFile(zip_filepath, "w") as zip_file:
            for i in range(len(fullfns)):
                if os.path.isfile(fullfns[i]):
                    with open(fullfns[i], mode="rb") as f:
                        zip_file.writestr(filenames[i], f.read())
        fullfns.insert(0, zip_filepath)
    return gr.File.update(value=fullfns, visible=True), plaintext_to_html(f"Saved: {filenames[0] if len(filenames) > 0 else 'none'}", ['performance'])


def open_folder(result_gallery, gallery_index = 0):
    try:
        folder = os.path.dirname(result_gallery[gallery_index]['name'])
    except Exception:
        folder = shared.opts.outdir_samples
    if not os.path.exists(folder):
        shared.log.warning(f'Folder open: folder="{folder}" does not exist')
        return
    elif not os.path.isdir(folder):
        shared.log.warning(f'Folder open: folder="{folder}" not a folder')
        return

    if not shared.cmd_opts.hide_ui_dir_config:
        path = os.path.normpath(folder)
        if platform.system() == "Windows":
            os.startfile(path) # pylint: disable=no-member
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path]) # pylint: disable=consider-using-with
        elif "microsoft-standard-WSL2" in platform.uname().release:
            subprocess.Popen(["wsl-open", path]) # pylint: disable=consider-using-with
        else:
            subprocess.Popen(["xdg-open", path]) # pylint: disable=consider-using-with


def create_output_panel(tabname, preview=True, prompt=None, height=None, transfer=True, scale=1, result_info=None):
    with gr.Column(variant='panel', elem_id=f"{tabname}_results", scale=scale):
        with gr.Group(elem_id=f"{tabname}_gallery_container"):
            if tabname == "txt2img":
                gr.HTML(value="", elem_id="main_info", visible=False, elem_classes=["main-info"])
            result_gallery = gr.Gallery(value=[],
                                        label='Output',
                                        show_label=False,
                                        show_download_button=True,
                                        allow_preview=True,
                                        container=False,
                                        preview=preview,
                                        columns=shared.opts.ui_columns,
                                        object_fit='scale-down',
                                        height=height,
                                        elem_id=f"{tabname}_gallery",
                                        elem_classes=["gallery_main"],
                                       )
            if prompt is not None:
                ui_sections.create_caption_button(tab=tabname, inputs=result_gallery, outputs=prompt, what='output')
            button_image_fit = gr.Button(ui_symbols.resize, elem_id=f"{tabname}_image_fit", elem_classes=['image-fit'])
            button_image_fit.click(fn=None, _js="cycleImageFit", inputs=[], outputs=[])

        with gr.Column(elem_id=f"{tabname}_footer", elem_classes="gallery_footer"):
            dummy_component = gr.Label(visible=False)
            with gr.Row(elem_id=f"image_buttons_{tabname}", elem_classes="image-buttons"):
                if not shared.cmd_opts.listen:
                    open_folder_button = gr.Button('Show', visible=not shared.cmd_opts.hide_ui_dir_config, elem_id=f'open_folder_{tabname}')
                    open_folder_button.click(open_folder, _js="(gallery, dummy) => [gallery, selected_gallery_index()]", inputs=[result_gallery, dummy_component], outputs=[])
                else:
                    clip_files = gr.Button('Copy', elem_id=f'open_folder_{tabname}')
                    clip_files.click(fn=None, _js='clip_gallery_urls', inputs=[result_gallery], outputs=[])
                save = gr.Button('Save', elem_id=f'save_{tabname}')
                delete = gr.Button('Delete', elem_id=f'delete_{tabname}')
                if transfer:
                    buttons = generation_parameters_copypaste.create_buttons(["control", "txt2img", "img2img", "extras", "caption"])
                else:
                    buttons = None

            download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False, elem_id=f'download_files_{tabname}')
            with gr.Group():
                html_info = gr.HTML(elem_id=f'html_info_{tabname}', elem_classes="infotext", visible=False) # contains raw infotext as returned by wrapped call
                html_info_formatted = gr.HTML(elem_id=f'html_info_formatted_{tabname}', elem_classes="infotext", visible=True) # contains html formatted infotext
                html_info.change(fn=infotext_to_html, inputs=[html_info], outputs=[html_info_formatted], show_progress='hidden')
                html_log = gr.HTML(elem_id=f'html_log_{tabname}')
                generation_info = gr.Textbox(visible=False, elem_id=f'generation_info_{tabname}')
                generation_info_button = gr.Button(visible=False, elem_id=f"{tabname}_generation_info_button")

                result_field = result_info or html_info_formatted
                generation_info_button.click(fn=update_generation_info, show_progress='hidden',
                    _js="(x, y, z) => [x, y, selected_gallery_index()]", # triggered on gallery change from js
                    inputs=[generation_info, html_info, html_info],
                    outputs=[html_info, result_field],
                )
                save.click(fn=call_queue.wrap_gradio_call(save_files), show_progress='hidden',
                    _js="(x, y, z, i) => [x, y, z, selected_gallery_index()]",
                    inputs=[generation_info, result_gallery, html_info, html_info],
                    outputs=[download_files, html_log],
                )
                delete.click(fn=call_queue.wrap_gradio_call(delete_files), show_progress='hidden',
                    _js=f'(x, y, i, j) => [x, y, ...selected_gallery_files("{tabname}")]',
                    inputs=[generation_info, result_gallery, html_info, html_info],
                    outputs=[result_gallery, html_log],
                )

            if tabname == "txt2img":
                paste_field_names = scripts_manager.scripts_txt2img.paste_field_names
            elif tabname == "img2img":
                paste_field_names = scripts_manager.scripts_img2img.paste_field_names
            elif tabname == "control":
                paste_field_names = scripts_manager.scripts_control.paste_field_names
            else:
                paste_field_names = []
            debug(f'Paste field: tab={tabname} fields={paste_field_names}')
            if buttons is not None:
                for paste_tabname, paste_button in buttons.items():
                    debug(f'Create output panel: source={tabname} target={paste_tabname} button={paste_button}')
                    bindings = generation_parameters_copypaste.ParamBinding(
                        paste_button=paste_button,
                        tabname=paste_tabname,
                        source_tabname=tabname,
                        source_image_component=result_gallery,
                        paste_field_names=paste_field_names,
                        source_text_component=prompt or generation_info
                    )
                    generation_parameters_copypaste.register_paste_params_button(bindings)
            return result_gallery, generation_info, html_info, html_info_formatted, html_log


def create_refresh_button(refresh_component, refresh_method, refreshed_args = None, elem_id = None, visible: bool = True):
    def refresh():
        refresh_method()
        if refreshed_args is None:
            args = {"choices": refresh_method()} # pylint: disable=unnecessary-lambda-assignment
        elif callable(refreshed_args):
            args = refreshed_args()
        else:
            args = refreshed_args
        for k, v in args.items():
            setattr(refresh_component, k, v)
        return gr.update(**args)

    refresh_button = ui_components.ToolButton(value=ui_symbols.refresh, elem_id=elem_id, visible=visible)
    refresh_button.click(fn=refresh, inputs=[], outputs=[refresh_component], show_progress='hidden')
    return refresh_button


def create_override_inputs(tab): # pylint: disable=unused-argument
    with gr.Row(elem_id=f"{tab}_override_settings_row"):
        visible = tab == 'control'
        override_settings = gr.Dropdown([], value=None, label="Override settings", visible=visible, elem_id=f"{tab}_override_settings", multiselect=True)
        override_settings.change(fn=lambda x: gr.Dropdown.update(visible=len(x) > 0), inputs=[override_settings], outputs=[override_settings])
    return override_settings


def reuse_seed(seed_component: gr.Number, reuse_button: gr.Button, subseed:bool=False):
    def reuse_click(selected_gallery_index):
        selected_gallery_index = int(selected_gallery_index)
        if processing.processed is None:
            seed = -1
        elif len(processing.processed.images) > len(processing.processed.all_seeds): # if we have more images than seeds it is likely the grid image
            selected_gallery_index -= (len(processing.processed.images) - len(processing.processed.all_seeds))
            seed = processing.processed.all_seeds[selected_gallery_index] if not subseed else processing.processed.all_subseeds[selected_gallery_index]
        elif selected_gallery_index <= len(processing.processed.all_seeds):
            seed = processing.processed.all_seeds[selected_gallery_index] if not subseed else processing.processed.all_subseeds[selected_gallery_index]
        elif len(processing.processed.all_seeds) > 0:
            seed = processing.processed.all_seeds[0] if not subseed else processing.processed.all_subseeds[0]
        else:
            seed = -1
        shared.log.debug(f'Reuse seed: index={selected_gallery_index} seed={seed} subseed={subseed}')
        return seed

    reuse_button.click(fn=reuse_click, _js="selected_gallery_index", inputs=[seed_component], outputs=[seed_component], show_progress='hidden')


def connect_reuse_seed(seed: gr.Number, reuse_seed_btn: gr.Button, generation_info: gr.Textbox, is_subseed, subseed_strength=None):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index: int):
        restore_seed = -1
        restore_strength = -1
        try:
            gen_info = json.loads(gen_info_string)
            shared.log.debug(f'Reuse: info={gen_info}')
            index -= gen_info.get('index_of_first_image', 0)
            index = int(index)
            if is_subseed:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                restore_seed = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
                restore_strength = gen_info.get('subseed_strength', 0)
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                restore_seed = all_seeds[index if 0 <= index < len(all_seeds) else 0]
        except json.decoder.JSONDecodeError:
            if gen_info_string != '':
                shared.log.error(f"Error parsing JSON generation info: {gen_info_string}")
        if is_subseed is not None:
            return [restore_seed, gr_show(False), restore_strength]
        else:
            return [restore_seed, gr_show(False)]
    dummy_component = gr.Number(visible=False, value=0)
    if subseed_strength is None:
        reuse_seed_btn.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress='hidden', inputs=[generation_info, dummy_component], outputs=[seed, dummy_component])
    else:
        reuse_seed_btn.click(fn=copy_seed, _js="(x, y) => [x, selected_gallery_index()]", show_progress='hidden', inputs=[generation_info, dummy_component], outputs=[seed, dummy_component, subseed_strength])


def update_token_counter(text):
    token_count = 0
    max_length = 75
    if shared.state.job_count > 0:
        shared.log.debug('Tokenizer busy')
        return f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>"
    from modules import extra_networks
    if isinstance(text, list):
        prompt, _ = extra_networks.parse_prompts(text)
    else:
        prompt, _ = extra_networks.parse_prompt(text)
    if shared.sd_loaded and hasattr(shared.sd_model, 'tokenizer') and shared.sd_model.tokenizer is not None:
        tokenizer = shared.sd_model.tokenizer
        # For multi-modal processors (e.g., PixtralProcessor), use the underlying text tokenizer
        if hasattr(tokenizer, 'tokenizer') and tokenizer.tokenizer is not None:
            tokenizer = tokenizer.tokenizer
        has_bos_token = hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None
        has_eos_token = hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None
        try:
            ids = tokenizer(prompt)
            ids = getattr(ids, 'input_ids', [])
        except Exception:
            ids = []
        token_count = len(ids) - int(has_bos_token) - int(has_eos_token)
        model_max_length = getattr(tokenizer, 'model_max_length', 0)
        max_length = model_max_length - int(has_bos_token) - int(has_eos_token)
        if max_length is None or max_length < 0 or max_length > 10000:
            max_length = 0
    return gr.update(value=f"<span class='gr-box gr-text-input'>{token_count}/{max_length}</span>", visible=token_count > 0)
