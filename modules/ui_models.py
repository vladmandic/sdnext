import os
import inspect
from typing import cast
import gradio as gr
from modules import errors, sd_models, sd_vae, extras, sd_samplers, ui_symbols, modelstats
from modules.ui_components import ToolButton
from modules.ui_common import create_refresh_button
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import opts, log


extra_ui = []


def create_ui():
    log.debug('UI initialize: tab=models')
    dummy_component = gr.Label(visible=False)
    with gr.Row(elem_id="models_tab"):
        with gr.Column(elem_id='models_output_container', scale=1):
            models_outcome = gr.HTML(elem_id="models_outcome", value="")
            models_file = gr.File(label='', visible=False)

        with gr.Column(elem_id='models_input_container', scale=3):

            with gr.Tab(label="Current", elem_id="models_current_tab"):
                def create_modules_table(rows: list):
                    html = """
                        <table class="simple-table">
                            <thead>
                                <tr><th>Module</th><th>Class</th><th>Device</th><th>Dtype</th><th>Quant</th><th>Params</th><th>Modules</th><th>Config</th></tr>
                            </thead>
                            <tbody>
                                {tbody}
                            </tbody>
                        </table>
                    """
                    tbody = ''
                    for row in rows:
                        try:
                            config = str(row.config)
                        except Exception:
                            config = '{}'
                        try:
                            tbody += f"""
                                <tr>
                                    <td>{row.name}</td>
                                    <td>{row.cls}</td>
                                    <td>{row.device}</td>
                                    <td>{row.dtype}</td>
                                    <td>{row.quant}</td>
                                    <td>{row.params}</td>
                                    <td>{row.modules}</td>
                                    <td><div class='model-config'>{config}</div></td>
                                </tr>
                            """
                        except Exception as e:
                            log.error(f'Model list: row={vars(row)} {e}')
                    return html.format(tbody=tbody)

                def analyze():
                    model = modelstats.analyze()
                    if model is None:
                        return ["Model not loaded", {}]
                    meta = model.meta
                    html = create_modules_table(model.modules)
                    return [html, meta]

                with gr.Row():
                    model_analyze = gr.Button(value="Analyze model", variant='primary')
                with gr.Row():
                    model_desc = gr.HTML(value="", elem_id="model_desc")
                with gr.Accordion(label="Save model", open=False):
                    with gr.Row():
                        save_name = gr.Textbox(label="Model name", placeholder="Model name to save as")
                    with gr.Row():
                        save_path = gr.Textbox(label="Model base path", placeholder="Path to save model to", value=opts.diffusers_dir)
                    with gr.Row():
                        save_shard = gr.Textbox(label="Max shard size", placeholder="Maximum shard size", value="10GB")
                        save_overwrite = gr.Checkbox(label="Overwrite existing", value=False)
                    with gr.Row():
                        save_result = gr.HTML(value="", elem_id="model_save_outcome")
                    with gr.Row():
                        model_save = gr.Button(value="Save model", variant='primary')
                        model_save.click(fn=sd_models.save_model, inputs=[save_name, save_path, save_shard, save_overwrite], outputs=[save_result])
                with gr.Accordion(label="Metadata", open=False):
                    model_meta = gr.JSON(label="Metadata", value={}, elem_id="model_meta")

                model_analyze.click(fn=analyze, inputs=[], outputs=[model_desc, model_meta])

            with gr.Tab(label="List", elem_id="models_list_tab"):
                def create_models_table(rows: list):
                    from modules import sd_detect
                    html = """
                        <table class="simple-table">
                            <thead>
                                <tr><th>Name</th><th>Type</th><th>Detect</th><th>Pipeline</th><th>Hash</th><th>Size</th><th>MTime</th></tr>
                            </thead>
                            <tbody>
                                {tbody}
                            </tbody>
                        </table>
                    """
                    tbody = ''
                    for row in rows:
                        try:
                            f = row.filename
                            stat_size, stat_mtime = modelstats.stat(f)
                            if os.path.isfile(f):
                                typ = os.path.splitext(f)[1][1:]
                                size = f"{round(stat_size / 1024 / 1024 / 1024, 3)} gb"
                            elif os.path.isdir(f):
                                typ = 'diffusers'
                                size = 'folder'
                            else:
                                typ = 'unknown'
                                size = 'unknown'
                            guess = 'Diffusion' # set default guess
                            guess = sd_detect.guess_by_size(f, guess)
                            guess = sd_detect.guess_by_name(f, guess)
                            guess, pipeline = sd_detect.guess_by_diffusers(f, guess)
                            guess = sd_detect.guess_variant(f, guess)
                            pipeline = sd_detect.shared_items.get_pipelines().get(guess, None) if pipeline is None else pipeline
                            tbody += f"""
                                <tr>
                                    <td>{row.model_name}</td>
                                    <td>{typ}</td>
                                    <td>{guess}</td>
                                    <td>{pipeline.__name__ if pipeline else '(unknown)'}</td>
                                    <td>{row.shorthash}</td>
                                    <td>{size}</td>
                                    <td>{stat_mtime}</td>
                                </tr>
                            """
                        except Exception as e:
                            log.error(f'Model list: row={vars(row)} {e}')
                    return html.format(tbody=tbody)

                with gr.Row():
                    gr.HTML('<h2>List all locally available models</h2><br>')
                with gr.Row():
                    model_list_btn = gr.Button(value="List models", variant='primary')
                    model_checkhash_btn = gr.Button(value="Calculate missing hashes", variant='secondary')
                with gr.Row():
                    model_table = gr.HTML(value='', elem_id="model_list_table")

                model_checkhash_btn.click(fn=sd_models.update_model_hashes, inputs=[], outputs=[model_table])
                model_list_btn.click(fn=lambda: create_models_table(list(sd_models.checkpoints_list.values())), inputs=[], outputs=[model_table])

            with gr.Tab(label="Metadata", elem_id="models_metadata_tab"):
                from modules.civitai.metadata_civitai import civit_search_metadata, civit_update_metadata
                with gr.Row():
                    gr.HTML('<h2>Fetch model preview metadata</h2><br>')
                with gr.Row():
                    civit_previews_btn = gr.Button(value="Scan missing", variant='primary')
                    civit_update_btn = gr.Button(value="Update all", variant='primary')
                with gr.Row():
                    civit_metadata = gr.HTML(value='', elem_id="civit_metadata")
                civit_previews_btn.click(fn=civit_search_metadata, inputs=[], outputs=[civit_metadata])
                civit_update_btn.click(fn=civit_update_metadata, inputs=[], outputs=[civit_metadata])


            with gr.Tab(label="Loader", elem_id="models_loader_tab"):
                from modules import ui_models_load
                ui_models_load.create_ui(models_outcome, models_file)

            with gr.Tab(label="Merge", elem_id="models_merge_tab"):
                from modules.merging import merge_methods
                from modules.merging.merge_utils import BETA_METHODS, TRIPLE_METHODS, interpolate
                from modules.merging.merge_presets import BLOCK_WEIGHTS_PRESETS, SDXL_BLOCK_WEIGHTS_PRESETS

                def sd_model_choices():
                    return ['None'] + sd_models.checkpoint_titles()

                with gr.Row():
                    gr.HTML('<h2>&nbspMerge multiple models<br></h2>')
                with gr.Row(equal_height=False):
                    with gr.Column(variant='compact'):
                        with gr.Row():
                            custom_name = gr.Textbox(label="New model name")
                        with gr.Row():
                            merge_mode = gr.Dropdown(choices=merge_methods.__all__, value="weighted_sum", label="Interpolation Method")
                            merge_mode_docs = gr.HTML(value=merge_methods.weighted_sum.__doc__.strip().replace("\n", "<br>")) # pylint: disable=no-member # pyright: ignore[reportOptionalMemberAccess]
                        with gr.Row():
                            primary_model_name = gr.Dropdown(sd_model_choices(), label="Primary model", value="None")
                            create_refresh_button(primary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "checkpoint_A_refresh")
                            secondary_model_name = gr.Dropdown(sd_model_choices(), label="Secondary model", value="None")
                            create_refresh_button(secondary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "checkpoint_B_refresh")
                            tertiary_model_name = gr.Dropdown(sd_model_choices(), label="Tertiary model", value="None", visible=False)
                            tertiary_refresh = create_refresh_button(tertiary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "checkpoint_C_refresh", visible=False)
                        with gr.Row():
                            with gr.Tabs() as tabs:
                                with gr.TabItem(label="Simple Merge", id=0):
                                    with gr.Row():
                                        alpha = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Alpha Ratio', value=0.5)
                                        beta = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Beta Ratio', value=None, visible=False)
                                with gr.TabItem(label="Preset Block Merge", id=1):
                                    with gr.Row():
                                        sdxl = gr.Checkbox(label="SDXL")
                                    with gr.Row():
                                        alpha_preset = gr.Dropdown(
                                            choices=["None"] + list(BLOCK_WEIGHTS_PRESETS.keys()), value=None,
                                            label="ALPHA Block Weight Preset", multiselect=True, max_choices=2)
                                        alpha_preset_lambda = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Preset Interpolation Ratio', value=None, visible=False)
                                        apply_preset = ToolButton('⇨', visible=True)
                                    with gr.Row():
                                        beta_preset = gr.Dropdown(choices=["None"] + list(BLOCK_WEIGHTS_PRESETS.keys()), value=None, label="BETA Block Weight Preset", multiselect=True, max_choices=2, interactive=True, visible=False)
                                        beta_preset_lambda = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label='Preset Interpolation Ratio', value=None, interactive=True, visible=False)
                                        beta_apply_preset = ToolButton('⇨', interactive=True, visible=False)
                                with gr.TabItem(label="Manual Block Merge", id=2):
                                    with gr.Row():
                                        alpha_label = gr.Markdown("# Alpha")
                                    with gr.Row():
                                        alpha_base = gr.Textbox(value=None, label="Base", min_width=70, scale=1)
                                        alpha_in_blocks = gr.Textbox(value=None, label="In Blocks", scale=15)
                                        alpha_mid_block = gr.Textbox(value=None, label="Mid Block", min_width=80, scale=1)
                                        alpha_out_blocks = gr.Textbox(value=None, label="Out Block", scale=15)
                                    with gr.Row():
                                        beta_label = gr.Markdown("# Beta", visible=False)
                                    with gr.Row():
                                        beta_base = gr.Textbox(value=None, label="Base", min_width=70, scale=1, interactive=True, visible=False)
                                        beta_in_blocks = gr.Textbox(value=None, label="In Blocks", interactive=True, scale=15, visible=False)
                                        beta_mid_block = gr.Textbox(value=None, label="Mid Block", min_width=80, interactive=True, scale=1, visible=False)
                                        beta_out_blocks = gr.Textbox(value=None, label="Out Block", interactive=True, scale=15, visible=False)
                        with gr.Row():
                            overwrite = gr.Checkbox(label="Overwrite model")
                        with gr.Row():
                            save_metadata = gr.Checkbox(value=True, label="Save metadata")
                        with gr.Row():
                            weights_clip = gr.Checkbox(label="Weights clip")
                            prune = gr.Checkbox(label="Prune", value=True, visible=False)
                        with gr.Row():
                            re_basin = gr.Checkbox(label="ReBasin")
                            re_basin_iterations = gr.Slider(minimum=0, maximum=25, step=1, label='Number of ReBasin Iterations', value=None, visible=False)
                        with gr.Row():
                            checkpoint_format = gr.Radio(choices=["ckpt", "safetensors"], value="safetensors", visible=False, label="Model format")
                        with gr.Row():
                            precision = gr.Radio(choices=["fp16", "fp32"], value="fp16", label="Model precision")
                        with gr.Row():
                            device = gr.Radio(choices=["cpu", "shuffle", "gpu"], value="cpu", label="Merge Device")
                            unload = gr.Checkbox(label="Unload Current Model from VRAM", value=False, visible=False)
                        with gr.Row():
                            bake_in_vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", interactive=True, label="Replace VAE")
                            create_refresh_button(bake_in_vae, sd_vae.refresh_vae_list,
                                                  lambda: {"choices": ["None"] + list(sd_vae.vae_dict)},
                                                  "modelmerger_bake_in_vae_refresh")
                        with gr.Row():
                            modelmerger_merge = gr.Button(value="Merge", variant='primary')

                def modelmerger(dummy_component, # dummy function just to get argspec later
                                overwrite, # pylint: disable=unused-argument
                                primary_model_name, # pylint: disable=unused-argument
                                secondary_model_name, # pylint: disable=unused-argument
                                tertiary_model_name, # pylint: disable=unused-argument
                                merge_mode, # pylint: disable=unused-argument
                                alpha, # pylint: disable=unused-argument
                                beta, # pylint: disable=unused-argument
                                alpha_preset, # pylint: disable=unused-argument
                                alpha_preset_lambda, # pylint: disable=unused-argument
                                alpha_base, # pylint: disable=unused-argument
                                alpha_in_blocks, # pylint: disable=unused-argument
                                alpha_mid_block, # pylint: disable=unused-argument
                                alpha_out_blocks, # pylint: disable=unused-argument
                                beta_preset, # pylint: disable=unused-argument
                                beta_preset_lambda, # pylint: disable=unused-argument
                                beta_base, # pylint: disable=unused-argument
                                beta_in_blocks, # pylint: disable=unused-argument
                                beta_mid_block, # pylint: disable=unused-argument
                                beta_out_blocks, # pylint: disable=unused-argument
                                precision, # pylint: disable=unused-argument
                                custom_name, # pylint: disable=unused-argument
                                checkpoint_format, # pylint: disable=unused-argument
                                save_metadata, # pylint: disable=unused-argument
                                weights_clip, # pylint: disable=unused-argument
                                prune, # pylint: disable=unused-argument
                                re_basin, # pylint: disable=unused-argument
                                re_basin_iterations, # pylint: disable=unused-argument
                                device, # pylint: disable=unused-argument
                                unload, # pylint: disable=unused-argument
                                bake_in_vae): # pylint: disable=unused-argument
                    kwargs = {}
                    for x in inspect.getfullargspec(modelmerger)[0]:
                        kwargs[x] = locals()[x]
                    for key in list(kwargs.keys()):
                        if kwargs[key] in [None, "None", "", 0, []]:
                            del kwargs[key]
                    del kwargs['dummy_component']
                    if kwargs.get("custom_name", None) is None:
                        log.error('Merge: no output model specified')
                        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_titles()) for _ in range(4)], "No output model specified"]
                    elif kwargs.get("primary_model_name", None) is None or kwargs.get("secondary_model_name", None) is None:
                        log.error('Merge: no models selected')
                        return [*[gr.Dropdown.update(choices=sd_models.checkpoint_titles()) for _ in range(4)], "No models selected"]
                    else:
                        log.debug(f'Merge start: {kwargs}')
                        try:
                            results = extras.run_modelmerger(dummy_component, **kwargs)
                        except Exception as e:
                            errors.display(e, 'Merge')
                            sd_models.list_models()  # to remove the potentially missing models from the list
                            return [*[gr.Dropdown.update(choices=sd_models.checkpoint_titles()) for _ in range(4)], f"Error merging checkpoints: {e}"]
                        return results

                def tertiary(mode):
                    if mode in TRIPLE_METHODS:
                        return [gr.update(visible=True) for _ in range(2)]
                    else:
                        return [gr.update(visible=False) for _ in range(2)]

                def beta_visibility(mode):
                    if mode in BETA_METHODS:
                        return [gr.update(visible=True) for _ in range(9)]
                    else:
                        return [gr.update(visible=False) for _ in range(9)]

                def show_iters(show):
                    if show:
                        return gr.Slider.update(value=5, visible=True)
                    else:
                        return gr.Slider.update(value=None, visible=False)

                def show_help(mode):
                    try:
                        doc = getattr(merge_methods, mode).__doc__.strip().replace("\n", "<br>")
                    except AttributeError:
                        log.warning(f'Merge mode "{mode}" is missing documentation')
                        doc = "Error: Documentation missing"
                    return gr.update(value=doc, visible=True)

                def show_unload(device):
                    if device == "gpu":
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)


                def preset_visiblility(x):
                    if len(x) == 2:
                        return gr.Slider.update(value=0.5, visible=True)
                    else:
                        return gr.Slider.update(value=None, visible=False)

                def load_presets(presets, ratio):
                    for i, p in enumerate(presets):
                        presets[i] = BLOCK_WEIGHTS_PRESETS[p]
                    if len(presets) == 2:
                        preset = interpolate(presets, ratio)
                    else:
                        preset = presets[0]
                    preset = [f'{x:.3f}' if int(x) != x else str(x) for x in preset] # pylint: disable=consider-using-f-string
                    preset = [preset[0], ",".join(preset[1:13]), preset[13], ",".join(preset[14:])]
                    return [gr.update(value=x) for x in preset] + [gr.update(selected=2)]

                def preset_choices(sdxl):
                    if sdxl:
                        return [gr.update(choices=["None"] + list(SDXL_BLOCK_WEIGHTS_PRESETS.keys())) for _ in range(2)]
                    else:
                        return [gr.update(choices=["None"] + list(BLOCK_WEIGHTS_PRESETS.keys())) for _ in range(2)]
                device.change(fn=show_unload, inputs=device, outputs=unload)
                merge_mode.change(fn=show_help, inputs=merge_mode, outputs=merge_mode_docs)
                sdxl.change(fn=preset_choices, inputs=sdxl, outputs=[alpha_preset, beta_preset])
                alpha_preset.change(fn=preset_visiblility, inputs=alpha_preset, outputs=alpha_preset_lambda)
                beta_preset.change(fn=preset_visiblility, inputs=alpha_preset, outputs=beta_preset_lambda)
                merge_mode.input(fn=tertiary, inputs=merge_mode, outputs=[tertiary_model_name, tertiary_refresh])
                merge_mode.input(fn=beta_visibility, inputs=merge_mode, outputs=[beta, alpha_label, beta_label, beta_apply_preset, beta_preset, beta_base, beta_in_blocks, beta_mid_block, beta_out_blocks])
                re_basin.change(fn=show_iters, inputs=re_basin, outputs=re_basin_iterations)
                apply_preset.click(fn=load_presets, inputs=[alpha_preset, alpha_preset_lambda], outputs=[alpha_base, alpha_in_blocks, alpha_mid_block, alpha_out_blocks, cast("gr.components.Component", tabs)]) # Casting because Tabs has an update method.
                beta_apply_preset.click(fn=load_presets, inputs=[beta_preset, beta_preset_lambda], outputs=[beta_base, beta_in_blocks, beta_mid_block, beta_out_blocks, cast("gr.components.Component", tabs)]) # Casting because Tabs has an update method.

                modelmerger_merge.click(
                    fn=wrap_gradio_gpu_call(modelmerger, extra_outputs=lambda: [gr.update() for _ in range(4)], name='Models'),
                    _js='modelmerger',
                    inputs=[
                        dummy_component,
                        overwrite,
                        primary_model_name,
                        secondary_model_name,
                        tertiary_model_name,
                        merge_mode,
                        alpha,
                        beta,
                        alpha_preset,
                        alpha_preset_lambda,
                        alpha_base,
                        alpha_in_blocks,
                        alpha_mid_block,
                        alpha_out_blocks,
                        beta_preset,
                        beta_preset_lambda,
                        beta_base,
                        beta_in_blocks,
                        beta_mid_block,
                        beta_out_blocks,
                        precision,
                        custom_name,
                        checkpoint_format,
                        save_metadata,
                        weights_clip,
                        prune,
                        re_basin,
                        re_basin_iterations,
                        device,
                        unload,
                        bake_in_vae,
                    ],
                    outputs=[
                        primary_model_name,
                        secondary_model_name,
                        tertiary_model_name,
                        dummy_component,
                        models_outcome,
                    ]
                )

            with gr.Tab(label="Replace", elem_id="models_replace_tab"):
                with gr.Row():
                    gr.HTML('<h2>&nbspReplace model components<br></h2>')
                with gr.Row():
                    with gr.Column(scale=3):
                        model_type = gr.Dropdown(label="Base model type", choices=['sd15', 'sdxl', 'sd21', 'sd35', 'flux.1'], value='sdxl', interactive=False)
                    with gr.Column(scale=5):
                        with gr.Row():
                            model_name = gr.Dropdown(sd_models.checkpoint_titles(), label="Input model")
                            create_refresh_button(model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_titles()}, "checkpoint_Z_refresh")
                    with gr.Column(scale=5):
                        custom_name = gr.Textbox(label="Output model", placeholder="Output model path")
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML('Model components<br><span style="color: var(--body-text-color-subdued)">Specify the components to include<br>Paths can be relative or absolute</span><br>')
                    with gr.Column(scale=5):
                        comp_unet = gr.Textbox(placeholder="UNet model", show_label=False)
                        comp_vae = gr.Textbox(placeholder="VAE model", show_label=False)
                    with gr.Column(scale=5):
                        comp_te1 = gr.Textbox(placeholder="Text encoder 1", show_label=False)
                        comp_te2 = gr.Textbox(placeholder="Text encoder 2", show_label=False)
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML('Model settings<br>')
                    with gr.Column(scale=10):
                        with gr.Row():
                            precision = gr.Dropdown(label="Model precision", choices=["fp32", "fp16", "bf16"], value="fp16")
                            comp_scheduler = gr.Dropdown(label="Sampler", choices=[s.name for s in sd_samplers.samplers if s.constructor is not None])
                            comp_prediction = gr.Dropdown(label="Prediction type", choices=["epsilon", "v"], value="epsilon")
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML('Merge LoRA<br>')
                    with gr.Column(scale=9):
                        comp_lora = gr.Textbox(label="Comma separated list with optional strength per LoRA", placeholder="LoRA models")
                    with gr.Column(scale=1):
                        comp_fuse = gr.Number(label="Fuse strength", value=1.0)

                with gr.Row():
                    gr.HTML('<br>')
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML('Model metadata<br>')
                    with gr.Column(scale=5):
                        meta_author = gr.Textbox(placeholder="Author name", show_label=False)
                        meta_version = gr.Textbox(placeholder="Model version", show_label=False)
                        meta_license = gr.Textbox(placeholder="Model license", show_label=False)
                    with gr.Column(scale=5):
                        meta_desc = gr.Textbox(placeholder="Model description", lines=3, show_label=False)
                        meta_hint = gr.Textbox(placeholder="Model hint", lines=3, show_label=False)
                    with gr.Column(scale=3):
                        meta_thumbnail = gr.Image(label="Thumbnail", type='pil')
                with gr.Row():
                    gr.HTML('Note: Save is optional as you can merge in-memory and use newly created model immediately')
                with gr.Row():
                    create_diffusers = gr.Checkbox(label="Save diffusers", value=True)
                    create_safetensors = gr.Checkbox(label="Save safetensors", value=True)
                    debug = gr.Checkbox(label="Debug info", value=False)

                model_modules_btn = gr.Button(value="Merge Modules", variant='primary')
                model_modules_btn.click(
                    fn=extras.run_model_modules,
                    inputs=[
                        model_type, model_name, custom_name,
                        comp_unet, comp_vae, comp_te1, comp_te2,
                        precision, comp_scheduler, comp_prediction,
                        comp_lora, comp_fuse,
                        meta_author, meta_version, meta_license, meta_desc, meta_hint, meta_thumbnail,
                        create_diffusers, create_safetensors, debug,
                    ],
                    outputs=[models_outcome]
                )

            with gr.Tab(label="CivitAI", elem_id="models_civitai_tab"):
                from modules.civitai.search_civitai import search_civitai, create_model_cards, base_models

                def civitai_search(civit_search_text, civit_search_tag, civit_nsfw, civit_type, civit_base, civit_token):
                    results = search_civitai(query=civit_search_text, tag=civit_search_tag, nsfw=civit_nsfw, types=civit_type, base=civit_base, token=civit_token)
                    html = create_model_cards(results)
                    return html

                def civitai_update_token(token):
                    log.debug('CivitAI update token')
                    opts.civitai_token = token
                    opts.save()

                def civitai_download(model_urls, model_names, model_types, model_path, civit_token, model_output):
                    from modules.civitai.download_civitai import download_civit_model
                    for model_url, model_name, model_type in zip(model_urls, model_names, model_types, strict=False):
                        msg = f"<h4>Initiating download</h4><div>{model_name} | {model_type} | <a href='{model_url}'>{model_url}</a></div><br>"
                        yield msg + model_output
                        download_civit_model(model_url, model_name, model_path, model_type, civit_token)
                        yield model_output

                with gr.Row():
                    gr.HTML('<h2>Search & Download</h2>')
                with gr.Row(elem_id='civitai_search_row'):
                    civit_search_text = gr.Textbox(label='', placeholder='keyword', elem_id="civit_search_text")
                    civit_search_tag = gr.Textbox(label='', placeholder='tag', elem_id="civit_search_text")
                    civit_search_text_btn = ToolButton(value=ui_symbols.search, interactive=True, elem_id="civit_text_search")
                with gr.Accordion(label='Advanced', open=False, elem_id="civitai_search_options"):
                    civit_download_btn = gr.Button(value="Download model", variant='primary', elem_id="civitai_download_btn", visible=False)
                    with gr.Row():
                        civit_token = gr.Textbox(opts.civitai_token, label='CivitAI token', placeholder='optional access token for private or gated models', elem_id="civitai_token")
                    with gr.Row():
                        civit_nsfw = gr.Checkbox(label='NSFW allowed', value=True)
                    with gr.Row():
                        civit_type = gr.Textbox(label='Target model type', placeholder='Checkpoint, LORA, ...', value='')
                    with gr.Row():
                        # civit_base = gr.Textbox(label='Base model', placeholder='SDXL, ...')
                        civit_base = gr.Dropdown(choices=base_models, label='Base model', value='')
                    with gr.Row():
                        civit_folder = gr.Textbox(label='Download folder', placeholder='optional folder for downloads')
                with gr.Row():
                    civitai_models_output = gr.HTML('', elem_id="civitai_models_output")
                # sort, period, limit
                _dummy = gr.Label(visible=False)  # dummy component to get argspec later
                civit_inputs = [civit_search_text, civit_search_tag, civit_nsfw, civit_type, civit_base, civit_token]
                civit_search_text_btn.click(fn=civitai_search, inputs=civit_inputs, outputs=[civitai_models_output])
                civit_search_text.submit(fn=civitai_search, inputs=civit_inputs, outputs=[civitai_models_output])
                civit_search_tag.submit(fn=civitai_search, inputs=civit_inputs, outputs=[civitai_models_output])
                civit_token.change(fn=civitai_update_token, inputs=[civit_token], outputs=[])
                civit_download_btn.click(
                    fn=civitai_download,
                    _js="downloadCivitModel",
                    inputs=[_dummy, _dummy, _dummy, civit_folder, civit_token, civitai_models_output],
                    outputs=[civitai_models_output],
                    show_progress='full',
                )

            with gr.Tab(label="Huggingface", elem_id="models_huggingface_tab"):
                from modules.models_hf import hf_search, hf_select, hf_download_model, hf_update_token
                with gr.Column(scale=6):
                    with gr.Row():
                        gr.HTML('<h2>&nbspDownload model from huggingface<br></h2>')
                    with gr.Row():
                        hf_search_text = gr.Textbox('', label='Search models', placeholder='search huggingface models')
                        hf_search_btn = ToolButton(value=ui_symbols.search, interactive=True, elem_id="hf_text_search")
                    with gr.Row():
                        hf_selected = gr.Textbox('', label='Select model', placeholder='select model from search results or enter model name manually')
                    with gr.Accordion(label='Advanced', open=False, elem_id="hf_search_options"):
                        with gr.Row():
                            hf_token = gr.Textbox(opts.huggingface_token, label='Huggingface token', placeholder='optional access token for private or gated models', elem_id="hf_token")
                        with gr.Row():
                            hf_variant = gr.Textbox('', label='Specify model variant', placeholder='')
                            hf_revision = gr.Textbox('', label='Specify model revision', placeholder='')
                        with gr.Row():
                            hf_mirror = gr.Textbox('', label='Huggingface mirror', placeholder='optional mirror site for downloads')
                            hf_custom_pipeline = gr.Textbox('', label='Custom pipeline', placeholder='optional pipeline for downloads')
                with gr.Column(scale=1):
                    gr.HTML('<br>')
                    hf_download_model_btn = gr.Button(value="Download model", variant='primary')

                with gr.Row():
                    hf_headers = ['Name', 'Pipeline', 'Tags', 'Downloads', 'Updated', 'URL']
                    hf_types = ['str', 'str', 'str', 'number', 'date', 'markdown']
                    hf_results = gr.DataFrame(None, label='Search results', show_label=True, interactive=False, wrap=True, headers=hf_headers, datatype=hf_types)

                hf_search_text.submit(fn=hf_search, inputs=[hf_search_text], outputs=[hf_results])
                hf_search_btn.click(fn=hf_search, inputs=[hf_search_text], outputs=[hf_results])
                hf_results.select(fn=hf_select, inputs=[hf_results], outputs=[hf_selected])
                hf_download_model_btn.click(fn=hf_download_model, inputs=[hf_selected, hf_token, hf_variant, hf_revision, hf_mirror, hf_custom_pipeline], outputs=[models_outcome])
                hf_token.change(fn=hf_update_token, inputs=[hf_token], outputs=[])

            from modules.lora.lora_extract import create_ui as lora_extract_ui
            lora_extract_ui()

            for ui in extra_ui:
                if callable(ui):
                    ui()
