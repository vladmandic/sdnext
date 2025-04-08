import os
import time
import json
import inspect
from datetime import datetime
import gradio as gr
from modules import errors, sd_models, sd_vae, extras, sd_samplers, ui_symbols, hashes
from modules.ui_components import ToolButton
from modules.ui_common import create_refresh_button
from modules.call_queue import wrap_gradio_gpu_call
from modules.shared import opts, log, req, readfile, max_workers, native
from modules.merging import merge_methods
from modules.merging.merge_utils import BETA_METHODS, TRIPLE_METHODS, interpolate
from modules.merging.merge_presets import BLOCK_WEIGHTS_PRESETS, SDXL_BLOCK_WEIGHTS_PRESETS


search_metadata_civit = None
extra_ui = []


def create_ui():
    dummy_component = gr.Label(visible=False)
    with gr.Row(elem_id="models_tab"):
        with gr.Column(elem_id='models_output_container', scale=1):
            # models_output = gr.Text(elem_id="models_output", value="", show_label=False)
            gr.HTML(elem_id="models_progress", value="")
            models_image = gr.Image(elem_id="models_image", show_label=False, interactive=False, type='pil')
            models_outcome = gr.HTML(elem_id="models_error", value="")

        with gr.Column(elem_id='models_input_container', scale=3):

            with gr.Tab(label="Current"):
                def analyze():
                    from modules import modelstats
                    model = modelstats.analyze()
                    desc = f"Model: {model.name}<br>Type: {model.type}<br>Class: {model.cls}<br>Size: {model.size} bytes<br>Modified: {model.mtime}<br>"
                    meta = model.meta
                    components = [(m.name, m.cls, m.device, m.dtype, m.params, m.modules, str(m.config)) for m in model.modules]
                    return [desc, components, meta]

                with gr.Row():
                    gr.HTML('<h2>&nbspAnalyze currently loaded model<br></h2>')
                with gr.Row():
                    model_analyze = gr.Button(value="Analyze", variant='primary')
                with gr.Row():
                    model_desc = gr.HTML(value="", elem_id="model_desc")
                with gr.Row():
                    module_headers = ['Module', 'Class', 'Device', 'DType', 'Params', 'Modules', 'Config']
                    module_types = ['str', 'str', 'str', 'str', 'number', 'number', 'str']
                    model_modules = gr.DataFrame(value=None, label=None, show_label=False, interactive=False, wrap=True, headers=module_headers, datatype=module_types, type='array')
                with gr.Row():
                    model_meta = gr.JSON(label="Metadata", value={}, elem_id="model_meta")

                model_analyze.click(fn=analyze, inputs=[], outputs=[model_desc, model_modules, model_meta])

            with gr.Tab(label="Custom"):
                from modules import ui_models_load
                ui_models_load.create_ui(models_outcome)

            with gr.Tab(label="Merge"):
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
                            merge_mode_docs = gr.HTML(value=getattr(merge_methods, "weighted_sum", "").__doc__.replace("\n", "<br>"))
                        with gr.Row():
                            primary_model_name = gr.Dropdown(sd_model_choices(), label="Primary model", value="None")
                            create_refresh_button(primary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "refresh_checkpoint_A")
                            secondary_model_name = gr.Dropdown(sd_model_choices(), label="Secondary model", value="None")
                            create_refresh_button(secondary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "refresh_checkpoint_B")
                            tertiary_model_name = gr.Dropdown(sd_model_choices(), label="Tertiary model", value="None", visible=False)
                            tertiary_refresh = create_refresh_button(tertiary_model_name, sd_models.list_models, lambda: {"choices": sd_model_choices()}, "refresh_checkpoint_C", visible=False)
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
                                                  "modelmerger_refresh_bake_in_vae")
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
                    doc = getattr(merge_methods, mode).__doc__.replace("\n", "<br>")
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
                    preset = ['%.3f' % x if int(x) != x else str(x) for x in preset] # pylint: disable=consider-using-f-string
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
                apply_preset.click(fn=load_presets, inputs=[alpha_preset, alpha_preset_lambda], outputs=[alpha_base, alpha_in_blocks, alpha_mid_block, alpha_out_blocks, tabs])
                beta_apply_preset.click(fn=load_presets, inputs=[beta_preset, beta_preset_lambda], outputs=[beta_base, beta_in_blocks, beta_mid_block, beta_out_blocks, tabs])

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

            with gr.Tab(label="Modules"):
                with gr.Row():
                    gr.HTML('<h2>&nbspReplace model components<br></h2>')
                with gr.Row():
                    with gr.Column(scale=3):
                        model_type = gr.Dropdown(label="Model type", choices=['sd15', 'sdxl', 'sd21', 'sd35', 'flux.1'], value='sdxl', interactive=False)
                    with gr.Column(scale=5):
                        with gr.Row():
                            model_name = gr.Dropdown(sd_models.checkpoint_titles(), label="Input model")
                            create_refresh_button(model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_titles()}, "refresh_checkpoint_Z")
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
                            comp_prediction = gr.Dropdown(Label="Prediction type", choices=["epsilon", "v"], value="epsilon")
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
                        meta_thumbnail = gr.Image(label="Thumbnail", type='pil', source='upload')
                with gr.Row():
                    gr.HTML('Note: Save is optional as you can merge in-memory and use newly created model immediately')
                with gr.Row():
                    create_diffusers = gr.Checkbox(label="Save diffusers", value=True)
                    create_safetensors = gr.Checkbox(label="Save safetensors", value=True)
                    debug = gr.Checkbox(label="Debug info", value=False)

                model_modules_btn = gr.Button(label="Modules", variant='primary')
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

            with gr.Tab(label="Validate"):
                model_headers = ['name', 'type', 'filename', 'hash', 'added', 'size', 'metadata']
                model_data = []

                with gr.Row():
                    gr.HTML('<h2>&nbspList all models <br></h2>')
                with gr.Row():
                    model_list_btn = gr.Button(value="List model details", variant='primary')
                    model_checkhash_btn = gr.Button(value="Calculate hash for all models", variant='primary')
                    model_checkhash_btn.click(fn=sd_models.update_model_hashes, inputs=[], outputs=[models_outcome])
                with gr.Row():
                    model_table = gr.DataFrame(
                        value=None,
                        headers=model_headers,
                        label='Model data',
                        show_label=True,
                        interactive=False,
                        wrap=True,
                        overflow_row_behaviour='paginate',
                        max_rows=50,
                    )

                def list_models():
                    total_size = 0
                    model_data.clear()
                    txt = ''
                    for m in sd_models.checkpoints_list.values():
                        try:
                            stat = os.stat(m.filename)
                            m_name = m.name.replace('.ckpt', '').replace('.safetensors', '')
                            m_type = 'ckpt' if m.name.endswith('.ckpt') else 'safe'
                            m_meta = len(json.dumps(m.metadata)) - 2
                            m_size = round(stat.st_size / 1024 / 1024 / 1024, 3)
                            m_time = datetime.fromtimestamp(stat.st_mtime)
                            model_data.append([m_name, m_type, m.filename, m.shorthash, m_time, m_size, m_meta])
                            total_size += stat.st_size
                        except Exception as e:
                            txt += f"Error: {m.name} {e}<br>"
                    txt += f"Model list enumerated {len(sd_models.checkpoints_list.keys())} models in {round(total_size / 1024 / 1024 / 1024, 3)} GB<br>"
                    return model_data, txt

                model_list_btn.click(fn=list_models, inputs=[], outputs=[model_table, models_outcome])

            with gr.Tab(label="Huggingface"):
                data = []
                os.environ.setdefault('HF_HUB_DISABLE_EXPERIMENTAL_WARNING', '1')
                os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
                os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
                os.environ.setdefault('HUGGINGFACE_HUB_VERBOSITY', 'warning')

                def hf_search(keyword):
                    import huggingface_hub as hf
                    hf_api = hf.HfApi()
                    models = hf_api.list_models(model_name=keyword, full=True, library="diffusers", limit=50, sort="downloads", direction=-1)
                    data.clear()
                    for model in models:
                        tags = [t for t in model.tags if not t.startswith('diffusers') and not t.startswith('license') and not t.startswith('arxiv') and len(t) > 2]
                        data.append([model.id, model.pipeline_tag, tags, model.downloads, model.lastModified, f'https://huggingface.co/{model.id}'])
                    return data

                def hf_select(evt: gr.SelectData, data):
                    return data[evt.index[0]][0]

                def hf_download_model(hub_id: str, token, variant, revision, mirror, custom_pipeline):
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

                with gr.Column(scale=6):
                    with gr.Row():
                        gr.HTML('<h2>&nbspDownload model from huggingface<br></h2>')
                    with gr.Row():
                        hf_search_text = gr.Textbox('', label='Search models', placeholder='search huggingface models')
                        hf_search_btn = ToolButton(value=ui_symbols.search, label="Search")
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                hf_selected = gr.Textbox('', label='Select model', placeholder='select model from search results or enter model name manually')
                        with gr.Column(scale=1):
                            with gr.Row():
                                hf_variant = gr.Textbox('', label='Specify model variant', placeholder='')
                                hf_revision = gr.Textbox('', label='Specify model revision', placeholder='')
                    with gr.Row():
                        hf_token = gr.Textbox(opts.huggingface_token, label='Huggingface token', placeholder='optional access token for private or gated models')
                        hf_mirror = gr.Textbox('', label='Huggingface mirror', placeholder='optional mirror site for downloads')
                        hf_custom_pipeline = gr.Textbox('', label='Custom pipeline', placeholder='optional pipeline for downloads')
                with gr.Column(scale=1):
                    gr.HTML('<br>')
                    hf_download_model_btn = gr.Button(value="Download model", variant='primary')

                with gr.Row():
                    hf_headers = ['Name', 'Pipeline', 'Tags', 'Downloads', 'Updated', 'URL']
                    hf_types = ['str', 'str', 'str', 'number', 'date', 'markdown']
                    hf_results = gr.DataFrame(None, label='Search results', show_label=True, interactive=False, wrap=True, overflow_row_behaviour='paginate', max_rows=10, headers=hf_headers, datatype=hf_types, type='array')

                hf_search_text.submit(fn=hf_search, inputs=[hf_search_text], outputs=[hf_results])
                hf_search_btn.click(fn=hf_search, inputs=[hf_search_text], outputs=[hf_results])
                hf_results.select(fn=hf_select, inputs=[hf_results], outputs=[hf_selected])
                hf_download_model_btn.click(fn=hf_download_model, inputs=[hf_selected, hf_token, hf_variant, hf_revision, hf_mirror, hf_custom_pipeline], outputs=[models_outcome])
                hf_token.change(fn=hf_update_token, inputs=[hf_token], outputs=[])

            with gr.Tab(label="CivitAI"):
                data = []

                def civit_search_model(name, tag, model_type):
                    # types = 'LORA' if model_type == 'LoRA' else 'Checkpoint'
                    url = 'https://civitai.com/api/v1/models?limit=25&Sort=Newest'
                    if model_type == 'Model':
                        url += '&types=Checkpoint'
                    elif model_type == 'LoRA':
                        url += '&types=LORA&types=DoRA&types=LoCon'
                    elif model_type == 'Embedding':
                        url += '&types=TextualInversion'
                    elif model_type == 'VAE':
                        url += '&types=VAE'
                    if name is not None and len(name) > 0:
                        url += f'&query={name}'
                    if tag is not None and len(tag) > 0:
                        url += f'&tag={tag}'
                    r = req(url)
                    log.debug(f'CivitAI search: type={model_type} name="{name}" tag={tag or "none"} url="{url}" status={r.status_code}')
                    if r.status_code != 200:
                        log.warning(f'CivitAI search: name="{name}" tag={tag} status={r.status_code}')
                        return [], gr.update(visible=False, value=[]), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
                    try:
                        body = r.json()
                    except Exception as e:
                        log.error(f'CivitAI search: name="{name}" tag={tag} {e}')
                        return [], gr.update(visible=False, value=[]), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
                    nonlocal data
                    data = body.get('items', [])
                    data1 = []
                    for model in data:
                        found = 0
                        if model_type == 'LoRA' and model['type'].lower() in ['lora', 'locon', 'dora', 'lycoris']:
                            found += 1
                        elif model_type == 'Embedding' and model['type'].lower() in ['textualinversion', 'embedding']:
                            found += 1
                        elif model_type == 'Model' and model['type'].lower() in ['checkpoint']:
                            found += 1
                        elif model_type == 'VAE' and model['type'].lower() in ['vae']:
                            found += 1
                        elif model_type == 'Other':
                            found += 1
                        if found > 0:
                            data1.append([
                                model['id'],
                                model['name'],
                                ', '.join(model['tags']),
                                model['stats']['downloadCount'],
                                model['stats']['rating']
                            ])
                    res = f'Search result: name={name} tag={tag or "none"} type={model_type} models={len(data1)}'
                    return res, gr.update(visible=len(data1) > 0, value=data1 if len(data1) > 0 else []), gr.update(visible=False, value=None), gr.update(visible=False, value=None)

                def civit_select1(evt: gr.SelectData, in_data):
                    model_id = in_data[evt.index[0]][0]
                    data2 = []
                    preview_img = None
                    for model in data:
                        if model['id'] == model_id:
                            for d in model['modelVersions']:
                                try:
                                    if d.get('images') is not None and len(d['images']) > 0 and len(d['images'][0]['url']) > 0:
                                        preview_img = d['images'][0]['url']
                                    data2.append([d.get('id', None), d.get('modelId', None) or model_id, d.get('name', None), d.get('baseModel', None), d.get('createdAt', None) or d.get('publishedAt', None)])
                                except Exception as e:
                                    log.error(f'CivitAI select: model="{in_data[evt.index[0]]}" {e}')
                                    log.error(f'CivitAI version data={type(d)}: {d}')
                    log.debug(f'CivitAI select: model="{in_data[evt.index[0]]}" versions={len(data2)}')
                    return data2, None, preview_img

                def civit_select2(evt: gr.SelectData, in_data):
                    variant_id = in_data[evt.index[0]][0]
                    model_id = in_data[evt.index[0]][1]
                    data3 = []
                    for model in data:
                        if model['id'] == model_id:
                            for variant in model['modelVersions']:
                                if variant['id'] == variant_id:
                                    for f in variant['files']:
                                        try:
                                            if os.path.splitext(f['name'])[1].lower() in ['.safetensors', '.ckpt', '.pt', '.pth', '.bin']:
                                                data3.append([f['name'], round(f['sizeKB']), json.dumps(f['metadata']), f['downloadUrl']])
                                        except Exception:
                                            pass
                    log.debug(f'CivitAI select: model="{in_data[evt.index[0]]}" files={len(data3)}')
                    return data3

                def civit_select3(evt: gr.SelectData, in_data):
                    log.debug(f'CivitAI select: variant={in_data[evt.index[0]]}')
                    return in_data[evt.index[0]][3], in_data[evt.index[0]][0], gr.update(interactive=True)

                def civit_download_model(model_url: str, model_name: str, model_path: str, model_type: str, token: str = None):
                    if model_url is None or len(model_url) == 0:
                        return 'No model selected'
                    try:
                        from modules.modelloader import download_civit_model
                        res = download_civit_model(model_url, model_name, model_path, model_type, token=token)
                    except Exception as e:
                        res = f"CivitAI model downloaded error: model={model_url} {e}"
                        log.error(res)
                        return res
                    from modules.sd_models import list_models  # pylint: disable=W0621
                    list_models()
                    return res

                def atomic_civit_search_metadata(item, res, rehash):
                    from modules.modelloader import download_civit_preview, download_civit_meta
                    if item is None:
                        return
                    meta = os.path.splitext(item['filename'])[0] + '.json'
                    has_meta = os.path.isfile(meta) and os.stat(meta).st_size > 0
                    if ('card-no-preview.png' in item['preview'] or not has_meta) and os.path.isfile(item['filename']):
                        sha = item.get('hash', None)
                        found = False
                        if sha is not None and len(sha) > 0:
                            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{sha}')
                            log.debug(f'CivitAI search: name="{item["name"]}" hash={sha} status={r.status_code}')
                            if r.status_code == 200:
                                d = r.json()
                                res.append(download_civit_meta(item['filename'], d['modelId']))
                                if d.get('images') is not None:
                                    for i in d['images']:
                                        preview_url = i['url']
                                        img_res = download_civit_preview(item['filename'], preview_url)
                                        res.append(img_res)
                                        if 'error' not in img_res:
                                            found = True
                                            break
                        if not found and rehash and os.stat(item['filename']).st_size < (1024 * 1024 * 1024):
                            sha = hashes.calculate_sha256(item['filename'], quiet=True)[:10]
                            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{sha}')
                            log.debug(f'CivitAI search: name="{item["name"]}" hash={sha} status={r.status_code}')
                            if r.status_code == 200:
                                d = r.json()
                                res.append(download_civit_meta(item['filename'], d['modelId']))
                                if d.get('images') is not None:
                                    for i in d['images']:
                                        preview_url = i['url']
                                        img_res = download_civit_preview(item['filename'], preview_url)
                                        res.append(img_res)
                                        if 'error' not in img_res:
                                            found = True
                                            break

                def civit_search_metadata(rehash, title):
                    log.debug(f'CivitAI search metadata: {title if type(title) == str else "all"}')
                    from modules.ui_extra_networks import get_pages
                    res = []
                    i = 0
                    t0 = time.time()
                    candidates = []
                    for page in get_pages():
                        if type(title) == str:
                            if page.title != title:
                                continue
                        if page.name == 'style':
                            continue
                        for item in page.list_items():
                            i += 1
                            candidates.append(item)
                            # atomic_civit_search_metadata(item, res, rehash)
                    import concurrent
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        for fn in candidates:
                            executor.submit(atomic_civit_search_metadata, fn, res, rehash)
                    atomic_civit_search_metadata(None, res, rehash)
                    t1 = time.time()
                    log.debug(f'CivitAI search metadata: items={i} time={t1-t0:.2f}')
                    txt = '<br>'.join([r for r in res if len(r) > 0])
                    return txt

                global search_metadata_civit  # pylint: disable=global-statement
                search_metadata_civit = civit_search_metadata

                def civitai_update_token(token):
                    log.debug('CivitAI update token')
                    opts.civitai_token = token
                    opts.save()

                with gr.Row():
                    gr.HTML('<h2>&nbspCivitAI fetch metadata<br></h2>')
                    gr.HTML('Fetches preview and metadata information for all models with missing information<br>Models with existing previews and information are not updated<br>')
                with gr.Row():
                    civit_previews_btn = gr.Button(value="Start", variant='primary')
                with gr.Row():
                    civit_previews_rehash = gr.Checkbox(value=True, label="Check alternative hash")

                with gr.Row():
                    gr.HTML('<h2>Search for models</h2>')
                with gr.Row():
                    with gr.Column(scale=1):
                        civit_model_type = gr.Dropdown(label='CivitAI model type', choices=['Model', 'LoRA', 'Embedding', 'VAE', 'Other'], value='Model')
                    with gr.Column(scale=15):
                        with gr.Row():
                            civit_search_text = gr.Textbox('', label='Search models', placeholder='keyword')
                            civit_search_tag = gr.Textbox('', label='', placeholder='tags')
                            civit_search_btn = ToolButton(value=ui_symbols.search, label="Search", interactive=True)
                        with gr.Row():
                            civit_search_res = gr.HTML('')
                with gr.Row():
                    gr.HTML('<h2>&nbspCivitAI download model<br></h2>')
                with gr.Row():
                    civit_download_model_btn = gr.Button(value="Download", variant='primary')
                    gr.HTML('<span style="line-height: 2em">Select a model, model version and and model variant from the search results to download or enter model URL manually</span><br>')
                with gr.Row():
                    civit_token = gr.Textbox(opts.civitai_token, label='CivitAI token', placeholder='optional access token for private or gated models')
                    civit_token.change(fn=civitai_update_token, inputs=[civit_token], outputs=[])
                with gr.Row():
                    civit_name = gr.Textbox('', label='Model name', placeholder='select model from search results', visible=True)
                    civit_selected = gr.Textbox('', label='Model URL', placeholder='select model from search results', visible=True)
                    civit_path = gr.Textbox('', label='Download path', placeholder='optional subfolder path where to save model', visible=True)
                with gr.Row():
                    gr.HTML('<h2>Search results</h2>')
                with gr.Row():
                    civit_headers1 = ['ID', 'Name', 'Tags', 'Downloads', 'Rating']
                    civit_types1 = ['number', 'str', 'str', 'number', 'number']
                    civit_results1 = gr.DataFrame(value=None, label=None, show_label=False, interactive=False,
                                                  wrap=True, overflow_row_behaviour='paginate', max_rows=10,
                                                  headers=civit_headers1, datatype=civit_types1, type='array',
                                                  visible=False)
                with gr.Row():
                    with gr.Column():
                        civit_headers2 = ['ID', 'ModelID', 'Name', 'Base', 'Created', 'Preview']
                        civit_types2 = ['number', 'number', 'str', 'str', 'date', 'str']
                        civit_results2 = gr.DataFrame(value=None, label='Model versions', show_label=True,
                                                      interactive=False, wrap=True, overflow_row_behaviour='paginate',
                                                      max_rows=10, headers=civit_headers2, datatype=civit_types2,
                                                      type='array', visible=False)
                    with gr.Column():
                        civit_headers3 = ['Name', 'Size', 'Metadata', 'URL']
                        civit_types3 = ['str', 'number', 'str', 'str']
                        civit_results3 = gr.DataFrame(value=None, label='Model variants', show_label=True,
                                                      interactive=False, wrap=True, overflow_row_behaviour='paginate',
                                                      max_rows=10, headers=civit_headers3, datatype=civit_types3,
                                                      type='array', visible=False)

                def is_visible(component):
                    visible = len(component) > 0 if component is not None else False
                    return gr.update(visible=visible)

                civit_search_text.submit(fn=civit_search_model, inputs=[civit_search_text, civit_search_tag, civit_model_type], outputs=[civit_search_res, civit_results1, civit_results2, civit_results3])
                civit_search_tag.submit(fn=civit_search_model, inputs=[civit_search_text, civit_search_tag, civit_model_type], outputs=[civit_search_res, civit_results1, civit_results2, civit_results3])
                civit_search_btn.click(fn=civit_search_model, inputs=[civit_search_text, civit_search_tag, civit_model_type], outputs=[civit_search_res, civit_results1, civit_results2, civit_results3])
                civit_results1.select(fn=civit_select1, inputs=[civit_results1], outputs=[civit_results2, civit_results3, models_image])
                civit_results2.select(fn=civit_select2, inputs=[civit_results2], outputs=[civit_results3])
                civit_results3.select(fn=civit_select3, inputs=[civit_results3], outputs=[civit_selected, civit_name, civit_search_btn])
                civit_results1.change(fn=is_visible, inputs=[civit_results1], outputs=[civit_results1])
                civit_results2.change(fn=is_visible, inputs=[civit_results2], outputs=[civit_results2])
                civit_results3.change(fn=is_visible, inputs=[civit_results3], outputs=[civit_results3])
                civit_download_model_btn.click(fn=civit_download_model, inputs=[civit_selected, civit_name, civit_path, civit_model_type, civit_token], outputs=[models_outcome])
                civit_previews_btn.click(fn=civit_search_metadata, inputs=[civit_previews_rehash, civit_previews_rehash], outputs=[models_outcome])

            with gr.Tab(label="Update"):
                with gr.Row():
                    gr.HTML('<h2>&nbspScan CivitAI for information on latest available model versions<br></h2>')
                with gr.Row():
                    civit_update_btn = gr.Button(value="Update", variant='primary')
                with gr.Row():
                    gr.HTML('<h2>Update scan results</h2>')
                with gr.Row():
                    civit_headers4 = ['ID', 'File', 'Name', 'Versions', 'Current', 'Latest', 'Update']
                    civit_types4 = ['number', 'str', 'str', 'number', 'str', 'str', 'str']
                    civit_widths4 = ['10%', '25%', '25%', '5%', '10%', '10%', '15%']
                    civit_results4 = gr.DataFrame(value=None, label=None, show_label=False, interactive=False, wrap=True, overflow_row_behaviour='paginate',
                                                  row_count=20, max_rows=100, headers=civit_headers4, datatype=civit_types4, type='array', column_widths=civit_widths4)
                with gr.Row():
                    gr.HTML('<h3>Select model from the list and download update if available</h3>')
                with gr.Row():
                    civit_update_download_btn = gr.Button(value="Download", variant='primary', visible=False)

                class CivitModel:
                    def __init__(self, name, fn, sha = None, meta = {}):
                        self.name = name
                        self.id = meta.get('id', 0)
                        self.fn = fn
                        self.sha = sha
                        self.meta = meta
                        self.versions = 0
                        self.vername = ''
                        self.latest = ''
                        self.latest_hashes = []
                        self.latest_name = ''
                        self.url = None
                        self.status = 'Not found'
                    def array(self):
                        return [self.id, self.fn, self.name, self.versions, self.vername, self.latest, self.status]

                selected_model: CivitModel = None
                update_data = []

                def civit_update_metadata():
                    nonlocal update_data
                    log.debug('CivitAI update metadata: models')
                    from modules import ui_extra_networks, modelloader
                    res = []
                    pages = ui_extra_networks.get_pages('Model')
                    if len(pages) == 0:
                        return 'CivitAI update metadata: no models found'
                    page: ui_extra_networks.ExtraNetworksPage = pages[0]
                    table_data = []
                    update_data.clear()
                    all_hashes = [(item.get('hash', None) or 'XXXXXXXX').upper()[:8] for item in page.list_items()]
                    for item in page.list_items():
                        model = CivitModel(name=item['name'], fn=item['filename'], sha=item.get('hash', None), meta=item.get('metadata', {}))
                        if model.sha is None or len(model.sha) == 0:
                            res.append(f'CivitAI skip search: name="{model.name}" hash=None')
                        else:
                            r = req(f'https://civitai.com/api/v1/model-versions/by-hash/{model.sha}')
                            res.append(f'CivitAI search: name="{model.name}" hash={model.sha} status={r.status_code}')
                            if r.status_code == 200:
                                d = r.json()
                                model.id = d['modelId']
                                modelloader.download_civit_meta(model.fn, model.id)
                                fn = os.path.splitext(item['filename'])[0] + '.json'
                                model.meta = readfile(fn, silent=True)
                                model.name = model.meta.get('name', model.name)
                                model.versions = len(model.meta.get('modelVersions', []))
                        versions = model.meta.get('modelVersions', [])
                        if len(versions) > 0:
                            model.latest = versions[0].get('name', '')
                            model.latest_hashes.clear()
                            for v in versions[0].get('files', []):
                                for h in v.get('hashes', {}).values():
                                    model.latest_hashes.append(h[:8].upper())
                        for ver in versions:
                            for f in ver.get('files', []):
                                for h in f.get('hashes', {}).values():
                                    if h[:8].upper() == model.sha[:8].upper():
                                        model.vername = ver.get('name', '')
                                        model.url = f.get('downloadUrl', None)
                                        model.latest_name = f.get('name', '')
                                        if model.vername == model.latest:
                                            model.status = 'Latest'
                                        elif any(map(lambda v: v in model.latest_hashes, all_hashes)): # pylint: disable=cell-var-from-loop # noqa: C417
                                            model.status = 'Downloaded'
                                        else:
                                            model.status = 'Available'
                                        break
                        log.debug(res[-1])
                        update_data.append(model)
                        table_data.append(model.array())
                        yield gr.update(value=table_data), '<br>'.join([r for r in res if len(r) > 0])
                    return '<br>'.join([r for r in res if len(r) > 0])

                def civit_update_select(evt: gr.SelectData, in_data):
                    nonlocal selected_model, update_data
                    try:
                        selected_model = next([m for m in update_data if m.fn == in_data[evt.index[0]][1]])
                    except Exception:
                        selected_model = None
                    if selected_model is None or selected_model.url is None or selected_model.status != 'Available':
                        return [gr.update(value='Model update not available'), gr.update(visible=False)]
                    else:
                        return [gr.update(), gr.update(visible=True)]

                def civit_update_download():
                    if selected_model is None or selected_model.url is None or selected_model.status != 'Available':
                        return 'Model update not available'
                    if selected_model.latest_name is None or len(selected_model.latest_name) == 0:
                        model_name = f'{selected_model.name} {selected_model.latest}.safetensors'
                    else:
                        model_name = selected_model.latest_name
                    return civit_download_model(selected_model.url, model_name, model_path='', model_type='Model')

                civit_update_btn.click(fn=civit_update_metadata, inputs=[], outputs=[civit_results4, models_outcome])
                civit_results4.select(fn=civit_update_select, inputs=[civit_results4], outputs=[models_outcome, civit_update_download_btn])
                civit_update_download_btn.click(fn=civit_update_download, inputs=[], outputs=[models_outcome])

            if native:
                from modules.lora.lora_extract import create_ui as lora_extract_ui
                lora_extract_ui()

            for ui in extra_ui:
                if callable(ui):
                    ui()
