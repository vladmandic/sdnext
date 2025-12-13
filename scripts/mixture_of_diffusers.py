import gradio as gr
from modules import scripts_manager, processing, shared, sd_models


supported_models = ['sdxl']
max_xtiles = 4
max_ytiles = 4


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.orig_attn = None

    def title(self):
        return 'Mixture-of-Diffusers: Tile Control'

    def show(self, is_img2img): # pylint: disable=unused-argument
        return True

    def update_ui(self, x_tiles, y_tiles):
        updates = []
        for x in range(max_xtiles):
            for y in range(max_ytiles):
                updates.append(gr.update(visible=(x < x_tiles) and (y < y_tiles)))
        return updates

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://huggingface.co/posts/elismasilva/251775641926329">&nbsp Mixture-of-Diffusers</a><br>')
        with gr.Row():
            gr.HTML('<span>&nbsp Use base prompt to define image background and common elements<br>&nbsp Set image width and height to final image size</span>')
        with gr.Row():
            x_tiles = gr.Slider(minimum=1, maximum=max_xtiles, step=1, value=1, label="X-axis tiles")
            y_tiles = gr.Slider(minimum=1, maximum=max_ytiles, step=1, value=1, label="Y-axis tiles")
        with gr.Row():
            x_overlap = gr.Slider(minimum=0, maximum=512, value=128, label="X-axis tile overlap")
            y_overlap = gr.Slider(minimum=0, maximum=512, value=128, label="Y-axis tile overlap")
        prompts = []
        for x in range(max_xtiles):
            for y in range(max_ytiles):
                with gr.Row():
                    prompts.append(gr.Textbox('', label=f"Tile prompt: x={x+1} y={y+1}", placeholder='Prompt for tile', visible=False, lines=2))
        x_tiles.change(fn=self.update_ui, inputs=[x_tiles, y_tiles], outputs=prompts)
        y_tiles.change(fn=self.update_ui, inputs=[x_tiles, y_tiles], outputs=prompts)
        return [x_tiles, y_tiles, x_overlap, y_overlap] + prompts

    def calc_size(self, size, tiles, overlap):
        tile_size = (size / tiles) + (overlap / 2) if tiles > 1 else size
        return 8 * int(tile_size // 8)

    def get_prompts(self, x_tiles, y_tiles, prompts, base_prompt, guidance):
        y_prompts = []
        y_guidance = []
        for y in range(max_ytiles):
            x_prompts = []
            x_guidance = []
            for x in range(max_xtiles):
                if (x < x_tiles) and (y < y_tiles):
                    prompt = prompts[x * max_xtiles + y] + ' ' + base_prompt
                    x_prompts.append(prompt.strip())
                    x_guidance.append(guidance)
            if len(x_prompts) > 0:
                y_prompts.append(x_prompts)
                y_guidance.append(x_guidance)
        return y_prompts, y_guidance

    def check_dependencies(self):
        from installer import install
        install('ligo-segments')
        try:
            from ligo.segments import segment # pylint: disable=unused-import
            return True
        except Exception as e:
            shared.log.error(f'MoD: {e}')
        return False

    def run(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ, unused-argument
        if MODELDATA.sd_model_type not in supported_models:
            shared.log.warning(f'MoD: class={MODELDATA.sd_model.__class__.__name__} model={MODELDATA.sd_model_type} required={supported_models}')
            return None
        if not self.check_dependencies():
            return None
        [x_tiles, y_tiles, x_overlap, y_overlap], prompts = args[:4], args[4:]
        if max(x_tiles, y_tiles) <= 1:
            return None
        from scripts.mod import StableDiffusionXLTilingPipeline # pylint: disable=no-name-in-module
        self.orig_pipe = MODELDATA.sd_model
        self.orig_attn = shared.opts.prompt_attention

        p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
        shared.prompt_styles.apply_styles_to_extra(p)
        p.styles = []
        p.prompts, guidance = self.get_prompts(x_tiles, y_tiles, prompts, p.prompt, p.cfg_scale)
        p.all_prompts = p.prompts
        p.task_args['prompts'] = p.prompts
        p.task_args['negative_prompt'] = p.negative_prompt
        p.task_args['tile_col_overlap'] = x_overlap if x_tiles > 1 else 0
        p.task_args['tile_row_overlap'] = y_overlap if y_tiles > 1 else 0
        p.task_args['tile_width'] = self.calc_size(p.width, x_tiles, x_overlap)
        p.task_args['tile_height'] = self.calc_size(p.height, y_tiles, y_overlap)
        p.task_args['guidance_scale_tiles'] = guidance
        p.task_args['width'] = p.width
        p.task_args['height'] = p.height
        p.extra_generation_params["MoD X"] = f'{x_tiles}/{p.task_args["tile_width"]}/{p.task_args["tile_col_overlap"]}'
        p.extra_generation_params["MoD Y"] = f'{y_tiles}/{p.task_args["tile_height"]}/{p.task_args["tile_row_overlap"]}'
        p.keep_prompts = True
        shared.opts.prompt_attention = 'fixed'
        shared.log.info(f'MoD: xtiles={x_tiles} ytiles={y_tiles} xoverlap={p.task_args["tile_col_overlap"]} yoverlap={p.task_args["tile_row_overlap"]} xsize={p.task_args["tile_width"]} ysize={p.task_args["tile_height"]}')

        MODELDATA.sd_model = sd_models.switch_pipe(StableDiffusionXLTilingPipeline, MODELDATA.sd_model)
        sd_models.set_diffuser_options(MODELDATA.sd_model)
        sd_models.apply_balanced_offload(MODELDATA.sd_model)
        return None

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args): # pylint: disable=arguments-differ, unused-argument
        if self.orig_pipe is None:
            return processed
        if MODELDATA.sd_model_type == "sdxl":
            MODELDATA.sd_model = self.orig_pipe
        if self.orig_attn is not None:
            shared.opts.prompt_attention = self.orig_attn
        self.orig_pipe = None
        self.orig_attn = None
        return processed
