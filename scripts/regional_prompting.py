# https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline
# https://github.com/huggingface/diffusers/blob/main/examples/community/regional_prompting_stable_diffusion.py

import gradio as gr
from diffusers.pipelines import pipeline_utils
from modules import shared, devices, scripts_manager, processing, sd_models, prompt_parser_diffusers


def hijack_register_modules(self, **kwargs):
    for name, module in kwargs.items():
        register_dict = None
        if module is None or (isinstance(module, (tuple, list)) and module[0] is None):
            register_dict = {name: (None, None)}
        elif isinstance(module, bool):
            pass
        else:
            library, class_name = pipeline_utils._fetch_class_library_tuple(module) # pylint: disable=protected-access
            register_dict = {name: (library, class_name)}
        if register_dict is not None:
            self.register_to_config(**register_dict)
        setattr(self, name, module)


class Script(scripts_manager.Script):
    def title(self):
        return 'Regional prompting'

    def show(self, is_img2img):
        return not is_img2img

    def change(self, mode):
        return [gr.update(visible='Col' in mode or 'Row' in mode), gr.update(visible='Prompt' in mode)]

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#regional-prompting-pipeline">&nbsp Regional prompting</a><br>')
        with gr.Row():
            mode = gr.Radio(label='Mode', choices=['None', 'Prompt', 'Prompt EX', 'Columns', 'Rows'], value='None')
        with gr.Row():
            power = gr.Slider(label='Power', minimum=0, maximum=1, value=1.0, step=0.01)
            threshold = gr.Textbox('', label='Prompt thresholds', visible=False)
            grid = gr.Textbox('', label='Grid sections', visible=False)
        mode.change(fn=self.change, inputs=[mode], outputs=[grid, threshold])
        return mode, grid, power, threshold

    def run(self, p: processing.StableDiffusionProcessing, mode, grid, power, threshold): # pylint: disable=arguments-differ
        if mode is None or mode == 'None':
            return None
        # backup pipeline and params
        orig_pipeline = MODELDATA.sd_model
        orig_dtype = devices.dtype
        orig_prompt_attention = shared.opts.prompt_attention
        # create pipeline
        if MODELDATA.sd_model_type != 'sd':
            shared.log.error(f'Regional prompting: incorrect base model: {MODELDATA.sd_model.__class__.__name__}')
            return None

        pipeline_utils.DiffusionPipeline.register_modules = hijack_register_modules
        prompt_parser_diffusers.EmbeddingsProvider._encode_token_ids_to_embeddings = prompt_parser_diffusers.orig_encode_token_ids_to_embeddings # pylint: disable=protected-access

        MODELDATA.sd_model = sd_models.switch_pipe('regional_prompting_stable_diffusion', MODELDATA.sd_model)
        if MODELDATA.sd_model.__class__.__name__ != 'RegionalPromptingStableDiffusionPipeline': # switch failed
            shared.log.error(f'Regional prompting: not a tiling pipeline: {MODELDATA.sd_model.__class__.__name__}')
            MODELDATA.sd_model = orig_pipeline
            return None
        sd_models.set_diffuser_options(MODELDATA.sd_model)
        shared.opts.data['prompt_attention'] = 'fixed' # this pipeline is not compatible with embeds
        processing.fix_seed(p)
        # set pipeline specific params, note that standard params are applied when applicable
        rp_args = {
            'mode': mode.lower(),
            'power': power,
        }
        if 'prompt' in mode.lower():
            rp_args['th'] = threshold
        else:
            rp_args['div'] = grid
        p.task_args = {
            **p.task_args,
            'prompt': p.prompt,
            'rp_args': rp_args,
        }
        # run pipeline
        shared.log.debug(f'Regional: args={p.task_args}')
        p.task_args['prompt'] = p.prompt
        processed: processing.Processed = processing.process_images(p) # runs processing using main loop

        # restore pipeline and params
        prompt_parser_diffusers.EmbeddingsProvider._encode_token_ids_to_embeddings = prompt_parser_diffusers.compel_hijack # pylint: disable=protected-access
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        MODELDATA.sd_model = orig_pipeline
        MODELDATA.sd_model.to(orig_dtype)
        return processed
