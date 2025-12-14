# https://github.com/showlab/X-Adapter

import torch
import diffusers
import gradio as gr
import huggingface_hub as hf
from sdnext_core import MODELDATA
from modules import errors, shared, devices, scripts_manager, processing, sd_models, sd_samplers


adapter = None


class Script(scripts_manager.Script):
    def title(self):
        return 'X-Adapter'

    def show(self, is_img2img):
        return False

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/showlab/X-Adapter">&nbsp X-Adapter</a><br>')
        with gr.Row():
            model = gr.Dropdown(label='Adapter model', choices=['None'] + sd_models.checkpoint_titles(), value='None')
            sampler = gr.Dropdown(label='Adapter sampler', choices=[s.name for s in sd_samplers.samplers], value='Default')
        with gr.Row():
            width = gr.Slider(label='Adapter width', minimum=64, maximum=2048, step=8, value=1024)
            height = gr.Slider(label='Adapter height', minimum=64, maximum=2048, step=8, value=1024)
        with gr.Row():
            start = gr.Slider(label='Adapter start', minimum=0.0, maximum=1.0, step=0.01, value=0.5)
            scale = gr.Slider(label='Adapter scale', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        with gr.Row():
            lora = gr.Textbox('', label='Adapter LoRA', default='')
        return model, sampler, width, height, start, scale, lora

    def run(self, p: processing.StableDiffusionProcessing, model, sampler, width, height, start, scale, lora): # pylint: disable=arguments-differ, unused-argument
        from scripts.xadapter.xadapter_hijacks import PositionNet
        diffusers.models.embeddings.PositionNet = PositionNet # patch diffusers==0.26 from diffusers==0.20
        from scripts.xadapter.adapter import Adapter_XL
        from scripts.xadapter.pipeline_sd_xl_adapter import StableDiffusionXLAdapterPipeline
        from scripts.xadapter.unet_adapter import UNet2DConditionModel as UNet2DConditionModelAdapter

        global adapter # pylint: disable=global-statement
        if model == 'None':
            return
        else:
            shared.opts.sd_model_refiner = model
        if MODELDATA.sd_model_type != 'sdxl':
            shared.log.error(f'X-Adapter: incorrect base model: {MODELDATA.sd_model.__class__.__name__}')
            return

        if adapter is None:
            shared.log.debug('X-Adapter: adapter loading')
            adapter = Adapter_XL()
            adapter_path = hf.hf_hub_download(repo_id='Lingmin-Ran/X-Adapter', filename='X_Adapter_v1.bin')
            adapter_dict = torch.load(adapter_path)
            adapter.load_state_dict(adapter_dict)
            try:
                if adapter is not None:
                    sd_models.move_model(adapter, devices.device)
            except Exception:
                pass
        if adapter is None:
            shared.log.error('X-Adapter: adapter loading failed')
            return

        sd_models.unload_model_weights(op='model')
        sd_models.unload_model_weights(op='refiner')
        orig_unetcondmodel = diffusers.models.unets.unet_2d_condition.UNet2DConditionModel
        diffusers.models.UNet2DConditionModel = UNet2DConditionModelAdapter # patch diffusers with x-adapter
        diffusers.models.unets.unet_2d_condition.UNet2DConditionModel = UNet2DConditionModelAdapter # patch diffusers with x-adapter
        sd_models.reload_model_weights(op='model')
        sd_models.reload_model_weights(op='refiner')
        diffusers.models.unets.unet_2d_condition.UNet2DConditionModel = orig_unetcondmodel # unpatch diffusers
        diffusers.models.UNet2DConditionModel = orig_unetcondmodel # unpatch diffusers

        if MODELDATA.sd_refiner_type != 'sd':
            shared.log.error(f'X-Adapter: incorrect adapter model: {MODELDATA.sd_model.__class__.__name__}')
            return

        # backup pipeline and params
        orig_pipeline = MODELDATA.sd_model
        orig_prompt_attention = shared.opts.prompt_attention
        pipe = None

        try:
            shared.log.debug('X-Adapter: creating pipeline')
            pipe = StableDiffusionXLAdapterPipeline(
                vae=MODELDATA.sd_model.vae,
                text_encoder=MODELDATA.sd_model.text_encoder,
                text_encoder_2=MODELDATA.sd_model.text_encoder_2,
                tokenizer=MODELDATA.sd_model.tokenizer,
                tokenizer_2=MODELDATA.sd_model.tokenizer_2,
                unet=MODELDATA.sd_model.unet,
                scheduler=MODELDATA.sd_model.scheduler,
                vae_sd1_5=MODELDATA.sd_refiner.vae,
                text_encoder_sd1_5=MODELDATA.sd_refiner.text_encoder,
                tokenizer_sd1_5=MODELDATA.sd_refiner.tokenizer,
                unet_sd1_5=MODELDATA.sd_refiner.unet,
                scheduler_sd1_5=MODELDATA.sd_refiner.scheduler,
                adapter=adapter,
            )
            sd_models.copy_diffuser_options(pipe, MODELDATA.sd_model)
            sd_models.set_diffuser_options(pipe)
            try:
                pipe.to(device=devices.device, dtype=devices.dtype)
            except Exception:
                pass
            shared.opts.data['prompt_attention'] = 'fixed'
            prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
            negative = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
            shared.prompt_styles.apply_styles_to_extra(p)
            p.styles = []
            p.task_args['prompt'] = prompt
            p.task_args['negative_prompt'] = negative
            p.task_args['prompt_sd1_5'] = prompt
            p.task_args['width_sd1_5'] = width
            p.task_args['height_sd1_5'] = height
            p.task_args['adapter_guidance_start'] = start
            p.task_args['adapter_condition_scale'] = scale
            p.task_args['fusion_guidance_scale'] = 1.0 # ???
            if sampler != 'Default':
                pipe.scheduler_sd1_5 = sd_samplers.create_sampler(sampler, MODELDATA.sd_refiner)
            else:
                pipe.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                pipe.scheduler_sd1_5 = diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler_sd1_5.config)
                pipe.scheduler_sd1_5.config.timestep_spacing = "leading"
            shared.log.debug(f'X-Adapter: pipeline={pipe.__class__.__name__} args={p.task_args}')
            MODELDATA.sd_model = pipe
        except Exception as e:
            shared.log.error(f'X-Adapter: pipeline creation failed: {e}')
            errors.display(e, 'X-Adapter: pipeline creation failed')
            MODELDATA.sd_model = orig_pipeline

        # run pipeline
        processed: processing.Processed = processing.process_images(p) # runs processing using main loop

        # restore pipeline and params
        try:
            if adapter is not None:
                adapter.to(devices.cpu)
        except Exception:
            pass
        pipe = None
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        MODELDATA.sd_model = orig_pipeline
        devices.torch_gc()
        return processed
