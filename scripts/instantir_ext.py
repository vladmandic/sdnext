import gradio as gr
import torch
import diffusers
from huggingface_hub import hf_hub_download
from modules import scripts_manager, processing, shared, sd_models, devices, ipadapter


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.orig_pipe = None
        self.orig_ip_unapply = None

    def title(self):
        return 'InstantIR: Image Restoration'

    def show(self, is_img2img):
        return is_img2img if shared.native else False

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/instantX-research/InstantIR">&nbsp InstantIR: Image Restoration</a><br>')
        with gr.Row():
            start = gr.Slider(label='Preview start', minimum=0.0, maximum=1.0, step=0.01, value=0.0)
            end = gr.Slider(label='Preview end', minimum=0.0, maximum=1.0, step=0.01, value=1.0)
        with gr.Row():
            hq = gr.Checkbox(label='HQ init latents', value=False)
            multistep = gr.Checkbox(label='Multistep restore', value=False)
            adastep = gr.Checkbox(label='Adaptive restore', value=False)
        with gr.Row():
            image = gr.Image(label='Override guidance image')
        return [start, end, hq, multistep, adastep, image]

    def run(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        supported_model_list = ['sdxl']
        if not hasattr(p, 'init_images') or len(p.init_images) == 0:
            shared.log.warning('InstantIR: no image')
            return None
        if shared.sd_model_type not in supported_model_list and shared.sd_model.__class__.__name__ != "InstantIRPipeline":
            shared.log.warning(f'InstantIR: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None
        start, end, hq, multistep, adastep, image = args
        from scripts import instantir as ir
        if shared.sd_model_type == "sdxl":
            if shared.sd_model.__class__.__name__ != "InstantIRPipeline":
                self.orig_pipe = shared.sd_model
                self.orig_ip_unapply = ipadapter.unapply
                shared.sd_model = sd_models.switch_pipe(ir.InstantIRPipeline, shared.sd_model)
                adapter_file = hf_hub_download('InstantX/InstantIR', subfolder='models', filename='adapter.pt', cache_dir=shared.opts.hfcache_dir)
                aggregator_file = hf_hub_download('InstantX/InstantIR', subfolder='models', filename='aggregator.pt', cache_dir=shared.opts.hfcache_dir)
                previewer_file = hf_hub_download('InstantX/InstantIR', subfolder='models', filename='previewer_lora_weights.bin', cache_dir=shared.opts.hfcache_dir)
                shared.log.debug(f'InstantIR: adapter="{adapter_file}" aggregator="{aggregator_file}" previewer="{previewer_file}"')
                ir.load_adapter_to_pipe(
                    pipe=shared.sd_model,
                    pretrained_model_path_or_dict=adapter_file,
                    image_encoder_or_path='facebook/dinov2-large',
                    use_lcm=False,
                    use_adaln=True,
                    )
                shared.sd_model.prepare_previewers(previewer_file)
                shared.sd_model.scheduler = diffusers.DDPMScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder="scheduler")
                pretrained_state_dict = torch.load(aggregator_file)
                shared.sd_model.aggregator.load_state_dict(pretrained_state_dict)
                shared.sd_model.aggregator.to(device=devices.device, dtype=devices.dtype)

        shared.log.info(f'InstantIR: class={shared.sd_model.__class__.__name__} start={start} end={end} multistep={multistep} adastep={adastep} hq={hq} cache={shared.opts.hfcache_dir}')
        p.sampler_name = 'Default' # ir has its own sampler
        p.init() # run init early to take care of resizing
        p.task_args['previewer_scheduler'] = ir.LCMSingleStepScheduler.from_config(shared.sd_model.scheduler.config)
        p.task_args['image'] = p.init_images
        p.task_args['save_preview_row'] = False
        p.task_args['init_latents_with_lq'] = not hq
        p.task_args['multistep_restore'] = multistep
        p.task_args['adastep_restore'] = adastep
        p.task_args['preview_start'] = start
        p.task_args['preview_end'] = end
        p.task_args['ip_adapter_image'] = image
        p.extra_generation_params["InstantIR"] = f'Start={start} End={end} HQ={hq} Multistep={multistep} Adastep={adastep}'
        ipadapter.unapply = lambda x: x # disable as main processing unloads ipadapter as it thinks its not needed
        devices.torch_gc()

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args): # pylint: disable=arguments-differ, unused-argument
        # TODO instantir: a mess to unload
        """
        if self.orig_pipe is None:
            return processed
        if hasattr(shared.sd_model, 'aggregator'):
            shared.sd_model.aggregator = None
        shared.log.debug(f'InstantIR restore: class={shared.sd_model.__class__.__name__}')
        shared.sd_model = self.orig_pipe
        self.orig_pipe = None
        shared.sd_model.unet.register_to_config(encoder_hid_dim_type=None)
        ipadapter.unapply = self.orig_ip_unapply
        ipadapter.unapply(shared.sd_model)
        devices.torch_gc()
        """
        return processed
