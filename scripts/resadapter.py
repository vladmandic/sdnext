from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import gradio as gr
from modules import scripts_manager, processing, shared, sd_models, devices


repo = 'jiaxiangc/res-adapter'
models = {
    'None': '',
    'SD15 v2 general': 'resadapter_v2_sd1.5',
    'SDXL v2 general': 'resadapter_v2_sdxl',
    'SD15 v1 general': 'resadapter_v1_sd1.5',
    'SD15 v1 extrapolation': 'resadapter_v1_sd1.5_extrapolation',
    'SD15 v1 interpolation': 'resadapter_v1_sd1.5_interpolation',
    'SDXL v1 general': 'resadapter_v1_sdxl',
    'SDXL v1 extrapolation': 'resadapter_v1_sdxl_extrapolation',
    'SDXL v1 interpolation': 'resadapter_v1_sdxl_interpolation',
}

class Script(scripts_manager.Script):
    def title(self):
        return 'ResAdapter: Domain Consistent Resolution'

    def show(self, is_img2img):
        return not is_img2img if shared.native else False

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/bytedance/res-adapter">&nbsp ResAdapter: Domain Consistent Resolution</a><br>')
        with gr.Row():
            model = gr.Dropdown(label="Model", choices=list(models), value="None")
            weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, label="Weight", value=1.0)
        return [model, weight]

    def run(self, p: processing.StableDiffusionProcessing, model, weight): # pylint: disable=arguments-differ
        if not shared.native or model == 'None':
            return None
        if shared.sd_model_type == 'sd':
            if not model.startswith('SD15'):
                shared.log.warning(f'ResAdapter: pipeline={shared.sd_model_type} selected={model}')
                return None
        if shared.sd_model_type == 'sdxl':
            if not model.startswith('SDXL'):
                shared.log.warning(f'ResAdapter: pipeline={shared.sd_model_type} selected={model}')
                return None

        old_pipe = shared.sd_model
        shared.sd_model.load_lora_weights(hf_hub_download(repo_id=repo, subfolder=models[model], filename="pytorch_lora_weights.safetensors"), adapter_name="res_adapter")
        shared.sd_model.set_adapters(["res_adapter"], adapter_weights=[weight])
        shared.sd_model.unet.load_state_dict(load_file(hf_hub_download(repo_id=repo, subfolder=models[model], filename="diffusion_pytorch_model.safetensors")), strict=False)
        sd_models.move_model(shared.sd_model, devices.device) # move pipeline to device
        sd_models.set_diffuser_options(shared.sd_model, vae=None, op='model')
        shared.log.debug(f'ResAdapter: pipeline={shared.sd_model.__class__.__name__} model="{model}" weight={weight} file="{models[model]}"')
        processed = processing.process_images(p)
        shared.sd_model = old_pipe
        return processed
