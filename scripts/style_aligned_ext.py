import gradio as gr
import torch
import numpy as np
import diffusers
from modules import scripts_manager, processing, shared, devices


handler = None
zts = None
supported_model_list = ['sdxl']
orig_prompt_attention = None


class Script(scripts_manager.Script):
    def title(self):
        return 'Style Aligned Image Generation'

    def show(self, is_img2img):
        return shared.native

    def reset(self):
        global handler, zts # pylint: disable=global-statement
        handler = None
        zts = None
        shared.log.info('SA: image upload')

    def preset(self, preset):
        if preset == 'text':
            return [['attention', 'adain_queries', 'adain_keys'], 1.0, 0, 0.0]
        elif preset == 'image':
            return [['group_norm', 'layer_norm', 'attention', 'adain_queries', 'adain_keys'], 1.0, 2, 0.0]
        else:
            return [['group_norm', 'layer_norm', 'attention', 'adain_queries', 'adain_keys', 'adain_values', 'full_attention_share'], 1.0, 1, 0.5]

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/google/style-aligned">&nbsp Style Aligned Image Generation</a><br><br>')
        with gr.Row():
            preset = gr.Dropdown(label="Preset", choices=['text', 'image', 'all'], value='text')
            scheduler = gr.Checkbox(label="Override scheduler", value=False)
        with gr.Row():
            shared_opts = gr.Dropdown(label="Shared options",
                                      multiselect=True,
                                      choices=['group_norm', 'layer_norm', 'attention', 'adain_queries', 'adain_keys', 'adain_values', 'full_attention_share'],
                                      value=['attention', 'adain_queries', 'adain_keys'],
                                    )
        with gr.Row():
            shared_score_scale = gr.Slider(label="Scale", minimum=0.0, maximum=2.0, step=0.01, value=1.0)
            shared_score_shift = gr.Slider(label="Shift", minimum=0, maximum=10, step=1, value=0)
            only_self_level = gr.Slider(label="Level", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
        with gr.Row():
            prompt = gr.Textbox(lines=1, label='Optional image description', placeholder='use the style from the image')
        with gr.Row():
            image = gr.Image(label='Optional image', type='pil')

        image.change(self.reset)
        preset.change(self.preset, inputs=[preset], outputs=[shared_opts, shared_score_scale, shared_score_shift, only_self_level])

        return [image, prompt, scheduler, shared_opts, shared_score_scale, shared_score_shift, only_self_level]

    def run(self, p: processing.StableDiffusionProcessing, image, prompt, scheduler, shared_opts, shared_score_scale, shared_score_shift, only_self_level): # pylint: disable=arguments-differ
        global handler, zts, orig_prompt_attention # pylint: disable=global-statement
        if shared.sd_model_type not in supported_model_list:
            shared.log.warning(f'SA: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None

        from scripts.style_aligned import sa_handler, inversion # pylint: disable=no-name-in-module

        handler = sa_handler.Handler(shared.sd_model)
        sa_args = sa_handler.StyleAlignedArgs(
            share_group_norm='group_norm' in shared_opts,
            share_layer_norm='layer_norm' in shared_opts,
            share_attention='attention' in shared_opts,
            adain_queries='adain_queries' in shared_opts,
            adain_keys='adain_keys' in shared_opts,
            adain_values='adain_values' in shared_opts,
            full_attention_share='full_attention_share' in shared_opts,
            shared_score_scale=float(shared_score_scale),
            shared_score_shift=np.log(shared_score_shift) if shared_score_shift > 0 else 0,
            only_self_level=1 if only_self_level else 0,
            )
        handler.register(sa_args)

        if scheduler:
            shared.sd_model.scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            p.sampler_name = 'None'

        if image is not None and zts is None:
            shared.log.info(f'SA: inversion image={image} prompt="{prompt}"')
            image = image.resize((1024, 1024))
            x0 = np.array(image).astype(np.float32) / 255.0
            shared.sd_model.scheduler = diffusers.DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            zts = inversion.ddim_inversion(shared.sd_model, x0, prompt, num_inference_steps=50, guidance_scale=2)

        p.prompt = p.prompt.splitlines()
        p.batch_size = len(p.prompt)
        orig_prompt_attention = shared.opts.prompt_attention
        shared.opts.data['prompt_attention'] = 'fixed' # otherwise need to deal with class_tokens_mask

        if zts is not None:
            processing.fix_seed(p)
            zT, inversion_callback = inversion.make_inversion_callback(zts, offset=0)
            generator = torch.Generator(device='cpu')
            generator.manual_seed(p.seed)
            latents = torch.randn(p.batch_size, 4, 128, 128, device='cpu', generator=generator, dtype=devices.dtype,).to(devices.device)
            latents[0] = zT
            p.task_args['latents'] = latents
            p.task_args['callback_on_step_end'] = inversion_callback

        shared.log.info(f'SA: batch={p.batch_size} type={"image" if zts is not None else "text"} config={sa_args.__dict__}')
        return None

    def after(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=unused-argument
        global handler # pylint: disable=global-statement
        if handler is not None:
            handler.remove()
            handler = None
            shared.opts.data['prompt_attention'] = orig_prompt_attention
