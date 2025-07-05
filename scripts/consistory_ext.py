"""
original code from <https://github.com/NVlabs/consistory>
ported to modules/consistory
- make it non-cuda exclusive
- separate create anchors and create extra
- do not force-load pipeline and unet, use existing model
- uses diffusers==0.25 class definitions, needed quite an update
- forces uses of xformers, converted attention calls to sdp
- unsafe tensor to numpy breaks with bfloat16
- removed debug print statements
"""
import time
import gradio as gr
import diffusers
from modules import scripts_manager, devices, errors, processing, shared, sd_models, sd_samplers


class Script(scripts_manager.Script):
    def __init__(self):
        super().__init__()
        self.anchor_cache_first_stage = None
        self.anchor_cache_second_stage = None

    def title(self):
        return 'ConsiStory: Consistent Image Generation'

    def show(self, is_img2img):
        return not is_img2img

    def reset(self):
        self.anchor_cache_first_stage = None
        self.anchor_cache_second_stage = None
        shared.log.debug('ConsiStory reset anchors')

    def ui(self, _is_img2img): # ui elements
        with gr.Row():
            gr.HTML('<a href="https://github.com/NVlabs/consistory">&nbsp ConsiStory: Consistent Image Generation</a><br>')
        with gr.Row():
            gr.HTML('<br> ▪ Anchors are created on first run<br> ▪ Subsequent generate will use anchors and apply to main prompt<br> ▪ Main prompts are separated by newlines')
        with gr.Row():
            subject = gr.Textbox(label="Subject", placeholder='short description of a subject', value='')
        with gr.Row():
            concepts = gr.Textbox(label="Concept Tokens", placeholder='one or more concepts to extract from subject', value='')
        with gr.Row():
            prompts = gr.Textbox(label="Anchor settings", lines=2, placeholder='two scene settings to place subject in', value='')
        with gr.Row():
            reset = gr.Button(value="Reset anchors", variant='primary')
            reset.click(fn=self.reset, inputs=[], outputs=[])
        with gr.Row():
            dropout = gr.Slider(label="Mask Dropout", minimum=0.0, maximum=1.0, step=0.1, value=0.5)
        with gr.Row():
            sampler = gr.Checkbox(label="Override sampler", value=True)
            steps = gr.Checkbox(label="Override steps", value=True)
        with gr.Row():
            same = gr.Checkbox(label="Same latent", value=False)
            queries = gr.Checkbox(label="Share queries", value=True)
        with gr.Row():
            sdsa = gr.Checkbox(label="Perform SDSA", value=True)
        with gr.Row():
            freeu = gr.Checkbox(label="Enable FreeU", value=False)
            freeu_preset = gr.Textbox(label="FreeU preset", value='0.6, 0.4, 1.1, 1.2')
        with gr.Row():
            injection = gr.Checkbox(label="Perform Injection", value=False)
            alpha = gr.Textbox(label="Alpha preset", value='10, 20, 0.8')
        return [subject, concepts, prompts, dropout, sampler, steps, same, queries, sdsa, freeu, freeu_preset, alpha, injection]

    def create_model(self):
        diffusers.models.embeddings.PositionNet = diffusers.models.embeddings.GLIGENTextBoundingboxProjection # patch as renamed in https://github.com/huggingface/diffusers/pull/6244/files
        import scripts.consistory as cs
        if shared.sd_model.__class__.__name__ != 'ConsistoryExtendAttnSDXLPipeline':
            shared.log.debug('ConsiStory init')
            t0 = time.time()
            state_dict = shared.sd_model.unet.state_dict() # save existing unet
            shared.sd_model = sd_models.switch_pipe(cs.ConsistoryExtendAttnSDXLPipeline, shared.sd_model)
            shared.sd_model.unet = cs.ConsistorySDXLUNet2DConditionModel.from_config(shared.sd_model.unet.config)
            shared.sd_model.unet.load_state_dict(state_dict) # now load it into new class
            shared.sd_model.unet.to(dtype=devices.dtype)
            state_dict = None
            # sd_models.set_diffuser_options(shared.sd_model)
            sd_models.move_model(shared.sd_model, devices.device)
            sd_models.move_model(shared.sd_model.unet, devices.device)
            t1 = time.time()
            shared.log.debug(f'ConsiStory load: model={shared.sd_model.__class__.__name__} time={t1-t0:.2f}')
        devices.torch_gc(force=True)

    def set_args(self, p: processing.StableDiffusionProcessing, *args):
        subject, concepts, prompts, dropout, sampler, steps, same, queries, sdsa, freeu, freeu_preset, alpha, injection = args # pylint: disable=unused-variable
        processing.fix_seed(p)
        if sampler:
            shared.sd_model.scheduler = diffusers.DDIMScheduler.from_config(shared.sd_model.scheduler.config)
        else:
            sd_samplers.create_sampler(p.sampler_name, shared.sd_model)
        if freeu:
            try:
                freeu_preset = [float(f.strip()) for f in freeu_preset.split(',')]
            except Exception:
                freeu_preset = []
                shared.log.warning(f'ConsiStory: freeu="{freeu_preset}" invalid')
            if len(freeu) == 4:
                shared.sd_model.enable_freeu(s1=freeu[0], s2=freeu[0], b1=freeu[0], b2=freeu[0])
        steps = 50 if steps else p.steps
        if injection:
            try:
                alpha = [a.strip() for a in alpha.split(',')]
                if len(alpha) == 3:
                    alpha = (int(alpha[0]), int(alpha[1]), float(alpha[2]))
            except Exception:
                alpha=(10, 20, 0.8)
                shared.log.warning(f'ConsiStory: alpha="{alpha}" invalid')
        else:
            alpha=(10, 20, 0.8)
        seed = p.seed
        concepts = [c.strip() for c in concepts.split(',') if c.strip() != '']
        for c in concepts:
            if c not in subject:
                shared.log.warning(f'ConsiStory: concept="{c}" not in subject')
                subject = f'{subject} {c}'
        settings = [p.strip() for p in prompts.split('\n') if p.strip() != '']
        anchors = [f'{subject} {p}' for p in settings]
        prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
        shared.prompt_styles.apply_styles_to_extra(p)
        p.styles = []
        prompts = [p.strip() for p in prompt.split('\n') if p.strip() != '']
        for i, prompt in enumerate(prompts):
            if subject not in prompt:
                prompts[i] = f'{subject} {prompt}'
        shared.log.debug(f'ConsiStory args: sampler={shared.sd_model.scheduler.__class__.__name__} steps={steps} sdsa={sdsa} queries={queries} same={same} dropout={dropout} freeu={freeu_preset if freeu else None} alpha={alpha if injection else None}')
        return concepts, anchors, prompts, alpha, steps, seed

    def create_anchors(self, anchors, concepts, seed, steps, dropout, same, queries, sdsa, injection, alpha):
        import scripts.consistory as cs
        t0 = time.time()
        if len(anchors) == 0:
            shared.log.warning('ConsiStory: no anchors')
            return []
        shared.log.debug(f'ConsiStory anchors: concepts={concepts} anchors={anchors}')
        with devices.inference_context():
            try:
                images, self.anchor_cache_first_stage, self.anchor_cache_second_stage = cs.run_anchor_generation(
                    story_pipeline=shared.sd_model,
                    prompts=anchors,
                    concept_token=concepts,
                    seed=seed,
                    n_steps=steps,
                    mask_dropout=dropout,
                    same_latent=same,
                    share_queries=queries,
                    perform_sdsa=sdsa,
                    inject_range_alpha=alpha,
                    perform_injection=injection,
                )
            except Exception as e:
                shared.log.error(f'ConsiStory: {e}')
                errors.display(e, 'ConsiStory')
                images = []
            devices.torch_gc()
        t1 = time.time()
        shared.log.debug(f'ConsiStory anchors: images={len(images)} time={t1-t0:.2f}')
        return images

    def create_extra(self, prompt, concepts, seed, steps, dropout, same, queries, sdsa, injection, alpha):
        import scripts.consistory as cs
        t0 = time.time()
        images = []
        shared.log.debug(f'ConsiStory extra: concepts={concepts} prompt="{prompt}"')
        with devices.inference_context():
            try:
                images = cs.run_extra_generation(
                    story_pipeline=shared.sd_model,
                    prompts=[prompt],
                    concept_token=concepts,
                    anchor_cache_first_stage=self.anchor_cache_first_stage,
                    anchor_cache_second_stage=self.anchor_cache_second_stage,
                    seed=seed,
                    n_steps=steps,
                    mask_dropout=dropout,
                    same_latent=same,
                    share_queries=queries,
                    perform_sdsa=sdsa,
                    inject_range_alpha=alpha,
                    perform_injection=injection,
                )
            except Exception as e:
                shared.log.error(f'ConsiStory: {e}')
                errors.display(e, 'ConsiStory')
                images = []
            devices.torch_gc()
        t1 = time.time()
        shared.log.debug(f'ConsiStory extra: images={len(images)} time={t1-t0:.2f}')
        return images

    def run(self, p: processing.StableDiffusionProcessing, *args): # pylint: disable=arguments-differ
        supported_model_list = ['sdxl']
        if shared.sd_model_type not in supported_model_list and shared.sd_model.__class__.__name__ != 'ConsistoryExtendAttnSDXLPipeline':
            shared.log.warning(f'ConsiStory: class={shared.sd_model.__class__.__name__} model={shared.sd_model_type} required={supported_model_list}')
            return None

        subject, concepts, prompts, dropout, sampler, steps, same, queries, sdsa, freeu, freeu_preset, alpha, injection = args # pylint: disable=unused-variable

        self.create_model() # create model if not already done
        concepts, anchors, prompts, alpha, steps, seed = self.set_args(p, *args) # set arguments

        images = []
        if self.anchor_cache_first_stage is None or self.anchor_cache_second_stage is None: # create anchors if not cached
            images = self.create_anchors(anchors, concepts, seed, steps, dropout, same, queries, sdsa, injection, alpha)

        for prompt in prompts:
            extra_out_images = self.create_extra(prompt, concepts, seed, steps, dropout, same, queries, sdsa, injection, alpha)
            for image in extra_out_images:
                images.append(image)

        shared.sd_model.disable_freeu()
        processed = processing.Processed(p, images)
        return processed

    def after(self, p: processing.StableDiffusionProcessing, processed: processing.Processed, *args): # pylint: disable=arguments-differ, unused-argument
        return processed
