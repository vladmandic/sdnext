# https://github.com/mulanai/MuLan
# https://huggingface.co/mulanai/mulan-lang-adapter
# https://huggingface.co/OpenGVLab/InternVL-14B-224px

"""
- [MuLan](https://github.com/mulanai/MuLan) Multi-langunage prompts - wirte your prompts in ~110 auto-detected languages!
  Compatible with SD15 and SDXL
  Enable in scripts -> MuLan and set encoder to `InternVL-14B-224px` encoder
  (that is currently only supported encoder, but others will be added)
  Note: Model will be auto-downloaded on first use: note its huge size of 27GB
  Even executing it in FP16 context will require ~16GB of VRAM for text encoder alone
  *Note*: Uses fixed prompt parser, so no prompt attention will be used

Examples:
- English: photo of a beautiful woman wearing a white bikini on a beach with a city skyline in the background
- Croatian: fotografija lijepe žene u bijelom bikiniju na plaži s gradskim obzorom u pozadini
- Italian: Foto di una bella donna che indossa un bikini bianco su una spiaggia con lo skyline di una città sullo sfondo
- Spanish: Foto de una hermosa mujer con un bikini blanco en una playa con un horizonte de la ciudad en el fondo
- German: Foto einer schönen Frau in einem weißen Bikini an einem Strand mit einer Skyline der Stadt im Hintergrund
- Arabic: صورة لامرأة جميلة ترتدي بيكيني أبيض على شاطئ مع أفق المدينة في الخلفية
- Japanese: 街のスカイラインを背景にビーチで白いビキニを着た美しい女性の写真
- Chinese: 一个美丽的女人在海滩上穿着白色比基尼的照片, 背景是城市天际线
- Korean: 도시의 스카이라인을 배경으로 해변에서 흰색 비키니를 입은 아름 다운 여성의 사진
"""

import gradio as gr
from modules import shared, scripts_manager, processing, devices


ENCODERS =[
    # 'None',
    'OpenGVLab/InternVL-14B-224px',
    # 'OpenGVLab/InternViT-6B-224px',
    # 'OpenGVLab/InternViT-6B-448px-V1-0',
    # 'OpenGVLab/InternViT-6B-448px-V1-2',
    # 'OpenGVLab/InternViT-6B-448px-V1-5',
]
GITPATH = 'git+https://github.com/mulanai/MuLan'

pipe_type = None
adapter = None
text_encoder = None
tokenizer = None
text_encoder_path = None


class Script(scripts_manager.Script):
    def title(self):
        return 'MuLan: Multi Language Prompts'

    def show(self, is_img2img):
        return True

    def ui(self, _is_img2img):
        with gr.Row():
            gr.HTML('<a href="https://github.com/mulanai/MuLan">&nbsp MuLan: Multi Language Prompts</a><br>')
        with gr.Row():
            selected_encoder = gr.Dropdown(label='Encoder', choices=ENCODERS, value=ENCODERS[0])
        return [selected_encoder]

    def run(self, p: processing.StableDiffusionProcessing, selected_encoder): # pylint: disable=arguments-differ
        global pipe_type, adapter, text_encoder, tokenizer, text_encoder_path # pylint: disable=global-statement
        if not selected_encoder or selected_encoder == 'None':
            return None
        # create pipeline
        if shared.sd_model_type != 'sd' and shared.sd_model_type != 'sdxl':
            shared.log.error(f'MuLan: incorrect base model: {shared.sd_model.__class__.__name__}')
            return None

        adapter_path = None
        if shared.sd_model_type == 'sd':
            adapter_path = 'mulanai/mulan-lang-adapter::sd15_aesthetic.pth'
        if shared.sd_model_type == 'sdxl':
            adapter_path = 'mulanai/mulan-lang-adapter::sdxl_aesthetic.pth'
        if adapter_path is None:
            return None

        # install-on-demand
        import installer
        installer.install(GITPATH, 'mulankit')
        import mulankit

        # backup pipeline and params
        orig_pipeline = shared.sd_model
        orig_prompt_attention = shared.opts.prompt_attention

        # mulan only works with single image, single prompt and in fixed attention
        p.batch_size = 1
        p.n_iter = 1
        shared.opts.prompt_attention = 'fixed'
        if isinstance(p.prompt, list):
            p.prompt = p.prompt[0]
        p.task_args['prompt'] = p.prompt
        if isinstance(p.negative_prompt, list):
            p.prompt = p.negative_prompt[0]
        p.task_args['negative_prompt'] = p.negative_prompt

        if pipe_type != ('sd15' if shared.sd_model_type == 'sd' else 'sdxl'):
            pipe_type = 'sd15' if shared.sd_model_type == 'sd' else 'sdxl'
            adapter = None
        if text_encoder is None or tokenizer is None or text_encoder_path != selected_encoder:
            text_encoder_path = selected_encoder
            shared.log.debug(f'MuLan loading: encoder="{text_encoder_path}"')
            text_encoder = None
            tokenizer = None
            devices.torch_gc(force=True)
            text_encoder, tokenizer = mulankit.api.load_internvl(text_encoder_path, text_encoder, tokenizer, torch_dtype=shared.sd_model.text_encoder.dtype)
            devices.torch_gc(force=True)
        if adapter is None:
            shared.log.debug(f'MuLan loading: adapter="{adapter_path}"')
            adapter = None
            devices.torch_gc(force=True)
            adapter = mulankit.api.load_adapter(adapter_path, type=pipe_type)
            devices.torch_gc(force=True)

        if not getattr(shared.sd_model, 'mulan', False):
            shared.log.info(f'MuLan apply: adapter="{adapter_path}" encoder="{text_encoder_path}"')
            # mulankit.setup(force_sdxl_zero_empty_prompt=False, force_sdxl_zero_pool_prompt=False)
            shared.sd_model = mulankit.transform(shared.sd_model,
                adapter=adapter,
                adapter_path=adapter_path,
                text_encoder=text_encoder,
                text_encoder_path=text_encoder_path,
                pipe_type=pipe_type,
                replace=False)
            shared.sd_model.mulan = True
            devices.torch_gc(force=True)

        processing.fix_seed(p)
        processed: processing.Processed = processing.process_images(p) # runs processing using main loop

        # restore pipeline and params
        shared.opts.data['prompt_attention'] = orig_prompt_attention
        shared.sd_model = orig_pipeline
        return processed
