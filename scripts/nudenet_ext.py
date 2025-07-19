# built-in imports and third party imports
import gradio as gr
# import required modules from sdnext
from modules import scripts, scripts_postprocessing, script_callbacks, processing, images # pylint: disable=import-error
# import actual nudenet module relative to extension root
from  scripts.nudenet import nudenet # pylint: disable=wrong-import-order
from  scripts.nudenet import langdetect # pylint: disable=wrong-import-order
from  scripts.nudenet import imageguard # pylint: disable=wrong-import-order
from  scripts.nudenet import bannedwords # pylint: disable=wrong-import-order


# main ui
def create_ui(accordion=True):
    def update_ui(checked):
        return gr.update(visible=checked)

    with gr.Accordion('NudeNet', open = False, elem_id='nudenet') if accordion else gr.Group():
        with gr.Row():
            enabled = gr.Checkbox(label = 'Enabled', value = False)
        with gr.Group(visible=False) as gr_censor:
            with gr.Row():
                copy = gr.Checkbox(label = 'Save as copy', value = False)
            with gr.Row():
                score = gr.Slider(label = 'Sensitivity', value = 0.2, mininimum = 0, maximum = 1, step = 0.01, interactive=True)
                blocks = gr.Slider(label = 'Block size', value = 3, minimum = 1, maximum = 10, step = 1, interactive=True)
            with gr.Row():
                censor = gr.Dropdown(label = 'Censor', value = [], choices = sorted(nudenet.labels), multiselect=True, interactive=True)
                method = gr.Dropdown(label = 'Method', value = 'pixelate', choices = ['none', 'pixelate', 'blur', 'image', 'block'], interactive=True)
            with gr.Row():
                overlay = gr.Textbox(label = 'Overlay', value = '', placeholder = 'Path to image or leave default', interactive=True)
        with gr.Row():
            metadata = gr.Checkbox(label = 'Add metadata', value = True)
        with gr.Row():
            lang = gr.Checkbox(label = 'Check language', value = False)
        with gr.Group(visible=False) as gr_lang:
            with gr.Row():
                allowed = gr.Textbox(label = 'Allowed languages', value = 'eng', placeholder = 'Comma separated list of allowed languages', interactive=True)
                alphabet = gr.Textbox(label = 'Allowed alphabets', value = 'latn', placeholder = 'Comma separated list of allowed alphabets', interactive=True)
        with gr.Row():
            policy = gr.Checkbox(label = 'Check policy violations', value = False)
        with gr.Row():
            banned = gr.Checkbox(label = 'Check banned words', value = False)
        with gr.Group(visible=False) as gr_banned:
            with gr.Row():
                words = gr.Textbox(label = 'Banned words', value = '', placeholder = 'Comma separated list of banned words', interactive=True)
        enabled.change(fn=update_ui, inputs=[enabled], outputs=[gr_censor])
        lang.change(fn=update_ui, inputs=[lang], outputs=[gr_lang])
        banned.change(fn=update_ui, inputs=[banned], outputs=[gr_banned])
    return [enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words]


# main processing used in both modes
def process(
        p: processing.StableDiffusionProcessing=None,
        pp: scripts.PostprocessImageArgs=None,
        enabled=True,
        lang=False,
        policy=False,
        banned=False,
        metadata=True,
        copy=False,
        score=0.2,
        blocks=3,
        censor=[],
        method='pixelate',
        overlay='',
        allowed='eng',
        alphabet='latn',
        words='',
    ):
    from modules.shared import state, log
    if enabled and p is not None and pp is not None and pp.image is not None:
        if nudenet.detector is None:
            nudenet.detector = nudenet.NudeDetector(providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # loads and initializes model once
        nudes = nudenet.detector.censor(image=pp.image, method=method, min_score=score, censor=censor, blocks=blocks, overlay=overlay)
        if len(nudes.censored) > 0:  # Check if there are any censored areas
            if not copy:
                pp.image = nudes.output
            else:
                info = processing.create_infotext(p)
                images.save_image(nudes.output, path=p.outpath_samples, seed=p.seed, prompt=p.prompt, info=info, p=p, suffix="-censored")
        meta = '; '.join([f'{d["label"]}:{d["score"]}' for d in nudes.detections]) # add all metadata
        nsfw = any([d["label"] in nudenet.nsfw for d in nudes.detections]) # noqa:C419
        if metadata and p is not None:
            p.extra_generation_params["NudeNet"] = meta
            p.extra_generation_params["NSFW"] = nsfw
        if metadata and hasattr(pp, 'info'):
            pp.info['NudeNet'] = meta
            pp.info['NSFW'] = nsfw
        log.debug(f'NudeNet detect: {meta} nsfw={nsfw}')
    if lang and p is not None:
        prompts = '.\n'.join(p.all_prompts) if p.all_prompts else p.prompt
        allowed = [a.strip() for a in allowed.split(',')] if allowed else []
        alphabet = [a.strip() for a in alphabet.split(',')] if alphabet else []
        res = langdetect.lang_detect(prompts)
        res = ','.join(res) if isinstance(res, list) else res
        if len(allowed) > 0:
            if not any(a in res for a in allowed):
                log.error(f'NudeNet: lang={res} allowed={allowed} not allowed')
                state.interrupted = True
        if len(alphabet) > 0:
            if not any(a in res for a in alphabet):
                log.error(f'NudeNet: alphabet={res} allowed={alphabet} not allowed')
                state.interrupted = True
        if metadata and p is not None:
            p.extra_generation_params["Lang"] = res
    if banned and p is not None:
        prompts = '.\n'.join(p.all_prompts) if p.all_prompts else p.prompt
        found = bannedwords.check_banned(words=words, prompt=prompts)
        if len(found) > 0:
            log.error(f'NudeNet: banned={found}')
            state.interrupted = True
            if metadata and p is not None:
                p.extra_generation_params["Banned"] = ', '.join(found)
    if policy and p is not None and pp is not None and pp.image is not None:
        res = imageguard.image_guard(image=pp.image)
        if metadata and p is not None:
            p.extra_generation_params["Rating"] = res.get('rating', 'N/A')
            p.extra_generation_params["Category"] = res.get('category', 'N/A')
        if metadata and hasattr(pp, 'info'):
            pp.info["Rating"] = res.get('rating', 'N/A')
            pp.info["Category"] = res.get('category', 'N/A')


# defines script for dual-mode usage
class Script(scripts.Script):
    # see below for all available options and callbacks
    # <https://github.com/vladmandic/automatic/blob/master/modules/scripts.py#L26>

    def title(self):
        return 'NudeNet'

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    # return signature is array of gradio components
    def ui(self, _is_img2img):
        return create_ui(accordion=True)

    # triggered by callback
    def before_process(self, p: processing.StableDiffusionProcessing, enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words): # pylint: disable=arguments-differ
        process(p, None, enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words)

    # triggered by callback
    def postprocess_image(self, p: processing.StableDiffusionProcessing, pp: scripts.PostprocessImageArgs, enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words): # pylint: disable=arguments-differ
        process(p, pp, enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words)


# defines postprocessing script for dual-mode usage
class ScriptPostprocessing(scripts_postprocessing.ScriptPostprocessing):
    name = 'NudeNet'
    order = 10000

    # return signature is object with gradio components
    def ui(self):
        enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words = create_ui(accordion=True)
        return { 'enabled': enabled, 'lang': lang, 'policy': policy, 'banned': banned, 'metadata': metadata, 'copy': copy, 'score': score, 'blocks': blocks, 'censor': censor, 'method': method, 'overlay': overlay, 'allowed': allowed, 'alphabet': alphabet, 'words': words}

    # triggered by callback
    def process(self, pp: scripts_postprocessing.PostprocessedImage, enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words): # pylint: disable=arguments-differ
        process(None, pp, enabled, lang, policy, banned, metadata, copy, score, blocks, censor, method, overlay, allowed, alphabet, words)
