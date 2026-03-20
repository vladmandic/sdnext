import os
import gradio as gr
from PIL import Image
from modules.scripts_postprocessing import ScriptPostprocessing, PostprocessedImage


models = [
    "none",
    "ben2",
    "silueta",
    "u2net",
    "u2net_human_seg",
    "isnet-general-use",
    "isnet-anime",
    # "u2netp",
    # "u2net_cloth_seg",
    # "sam",
]


class ScriptPostprocessingUpscale(ScriptPostprocessing):
    name = "Remove background"
    order = 20000
    model = None

    def ui(self):
        def visible(model):
            return gr.update(visible=model not in ["none"]), gr.update(visible=model in ["ben2"]), gr.update(visible=model not in ["none", "ben2"])

        with gr.Accordion('Remove background', open = True, elem_id="postprocess_rembg_accordion"):
            with gr.Row():
                model = gr.Dropdown(label="Model", choices=models, value="none", elem_id="extras_rembg_model")
            with gr.Group(visible=False) as group_base:
                with gr.Row():
                    merge_alpha = gr.Checkbox(label="Merge alpha", value=False)
                    mask_only = gr.Checkbox(label="Mask only", value=False)
            with gr.Group(visible=False) as group_ben2:
                with gr.Row():
                    refine = gr.Checkbox(label="Refine foreground", value=False)
            with gr.Group(visible=False) as group_rembg:
                with gr.Row():
                    postprocess_mask = gr.Checkbox(label="Postprocess mask", value=False, elem_id="extras_rembg_process_mask")
                    alpha_matting = gr.Checkbox(label="Alpha matting", value=False, elem_id="extras_rembg_alpha")
                with gr.Row(visible=True) as alpha_mask_row:
                    alpha_matting_erode_size = gr.Slider(label="Erode size", minimum=0, maximum=40, step=1, value=10, elem_id="extras_rembg_alpha_erode")
                    alpha_matting_foreground_threshold = gr.Slider(label="Foreground threshold", minimum=0, maximum=255, step=1, value=240, elem_id="extras_rembg_alpha_foreground")
                    alpha_matting_background_threshold = gr.Slider(label="Background threshold", minimum=0, maximum=255, step=1, value=10, elem_id="extras_rembg_alpha_background")
                alpha_matting.change(fn=lambda x: gr.update(visible=x), inputs=[alpha_matting], outputs=[alpha_mask_row])
            model.change(fn=lambda x: visible(x), inputs=[model], outputs=[group_base, group_ben2, group_rembg])
            return {
                "model": model,
                "merge_alpha": merge_alpha,
                "refine": refine,
                "mask_only": mask_only,
                "postprocess_mask": postprocess_mask,
                "alpha_matting": alpha_matting,
                "alpha_matting_foreground_threshold": alpha_matting_foreground_threshold,
                "alpha_matting_background_threshold": alpha_matting_background_threshold,
                "alpha_matting_erode_size": alpha_matting_erode_size,
            }

    def process(self, pp: PostprocessedImage, model, merge_alpha, refine, mask_only, postprocess_mask, alpha_matting, alpha_matting_foreground_threshold, alpha_matting_background_threshold, alpha_matting_erode_size): # pylint: disable=arguments-differ
        from modules.logger import log
        if not model or model == "none":
            return pp
        if isinstance(pp, Image.Image):
            image = pp
            info = {}
        else:
            image = pp.image
            info = pp.info

        log.info(f'RemoveBackground: model={model} merge_alpha={merge_alpha} refine={refine} mask_only={mask_only} postprocess_mask={postprocess_mask} alpha_matting={alpha_matting} alpha_matting_foreground_threshold={alpha_matting_foreground_threshold} alpha_matting_background_threshold={alpha_matting_background_threshold} alpha_matting_erode_size={alpha_matting_erode_size}')
        if model == 'ben2':
            try:
                from modules.rembg import ben2
                image = ben2.remove(image, refine=refine)
            except Exception as e:
                log.error(f'RemoveBackground: model={model} {e}')
                return pp
        else:
            try:
                from installer import install
                for pkg in ["dctorch==0.1.2", "pymatting", "pooch", "rembg"]:
                    install(pkg, no_deps=True, ignore=False)
                import rembg
                if "U2NET_HOME" not in os.environ:
                    from modules.paths import models_path
                    os.environ["U2NET_HOME"] = os.path.join(models_path, "Rembg")
                image = rembg.remove(image,
                                    post_process_mask=postprocess_mask,
                                    alpha_matting=alpha_matting,
                                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                                    alpha_matting_erode_size=alpha_matting_erode_size,
                                    session=rembg.new_session(model))
            except Exception as e:
                log.error(f'RemoveBackground: model={model} {e}')
                return pp

        if mask_only and pp.image.mode == "RGBA":
            _r, _g, _b, alpha = pp.image.split()
            image = alpha
        if merge_alpha and pp.image.mode == "RGBA":
            flattened = Image.new("RGBA", pp.image.size, "BLACK")
            flattened.paste(pp.image, mask=pp.image)
            flattened.convert("RGB")
            image = flattened

        if isinstance(pp, Image.Image):
            pp = PostprocessedImage(image=image, info={ **info, 'Rembg': model })
        else:
            pp.image = image
            pp.info['Rembg'] = model
        return pp
