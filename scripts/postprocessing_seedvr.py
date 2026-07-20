from typing import TYPE_CHECKING
import gradio as gr
from modules import scripts_postprocessing
if TYPE_CHECKING:
    from modules.postprocess.seedvr_model import UpscalerSeedVR


class ScriptSeedVR(scripts_postprocessing.ScriptPostprocessing):
    name = "SeedVR"

    def ui(self):
        from modules.postprocess.seedvr_model import MODELS_MAP
        with gr.Accordion(self.name, open = False, elem_id="postprocess_seedvr_accordion"):
            with gr.Row():
                seedvr_enabled = gr.Checkbox(label="Enable SeedVR", value=False, elem_id="extras_seedvr_enabled")
                seedvr_offload = gr.Checkbox(label="Offload model", value=True, elem_id="extras_seedvr_offload")
            with gr.Row():
                seedvr_selected = gr.Dropdown(label="SeedVR model", choices=list(MODELS_MAP.keys()), value=list(MODELS_MAP.keys())[0], elem_id="extras_seedvr_model")
            with gr.Row():
                seedvr_scale = gr.Slider(minimum=1, maximum=16, step=0.1, value=2, label="SeedVR scale", elem_id="extras_seedvr_scale")
                seedvr_steps = gr.Slider(step=1, value=1, minimum=1, maximum=99, label="SeedVR steps", elem_id="extras_seedvr_steps")
            with gr.Row():
                seedvr_seed = gr.Number(step=1, value=-1, label="SeedVR seed", elem_id="extras_seedvr_seed")
            with gr.Row():
                seedvr_cfg_scale = gr.Slider(minimum=0.0, maximum=15.0, step=0.01, value=1.5, label="SeedVR guidance scale", elem_id="extras_seedvr_cfg_scale")
                seedvr_cfg_rescale = gr.Slider(minimum=0.0, maximum=15.0, step=0.01, value=0.0, label="SeedVR guidance rescale", elem_id="extras_seedvr_cfg_rescale")
            with gr.Row():
                seedvr_tile_size = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="SeedVR tile size", elem_id="extras_seedvr_tile_size")
                seedvr_tile_overlap = gr.Slider(minimum=0, maximum=1.0, step=0.01, value=0.25, label="SeedVR tile overlap", elem_id="extras_seedvr_tile_overlap")
            with gr.Row():
                seedvr_batch_size = gr.Slider(minimum=1, maximum=64, step=1, value=1, label="SeedVR batch size", elem_id="extras_seedvr_batch_size")
                seedvr_batch_overlap = gr.Slider(minimum=0, maximum=16, step=1, value=0, label="SeedVR batch overlap", elem_id="extras_seedvr_batch_overlap")
        return {
            "seedvr_enabled": seedvr_enabled,
            "seedvr_selected": seedvr_selected,
            "seedvr_scale": seedvr_scale,
            "seedvr_seed": seedvr_seed,
            "seedvr_steps": seedvr_steps,
            "seedvr_cfg_scale": seedvr_cfg_scale,
            "seedvr_cfg_rescale": seedvr_cfg_rescale,
            "seedvr_tile_size": seedvr_tile_size,
            "seedvr_tile_overlap": seedvr_tile_overlap,
            "seedvr_batch_size": seedvr_batch_size,
            "seedvr_batch_overlap": seedvr_batch_overlap,
            "seedvr_offload": seedvr_offload,
        }

    def process(self,
                pp: scripts_postprocessing.PostprocessedImage,
                seedvr_enabled: bool,
                seedvr_selected: str,
                seedvr_scale: int,
                seedvr_seed: int,
                seedvr_steps: int,
                seedvr_cfg_scale: float,
                seedvr_cfg_rescale: float,
                seedvr_tile_size: int,
                seedvr_tile_overlap: float,
                seedvr_batch_size: int,
                seedvr_batch_overlap: int,
                seedvr_offload: bool
               ): # pylint: disable=arguments-differ
        if not seedvr_enabled:
            return
        from modules import shared, upscaler
        from modules.logger import log
        _input = pp.image or pp.video
        if _input is None:
            return
        instance: upscaler.UpscalerData = next(iter([x for x in shared.sd_upscalers if x.name == seedvr_selected]), None)
        scaler: UpscalerSeedVR = instance.scaler

        log.info(f'Upscaler: type="SeedVR" model="{seedvr_selected}" scale={seedvr_scale} seed={seedvr_seed} steps={seedvr_steps} cfg_scale={seedvr_cfg_scale} cfg_rescale={seedvr_cfg_rescale} tile_size={seedvr_tile_size} tile_overlap={seedvr_tile_overlap} batch_size={seedvr_batch_size} batch_overlap={seedvr_batch_overlap}')

        jobid = shared.state.begin('Upscale')

        scaler.scale = float(seedvr_scale)
        upscaled = scaler.do_upscale(_input,
                                     seedvr_selected,
                                     cfg_scale=seedvr_cfg_scale,
                                     cfg_rescale=seedvr_cfg_rescale,
                                     steps=seedvr_steps,
                                     seed=seedvr_seed,
                                     tile_size=seedvr_tile_size,
                                     tile_overlap=seedvr_tile_overlap,
                                     batch_size=seedvr_batch_size,
                                     batch_overlap=seedvr_batch_overlap,
                                     offload=seedvr_offload
                                    )
        shared.state.end(jobid)

        if isinstance(upscaled, str):
            pp.video = upscaled
        else:
            pp.image = upscaled
        pp.info["SeedVR"] = f"Scale={seedvr_scale} Seed={seedvr_seed} CFG Scale={seedvr_cfg_scale} CFG Rescale={seedvr_cfg_rescale}"
