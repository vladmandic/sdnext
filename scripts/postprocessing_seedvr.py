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
                seedvr_scale = gr.Slider(minimum=1, maximum=16, step=0.1, value=2, label="SeedVR scale", elem_id="extras_seedvr_scale")
            with gr.Accordion('SeedVR advanced', open = False, elem_id="postprocess_seedvr_advanced_accordion"):
                with gr.Row():
                    seedvr_steps = gr.Slider(step=1, value=1, minimum=1, maximum=99, label="SeedVR steps", elem_id="extras_seedvr_steps")
                    seedvr_seed = gr.Number(step=1, value=-1, label="SeedVR seed", elem_id="extras_seedvr_seed")
                with gr.Row():
                    seedvr_cfg_scale = gr.Slider(minimum=0.0, maximum=15.0, step=0.01, value=1.5, label="SeedVR guidance scale", elem_id="extras_seedvr_cfg_scale")
                    seedvr_cfg_rescale = gr.Slider(minimum=0.0, maximum=15.0, step=0.01, value=0.0, label="SeedVR guidance rescale", elem_id="extras_seedvr_cfg_rescale")
            with gr.Accordion('SeedVR VAE', open = False, elem_id="postprocess_seedvr_vae_accordion"):
                with gr.Row():
                    seedvr_vae_tile_encode = gr.Checkbox(label="VAE tiled encode", value=True, elem_id="extras_seedvr_vae_tile_encode")
                    seedvr_vae_tile_decode = gr.Checkbox(label="VAE tiled decode", value=True, elem_id="extras_seedvr_vae_tile_decode")
                with gr.Row():
                    seedvr_tile_size = gr.Slider(minimum=64, maximum=4096, step=8, value=1024, label="SeedVR tile size", elem_id="extras_seedvr_tile_size")
                    seedvr_tile_overlap = gr.Slider(minimum=0, maximum=1.0, step=0.01, value=0.25, label="SeedVR tile overlap", elem_id="extras_seedvr_tile_overlap")
                with gr.Row():
                    seedvr_vae_memory = gr.Slider(minimum=0.1, maximum=1.0, step=0.01, value=1.0, label="SeedVR VAE memory", elem_id="extras_seedvr_vae_memory")
            with gr.Accordion('SeedVR video', open = False, elem_id="postprocess_seedvr_video_accordion"):
                with gr.Row():
                    seedvr_batch_size = gr.Slider(minimum=1, maximum=64, step=1, value=1, label="SeedVR batch size", elem_id="extras_seedvr_batch_size")
                    seedvr_batch_overlap = gr.Slider(minimum=0, maximum=16, step=1, value=0, label="SeedVR batch overlap", elem_id="extras_seedvr_batch_overlap")
                with gr.Row():
                    seedvr_interpolate = gr.Slider(label="RIFE interpolate frames", minimum=0, maximum=4, step=1, value=0, elem_id="extras_seedvr_interpolate")
                with gr.Row():
                    from modules.video_models.video_utils import get_codecs
                    from modules.ui_common import create_refresh_button
                    seedvr_codec = gr.Dropdown(label="Video codec", choices=['none', 'libx264'], value='libx264', type='value')
                    create_refresh_button(seedvr_codec, get_codecs, elem_id="video_mp4_codec_refresh")
                    seedvr_codec_opt = gr.Textbox(label="Video options", value='crf:16', elem_id="video_mp4_opt")

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
            "seedvr_interpolate": seedvr_interpolate,
            "seedvr_codec": seedvr_codec,
            "seedvr_codec_opt": seedvr_codec_opt,
            "seedvr_vae_memory": seedvr_vae_memory,
            "seedvr_vae_tile_encode": seedvr_vae_tile_encode,
            "seedvr_vae_tile_decode": seedvr_vae_tile_decode,
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
                seedvr_offload: bool,
                seedvr_interpolate: int,
                seedvr_codec: str,
                seedvr_codec_opt: str,
                seedvr_vae_memory: float,
                seedvr_vae_tile_encode: bool,
                seedvr_vae_tile_decode: bool,
               ): # pylint: disable=arguments-differ
        if not seedvr_enabled:
            return
        from modules import shared, upscaler
        _input = pp.image or pp.video
        if _input is None:
            return
        instance: upscaler.UpscalerData = next(iter([x for x in shared.sd_upscalers if x.name == seedvr_selected]), None)
        scaler: UpscalerSeedVR = instance.scaler

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
                                     offload=seedvr_offload,
                                     interpolate=seedvr_interpolate,
                                     codec=seedvr_codec,
                                     codec_opt=seedvr_codec_opt,
                                     vae_memory=seedvr_vae_memory,
                                     vae_tile_encode=seedvr_vae_tile_encode,
                                     vae_tile_decode=seedvr_vae_tile_decode,
                                    )
        shared.state.end(jobid)

        if isinstance(upscaled, str):
            pp.video = upscaled
        else:
            pp.image = upscaled
        pp.info["SeedVR"] = f"Scale={seedvr_scale} Seed={seedvr_seed} CFG Scale={seedvr_cfg_scale} CFG Rescale={seedvr_cfg_rescale}"
