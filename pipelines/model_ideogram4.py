import json
import diffusers
from transformers.models.qwen3_vl import Qwen3VLModel
from modules import shared, devices, sd_models, model_quant
from modules.logger import log
from pipelines import generic


def prompt_to_json(prompt):
    """Normalize a JSON caption to the compact form Ideogram 4 trained on, or wrap plain text.

    Ideogram 4 expects a structured JSON caption serialized compactly. A valid JSON prompt is
    re-serialized to that form; a plain-text prompt is wrapped in a minimal caption so it stays
    in distribution instead of tripping the weight-baked "blocked by safety filter" placeholder.
    """
    if isinstance(prompt, list):
        return [prompt_to_json(p) for p in prompt]
    if not isinstance(prompt, str) or len(prompt) == 0:
        return prompt
    try:
        return json.dumps(json.loads(prompt), ensure_ascii=False, separators=(',', ':'))
    except ValueError:
        caption = {'high_level_description': prompt, 'compositional_deconstruction': {'background': prompt, 'elements': []}}
        return json.dumps(caption, ensure_ascii=False, separators=(',', ':'))


class Ideogram4Pipeline(diffusers.Ideogram4Pipeline):
    """SD.Next integration subclass for the diffusers-native Ideogram 4 pipeline.

    ``encode_prompt`` normalizes the prompt into the structured JSON the model expects, then
    drives the Qwen3-VL tap. The tap calls ``language_model`` submodules directly, bypassing the
    balanced-offload pre-forward hook, so the encoder is moved on-device for it and released after.
    """

    def encode_prompt(self, prompt, *args, **kwargs):
        prompt = prompt_to_json(prompt)
        self.text_encoder.to(self._execution_device)
        try:
            return super().encode_prompt(prompt, *args, **kwargs)
        finally:
            if shared.opts.diffusers_offload_mode != 'none':
                self.text_encoder.to(devices.cpu)


def pin_transformers(transformer, unconditional_transformer) -> bool:
    """Keep both transformers resident under balanced offload when they fit the budget.

    Every denoise step runs both transformers, so balanced offload ping-pongs them across
    PCIe each step. ``offload_never`` makes ``offload_allowed`` skip the per-step pre-sweep so
    they stay resident, but only when both fit ``gpu_memory * max watermark`` (which leaves the
    watermark headroom for activations); otherwise the normal offload path is kept.
    """
    if shared.opts.diffusers_offload_mode != 'balanced' or shared.gpu_memory <= 0:
        return False
    if transformer is None or unconditional_transformer is None:
        return False
    size_gb = sum(p.numel() * p.element_size() for m in (transformer, unconditional_transformer) for p in m.parameters()) / (1024 ** 3)
    budget_gb = shared.gpu_memory * shared.opts.diffusers_offload_max_gpu_memory
    fits = size_gb <= budget_gb
    if fits:
        transformer.offload_never = True
        unconditional_transformer.offload_never = True
    log.debug(f'Load model: type=Ideogram4 offload=balanced transformers={size_gb:.1f} budget={budget_gb:.1f} action={"pin" if fits else "default"}')
    return fits


def load_ideogram4(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)
    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=Ideogram4 repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    generic.set_pipeline('Ideogram4', Ideogram4Pipeline)
    if repo_id is None or repo_id.lower() == 'none':
        return None

    # Each transformer loads independently from its subfolder, so each gets its own SDNQ config.
    cls = diffusers.Ideogram4Transformer2DModel
    transformer = generic.load_transformer(repo_id, cls_name=cls, subfolder="transformer", load_config=diffusers_load_config)
    unconditional_transformer = generic.load_transformer(repo_id, cls_name=cls, subfolder="unconditional_transformer", load_config=diffusers_load_config)
    pin_transformers(transformer, unconditional_transformer)
    # shared_te_map redirects to the shared Qwen3-VL repo (deduped with VQA + prompt-enhance);
    # the bundled text_encoder is the fallback when sharing is off. The vae, tokenizer, and
    # scheduler load from the repo via from_pretrained.
    text_encoder = generic.load_text_encoder(repo_id, cls_name=Qwen3VLModel, load_config=diffusers_load_config)

    pipe = Ideogram4Pipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        unconditional_transformer=unconditional_transformer,
        text_encoder=text_encoder,
        **load_args,
    )
    # The pipeline decodes internally; the CFG scale slider drives guidance_scale, which is
    # mutually exclusive with the pipeline's default per-step guidance_schedule.
    pipe.task_args = {'output_type': 'pil', 'guidance_schedule': None} # pylint: disable=attribute-defined-outside-init
    # JSON captions must pass through verbatim; skip styles/wildcards that would strip the braces.
    pipe.keep_prompts = True # pylint: disable=attribute-defined-outside-init

    del transformer, unconditional_transformer, text_encoder
    devices.torch_gc(force=True, reason='load')
    return pipe
