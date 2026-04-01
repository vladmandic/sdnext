import time
import rich.progress as rp
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant, sd_hijack_te
from modules.logger import log, console
from pipelines import generic


class GLMTokenProgressProcessor(transformers.LogitsProcessor):
    """LogitsProcessor that tracks autoregressive token generation progress for GLM-Image."""

    def __init__(self):
        self.total_tokens = 0
        self.current_step = 0
        self.task_id = None
        self.pbar = None
        self.pbar_task = None
        self.start_time = 0

    def set_total(self, total_tokens: int):
        self.total_tokens = total_tokens
        self.current_step = 0

    def __call__(self, input_ids, scores):
        if self.current_step == 0:
            self.task_id = shared.state.begin('AR Generation')
            self.start_time = time.time()
            self.pbar = rp.Progress(
                rp.TextColumn('[cyan]AR Generation'),
                rp.TextColumn('{task.fields[speed]}'),
                rp.BarColumn(bar_width=40, complete_style='#327fba', finished_style='#327fba'),
                rp.TaskProgressColumn(),
                rp.MofNCompleteColumn(),
                rp.TimeElapsedColumn(),
                rp.TimeRemainingColumn(),
                console=console,
            )
            self.pbar.start()
            self.pbar_task = self.pbar.add_task(description='', total=self.total_tokens, speed='')
        self.current_step += 1
        shared.state.sampling_step = self.current_step
        shared.state.sampling_steps = self.total_tokens
        if self.pbar is not None and self.pbar_task is not None:
            elapsed = time.time() - self.start_time
            speed = f'{self.current_step / elapsed:.2f}tok/s' if elapsed > 0 else ''
            self.pbar.update(self.pbar_task, completed=self.current_step, speed=speed)
        if self.current_step >= self.total_tokens:
            if self.pbar is not None:
                self.pbar.stop()
                self.pbar = None
            if self.task_id is not None:
                shared.state.end(self.task_id)
                self.task_id = None
        return scores


def hijack_vision_language_generate(pipe):
    """Wrap vision_language_encoder.generate to add progress tracking."""
    if not hasattr(pipe, 'vision_language_encoder') or pipe.vision_language_encoder is None:
        return

    original_generate = pipe.vision_language_encoder.generate
    progress_processor = GLMTokenProgressProcessor()

    def wrapped_generate(*args, **kwargs):
        # Get max_new_tokens to determine total tokens
        max_new_tokens = kwargs.get('max_new_tokens', 0)
        progress_processor.set_total(max_new_tokens)

        # Add progress processor to logits_processor list
        existing_processors = kwargs.get('logits_processor', None)
        if existing_processors is None:
            existing_processors = []
        elif not isinstance(existing_processors, list):
            existing_processors = list(existing_processors)
        kwargs['logits_processor'] = existing_processors + [progress_processor]

        return original_generate(*args, **kwargs)

    pipe.vision_language_encoder.generate = wrapped_generate


def load_glm_image(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    if not hasattr(transformers, 'GlmImageForConditionalGeneration'):
        log.error(f'Load model: type=GLM-Image repo="{repo_id}" transformers={transformers.__version__} not supported')
        return None

    load_args, _quant_args = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=GLM-Image repo="{repo_id}" offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    # Load transformer (DiT decoder - 7B) with quantization support
    transformer = generic.load_transformer(
        repo_id,
        cls_name=diffusers.GlmImageTransformer2DModel,
        load_config=diffusers_load_config
    )

    # Load text encoder (ByT5 for glyph) - cannot use shared T5 as GLM-Image requires specific ByT5 encoder (1472 hidden size)
    text_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.T5EncoderModel,
        load_config=diffusers_load_config,
        allow_shared=False
    )

    # Load vision-language encoder (AR model - 9B)
    # Note: This is a conditional generation model, different from typical text encoders
    vision_language_encoder = generic.load_text_encoder(
        repo_id,
        cls_name=transformers.GlmImageForConditionalGeneration, # pylint: disable=no-member
        subfolder="vision_language_encoder",
        load_config=diffusers_load_config,
        allow_shared=False
    )

    pipe = diffusers.GlmImagePipeline.from_pretrained(
        repo_id,
        cache_dir=shared.opts.diffusers_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        vision_language_encoder=vision_language_encoder,
        **load_args,
    )

    pipe.task_args = {
        'output_type': 'np',
        'generate_kwargs': {
            'eos_token_id': None,  # Disable EOS early stopping to ensure all required tokens are generated
        },
    }

    del transformer, text_encoder, vision_language_encoder
    sd_hijack_te.init_hijack(pipe)
    hijack_vision_language_generate(pipe)  # Add progress tracking for AR token generation
    devices.torch_gc(force=True, reason='load')
    return pipe
