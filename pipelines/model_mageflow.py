import diffusers
import torch
import transformers
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from modules import shared, devices, sd_models, model_quant, sd_hijack_te, sd_hijack_vae
from modules.logger import log
from pipelines import generic
from pipelines.mageflow import MageFlowPipeline, MageFlowTransformer2DModel
from pipelines.mageflow.convert_mage_flow_to_diffusers import convert_transformer_state_dict
from pipelines.native_transformer import TransformerSpec


class MageFlowFP32Linear(torch.nn.Linear):
    """Linear that keeps fp32 parameters through module casts and returns output_dtype when set."""
    output_dtype: torch.dtype | None = None

    def _apply(self, fn, recurse=True):
        def keep_fp32(tensor):
            if not tensor.is_floating_point():
                return fn(tensor)
            return tensor.to(device=fn(tensor.new_empty(0)).device, dtype=torch.float32) # move without rounding through the cast dtype
        return super()._apply(keep_fp32, recurse=recurse)

    def forward(self, input): # pylint: disable=redefined-builtin
        output = torch.nn.functional.linear(input.to(torch.float32), self.weight, self.bias)
        return output if self.output_dtype is None else output.to(self.output_dtype)


class MageFlowLowRankTransformer2DModel(MageFlowTransformer2DModel):
    """Mage-Flow with optional shared low-rank AdaLN (mageflow-lowrank-modulation-v1).

    With ``modulation_rank`` set, ``modulation_down`` projects silu(temb) once and each block keeps an fp32 head
    in ``img_mod.1``/``txt_mod.1``; ``norm_out`` still reads the full temb. Unset, the model is stock Mage-Flow.
    """

    @register_to_config
    def __init__(self, in_channels: int = 128, out_channels: int = 128, context_in_dim: int = 3584, hidden_size: int = 3072, num_attention_heads: int = 24, num_layers: int = 32, axes_dim: list[int] | None = None, patch_size: int = 1, modulation_rank: int | None = None):
        super().__init__(in_channels=in_channels, out_channels=out_channels, context_in_dim=context_in_dim, hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_layers=num_layers, axes_dim=axes_dim or [16, 48, 48], patch_size=patch_size)
        self.modulation_rank = modulation_rank
        if not modulation_rank:
            return
        self.modulation_down = MageFlowFP32Linear(hidden_size, modulation_rank, dtype=torch.float32)
        for block in self.transformer_blocks:
            for mod in (block.img_mod, block.txt_mod): # silu moves ahead of the shared projection
                mod[0] = torch.nn.Identity()
                mod[1] = MageFlowFP32Linear(modulation_rank, 6 * hidden_size, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor = None, timestep: torch.Tensor = None, img_ids: torch.Tensor = None, joint_attention_kwargs: dict | None = None, return_dict: bool = True):
        if not self.modulation_rank:
            return super().forward(hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, img_ids=img_ids, joint_attention_kwargs=joint_attention_kwargs, return_dict=return_dict)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(self.context_embedder_norm(encoder_hidden_states))
        temb = self.time_text_embed(timestep.to(hidden_states.dtype), hidden_states)
        block_temb = self.modulation_down(torch.nn.functional.silu(temb))
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        image_rotary_emb = self.pos_embed(img_ids, *self.pos_embed._infer_grid_size(img_ids)) # pylint: disable=protected-access
        for block in self.transformer_blocks:
            block.img_mod[1].output_dtype = block.txt_mod[1].output_dtype = hidden_states.dtype
            encoder_hidden_states, hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=block_temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs)
        output = self.proj_out(self.norm_out(hidden_states, temb))
        return Transformer2DModelOutput(sample=output) if return_dict else (output,)


def infer_config(state_dict: dict) -> dict:
    """``modulation_rank`` from the shared down projection of low-rank checkpoints."""
    down = state_dict.get('modulation_down.weight')
    return {'modulation_rank': down.shape[0] if down is not None else None}


MAGEFLOW_SPEC = TransformerSpec(cls=MageFlowLowRankTransformer2DModel, converter=convert_transformer_state_dict, infer_config=infer_config) # original-format files are renamed, diffusers-format files pass through


def load_mageflow(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}

    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    load_args, _ = model_quant.get_dit_args(diffusers_load_config, allow_quant=False)
    log.debug(f'Load model: type=MageFlow repo="{repo_id}" config={diffusers_load_config} offload={shared.opts.diffusers_offload_mode} dtype={devices.dtype} args={load_args}')

    generic.set_pipeline('MageFlow', MageFlowPipeline)

    if repo_id is None or repo_id.lower() == 'none':
        return None

    transformer = generic.load_transformer(repo_id, cls_name=MageFlowLowRankTransformer2DModel, load_config=diffusers_load_config, native_spec=MAGEFLOW_SPEC)
    text_encoder = generic.load_text_encoder(repo_id, cls_name=transformers.Qwen3VLForConditionalGeneration, load_config=diffusers_load_config)
    tokenizer = transformers.Qwen2Tokenizer.from_pretrained(repo_id, subfolder="text_encoder", cache_dir=shared.opts.diffusers_dir)

    diffusers.pipelines.auto_pipeline.AUTO_TEXT2IMAGE_PIPELINES_MAPPING['mageflow'] = MageFlowPipeline
    diffusers.pipelines.auto_pipeline.AUTO_IMAGE2IMAGE_PIPELINES_MAPPING['mageflow'] = MageFlowPipeline

    pipe = MageFlowPipeline.from_pretrained(
        repo_id,
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        cache_dir=shared.opts.diffusers_dir,
        **load_args,
    )

    pipe.task_args = {
        'output_type': 'pil',
    }

    generic.load_vae_override(pipe, diffusers_load_config)
    del transformer
    del text_encoder
    del tokenizer

    sd_hijack_te.init_hijack(pipe)
    sd_hijack_vae.init_hijack(pipe)

    devices.torch_gc(force=True, reason='load')
    return pipe
