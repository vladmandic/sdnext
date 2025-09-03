import json
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ...xut.xut import XUDiT
from .base import *


class XUDiTConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size=2,
        input_dim=4,
        dim=1024,
        ctx_dim=1024,
        ctx_size=256,
        heads=16,
        dim_head=64,
        mlp_dim=3072,
        depth=8,
        enc_blocks=1,
        dec_blocks=2,
        dec_ctx=False,
        class_cond=0,
        shared_adaln=True,
        concat_ctx=True,
        use_dyt=False,
        double_t=False,
        addon_info_embs_dim=None,
        tread_config=None,
    ):
        super().__init__()
        self.model = XUDiT(
            patch_size=patch_size,
            input_dim=input_dim,
            dim=dim,
            ctx_dim=ctx_dim,
            ctx_size=ctx_size,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            depth=depth,
            enc_blocks=enc_blocks,
            dec_blocks=dec_blocks,
            dec_ctx=dec_ctx,
            class_cond=class_cond,
            shared_adaln=shared_adaln,
            concat_ctx=concat_ctx,
            use_dyt=use_dyt,
            double_t=double_t,
            addon_info_embs_dim=addon_info_embs_dim,
            tread_config=tread_config,
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any] | str) -> "XUDiTConditionModel":
        if isinstance(config, str):
            with open(config, "r") as f:
                config = json.load(f)
        return cls(**config)

    def enable_gradient_checkpointing(self):
        return self.model.set_grad_ckpt(True)

    def disable_gradient_checkpointing(self):
        return self.model.set_grad_ckpt(False)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        pos_map: Optional[torch.Tensor] = None,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        if added_cond_kwargs is None:
            added_cond_kwargs = {}
        result = self.model(
            sample, timestep, encoder_hidden_states, pos_map, **added_cond_kwargs
        )
        if return_dict:
            return UNet2DConditionOutput(sample=result)
        else:
            return (sample,)
