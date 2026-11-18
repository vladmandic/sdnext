import functools
from diffusers.models.attention_processor import Attention
from diffusers import ModelMixin, ConfigMixin
from .attention import GeneralizedLinearAttention


model_dict = {
    "runwayml/stable-diffusion-v1-5": "Yuanshi/LinFusion-1-5",
    "SG161222/Realistic_Vision_V4.0_noVAE": "Yuanshi/LinFusion-1-5",
    "Lykon/dreamshaper-8": "Yuanshi/LinFusion-1-5",
    "stabilityai/stable-diffusion-2-1": "Yuanshi/LinFusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0": "Yuanshi/LinFusion-XL",
}


def replace_submodule(model, module_name, new_submodule):
    path, attr = module_name.rsplit(".", 1)
    parent_module = functools.reduce(getattr, path.split("."), model)
    setattr(parent_module, attr, new_submodule)


class LinFusion(ModelMixin, ConfigMixin):
    def __init__(self, modules_list, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.modules_dict = {}
        self.register_to_config(modules_list=modules_list)

        for i, attention_config in enumerate(modules_list):
            dim_n = attention_config["dim_n"]
            heads = attention_config["heads"]
            projection_mid_dim = attention_config["projection_mid_dim"]
            linear_attention = GeneralizedLinearAttention(
                query_dim=dim_n,
                out_dim=dim_n,
                dim_head=dim_n // heads,
                projection_mid_dim=projection_mid_dim,
            )
            self.add_module(f"{i}", linear_attention)
            self.modules_dict[attention_config["module_name"]] = linear_attention

    @classmethod
    def get_default_config(
        cls,
        pipeline=None,
        unet=None,
    ):
        """
        Get the default configuration for the LinFusion model.
        (The `projection_mid_dim` is same as the `query_dim` by default.)
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        modules_list = []
        for module_name, module in unet.named_modules():
            if not isinstance(module, Attention):
                continue
            if "attn1" not in module_name:
                continue
            dim_n = module.to_q.weight.shape[0]
            # modules_list.append((module_name, dim_n, module.heads))
            modules_list.append(
                {
                    "module_name": module_name,
                    "dim_n": dim_n,
                    "heads": module.heads,
                    "projection_mid_dim": None,
                }
            )
        return {"modules_list": modules_list}

    @classmethod
    def construct_for(
        cls,
        pipeline=None,
        unet=None,
        load_pretrained=True,
        pretrained_model_name_or_path=None,
        pipe_name_path=None,
    ) -> "LinFusion":
        """
        Construct a LinFusion object for the given pipeline.
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        if load_pretrained:
            from modules.shared import opts

            # Load from pretrained
            if not pretrained_model_name_or_path:
                pipe_name_path = pipe_name_path or pipeline._internal_dict._name_or_path # pylint: disable=protected-access
                pretrained_model_name_or_path = model_dict.get(pipe_name_path, None)
                if pretrained_model_name_or_path:
                    pass
                else:
                    raise RuntimeError(
                        f"LinFusion not found for pipeline [{pipe_name_path}], please provide the path."
                    )
            linfusion = (
                LinFusion.from_pretrained(pretrained_model_name_or_path, local_files_only=opts.offline_mode)
                .to(unet.device)
                .to(unet.dtype)
            )
        else:
            # Create from scratch without pretrained parameters
            default_config = LinFusion.get_default_config(unet=unet)
            linfusion = LinFusion(**default_config).to(unet.device).to(unet.dtype)
        linfusion.mount_to(unet=unet)
        return linfusion

    def mount_to(self, pipeline=None, unet=None) -> None:
        """
        Mounts the modules in the `modules_dict` to the given `pipeline`.
        """
        assert unet is not None or pipeline.unet is not None
        unet = unet or pipeline.unet
        for module_name, module in self.modules_dict.items():
            replace_submodule(unet, module_name, module)
        self.to(unet.device).to(unet.dtype)
