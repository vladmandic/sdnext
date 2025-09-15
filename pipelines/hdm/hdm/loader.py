import torch
from diffusers import (
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)

from .modules.text_encoders import BaseTextEncoder, SimpleTextEncoder
from .trainer import DMTrainer, FlowTrainer
from .utils import instantiate


def model_loader(
    unet: UNet2DConditionModel | None = None,
    unet_class=UNet2DConditionModel,
    unet_config=None,
    te: BaseTextEncoder | None = None,
    te_class=SimpleTextEncoder,
    te_config={
        "te_name": "apple/DFN5B-CLIP-ViT-H-14-378",
        "device": "cpu",
        "dtype": torch.float32,
        "zero_for_padding": True,
    },
    te_name="",
    tokenizers: list[dict] | None = None,
    vae: AutoencoderKL | None = None,
    vae_class=AutoencoderKL,
    vae_config=None,
    vae_name="",
    scheduler: EulerDiscreteScheduler | None = None,
    scheduler_class=EulerDiscreteScheduler,
    scheduler_config=None,
    scheduler_name=None,
    type=None,
):
    if unet is None:
        unet = instantiate(unet_class)(**unet_config)
    else:
        unet = instantiate(unet)
    if te is None:
        if te_name is not None and te_name != "":
            te = instantiate(te_class).from_pretrained(te_name)
        else:
            te = instantiate(te_class)(**te_config)
    else:
        te = instantiate(te)
    if vae is not None:
        vae = instantiate(vae)
    elif vae_class is not None:
        if vae_name is not None and vae_name != "":
            vae = instantiate(vae_class).from_pretrained(vae_name)
        elif vae_config is not None:
            vae = instantiate(vae_class)(**vae_config)
    if scheduler is None:
        if scheduler_name is not None and scheduler_name != "":
            scheduler = instantiate(scheduler_class).from_pretrained(scheduler_name)
        elif scheduler_class is not None and scheduler_config is not None:
            scheduler = instantiate(scheduler_class)(**scheduler_config)
        else:
            scheduler = None
    else:
        scheduler = instantiate(scheduler)

    if hasattr(te, "tokenizers"):
        tokenizers = te.tokenizers
    elif hasattr(te, "tokenizer") and te.tokenizer is not None:
        tokenizers = [te.tokenizer]
    elif isinstance(tokenizers, str) and tokenizers != "":
        tokenizers = [instantiate(tokenizers)]
    elif isinstance(tokenizers, list):
        tokenizers = [instantiate(tokenizer) for tokenizer in tokenizers]
    else:
        tokenizers = None

    return unet, te, tokenizers, vae, scheduler


def load_trainer(conf: dict, unet=None, te=None, vae=None, scheduler=None, type=None):
    conf = dict(**conf)
    if unet is not None:
        conf["unet"] = unet
    if te is not None:
        conf["te"] = te
    if vae is not None:
        conf["vae"] = vae
    if scheduler is not None:
        conf["scheduler"] = scheduler
    type = type or conf.pop("type", "dm")
    if type == "dm":
        trainer = DMTrainer(**conf)
    elif type == "flow":
        conf.pop("scheduler")
        trainer = FlowTrainer(**conf)
    else:
        raise NotImplementedError
    return trainer


def load_model(conf: dict):
    """
    return unet(dit)/te/vae/scheduler
    """
    if "model" in conf:
        return model_loader(**conf["model"])
    return model_loader(**conf)


def load_dataset(conf: dict):
    dataset = instantiate(conf)
    return dataset


def load_all(conf: dict):
    dataset_conf = conf.pop("dataset")
    dataset = load_dataset(dataset_conf)
    model_conf = conf.pop("model")
    unet, te, tokenizers, vae, scheduler = load_model(model_conf)
    trainer = load_trainer(
        conf.pop("trainer"), unet=unet, te=te, vae=vae, scheduler=scheduler
    )
    # TODO: there might be a better way to handle this
    dataset.tokenizers = tokenizers
    return dataset, trainer, (unet, te, tokenizers, vae, scheduler)
