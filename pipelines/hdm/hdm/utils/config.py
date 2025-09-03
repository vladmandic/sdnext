import os
import toml
import omegaconf


def load_train_config(file):
    config = toml.load(file)

    model = config["model"]
    model["config"] = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.load(model["config"]), resolve=True
    )
    dataset = config["dataset"]
    trainer = config["trainer"]
    lightning = config["lightning"]

    if "logger" in lightning and not lightning["logger"].get("version", None):
        lightning["logger"]["version"] = os.urandom(4).hex()

    if "scaling_factor" in model and "scaling_factor" not in dataset:
        dataset["scaling_factor"] = model["scaling_factor"]
    if "scaling_factor" in dataset and "scaling_factor" not in model:
        model["scaling_factor"] = dataset["scaling_factor"]
    if "scaling_factor" not in model and "scaling_factor" not in dataset:
        model["scaling_factor"] = dataset["scaling_factor"] = 1.0

    if "latent_shift" in model and "latent_shift" not in dataset:
        dataset["latent_shift"] = model["latent_shift"]
    if "latent_shift" in dataset and "latent_shift" not in model:
        model["latent_shift"] = dataset["latent_shift"]
    if "latent_shift" not in model and "latent_shift" not in dataset:
        model["latent_shift"] = dataset["latent_shift"] = 0.0

    return model, dataset, trainer, lightning
