from typing import List, Literal, Optional
from pydantic.dataclasses import dataclass
from ..base import ModelConfig


@dataclass
class LBMConfig(ModelConfig):
    """This is the Config for LBM Model class which defines all the useful parameters to be used in the model.

    Args:

        source_key (str):
            Key for the source image. Defaults to "source_image"

        target_key (str):
            Key for the target image. Defaults to "target_image"

        mask_key (Optional[str]):
            Key for the mask showing the valid pixels. Defaults to None

        latent_loss_type (str):
            Loss type to use. Defaults to "l2". Choices are "l2", "l1"

        pixel_loss_type (str):
            Pixel loss type to use. Defaults to "l2". Choices are "l2", "l1", "lpips"

        pixel_loss_max_size (int):
            Maximum size of the image for pixel loss.
            The image will be cropped to this size to reduce decoding computation cost. Defaults to 512

        pixel_loss_weight (float):
            Weight of the pixel loss. Defaults to 0.0

        timestep_sampling (str):
            Timestep sampling to use. Defaults to "uniform". Choices are "uniform"

        input_key (str):
            Key for the input. Defaults to "image"

        controlnet_input_key (str):
            Key for the controlnet conditioning. Defaults to "controlnet_conditioning"

        adapter_input_key (str):
            Key for the adapter conditioning. Defaults to "adapter_conditioning"

        ucg_keys (Optional[List[str]]):
            List of keys for which we enforce zero_conditioning during Classifier-free guidance. Defaults to None

        prediction_type (str):
            Type of prediction to use. Defaults to "epsilon". Choices are "epsilon", "v_prediction", "flow

        logit_mean (Optional[float]):
            Mean of the logit for the log normal distribution. Defaults to 0.0

        logit_std (Optional[float]):
            Standard deviation of the logit for the log normal distribution. Defaults to 1.0

        guidance_scale (Optional[float]):
            The guidance scale. Useful for finetunning guidance distilled diffusion models. Defaults to None

        selected_timesteps (Optional[List[float]]):
            List of selected timesteps to be sampled from if using `custom_timesteps` timestep sampling. Defaults to None

        prob (Optional[List[float]]):
            List of probabilities for the selected timesteps if using `custom_timesteps` timestep sampling. Defaults to None
    """

    source_key: str = "source_image"
    target_key: str = "target_image"
    mask_key: Optional[str] = None
    latent_loss_weight: float = 1.0
    latent_loss_type: Literal["l2", "l1"] = "l2"
    pixel_loss_type: Literal["l2", "l1", "lpips"] = "l2"
    pixel_loss_max_size: int = 512
    pixel_loss_weight: float = 0.0
    timestep_sampling: Literal["uniform", "log_normal", "custom_timesteps"] = "uniform"
    logit_mean: Optional[float] = 0.0
    logit_std: Optional[float] = 1.0
    selected_timesteps: Optional[List[float]] = None
    prob: Optional[List[float]] = None
    bridge_noise_sigma: float = 0.001

    def __post_init__(self):
        super().__post_init__()
        if self.timestep_sampling == "log_normal":
            assert isinstance(self.logit_mean, float) and isinstance(
                self.logit_std, float
            ), "logit_mean and logit_std should be float for log_normal timestep sampling"

        if self.timestep_sampling == "custom_timesteps":
            assert isinstance(self.selected_timesteps, list) and isinstance(
                self.prob, list
            ), "timesteps and prob should be list for custom_timesteps timestep sampling"
            assert len(self.selected_timesteps) == len(
                self.prob
            ), "timesteps and prob should be of same length for custom_timesteps timestep sampling"
            assert (
                sum(self.prob) == 1
            ), "prob should sum to 1 for custom_timesteps timestep sampling"
