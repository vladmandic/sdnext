import torch.nn.functional as F
import torchvision.transforms as T
from transformers.models.clip.modeling_clip import (
    CLIPPreTrainedModel,
    CLIPModel,
)


class CLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
        cache_dir,
        variant=None,
        subfolder="image_prompt_encoder",
    ):
        # Inference loads from the ObjectClear checkpoint's ``image_prompt_encoder``
        # subfolder (the default). Training initialises from a plain CLIP repo such
        # as ``openai/clip-vit-large-patch14``, which has no subfolder — pass
        # ``subfolder=""`` (or None) in that case.
        model = CLIPModel.from_pretrained(
            global_model_name_or_path,
            subfolder=subfolder or "",
            cache_dir=cache_dir,
            variant=variant,
        )
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return CLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values):
        b, _c, h, w = object_pixel_values.shape

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        object_pixel_values = self.vision_processor(object_pixel_values)
        object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, 1, -1)
        return object_embeds
