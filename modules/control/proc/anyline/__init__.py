import numpy as np
from PIL import Image


class AnylineDetector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path="TheMistoAI/MistoLine", cache_dir=None, local_files_only=False):
        from installer import install
        install('controlnet-aux', quiet=True)
        from controlnet_aux import AnylineDetector as _AnylineDetector
        model = _AnylineDetector.from_pretrained(pretrained_model_or_path, cache_dir=cache_dir)
        return cls(model)

    def __call__(self, image, output_type="pil", **kwargs):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        result = self.model(image, output_type=output_type)
        return result
