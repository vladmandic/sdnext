from transformers import AutoConfig, Qwen2Config
from typing import Tuple


class XOmniConfig(Qwen2Config):
    model_type = "x-omni"

    def __init__(
        self,
        num_mm_adap_layers: int = 4,
        num_mm_head_layers: int = 4,
        mm_vocab_size: int = 16448,
        image_vocab_size: int = 16384,
        mm_special_tokens: Tuple[str] = ('<SOM>', '<EOM>', '<IMAGE>'),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mm_adap_layers = num_mm_adap_layers
        self.num_mm_head_layers = num_mm_head_layers
        self.mm_vocab_size = mm_vocab_size
        self.image_vocab_size = image_vocab_size
        self.mm_special_tokens = mm_special_tokens


AutoConfig.register("x-omni", XOmniConfig)
