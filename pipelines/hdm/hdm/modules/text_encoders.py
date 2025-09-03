from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel, Qwen2Model

from ..utils import remove_none, instantiate


class BaseTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.text_model = None

    def tokenize(self, text: str) -> list[int] | list[list[int]] | torch.LongTensor:
        raise NotImplementedError

    def encode(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, tokenizer_outputs: list[dict[str, torch.Tensor]]):
        raise NotImplementedError


class SimpleTextEncoder(BaseTextEncoder):
    def __init__(
        self,
        te_name: str = "apple/DFN5B-CLIP-ViT-H-14-378",
        te_cls: type = CLIPTextModel,
        te_kwargs: dict[str, Any] = {},
        zero_for_padding: bool = True,
        max_length: int = 256,
    ):
        super().__init__()
        self.tokenizers = [AutoTokenizer.from_pretrained(te_name, **te_kwargs)]
        for tokenizer in self.tokenizers:
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.model_max_length > max_length:
                tokenizer.model_max_length = max_length

        self.text_model = (
            instantiate(te_cls).from_pretrained(te_name).to(self.device_type)
        )
        self.zero_for_padding = zero_for_padding

    def tokenize(self, text, **kwargs):
        return [self.tokenizers[0](text, **kwargs)]

    def encode(self, text, **kwargs):
        return self.forward(self.tokenize(text, **kwargs))

    def forward(self, tokenizers_outputs):
        tokens = tokenizers_outputs[0]
        text_model = self.text_model

        input_ids = tokens["input_ids"].to(self.device_type.device)
        attn_mask = tokens["attention_mask"].to(self.device_type.device)

        # In CLIP we have `last_hidden_state = self.final_layer_norm(last_hidden_state)`
        # The pooled embedding is also normalized
        normed_embedding, pooled_embedding, *embeddings = text_model(
            input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        if len(embeddings):
            embedding = embeddings[-1][-1]
        else:
            embedding = pooled_embedding[-1]
            pooled_embedding = None
        if self.zero_for_padding:
            while embedding.ndim > attn_mask.ndim:
                attn_mask = attn_mask.unsqueeze(-1)
            embedding = embedding * attn_mask
            normed_embedding = normed_embedding * attn_mask
        return embedding, normed_embedding, pooled_embedding, attn_mask


class ConcatTextEncoders(BaseTextEncoder):
    DEFAULT_SETTINGS = {
        "disable_autocast": False,
        "concat_buckets": 0,
        "use_pooled": False,
        "need_mask": False,
        "layer_ids": -1,
    }

    def __init__(
        self,
        tokenizers: list[str] = [],
        text_models: list[dict] = [],
        zero_for_padding: bool = True,
        max_length: int = 256,
        model_dim: int = -1,
        output_dim: int = -1,
        pooled_dim: int = -1,
        extra_mlp: bool = False,
    ):
        """
        A text encoder wrapper for multiple tokenizers and text models.
        Can support tricky concat config like what SD3 need

        SDXL:
            tes: [CLIP-L, openCLIP-G]
            concat_buckets: [0, 0]
            use_pooled: [True, True]
            layer_index: [-1, -2]
        SD3:
            tes: [CLIP-L, openCLIP-G, T5-xxl]
            concat_buckets: [0, 0, 1]
            use_pooled: [True, True, False]
        """
        super().__init__()
        self.tokenizers = [
            AutoTokenizer.from_pretrained(tokenizer) for tokenizer in tokenizers
        ]
        for tokenizer in self.tokenizers:
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.model_max_length > max_length:
                tokenizer.model_max_length = max_length

        text_models_configs = [
            (instantiate(config.pop("model")), {**config}) for config in text_models
        ]
        self.max_bucket = max([i[1]["concat_buckets"] for i in text_models_configs])
        self.register_buffer("_device", torch.tensor(0), persistent=False)

        self.text_models = nn.ModuleList([i[0] for i in text_models_configs])
        self.configs = [i[1] for i in text_models_configs]
        self.zero_for_padding = zero_for_padding

        self.emb_mlp = self.pool_mlp = None
        if extra_mlp and model_dim != -1:
            if output_dim != -1:
                self.emb_mlp = nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, model_dim * 4),
                    nn.Mish(),
                    nn.Linear(model_dim * 4, output_dim),
                )
            if pooled_dim != -1:
                self.pool_mlp = nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, model_dim * 4),
                    nn.Mish(),
                    nn.Linear(model_dim * 4, pooled_dim),
                )

    def trainable_modules(self):
        results = []
        if self.emb_mlp is not None:
            results.append(self.emb_mlp)
        if self.pool_mlp is not None:
            results.append(self.pool_mlp)
        return results

    def trainable_params(self):
        results = []
        if self.emb_mlp is not None:
            results.extend(self.emb_mlp.parameters())
        if self.pool_mlp is not None:
            results.extend(self.pool_mlp.parameters())
        return results

    @property
    def device(self):
        return self._device.device

    def tokenize(self, text, **kwargs):
        results = []
        for tokenizer in self.tokenizers:
            results.append(tokenizer(text, **kwargs, return_tensors="pt"))
        return results

    def encode(self, text, **kwargs):
        return self.forward(self.tokenize(text, **kwargs))

    def forward(
        self, tokenizers_outputs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: torch.Tensor
            normed_embedding: torch.Tensor
            pooled_embedding: torch.Tensor
            attn_mask: torch.Tensor
        """
        attn_masks = [None for _ in range(self.max_bucket + 1)]
        text_embeddings = [[] for _ in range(self.max_bucket + 1)]
        normed_text_embeddings = [[] for _ in range(self.max_bucket + 1)]
        pooled_text_embeddings = [[] for _ in range(self.max_bucket + 1)]
        for idx, (tokens, text_model, config) in enumerate(
            zip(tokenizers_outputs, self.text_models, self.configs)
        ):
            bucket = config["concat_buckets"]
            need_mask = config["need_mask"]
            use_pooled = config["use_pooled"]
            layer_idx = config["layer_idx"]
            disable_autocast = config["disable_autocast"]

            input_ids = tokens["input_ids"].to(self.device)
            attn_mask = tokens["attention_mask"].to(self.device)
            if attn_masks[bucket] is None and need_mask:
                attn_masks[bucket] = attn_mask

            with torch.autocast("cuda", enabled=not disable_autocast):
                output = text_model(
                    input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                normed_embedding = output.last_hidden_state
                # The case of CLIP
                if hasattr(output, "pooler_output"):
                    # embeddings is tuple
                    embedding = output.hidden_states[layer_idx]
                    pooled_embedding = output.pooler_output
                # The case of T5 or other models
                else:
                    embedding = output.hidden_states[-1]
                    pooled_embedding = torch.zeros_like(embedding[:, 0, :])

            if self.zero_for_padding:
                while embedding.ndim > attn_mask.ndim:
                    attn_mask = attn_mask.unsqueeze(-1)
                embedding = embedding * attn_mask
                normed_embedding = normed_embedding * attn_mask
            text_embeddings[bucket].append(embedding)
            normed_text_embeddings[bucket].append(normed_embedding)
            if use_pooled:
                pooled_text_embeddings[bucket].append(pooled_embedding)

        for i in range(len(text_embeddings)):
            if text_embeddings[i] == []:
                text_embeddings[i] = None
                normed_text_embeddings[i] = None
                pooled_text_embeddings[i] = None
                continue
            text_embeddings[i] = torch.cat(text_embeddings[i], dim=-1)
            normed_text_embeddings[i] = torch.cat(normed_text_embeddings[i], dim=-1)
            if pooled_text_embeddings[i] == []:
                pooled_text_embeddings[i] = None
                continue
            pooled_text_embeddings[i] = torch.cat(pooled_text_embeddings[i], dim=-1)

        max_dim = max(
            embedding.size(-1) for embedding in text_embeddings if embedding is not None
        )
        for idx, embedding in enumerate(text_embeddings):
            if embedding is None:
                continue
            if embedding.size(-1) < max_dim:
                text_embeddings[idx] = torch.nn.functional.pad(
                    embedding, (0, max_dim - embedding.size(-1))
                )
        for idx, embedding in enumerate(normed_text_embeddings):
            if embedding is None:
                continue
            if embedding.size(-1) < max_dim:
                normed_text_embeddings[idx] = torch.nn.functional.pad(
                    embedding, (0, max_dim - embedding.size(-1))
                )
        if any(mask is not None for mask in attn_masks):
            for idx, embedding in enumerate(text_embeddings):
                if embedding is None:
                    continue
                elif attn_masks[idx] is None:
                    attn_masks[idx] = torch.ones(
                        embedding.size(0), embedding.size(1), device=embedding.device
                    ).long()
            attn_masks = torch.cat(remove_none(attn_masks), dim=1)
        else:
            attn_masks = None
        if any(pooled is not None for pooled in pooled_text_embeddings):
            pooled_text_embeddings = torch.cat(
                remove_none(pooled_text_embeddings), dim=-1
            )
        else:
            pooled_text_embeddings = None
        text_embeddings = torch.cat(remove_none(text_embeddings), dim=1)
        normed_text_embeddings = torch.cat(remove_none(normed_text_embeddings), dim=1)

        if self.emb_mlp is not None:
            text_embeddings = self.emb_mlp(text_embeddings)
            normed_text_embeddings = self.emb_mlp(normed_text_embeddings)
        if self.pool_mlp is not None and pooled_text_embeddings is not None:
            pooled_text_embeddings = self.pool_mlp(pooled_text_embeddings)

        return (
            normed_text_embeddings,
            text_embeddings,
            pooled_text_embeddings,
            attn_masks,
        )


if __name__ == "__main__":
    te = ConcatTextEncoders(
        tokenizers=[
            "openai/clip-vit-large-patch14",
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "google/t5-v1_1-xxl",
        ],
        text_models=[
            (CLIPTextModel, "openai/clip-vit-large-patch14", {}),
            (CLIPTextModel, "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", {}),
            (
                T5EncoderModel,
                "google/t5-v1_1-xxl",
                {},
            ),  # Need `pip install sentencepiece`
        ],
        concat_buckets=[0, 0, 1],
        use_pooled=[True, True, False],
        layer_idx=[-1, -2, -1],
        need_mask=[False, False, True],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    with torch.no_grad():
        text_embeddings, normed_text_embeddings, pooled_text_embeddings, attn_masks = (
            te.encode("hello")
        )
        print(text_embeddings.shape, normed_text_embeddings.shape)
        print(pooled_text_embeddings.shape)
