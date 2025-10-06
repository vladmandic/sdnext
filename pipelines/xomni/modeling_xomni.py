import os
from types import SimpleNamespace
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from transformers import Qwen2ForCausalLM, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding, Qwen2DecoderLayer, Qwen2Model, Qwen2PreTrainedModel

from .configuration_xomni import XOmniConfig
from .modeling_siglip_tokenizer import create_anyres_preprocess, SiglipTokenizer
from .modeling_siglip_flux import FluxTransformer2DModelWithSigLIP, FluxPipelineWithSigLIP
from .modeling_vit import create_siglip_vit


class XOmniDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: XOmniConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.is_lm_layer = config.num_mm_adap_layers <= layer_idx < config.num_hidden_layers - config.num_mm_head_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        hidden_states, multimodal_mask = torch.split(hidden_states, hidden_states.shape[-1] // 2, dim=-1)
        if self.is_lm_layer:
            output_hidden_states, *others = super().forward(hidden_states, **kwargs)
            output_hidden_states = torch.cat([output_hidden_states, multimodal_mask], dim=-1)
            return output_hidden_states, *others
        
        # mm_hidden_states = torch.where(multimodal_mask.bool(), hidden_states, torch.zeros_like(hidden_states))
        output_hidden_states, *others = super().forward(hidden_states, **kwargs)
        output_hidden_states = torch.where(multimodal_mask.bool(), output_hidden_states, hidden_states)
        output_hidden_states = torch.cat([output_hidden_states, multimodal_mask], dim=-1)
        return output_hidden_states, *others


class XOmniModel(Qwen2Model, Qwen2PreTrainedModel):
    model_type = "x-omni"
    config_class = XOmniConfig

    def __init__(self, config: XOmniConfig):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = -1
        self.vocab_size = config.vocab_size
        
        self.lm_embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.mm_embed_tokens = nn.Embedding(config.mm_vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [XOmniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.lm_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mm_norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.lm_embed_tokens

    def set_input_embeddings(self, value):
        self.lm_embed_tokens = value

    def embed_tokens(self, input_ids):
        (B, L), C = input_ids.shape, self.config.hidden_size
        multimodal_mask = input_ids >= self.config.vocab_size
        lm_input_ids = input_ids[~multimodal_mask][None, :]
        mm_input_ids = input_ids[multimodal_mask][None, :] - self.config.vocab_size
        lm_embeds = self.lm_embed_tokens(lm_input_ids)
        mm_embeds = self.mm_embed_tokens(mm_input_ids)

        inputs_embeds = lm_embeds.new_empty((B, L, C))
        multimodal_mask = multimodal_mask[:, :, None].expand_as(inputs_embeds)
        inputs_embeds[~multimodal_mask] = lm_embeds.reshape(-1)
        inputs_embeds[multimodal_mask] = mm_embeds.reshape(-1)

        inputs_embeds = torch.cat([inputs_embeds, multimodal_mask.to(inputs_embeds.dtype)], dim=-1)
        return inputs_embeds

    def norm(self, hidden_states):
        hidden_states, multimodal_mask = torch.split(hidden_states, hidden_states.shape[-1] // 2, dim=-1)
        return torch.where(multimodal_mask.bool(), self.mm_norm(hidden_states), self.lm_norm(hidden_states))


class XOmniForCausalLM(Qwen2ForCausalLM):
    model_type = "x-omni"
    config_class = XOmniConfig
    
    _keys_to_ignore_on_load_missing = r'image_tokenizer\.*'

    def __init__(self, config):
        super().__init__(config)
        self.model = XOmniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mm_head = nn.Linear(config.hidden_size, config.mm_vocab_size, bias=False)
        
        self.generation_mode = 'text'
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    def init_vision(self, flux_pipe_path, **kwargs):
        self.som_token = self.config.mm_special_tokens[0]
        self.eom_token = self.config.mm_special_tokens[1]
        self.img_token = self.config.mm_special_tokens[2]

        self.vision_config = SimpleNamespace(**self.config.vision_config)
        self.transform_config = SimpleNamespace(**self.vision_config.transform)
        self.encoder_config = SimpleNamespace(**self.vision_config.encoder)
        self.decoder_config = SimpleNamespace(**self.vision_config.decoder)

        dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
        self.vision_dtype = dtype_map[self.vision_config.dtype]

        self.image_transform = create_anyres_preprocess(**self.vision_config.transform)
        
        self.encoder_config.siglip_path = os.path.join(self.name_or_path, self.encoder_config.siglip_path) if os.path.isdir(self.name_or_path) else hf_hub_download(repo_id=self.name_or_path, filename=self.encoder_config.siglip_path)
        self.encoder_config.projector_path = os.path.join(self.name_or_path, self.encoder_config.projector_path) if os.path.isdir(self.name_or_path) else hf_hub_download(repo_id=self.name_or_path, filename=self.encoder_config.projector_path)

        self.image_tokenizer = SiglipTokenizer(**vars(self.encoder_config))
        self.image_tokenizer.to(self.device, self.vision_dtype)

        transformer = FluxTransformer2DModelWithSigLIP.from_pretrained(
            self.name_or_path, 
            siglip_channels=self.encoder_config.z_channels, 
            torch_dtype=self.vision_dtype,
            subfolder=self.decoder_config.model_path,
            **kwargs,
        )

        self.decoder_pipe = FluxPipelineWithSigLIP.from_pretrained(
            flux_pipe_path, 
            transformer=transformer,
            torch_dtype=self.vision_dtype,
        )
        self.decoder_pipe.set_progress_bar_config(disable=True)

    def set_generation_mode(self, mode):
        assert mode in ('text', 'image'), f'Invalid generation mode: {mode}'
        self.generation_mode = mode

    def mmencode(self, tokenizer, texts=None, images=None, **kwargs):
        texts = texts or []
        images = images or []
        doc = ''
        while len(texts) > 0 or len(images) > 0:
            if len(texts) > 0:
                doc += texts.pop(0)
            if len(images) > 0:
                doc += self.tokenize_image(images.pop(0))
        return tokenizer.encode(doc, **kwargs)
    
    def mmdecode(self, tokenizer, token_ids, force_text=None, **kwargs):
        force_text = force_text or []
        if isinstance(token_ids, torch.Tensor):
            if len(token_ids.shape) == 2:
                assert token_ids.shape[0] == 1
                token_ids = token_ids[0]
            assert len(token_ids.shape) == 1
        else:
            if not isinstance(token_ids[0], int):
                assert len(token_ids) == 1
                token_ids = token_ids[0]
            assert isinstance(token_ids[0], int)
        
        doc = tokenizer.decode(token_ids, **kwargs)
        doc = doc.replace(tokenizer.pad_token, '')
        doc = doc.replace('<SEP>', '')
        texts, images = [], []
        text_image_chunks = doc.split(self.eom_token)
        for chunk in text_image_chunks:
            text, image_str = chunk.split(self.som_token) \
                if self.som_token in chunk else (chunk, '')
            texts.append(text)
            if self.img_token in image_str:
                image_meta, token_str = image_str.split(self.img_token)
                H, W = tuple(map(int, image_meta.split(' ')))
                token_ids = list(map(
                    lambda x: int(x.split('>')[0]),
                    token_str.split('<MM-Token-')[1:H*W+1],
                ))
                if len(force_text) > 0:
                    image = self.detokenize_image([force_text.pop(0)], images, token_ids, (H, W))
                else:
                    image = self.detokenize_image(texts, images, token_ids, (H, W))
                images.append(image)
        return texts, images
    
    @torch.no_grad()
    def tokenize_image(self, image):
        assert hasattr(self, 'image_tokenizer'), 'Please call "init_vision" before that.'

        image_str = self.som_token
        image = self.image_transform(image)
        assert image is not None, f'Unsupported image aspect ratio (max {self.transform_config.max_aspect_ratio}) or image resolution is too low (min {self.transform_config.min_short_size})'

        image = image[None, ...].to(self.device, self.vision_dtype)
        tokens = self.image_tokenizer.encode(image)
        B, H, W = tokens.shape
        tokens = tokens.view(B, -1).cpu().tolist()[0]
        token_str = ''.join(map(lambda x: '<MM-Token-{token_id}>'.format(token_id=x), tokens))
        image_str = f'{self.som_token}{H} {W}{self.img_token}{token_str}{self.eom_token}'
        return image_str
    
    @torch.no_grad()
    def detokenize_image(self, texts, images, token_ids, shape):
        assert hasattr(self, 'image_tokenizer'), 'Please call "init_vision" before that.'
        assert len(texts) == 1 and len(images) == 0, 'Only support one image per sample.'
        H, W = shape
        tokens = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        latents = self.image_tokenizer.decode(tokens, (1, H, W, self.encoder_config.codebook_dim))
        upscale_factor = self.decoder_config.upscale_factor
        latents = latents.reshape(*latents.shape[:2], -1).transpose(1, 2).contiguous()
        image = self.decoder_pipe(
            latents,
            [texts[0]],
            negative_prompt=[''],
            height=H * upscale_factor, width=W * upscale_factor,
            num_inference_steps=self.decoder_config.num_inference_steps, 
            guidance_scale=1.0,
            true_cfg_scale=self.decoder_config.cfg_scale,
            true_cfg_scale_2=self.decoder_config.cfg_scale_2,
        ).images[0]


        return image
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        self.model.has_sliding_layers = False
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states[:, -num_logits_to_keep:, :]
        logits = hidden_states.new_full(
            (*hidden_states.shape[:-1], self.config.vocab_size + self.config.mm_vocab_size), 
            torch.finfo(hidden_states.dtype).min
        )
        if self.generation_mode == 'text':
            logits[:, :, :self.config.vocab_size] = self.lm_head(hidden_states)
        else:
            logits[:, :, self.config.vocab_size:self.config.vocab_size + self.config.image_vocab_size] = self.mm_head(hidden_states)[:, :, :self.config.image_vocab_size]
        
        logits = logits.float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoModel.register(XOmniConfig, XOmniModel)
AutoModelForCausalLM.register(XOmniConfig, XOmniForCausalLM)
