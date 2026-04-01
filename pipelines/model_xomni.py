import torch
import transformers
import diffusers
from modules import shared, devices, sd_models, model_quant
from modules.logger import log


class XOmniPipeline(diffusers.DiffusionPipeline):
    def __init__(
            self,
            tokenizer=None,
            model=None,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.register_modules(
            tokenizer=tokenizer,
            model=model,
        )

    def load(
            self,
            repo_id,
            load_config: dict = {},
        ):
        from pipelines.xomni import modeling_xomni
        load_args, quant_args = model_quant.get_dit_args(load_config, module='Model', device_map=True)
        log.debug(f'Load model: cls=XOmniPipeline module=tokenizer repo_id="{repo_id}"')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            repo_id,
            use_fast=True,
        )
        log.debug(f'Load model: cls=XOmniPipeline module=transformer repo_id="{repo_id}" args={load_args}')
        # self.model = transformers.AutoModelForCausalLM.from_pretrained(
        self.model = modeling_xomni.XOmniForCausalLM.from_pretrained(
            repo_id,
            # trust_remote_code=True,
            cache_dir=shared.opts.hfcache_dir,
            **load_args,
            **quant_args,
        )
        flux_repo_id = "black-forest-labs/FLUX.1-dev"
        log.debug(f'Load model: cls=XOmniPipeline module=vision repo_id="{flux_repo_id}"')
        self.model.init_vision(
            flux_repo_id,
            **quant_args,
        )
        self.model.set_generation_mode('image')

    def __call__(
            self,
            prompt: str = "",
            width: int = 1024,
            height: int = 1024,
            seed: int = -1,
            temperature: float = 1.0,
            downsample_size: int = 16,
            min_p: float = 0.03,
            top_p: float = 1.0,
            cfg_scale: float = 1.0,
        ):

        if isinstance(prompt, list):
            prompt = prompt[0]
        token_h, token_w = height // downsample_size, width // downsample_size
        image_prefix = f'<SOM>{token_h} {token_w}<IMAGE>'
        generation_config = transformers.generation.GenerationConfig(
            max_new_tokens=token_h * token_w,
            do_sample=True,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            guidance_scale=cfg_scale,
            suppress_tokens=self.tokenizer.convert_tokens_to_ids(self.model.config.mm_special_tokens),
        )

        # Sample inputs:
        tokens = self.tokenizer(
            [prompt + image_prefix],
            return_tensors='pt',
            padding='longest',
            padding_side='left',
        )
        input_ids = tokens.input_ids.to(devices.device)
        attention_mask = tokens.attention_mask.to(devices.device)
        negative_ids = self.tokenizer.encode(
            image_prefix,
            add_special_tokens=False,
            return_tensors='pt',
        ).to(devices.device).expand(1, -1)

        torch.manual_seed(seed)
        tokens = self.model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            negative_prompt_ids=negative_ids,
        )

        tokens = torch.nn.functional.pad(tokens, (0, 1), value=self.tokenizer.convert_tokens_to_ids('<EOM>'))
        torch.manual_seed(seed)
        _, images = self.model.mmdecode(self.tokenizer, tokens[0], skip_special_tokens=False)
        images[0].save('/tmp/xomni_out.png')
        return images


def load_xomni(checkpoint_info, diffusers_load_config=None):
    if diffusers_load_config is None:
        diffusers_load_config = {}
    repo_id = sd_models.path_to_repo(checkpoint_info)
    sd_models.hf_auth_check(checkpoint_info)

    pipe = XOmniPipeline()
    pipe.load(repo_id, load_config=diffusers_load_config)
    return pipe
