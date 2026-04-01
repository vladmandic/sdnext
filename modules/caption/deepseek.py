# source: <https://huggingface.co/deepseek-ai/deepseek-vl2-tiny>
# implementation: <https://github.com/deepseek-ai/DeepSeek-VL2/tree/main/deepseek_vl2/serve>
"""
- run `git clone https://github.com/deepseek-ai/DeepSeek-VL2 repositories/deepseek-vl2 --depth 1`
- remove hardcoded `python==3.9` requirement due to obsolete attrdict package dependency
- patch transformers due to internal changes as deepseek requires obsolete `transformers==4.38.2`
- deepseek requires `xformers`
- broken flash_attention
"""

import os
import sys
import importlib
from transformers import AutoModelForCausalLM
from modules import shared, devices, paths, sd_models
from modules.sd_offload_aux import register_aux, deregister_aux, move_aux_to_gpu, offload_aux
from modules.logger import log


# model_path = "deepseek-ai/deepseek-vl2-small"
vl_gpt = None
vl_chat_processor = None
loaded_repo = None


class fake_attrdict:
    class AttrDict(dict):  # dot notation access to dictionary attributes
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__


def load(repo: str):
    """Load DeepSeek VL2 model (experimental)."""
    global vl_gpt, vl_chat_processor, loaded_repo  # pylint: disable=global-statement
    if not shared.cmd_opts.experimental:
        log.error(f'Caption: type=vlm model="DeepSeek VL2" repo="{repo}" is experimental-only')
        return False
    folder = os.path.join(paths.script_path, 'repositories', 'deepseek-vl2')
    if not os.path.exists(folder):
        log.error(f'Caption: type=vlm model="DeepSeek VL2" repo="{repo}" deepseek-vl2 repo not found')
        return False
    if vl_gpt is None or loaded_repo != repo:
        # GLOBAL PATCHES (not reverted): DeepSeek VL2 requires attrdict and uses LlamaFlashAttention2
        # which may not be available. These patches persist for the lifetime of the process and may
        # affect other Llama model loads (forcing standard attention instead of flash attention).
        sys.modules['attrdict'] = fake_attrdict
        from transformers.models.llama import modeling_llama
        modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
        importlib.import_module('repositories.deepseek-vl2.deepseek_vl2')
        deekseek_vl_models = importlib.import_module('repositories.deepseek-vl2.deepseek_vl2.models')
        vl_chat_processor = deekseek_vl_models.DeepseekVLV2Processor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            use_safetensors=True,
            cache_dir=shared.opts.hfcache_dir,
        )
        vl_gpt.to(dtype=devices.dtype)
        vl_gpt.eval()  # required: trust_remote_code model
        register_aux('deepseek', vl_gpt)
        loaded_repo = repo
        devices.torch_gc()
        log.info(f'Caption: type=vlm model="DeepSeek VL2" repo="{repo}"')
    move_aux_to_gpu('deepseek')
    return True


def unload():
    """Release DeepSeek VL2 model from GPU/memory."""
    global vl_gpt, vl_chat_processor, loaded_repo  # pylint: disable=global-statement
    if vl_gpt is not None:
        log.debug(f'DeepSeek unload: model="{loaded_repo}"')
        deregister_aux('deepseek')
        sd_models.move_model(vl_gpt, devices.cpu, force=True)
        vl_gpt = None
        vl_chat_processor = None
        loaded_repo = None
        devices.torch_gc(force=True)
    else:
        log.debug('DeepSeek unload: no model loaded')


def predict(question, image, repo):
    if not load(repo):
        return ''

    if len(question) < 2:
        question = "Describe the image."
    question = question.replace('<', '').replace('>', '')
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n<|ref|>{question}<|/ref|>.",
            # "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=[image],
        force_batchify=True,
        system_prompt=""
    ).to(device=devices.device, dtype=devices.dtype)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    inputs_embeds = inputs_embeds.to(device=devices.device, dtype=devices.dtype)
    try:
        with devices.inference_context():
            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=shared.opts.caption_vlm_max_length,
                do_sample=False,
                use_cache=True
            )
    finally:
        offload_aux('deepseek')
    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer
