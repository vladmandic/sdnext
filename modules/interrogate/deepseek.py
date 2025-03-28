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


# model_path = "deepseek-ai/deepseek-vl2-small"
vl_gpt = None
vl_chat_processor = None


class fake_attrdict():
    class AttrDict(dict): # dot notation access to dictionary attributes
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

# def fake_is_flash_attn_2_available():
#     return False


def predict(question, image, repo):
    global vl_gpt, vl_chat_processor # pylint: disable=global-statement
    if not shared.cmd_opts.experimental:
        shared.log.error(f'Interrogate: type=vlm model="DeepSeek VL2" repo="{repo}" is experimental-only')
        return ''
    folder = os.path.join(paths.script_path, 'repositories', 'deepseek-vl2')
    if not os.path.exists(folder):
        shared.log.error(f'Interrogate: type=vlm model="DeepSeek VL2" repo="{repo}" deepseek-vl2 repo not found')
        return ''
    if vl_gpt is None:
        sys.modules['attrdict'] = fake_attrdict
        from transformers.models.llama import modeling_llama
        modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention
        _deekseek_vl = importlib.import_module('repositories.deepseek-vl2.deepseek_vl2')
        deekseek_vl_models = importlib.import_module('repositories.deepseek-vl2.deepseek_vl2.models')
        vl_chat_processor = deekseek_vl_models.DeepseekVLV2Processor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
        vl_gpt = AutoModelForCausalLM.from_pretrained(
            repo,
            trust_remote_code=True,
            cache_dir=shared.opts.hfcache_dir,
        )
        vl_gpt = vl_gpt.to(device=devices.device, dtype=devices.dtype).eval()

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
    sd_models.move_model(vl_gpt, devices.device)
    with devices.inference_context():
        outputs = vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
            bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
            eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
            max_new_tokens=shared.opts.interrogate_vlm_max_length,
            do_sample=False,
            use_cache=True
        )
    vl_gpt = vl_gpt.to(devices.cpu)
    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print('inputs', prepare_inputs['sft_format'][0])
    print('answer', answer)
    return answer
