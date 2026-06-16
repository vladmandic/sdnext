from .classes import Attention


def get_attention_config(self):
    attention = Attention(
        implementation=getattr(self.model.config, "_attn_implementation", "eager"),
        flash=getattr(self.model.config, "use_flash_attention", False)
    )
    if hasattr(self.model.config, "num_key_value_heads") and hasattr(self.model.config, "num_attention_heads"):
        kv_heads = self.model.config.num_key_value_heads
        attn_heads = self.model.config.num_attention_heads
        if kv_heads < attn_heads:
            attention.arch = f"arch=GQA kv_heads={kv_heads} attn_heads={attn_heads}"
        elif kv_heads == 1:
            attention.arch = "arch=MQA"
        else:
            attention.arch = "arch=MHA"
    return attention


def get_prompt_template(self, messages: list[str], tools) -> str:
    model_name = getattr(self.model.config, "_name_or_path", "").lower()
    if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
        template_kwargs = {
            "conversation": messages,
            "tokenize": False,
            "add_generation_prompt": True
        }
        if tools:
            template_kwargs["tools"] = tools
        prompt_str = self.tokenizer.apply_chat_template(**template_kwargs)
    else:
        prompt_str = ""
        if "llama" in model_name:
            prompt_str += "<|begin_of_text|>"
            for msg in messages:
                prompt_str += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            prompt_str += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        elif "gemma" in model_name:
            for msg in messages:
                role_tag = msg["role"] if msg["role"] != "system" else "user"
                prompt_str += f"<start_of_turn>{role_tag}\n{msg['content']}<end_of_turn>\n"
            prompt_str += "<start_of_turn>assistant\n"
        elif "qwen" in model_name:
            for msg in messages:
                prompt_str += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt_str += "<|im_start|>assistant\n"
        else:
            for msg in messages:
                prompt_str += f"<|{msg['role']}|>\n{msg['content']}\n"
            prompt_str += "<|assistant|>\n"
    return prompt_str
