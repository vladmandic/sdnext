import os
import json
import re
import threading
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import Request
from fastapi.responses import StreamingResponse
from transformers import StoppingCriteriaList, TextIteratorStreamer
from modules.logger import log
from .classes import CustomTokenStopCriteria, Stats


debug_log = log.debug if os.environ.get('SD_LLM_DEBUG', None) is not None else lambda *args, **kwargs: None


def enforce_rolling_context(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensures context bounds are respected while anchoring the base system prompt."""
    system_message = None
    chat_history = []
    for msg in messages:
        if msg.get("role") == "system":
            system_message = msg
        else:
            chat_history.append(msg)
    while len(chat_history) > 0:
        current_payload = ([system_message] if system_message else []) + chat_history
        try:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                test_str = self.tokenizer.apply_chat_template(current_payload, tokenize=False, add_generation_prompt=True)
            else:
                test_str = "".join([m["content"] for m in current_payload])
            total_len = len(self.tokenizer.encode(test_str))
            if (total_len + self.config.max_new_tokens) <= self.config.max_context_tokens:
                return current_payload
        except Exception:
            pass
        if len(chat_history) >= 2:
            chat_history = chat_history[2:]
            log.debug("OpenAI: Context('dropped oldest conversation')")
        elif len(chat_history) == 1:
            chat_history.pop(0)
            log.debug("OpenAI: Context('dropped history')")
        else:
            break
    return [system_message] if system_message else []


def parse_inline_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Parses model-specific text tags and builds OpenAI tool payload models."""
    tool_calls = []
    qwen_pattern = r"<\|tool_call_start\|>(.*?)(?:<\|tool_call_end\|>|$)"
    gemma_pattern = r"<tool_call>(.*?)(?:</tool_call>|$)"
    matches = re.findall(qwen_pattern, text, re.DOTALL) + re.findall(gemma_pattern, text, re.DOTALL)
    for idx, match_str in enumerate(matches):
        try:
            parsed = json.loads(match_str.strip())
            call_id = f"call_{int(time.time())}_{idx}"
            if "name" in parsed:
                args = parsed.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args)
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": parsed["name"], "arguments": args}
                })
        except Exception:
            continue
    return tool_calls


async def execute_generation(
    self,
    stats: Stats,
    request: Request,
    prompt: str,
    config: Dict[str, Any],
    stream: bool,
    stream_options: Optional[Dict[str, Any]] = None,
    images: Optional[List[Any]] = None,
):
    """Executes tensor processing, tracking streaming reasoning buffers and finish conditions."""
    start_time = time.time()
    if self.processor and images:
        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.model.device)
    else:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    model_name = getattr(self.model.config, "_name_or_path", "local-transformer")
    req_id = f"gen-{int(time.time())}"
    stop_ids = [self.tokenizer.eos_token_id]
    stop_targets = [
        "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "</tool_call>",
        "<|tool_call_end|>", "<|end|>", "<|start_header_id|>", "<|end_header_id|>",
        "<end_of_turn>"
    ]
    for t_str in stop_targets:
        t_id = self.tokenizer.convert_tokens_to_ids(t_str)
        if isinstance(t_id, int) and t_id > 0:
            stop_ids.append(t_id)
    stopping_criteria = StoppingCriteriaList([CustomTokenStopCriteria(list(set(stop_ids)))])
    chunks = { 'reasoning': [], 'content': [] }

    def execute_streaming_generation():
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=self.config.use_cache,
            **config
        )
        threading.Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True).start()

        async def production_stream_decorator() -> AsyncGenerator[str, None]:
            first_token_sent = False
            is_thinking = False
            token_count = 0
            generation_start = time.time()
            try:
                for chunk in streamer:
                    if await request.is_disconnected():
                        break
                    if not chunk:
                        continue
                    if any(t in chunk for t in ["<think>", "## Thought", "thought\n"]):
                        is_thinking = True
                        for t in ["<think>", "## Thought", "thought\n"]:
                            chunk = chunk.replace(t, "")
                    if "</think>" in chunk:
                        is_thinking = False
                        chunk = chunk.replace("</think>", "")
                    for tag in [
                        "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|tool_call_start|>",
                        "<tool_call>", "<|end|>", "<|start_header_id|>", "<|end_header_id|>",
                        "<end_of_turn>"
                    ]:
                        chunk = chunk.replace(tag, "")
                    if not chunk:
                        continue
                    if not first_token_sent:
                        stats.ttft = time.time() - start_time
                        first_token_sent = True
                    token_count += 1
                    choice_delta = {}
                    if is_thinking:
                        chunks['reasoning'].append(chunk)
                        choice_delta["reasoning_content"] = chunk
                    else:
                        chunks['content'].append(chunk)
                        choice_delta["content"] = chunk

                    finish_reason = "length" if token_count >= config.get("max_new_tokens", self.config.max_new_tokens) else None
                    payload = {
                        "id": req_id, "object": "chat.completion.chunk", "created": int(time.time()),
                        "model": model_name, "choices": [{"index": 0, "delta": choice_delta, "finish_reason": finish_reason}]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                final_payload = {
                    "id": req_id, "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_name, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                if stream_options and stream_options.get("include_usage"):
                    prompt_tokens = len(inputs.input_ids[0])
                    usage_payload = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": token_count,
                            "total_tokens": prompt_tokens + token_count
                        }
                    }
                    yield f"data: {json.dumps(usage_payload)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                duration = time.time() - generation_start
                if duration > 0 and token_count > 0:
                    stats.tps = token_count / duration
                    stats.chunks['reasoning'] = len(chunks['reasoning'])
                    stats.chunks['content'] = len(chunks['content'])
                    stats.latency = time.time() - start_time
                    stats.tools = 0
                    stats.tokens['prompt'] = len(inputs.input_ids[0])
                    stats.tokens['output'] = token_count
                    stats.streaming = True
                    stats.thinking = len(chunks['reasoning']) > 0
                    text_reasoning = " ".join(c.strip() for c in chunks["reasoning"] if len(c.strip()) > 0)
                    text_content = " ".join(c.strip() for c in chunks["content"] if len(c.strip()) > 0)
                    debug_log(f'OpenAI chunks: reasoning="{text_reasoning}"')
                    debug_log(f'OpenAI chunks: content="{text_content}"')
                    log.debug(f"OpenAI: {stats}")
        return StreamingResponse(production_stream_decorator(), media_type="text/event-stream")

    def execute_direct_generation():
        outputs = self.model.generate(
            **inputs,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=self.config.use_cache,
            **config
        )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        reasoning_text = ""
        for start_tag, end_tag in [("<think>", "</think>"), ("## Thought", "##"), ("thought\n", "\n\n")]:
            if start_tag in raw_text and end_tag in raw_text:
                parts = raw_text.split(end_tag)
                reasoning_text = parts[0].replace(start_tag, "").strip()
                raw_text = parts[1]
                break
        tool_calls = parse_inline_tool_calls(raw_text)
        clean_text = raw_text
        for tag in [
            "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "</tool_call>",
            "<|tool_call_end|>", "<|end|>", "<|start_header_id|>", "<|end_header_id|>",
            "<end_of_turn>"
        ]:
            clean_text = clean_text.replace(tag, "")
        if tool_calls:
            clean_text = re.sub(r"<\|tool_call_start\|>.*?<\|tool_call_end\|>", "", clean_text, flags=re.DOTALL)
            clean_text = re.sub(r"<tool_call>.*?</tool_call>", "", clean_text, flags=re.DOTALL).strip()

        stats.ttft = time.time() - start_time
        stats.latency = time.time() - start_time
        stats.tools = len(tool_calls)
        stats.tokens['prompt'] = len(inputs.input_ids[0])
        stats.tokens['reasoning'] = len(reasoning_text)
        stats.tokens['completion'] = len(generated_ids)
        stats.tokens['total'] = len(inputs.input_ids[0]) + len(generated_ids)
        stats.tps = stats.tokens['total'] / stats.latency if stats.latency > 0 else 0
        stats.streaming = False
        stats.thinking = len(reasoning_text) > 0
        log.debug(f"OpenAI: {stats}")

        debug_log(f'OpenAI response: "{clean_text}"')
        message_payload = {"role": "assistant", "content": clean_text if clean_text else None}
        if reasoning_text:
            debug_log(f'OpenAI reasoning: "{reasoning_text}"')
            message_payload["reasoning_content"] = reasoning_text
        if tool_calls:
            debug_log(f'OpenAI tools: {tool_calls}')
            message_payload["tool_calls"] = tool_calls
        f_reason = "tool_calls" if tool_calls else ("length" if len(generated_ids) >= config["max_new_tokens"] else "stop")

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "message": message_payload,
                "index": 0,
                "finish_reason": f_reason
            }],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(generated_ids),
                "total_tokens": len(inputs.input_ids[0]) + len(generated_ids)
            }
        }

    if stream:
        return execute_streaming_generation()
    else:
        return execute_direct_generation()
