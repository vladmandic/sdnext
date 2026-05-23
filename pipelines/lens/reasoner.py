"""Prompt reasoner - refines user prompts before they hit the text encoder.

Decision matrix (driven by ``enable`` and whether an OpenAI-compatible API
is configured):

| ``enable``        | OpenAI API set? | Behavior                                  |
| ----------------- | --------------- | ----------------------------------------- |
| ``False`` (default) | no            | identity (return prompts unchanged)       |
| ``False``         | yes             | refine via OpenAI-compatible API          |
| ``True``          | no              | refine via the local GPT-OSS              |
| ``True``          | yes             | refine via OpenAI-compatible API          |

The OpenAI path uses any chat-completion endpoint speaking the OpenAI v1
schema (e.g. ``vllm``, ``ollama --openai-compat``, ``together.ai``).
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import torch


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
HARMONY_FINAL_RE = re.compile(
    r"<\|start\|>assistant(?:<\|channel\|>final)?<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)
HARMONY_DIRECT_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
    re.DOTALL,
)
PLAIN_HARMONY_FINAL_MARKER_RE = re.compile(r"assistant\s*final\s*", re.IGNORECASE)
PLAIN_HARMONY_DIRECT_FINAL_RE = re.compile(r"(?:^|\n)\s*final\s*", re.IGNORECASE)


SYSTEM_PROMPT = """
You are a prompt rewriter for a text-to-image model.
Your task is to convert the user's input into a single, precise, descriptive image prompt suitable for a text-to-image model.
Follow these rules strictly:

1. The output must be a clear and accurate description of a single image scene, written in the style of a text-to-image prompt.
  - Do not include explanations, reasoning, commentary, or meta text.
  - Do not ask questions.
  - Do not output multiple options.
  - Do not use uncertain, speculative, or alternative wording such as "maybe", "possibly", "perhaps", "or", "might", or "could".

2. Preserve the user's intended scene faithfully.
  - Do not change the objects, entities, attributes, actions, relationships, or core setting explicitly described by the user.
  - You may add reasonable visual details only when they help make the image concrete and coherent.
  - Any added details must be consistent with the user's description and must not introduce new important objects or alter the meaning.

3. If the image contains many main subjects of the same kind, describe each subject in detail, including humans, animals, objects, and any other prominent elements.
  - For each subject, include its appearance, color, size, shape, material, pose, expression, and position if applicable in the scene.
  - Make sure every main subject is clearly distinguishable from the others, such as in a scene with "4 dogs," describing each dog separately.

4. The output must fully cover the scene implied by the user's input.
  - Include the main subjects, relevant attributes, actions, spatial relationships, environment, and visible details necessary to render the scene.
  - If the user input is already sufficiently detailed and already suitable for image generation, keep it unchanged or only make minimal edits for fluency and clarity.

5. Resolve content that requires simple inference into explicit visual results when the result is unambiguous and visually representable.
  - Example: if the user says "the answer to 2+2 is written on the blackboard", output should explicitly describe "the blackboard shows 2+2=4".
  - Use only direct, necessary inference that is clearly implied by the user input.
  - Do not invent hidden facts, backstory, or ambiguous details.

6. Language rule:
  - If the user input is not in English, output in the same language.
  - Otherwise, output in English.

7. Output format:
  - Output exactly one final rewritten prompt.
  - Do not use bullet points, numbering, JSON, XML, Markdown, or quotation marks unless they are part of the scene itself.

Your goal is to produce a prompt that is concrete, visual, faithful to the user intent, and directly usable as input to a text-to-image model.
""".strip()


def _extract_plain_harmony_final(text: str) -> Optional[str]:
    matches = list(PLAIN_HARMONY_FINAL_MARKER_RE.finditer(text))
    if matches:
        final_text = text[matches[-1].end() :].strip()
        return final_text or None

    if text.lstrip().lower().startswith("analysis"):
        matches = list(PLAIN_HARMONY_DIRECT_FINAL_RE.finditer(text))
        if matches:
            final_text = text[matches[-1].end() :].strip()
            return final_text or None
    return None


def _clean_reasoner_output(text: str) -> str:
    text = text.strip()
    final_match = None
    for match in HARMONY_FINAL_RE.finditer(text):
        final_match = match
    if final_match is not None:
        text = final_match.group(1).strip()
    else:
        direct_final_match = None
        for match in HARMONY_DIRECT_FINAL_RE.finditer(text):
            direct_final_match = match
        if direct_final_match is not None:
            text = direct_final_match.group(1).strip()
        else:
            plain_final = _extract_plain_harmony_final(text)
            if plain_final is not None:
                text = plain_final

    text = THINK_BLOCK_RE.sub("", text).strip()
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1].strip()
    plain_final = _extract_plain_harmony_final(text)
    if plain_final is not None:
        text = plain_final
    for token in (
        "<|channel|>analysis<|message|>",
        "<|start|>assistant<|channel|>analysis<|message|>",
        "<|channel|>final<|message|>",
        "<|start|>assistant<|channel|>final<|message|>",
        "<|start|>assistant<|message|>",
        "<|return|>",
        "<|end|>",
        "<|endoftext|>",
        "<|im_end|>",
    ):
        text = text.replace(token, "")

    text = text.strip()
    if re.match(r"^(?:analysis|assistant\s*analysis)(?:\b|[A-Z])", text, flags=re.IGNORECASE | re.DOTALL):
        return ""
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    return " ".join(text.split())


class PromptReasoner:
    """Optional prompt rewriter, used by ``LensPipeline.refine_prompt``."""

    def __init__(
        self,
        *,
        text_encoder=None,
        tokenizer=None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.openai_model = openai_model
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self._client = None  # lazily constructed

    @property
    def has_api(self) -> bool:
        return bool(self.openai_api_key and self.openai_model)

    def refine(self, prompts: Sequence[str], enable: bool) -> List[str]:
        prompts = list(prompts)
        # API takes precedence whenever it is configured.
        if self.has_api:
            return self._refine_via_api(prompts)
        if enable:
            if self.text_encoder is None or self.tokenizer is None:
                raise RuntimeError(
                    "Reasoner enabled with no API: both text_encoder and "
                    "tokenizer must be provided to use the local GPT-OSS as "
                    "the reasoner."
                )
            return self._refine_via_local(prompts)
        return prompts

    # ------------------------------------------------------------------
    # Local GPT-OSS path
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _refine_via_local(self, prompts: List[str]) -> List[str]:
        refined: List[str] = []
        for prompt in prompts:
            system_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                "Keep any reasoning private. The visible answer must contain only the final rewritten prompt."
            )
            conversation = [
                {"role": "system", "content": system_prompt, "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
            ]
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True, reasoning_effort="low"
            )
            input_ids = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=True
            ).input_ids
            out_ids = self.text_encoder.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0.0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = out_ids[0, input_ids.shape[1]:]
            text_out = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            clean_text_out = _clean_reasoner_output(text_out)
            refined.append(clean_text_out or prompt)
        return refined

    # ------------------------------------------------------------------
    # OpenAI-compatible API path
    # ------------------------------------------------------------------

    def _client_or_raise(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed. `pip install openai` to use "
                    "the API-based reasoner."
                ) from exc
            self._client = OpenAI(
                api_key=self.openai_api_key,
                base_url=self.openai_base_url,
            )
        return self._client

    def _refine_via_api(self, prompts: List[str]) -> List[str]:
        client = self._client_or_raise()
        out: List[str] = []
        for prompt in prompts:
            resp = client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_new_tokens,
            )
            text = (resp.choices[0].message.content or "").strip()
            out.append(text or prompt)
        return out
