import io
import os
import time
import json
import base64
import copy
import torch
import transformers
import transformers.dynamic_module_utils
from PIL import Image
from modules import shared, devices, errors, model_quant, sd_models, sd_models_compile, ui_symbols
from modules.caption import vqa_detection


# Debug logging - function-based to avoid circular import
debug_enabled = os.environ.get('SD_CAPTION_DEBUG', None) is not None

def debug(*args, **kwargs):
    if debug_enabled:
        shared.log.trace(*args, **kwargs)

vlm_default = "Alibaba Qwen 2.5 VL 3B"
vlm_models = {
    "Google Gemma 3 4B": "google/gemma-3-4b-it",
    "Google Gemma 3n E2B": "google/gemma-3n-E2B-it", # 1.5GB
    "Google Gemma 3n E4B": "google/gemma-3n-E4B-it", # 1.5GB
    "Nidum Gemma 3 4B Uncensored": "nidum/Nidum-Gemma-3-4B-it-Uncensored",
    "Allura Gemma 3 Glitter 4B": "allura-org/Gemma-3-Glitter-4B",
    "Alibaba Qwen 2.0 VL 2B": "Qwen/Qwen2-VL-2B-Instruct",
    "Alibaba Qwen 2.5 Omni 3B": "Qwen/Qwen2.5-Omni-3B",
    "Alibaba Qwen 2.5 VL 3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Alibaba Qwen 3 VL 2B": "Qwen/Qwen3-VL-2B-Instruct",
    f"Alibaba Qwen 3 VL 2B Thinking {ui_symbols.reasoning}": "Qwen/Qwen3-VL-2B-Thinking",
    "Alibaba Qwen 3 VL 4B": "Qwen/Qwen3-VL-4B-Instruct",
    f"Alibaba Qwen 3 VL 4B Thinking {ui_symbols.reasoning}": "Qwen/Qwen3-VL-4B-Thinking",
    "Alibaba Qwen 3 VL 8B": "Qwen/Qwen3-VL-8B-Instruct",
    f"Alibaba Qwen 3 VL 8B Thinking {ui_symbols.reasoning}": "Qwen/Qwen3-VL-8B-Thinking",
    "XiaomiMiMo MiMo VL 7B RL": "XiaomiMiMo/MiMo-VL-7B-RL-2508", # 8.3GB
    "Huggingface Smol VL2 0.5B": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "Huggingface Smol VL2 2B": "HuggingFaceTB/SmolVLM-Instruct",
    "Apple FastVLM 0.5B": "apple/FastVLM-0.5B",
    "Apple FastVLM 1.5B": "apple/FastVLM-1.5B",
    "Apple FastVLM 7B": "apple/FastVLM-7B",
    "Microsoft Florence 2 Base": "florence-community/Florence-2-base-ft", # 0.5GB
    "Microsoft Florence 2 Large": "florence-community/Florence-2-large-ft", # 1.5GB
    "MiaoshouAI PromptGen 1.5 Base": "Disty0/Florence-2-base-PromptGen-v1.5", # 0.5GB
    "MiaoshouAI PromptGen 1.5 Large": "Disty0/Florence-2-large-PromptGen-v1.5", # 1.5GB
    "MiaoshouAI PromptGen 2.0 Base": "Disty0/Florence-2-base-PromptGen-v2.0", # 0.5GB
    "MiaoshouAI PromptGen 2.0 Large": "Disty0/Florence-2-large-PromptGen-v2.0", # 1.5GB
    "CogFlorence 2.0 Large": "thwri/CogFlorence-2-Large-Freeze", # 1.6GB
    "CogFlorence 2.2 Large": "thwri/CogFlorence-2.2-Large", # 1.6GB
    f"Moondream 2 {ui_symbols.reasoning}": "vikhyatk/moondream2", # 3.7GB
    f"Moondream 3 Preview {ui_symbols.reasoning}": "moondream/moondream3-preview", # 9.3GB (gated)
    "Google Pix Textcaps": "google/pix2struct-textcaps-base", # 1.1GB
    "Google PaliGemma 2 3B": "google/paligemma2-3b-pt-224",
    "Salesforce BLIP Base": "Salesforce/blip-vqa-base", # 1.5GB
    "Salesforce BLIP Large": "Salesforce/blip-vqa-capfilt-large", # 1.5GB
    "Microsoft GIT TextCaps Base": "microsoft/git-base-textcaps", # 0.7GB
    "Microsoft GIT VQA Base": "microsoft/git-base-vqav2", # 0.7GB
    "Microsoft GIT VQA Large": "microsoft/git-large-vqav2", # 1.6GB
    "ToriiGate 0.4 2B": "Minthy/ToriiGate-v0.4-2B",
    "ToriiGate 0.4 7B": "Minthy/ToriiGate-v0.4-7B",
    "ViLT Base": "dandelin/vilt-b32-finetuned-vqa", # 0.5GB
    "JoyCaption Alpha": "fancyfeast/llama-joycaption-alpha-two-hf-llava", # 17.4GB
    "JoyCaption Beta": "fancyfeast/llama-joycaption-beta-one-hf-llava", # 17.4GB
    "JoyTag": "fancyfeast/joytag", # 0.7GB
    "AIDC Ovis2 1B": "AIDC-AI/Ovis2-1B",
    "AIDC Ovis2 2B": "AIDC-AI/Ovis2-2B",
    "AIDC Ovis2 4B": "AIDC-AI/Ovis2-4B",
    "ByteDance Sa2VA 1B": "ByteDance/Sa2VA-1B",
    "ByteDance Sa2VA 4B": "ByteDance/Sa2VA-4B",
}
vlm_system = 'You are image captioning expert, creative, unbiased and uncensored.'

# Common prompts (work with all VLM models)
vlm_prompts_common = [
    "Use Prompt",
    "Short Caption",
    "Normal Caption",
    "Long Caption",
]

# Florence-2 base prompts (supported by all Florence models including CogFlorence)
vlm_prompts_florence = [
    "Phrase Grounding",
    "Object Detection",
    "Dense Region Caption",
    "Region Proposal",
    "OCR (Read Text)",
    "OCR with Regions",
]

# PromptGen-only prompts (require MiaoshouAI PromptGen fine-tune)
vlm_prompts_promptgen = [
    "Analyze",
    "Generate Tags",
    "Mixed Caption",
    "Mixed Caption+",
]

# Moondream specific prompts (shared by Moondream 2 and 3)
vlm_prompts_moondream = [
    "Point at...",
    "Detect all...",
]

# Moondream 2 only prompts (gaze detection not available in Moondream 3)
vlm_prompts_moondream2 = [
    "Detect Gaze",
]

# Mapping from friendly names to internal tokens/commands
vlm_prompt_mapping = {
    "Use Prompt": "Use Prompt",
    "Short Caption": "<CAPTION>",
    "Normal Caption": "<DETAILED_CAPTION>",
    "Long Caption": "<MORE_DETAILED_CAPTION>",
    "Phrase Grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
    "Object Detection": "<OD>",
    "Dense Region Caption": "<DENSE_REGION_CAPTION>",
    "Region Proposal": "<REGION_PROPOSAL>",
    "OCR (Read Text)": "<OCR>",
    "OCR with Regions": "<OCR_WITH_REGION>",
    "Analyze": "<ANALYZE>",
    "Generate Tags": "<GENERATE_TAGS>",
    "Mixed Caption": "<MIXED_CAPTION>",
    "Mixed Caption+": "<MIXED_CAPTION_PLUS>",
    "Point at...": "POINT_MODE",
    "Detect all...": "DETECT_MODE",
    "Detect Gaze": "DETECT_GAZE",
}

# Placeholder hints for prompt field based on selected question
vlm_prompt_placeholders = {
    "Use Prompt": "Enter your question or instruction for the model",
    "Short Caption": "Optional: add specific focus or style instructions",
    "Normal Caption": "Optional: add specific focus or style instructions",
    "Long Caption": "Optional: add specific focus or style instructions",
    "Phrase Grounding": "Optional: specify phrases to ground in the image",
    "Object Detection": "Optional: specify object types to detect",
    "Dense Region Caption": "Optional: add specific instructions",
    "Region Proposal": "Optional: add specific instructions",
    "OCR (Read Text)": "Optional: add specific instructions",
    "OCR with Regions": "Optional: add specific instructions",
    "Analyze": "Optional: add specific analysis instructions",
    "Generate Tags": "Optional: add specific tagging instructions",
    "Mixed Caption": "Optional: add specific instructions",
    "Mixed Caption+": "Optional: add specific instructions",
    "Point at...": "Enter objects to locate, e.g., 'the red car' or 'all the eyes'",
    "Detect all...": "Enter object type to detect, e.g., 'cars' or 'faces'",
    "Detect Gaze": "No input needed - auto-detects face and gaze direction",
}

# Legacy list for backwards compatibility
vlm_prompts = vlm_prompts_common + vlm_prompts_florence + vlm_prompts_promptgen + vlm_prompts_moondream + vlm_prompts_moondream2

vlm_prefill = 'Answer: the image shows'


def get_prompts_for_model(model_name: str) -> list:
    """Get available prompts based on selected model."""
    if model_name is None:
        return vlm_prompts_common

    model_lower = model_name.lower()

    # Check for PromptGen models (MiaoshouAI fine-tunes with extra prompts)
    if 'promptgen' in model_lower:
        return vlm_prompts_common + vlm_prompts_florence + vlm_prompts_promptgen

    # Check for Florence-2 base / CogFlorence models (no PromptGen-specific prompts)
    if 'florence' in model_lower:
        return vlm_prompts_common + vlm_prompts_florence

    # Check for Moondream models (Moondream 2 has gaze detection, Moondream 3 does not)
    if 'moondream' in model_lower:
        if 'moondream3' in model_lower or 'moondream 3' in model_lower:
            return vlm_prompts_common + vlm_prompts_moondream
        else:  # Moondream 2 includes gaze detection
            return vlm_prompts_common + vlm_prompts_moondream + vlm_prompts_moondream2

    # Default: common prompts only
    return vlm_prompts_common


def get_internal_prompt(friendly_name: str, user_prompt: str = None) -> str:
    """Convert friendly prompt name to internal token/command."""
    internal = vlm_prompt_mapping.get(friendly_name, friendly_name)

    # Handle Moondream point/detect modes - prepend trigger phrase
    if internal == "POINT_MODE" and user_prompt:
        return f"Point at {user_prompt}"
    elif internal == "DETECT_MODE" and user_prompt:
        return f"Detect {user_prompt}"

    return internal


def get_prompt_placeholder(friendly_name: str) -> str:
    """Get placeholder text for the prompt field based on selected question."""
    return vlm_prompt_placeholders.get(friendly_name, "Enter your question or instruction")


def is_florence_task(question: str) -> bool:
    """Check if the question is a Florence-2 task token (either friendly name or internal token).

    This includes both base Florence prompts and PromptGen-specific prompts,
    since all are handled by the Florence handler.
    """
    if not question:
        return False
    # Check if it's a Florence-specific friendly name (base or PromptGen)
    if question in vlm_prompts_florence or question in vlm_prompts_promptgen:
        return True
    # Check if it's an internal Florence-2 task token (for backwards compatibility)
    florence_base_tokens = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>', '<CAPTION_TO_PHRASE_GROUNDING>',
                            '<OD>', '<DENSE_REGION_CAPTION>', '<REGION_PROPOSAL>', '<OCR>', '<OCR_WITH_REGION>']
    promptgen_tokens = ['<ANALYZE>', '<GENERATE_TAGS>', '<MIXED_CAPTION>', '<MIXED_CAPTION_PLUS>']
    return question in florence_base_tokens or question in promptgen_tokens


def is_thinking_model(model_name: str) -> bool:
    """Check if the model supports thinking mode based on its name."""
    if not model_name:
        return False
    model_lower = model_name.lower()
    # Check for known thinking models
    thinking_indicators = [
        'thinking',  # Qwen3-VL-*-Thinking models
        'moondream3',  # Moondream 3 supports thinking
        'moondream 3',
        'moondream2',  # Moondream 2 supports reasoning mode
        'moondream 2',
        'mimo',
    ]
    return any(indicator in model_lower for indicator in thinking_indicators)


def truncate_b64_in_conversation(conversation, front_chars=50, tail_chars=50, threshold=200):
    """
    Deep copy a conversation structure and truncate long base64 image strings for logging.
    Preserves front and tail of base64 strings with truncation indicator.
    """
    conv_copy = copy.deepcopy(conversation)

    def truncate_recursive(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "image" and isinstance(value, str) and len(value) > threshold:
                    # Truncate the base64 image string
                    truncated_count = len(value) - front_chars - tail_chars
                    obj[key] = f"{value[:front_chars]}...[{truncated_count} chars truncated]...{value[-tail_chars:]}"
                elif isinstance(value, (dict, list)):
                    truncate_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                truncate_recursive(item)

    truncate_recursive(conv_copy)
    return conv_copy


def keep_think_block_open(text_prompt: str) -> str:
    """Remove the closing </think> of the final assistant message so the model can continue reasoning."""
    think_open = "<think>"
    think_close = "</think>"
    last_open = text_prompt.rfind(think_open)
    if last_open == -1:
        return text_prompt
    close_index = text_prompt.find(think_close, last_open)
    if close_index == -1:
        return text_prompt
    # Skip any whitespace immediately following the closing tag
    end_close = close_index + len(think_close)
    while end_close < len(text_prompt) and text_prompt[end_close] in (' ', '\t'):
        end_close += 1
    while end_close < len(text_prompt) and text_prompt[end_close] in ('\r', '\n'):
        end_close += 1
    trimmed_prompt = text_prompt[:close_index] + text_prompt[end_close:]
    debug('VQA caption: keep_think_block_open applied to prompt segment near assistant reply')
    return trimmed_prompt


def b64(image):
    if image is None:
        return ''
    with io.BytesIO() as stream:
        image.save(stream, 'JPEG')
        values = stream.getvalue()
        encoded = base64.b64encode(values).decode()
        return encoded


def clean(response, question, prefill=None):
    strip = ['---', '\r', '\t', '**', '"', '"', '"', 'Assistant:', 'Caption:', '<|im_end|>', '<pad>']
    if isinstance(response, str):
        response = response.strip()
    elif isinstance(response, dict):
        text_response = ""
        if 'reasoning' in response and get_keep_thinking():
            r_text = response['reasoning']
            if isinstance(r_text, dict) and 'text' in r_text:
                r_text = r_text['text']
            text_response += f"Reasoning:\n{r_text}\n\nAnswer:\n"

        if 'answer' in response:
            text_response += response['answer']
        elif 'caption' in response:
            text_response += response['caption']
        elif 'task' in response:
            text_response += response['task']
        else:
            if not text_response:
                text_response = json.dumps(response)
        response = text_response
    elif isinstance(response, list):
        response = response[0]
    else:
        response = str(response)

    # Determine prefill text
    prefill_text = vlm_prefill if prefill is None else prefill
    if prefill_text is None:
        prefill_text = ""
    prefill_text = prefill_text.strip()

    question = question.replace('<', '').replace('>', '').replace('_', ' ')
    if question in response:
        response = response.split(question, 1)[1]
    while any(s in response for s in strip):
        for s in strip:
            response = response.replace(s, '')
    response = response.replace('  ', ' ').replace('*  ', '- ').strip()

    # Handle prefill retention/removal
    if get_keep_prefill():
        # Add prefill if it's missing from the cleaned response
        if len(prefill_text) > 0 and not response.startswith(prefill_text):
            sep = " "
            if not response or response[0] in ".,!?;:":
                sep = ""
            response = f"{prefill_text}{sep}{response}"
    else:
        # Remove prefill if it's present in the cleaned response
        if len(prefill_text) > 0 and response.startswith(prefill_text):
            response = response[len(prefill_text):].strip()

    return response


def _get_overrides():
    """Get generation overrides from VQA singleton if available."""
    if _instance is not None and _instance.generation_overrides is not None:
        return _instance.generation_overrides
    return {}


def get_keep_thinking():
    """Check if thinking trace should be kept, with per-request override support."""
    overrides = _get_overrides()
    if overrides.get('keep_thinking') is not None:
        return overrides['keep_thinking']
    return shared.opts.caption_vlm_keep_thinking


def strip_think_xml_tags(text: str, keep: bool = False) -> str:
    """Strip or reformat XML-style <think>...</think> blocks from model output.

    Applies to models that use HuggingFace chat templates with <think>/<\/think>
    tokens (Qwen, Gemma, SmolVLM). Models with structured reasoning APIs
    (e.g. Moondream) handle their reasoning output separately.

    The opening <think> tag is often in the prompt (not the response), so the
    response may only contain </think> without a matching <think>.

    Args:
        text: Model output text potentially containing <think>/<\/think> tags.
        keep: If True, reformat tags as human-readable Reasoning/Answer sections.
              If False, strip thinking blocks entirely.
    """
    if keep:
        if '</think>' in text and '<think>' not in text:
            text = 'Reasoning:\n' + text.replace('</think>', '\n\nAnswer:')
        else:
            text = text.replace('<think>', 'Reasoning:\n').replace('</think>', '\n\nAnswer:')
    else:
        while '</think>' in text:
            start = text.find('<think>')
            end = text.find('</think>')
            if start != -1 and start < end:
                text = text[:start] + text[end + 8:]
            else:
                text = text[end + 8:]
    return text


def get_keep_prefill():
    """Check if prefill should be kept in output, with per-request override support."""
    overrides = _get_overrides()
    if overrides.get('keep_prefill') is not None:
        return overrides['keep_prefill']
    return shared.opts.caption_vlm_keep_prefill


def get_kwargs():
    """Build generation kwargs from settings with per-request overrides from VQA instance.

    Checks the singleton VQA instance's generation_overrides for per-request overrides.
    Override keys: max_tokens, temperature, top_k, top_p, num_beams, do_sample
    None values are ignored, allowing selective override.
    """
    # Get overrides from VQA singleton if available
    overrides = _get_overrides()

    # Get base values from settings, apply overrides if provided
    max_tokens = overrides.get('max_tokens') if overrides.get('max_tokens') is not None else shared.opts.caption_vlm_max_length
    do_sample = overrides.get('do_sample') if overrides.get('do_sample') is not None else shared.opts.caption_vlm_do_sample
    num_beams = overrides.get('num_beams') if overrides.get('num_beams') is not None else shared.opts.caption_vlm_num_beams
    temperature = overrides.get('temperature') if overrides.get('temperature') is not None else shared.opts.caption_vlm_temperature
    top_k = overrides.get('top_k') if overrides.get('top_k') is not None else shared.opts.caption_vlm_top_k
    top_p = overrides.get('top_p') if overrides.get('top_p') is not None else shared.opts.caption_vlm_top_p

    kwargs = {
        'max_new_tokens': max_tokens,
        'do_sample': do_sample,
    }
    if num_beams > 0:
        kwargs['num_beams'] = num_beams
    if temperature > 0:
        kwargs['temperature'] = temperature
    if top_k > 0:
        kwargs['top_k'] = top_k
    if top_p > 0:
        kwargs['top_p'] = top_p
    return kwargs


class VQA:
    """Vision-Language Model interrogation class with per-model self-contained loading."""

    def __init__(self):
        self.processor = None
        self.model = None
        self.loaded: str = None
        self.last_annotated_image = None
        self.last_detection_data = None
        self._generation_overrides = None  # Per-request generation parameter overrides

    @property
    def generation_overrides(self):
        """Get current per-request generation parameter overrides."""
        return self._generation_overrides

    def unload(self):
        """Release VLM model from GPU/memory, including external handlers."""
        if self.model is not None:
            model_name = self.loaded
            shared.log.debug(f'VQA unload: unloading model="{model_name}"')
            sd_models.move_model(self.model, devices.cpu, force=True)
            self.model = None
            self.processor = None
            self.loaded = None
            devices.torch_gc(force=True, reason='vqa unload')
            shared.log.debug(f'VQA unload: model="{model_name}" unloaded')
        else:
            shared.log.debug('VQA unload: no internal model loaded')
        # External handlers manage their own module-level globals and are not covered by self.model
        from modules.caption import moondream3, joycaption, joytag, deepseek
        moondream3.unload()
        joycaption.unload()
        joytag.unload()
        deepseek.unload()

    def load(self, model_name: str = None):
        """Load VLM model into memory for the specified model name."""
        model_name = model_name or shared.opts.caption_vlm_model
        if not model_name:
            shared.log.warning('VQA load: no model specified')
            return
        repo = vlm_models.get(model_name)
        if repo is None:
            shared.log.error(f'VQA load: unknown model="{model_name}"')
            return

        shared.log.debug(f'VQA load: pre-loading model="{model_name}" repo="{repo}"')

        # Dispatch to appropriate loader (same logic as caption)
        repo_lower = repo.lower()
        if 'qwen' in repo_lower or 'torii' in repo_lower or 'mimo' in repo_lower:
            self._load_qwen(repo)
        elif 'gemma' in repo_lower and 'pali' not in repo_lower:
            self._load_gemma(repo)
        elif 'smol' in repo_lower:
            self._load_smol(repo)
        elif 'florence' in repo_lower:
            self._load_florence(repo)
        elif 'moondream2' in repo_lower:
            self._load_moondream(repo)
        elif 'git' in repo_lower:
            self._load_git(repo)
        elif 'blip' in repo_lower:
            self._load_blip(repo)
        elif 'vilt' in repo_lower:
            self._load_vilt(repo)
        elif 'pix' in repo_lower:
            self._load_pix(repo)
        elif 'paligemma' in repo_lower:
            self._load_paligemma(repo)
        elif 'ovis' in repo_lower:
            self._load_ovis(repo)
        elif 'sa2' in repo_lower:
            self._load_sa2(repo)
        elif 'fastvlm' in repo_lower:
            self._load_fastvlm(repo)
        elif 'moondream3' in repo_lower:
            from modules.caption import moondream3
            moondream3.load_model(repo)
            shared.log.info(f'VQA load: model="{model_name}" loaded (external handler)')
            return
        elif 'joytag' in repo_lower:
            from modules.caption import joytag
            joytag.load()
            shared.log.info(f'VQA load: model="{model_name}" loaded (external handler)')
            return
        elif 'joycaption' in repo_lower:
            from modules.caption import joycaption
            joycaption.load(repo)
            shared.log.info(f'VQA load: model="{model_name}" loaded (external handler)')
            return
        elif 'deepseek' in repo_lower:
            from modules.caption import deepseek
            deepseek.load(repo)
            shared.log.info(f'VQA load: model="{model_name}" loaded (external handler)')
            return
        else:
            shared.log.warning(f'VQA load: no pre-loader for model="{model_name}"')
            return

        sd_models.move_model(self.model, devices.device)
        shared.log.info(f'VQA load: model="{model_name}" loaded')

    def _load_fastvlm(self, repo: str):
        """Load FastVLM model and tokenizer."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            quant_args = model_quant.create_config(module='LLM')
            self.model = None
            self.processor = transformers.AutoTokenizer.from_pretrained(repo, trust_remote_code=True, cache_dir=shared.opts.hfcache_dir)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                trust_remote_code=True,
                cache_dir=shared.opts.hfcache_dir,
                **quant_args,
            )
            self.loaded = repo
            devices.torch_gc()

    def _fastvlm(self, question: str, image: Image.Image, repo: str, model_name: str = None):
        debug(f'VQA caption: handler=fastvlm model_name="{model_name}" repo="{repo}" question="{question}" image_size={image.size if image else None}')
        self._load_fastvlm(repo)
        sd_models.move_model(self.model, devices.device)
        if len(question) < 2:
            question = "Describe the image."
        question = question.replace('<', '').replace('>', '')
        IMAGE_TOKEN_INDEX = -200  # what the model code looks for
        messages = [{"role": "user", "content": f"<image>\n{question}"}]
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        pre, post = rendered.split("<image>", 1)
        pre_ids = self.processor(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.processor(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1)
        input_ids = input_ids.to(devices.device)
        attention_mask = torch.ones_like(input_ids, device=devices.device)
        px = self.model.get_vision_tower().image_processor(images=image, return_tensors="pt")
        px = px["pixel_values"].to(self.model.device, dtype=self.model.dtype)
        with devices.inference_context():
            outputs = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=128,
            )
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        return answer

    def _load_qwen(self, repo: str):
        """Load Qwen VL model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            if 'Qwen3-VL' in repo or 'Qwen3VL' in repo:
                cls_name = transformers.Qwen3VLForConditionalGeneration
            elif 'Qwen2.5-VL' in repo or 'Qwen2_5_VL' in repo or 'MiMo-VL' in repo:
                cls_name = transformers.Qwen2_5_VLForConditionalGeneration
            elif 'Qwen2-VL' in repo or 'Qwen2VL' in repo:
                cls_name = transformers.Qwen2VLForConditionalGeneration
            else:
                cls_name = transformers.AutoModelForCausalLM
            quant_args = model_quant.create_config(module='LLM')
            self.model = cls_name.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                **quant_args,
            )
            self.processor = transformers.AutoProcessor.from_pretrained(repo, max_pixels=1024*1024, cache_dir=shared.opts.hfcache_dir)
            if 'LLM' in shared.opts.cuda_compile:
                self.model = sd_models_compile.compile_torch(self.model)
            self.loaded = repo
            devices.torch_gc()

    def _qwen(self, question: str, image: Image.Image, repo: str, system_prompt: str = None, model_name: str = None, prefill: str = None, thinking_mode: bool = False):
        self._load_qwen(repo)
        sd_models.move_model(self.model, devices.device)
        # Get model class name for logging
        cls_name = self.model.__class__.__name__
        debug(f'VQA caption: handler=qwen model_name="{model_name}" model_class="{cls_name}" repo="{repo}" question="{question}" system_prompt="{system_prompt}" image_size={image.size if image else None}')

        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        system_prompt = system_prompt or shared.opts.caption_vlm_system
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": b64(image)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        # Add prefill if provided
        prefill_value = vlm_prefill if prefill is None else prefill
        prefill_text = prefill_value.strip()

        # Thinking models emit their own <think> tags via the chat template
        # Only models with thinking capability can use thinking mode
        is_thinking = is_thinking_model(model_name)

        # Standardize prefill
        prefill_value = vlm_prefill if prefill is None else prefill
        prefill_text = prefill_value.strip()
        use_prefill = len(prefill_text) > 0

        if debug_enabled:
            debug(f'VQA caption: handler=qwen conversation_roles={[msg["role"] for msg in conversation]}')
            debug(f'VQA caption: handler=qwen full_conversation={truncate_b64_in_conversation(conversation)}')
            debug(f'VQA caption: handler=qwen is_thinking={is_thinking} thinking_mode={thinking_mode} prefill="{prefill_text}"')

        # Generate base prompt using template
        # Qwen-Thinking template automatically adds "<|im_start|>assistant\n<think>\n" when add_generation_prompt=True
        try:
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError) as e:
            debug(f'VQA caption: handler=qwen chat_template fallback add_generation_prompt=True: {e}')
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Manually handle thinking tags and prefill
        if is_thinking:
            if not thinking_mode:
                # User wants to SKIP thinking.
                # Since template opened the block with <think>, we close it immediately.
                text_prompt += "</think>\n"
                if use_prefill:
                    text_prompt += prefill_text
            else:
                # User wants thinking. Prompt already ends in <think>.
                # If prefill is provided, it becomes part of the thought process.
                if use_prefill:
                    text_prompt += prefill_text
        else:
            # Standard model (not forcing <think>)
            if use_prefill:
                text_prompt += prefill_text

        if debug_enabled:
            debug(f'VQA caption: handler=qwen text_prompt="{text_prompt}"')
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(devices.device, devices.dtype)
        gen_kwargs = get_kwargs()
        debug(f'VQA caption: handler=qwen generation_kwargs={gen_kwargs} input_ids_shape={inputs.input_ids.shape}')
        with devices.inference_context():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        debug(f'VQA caption: handler=qwen output_ids_shape={output_ids.shape}')
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if debug_enabled:
            debug(f'VQA caption: handler=qwen response_before_clean="{response}"')
        if len(response) > 0:
            response[0] = strip_think_xml_tags(response[0], keep=get_keep_thinking())
        return response

    def _load_gemma(self, repo: str):
        """Load Gemma 3 model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            if '3n' in repo:
                cls = transformers.Gemma3nForConditionalGeneration  # pylint: disable=no-member
            else:
                cls = transformers.Gemma3ForConditionalGeneration
            quant_args = model_quant.create_config(module='LLM')
            self.model = cls.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                **quant_args,
            )
            if 'LLM' in shared.opts.cuda_compile:
                self.model = sd_models_compile.compile_torch(self.model)
            self.processor = transformers.AutoProcessor.from_pretrained(repo, max_pixels=1024*1024, cache_dir=shared.opts.hfcache_dir)
            self.loaded = repo
            devices.torch_gc()

    def _gemma(self, question: str, image: Image.Image, repo: str, system_prompt: str = None, model_name: str = None, prefill: str = None, thinking_mode: bool = False):
        self._load_gemma(repo)
        sd_models.move_model(self.model, devices.device)
        # Get model class name for logging
        cls_name = self.model.__class__.__name__
        debug(f'VQA caption: handler=gemma model_name="{model_name}" model_class="{cls_name}" repo="{repo}" question="{question}" system_prompt="{system_prompt}" image_size={image.size if image else None}')

        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        system_prompt = system_prompt or shared.opts.caption_vlm_system

        system_content = []
        if system_prompt is not None and len(system_prompt) > 4:
            system_content.append({"type": "text", "text": system_prompt})

        user_content = []
        if question is not None and len(question) > 4:
            user_content.append({"type": "text", "text": question})
        if image is not None:
            user_content.append({"type": "image", "image": b64(image)})
        conversation = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        # Add prefill if provided
        prefill_value = vlm_prefill if prefill is None else prefill
        prefill_text = prefill_value.strip()
        use_prefill = len(prefill_text) > 0
        # Thinking models emit their own <think> tags via the chat template
        # Use manual toggle OR auto-detection based on model name
        use_thinking = thinking_mode or is_thinking_model(model_name)
        if use_prefill:
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prefill_text}],
            })
            debug(f'VQA caption: handler=gemma prefill="{prefill_text}"')
        else:
            debug('VQA caption: handler=gemma prefill disabled (empty), relying on add_generation_prompt')
        if debug_enabled:
            debug(f'VQA caption: handler=gemma conversation_roles={[msg["role"] for msg in conversation]}')
            debug(f'VQA caption: handler=gemma full_conversation={truncate_b64_in_conversation(conversation)}')
            debug_prefill_mode = 'add_generation_prompt=False continue_final_message=True' if use_prefill else 'add_generation_prompt=True'
            debug(f'VQA caption: handler=gemma template_mode={debug_prefill_mode}')
        try:
            if use_prefill:
                text_prompt = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False,
                )
            else:
                text_prompt = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
        except (TypeError, ValueError) as e:
            debug(f'VQA caption: handler=gemma chat_template fallback add_generation_prompt=True: {e}')
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
        if use_prefill and use_thinking:
            text_prompt = keep_think_block_open(text_prompt)
        if debug_enabled:
            debug(f'VQA caption: handler=gemma text_prompt="{text_prompt}"')
        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device=devices.device, dtype=devices.dtype)
        input_len = inputs["input_ids"].shape[-1]
        gen_kwargs = get_kwargs()
        debug(f'VQA caption: handler=gemma generation_kwargs={gen_kwargs} input_len={input_len}')
        with devices.inference_context():
            generation = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        debug(f'VQA caption: handler=gemma output_ids_shape={generation.shape}')
        generation = generation[0][input_len:]
        response = self.processor.decode(generation, skip_special_tokens=True)
        if debug_enabled:
            debug(f'VQA caption: handler=gemma response_before_clean="{response}"')

        response = strip_think_xml_tags(response, keep=get_keep_thinking())
        return response

    def _load_paligemma(self, repo: str):
        """Load PaliGemma model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.processor = transformers.PaliGemmaProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
            self.model = None
            self.model = transformers.PaliGemmaForConditionalGeneration.from_pretrained(
                repo,
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
            )
            self.loaded = repo
            devices.torch_gc()

    def _paligemma(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        self._load_paligemma(repo)
        sd_models.move_model(self.model, devices.device)
        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        model_inputs = self.processor(text=question, images=image, return_tensors="pt").to(devices.device, devices.dtype)
        input_len = model_inputs["input_ids"].shape[-1]
        with devices.inference_context():
            generation = self.model.generate(
                **model_inputs,
                **get_kwargs(),
            )
        generation = generation[0][input_len:]
        response = self.processor.decode(generation, skip_special_tokens=True)
        return response

    def _load_ovis(self, repo: str):
        """Load Ovis model (requires flash-attn)."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                multimodal_max_length=32768,
                trust_remote_code=True,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.loaded = repo
            devices.torch_gc()

    def _ovis(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        try:
            import flash_attn  # pylint: disable=unused-import
        except Exception:
            shared.log.error(f'Caption: vlm="{repo}" flash-attn is not available')
            return ''
        self._load_ovis(repo)
        sd_models.move_model(self.model, devices.device)
        text_tokenizer = self.model.get_text_tokenizer()
        visual_tokenizer = self.model.get_visual_tokenizer()
        max_partition = 9
        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        question = f'<image>\n{question}'
        _prompt, input_ids, pixel_values = self.model.preprocess_inputs(question, [image], max_partition=max_partition)
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)
        pixel_values = [pixel_values]
        with devices.inference_context():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True,
                **get_kwargs())
            response = text_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

    def _load_smol(self, repo: str):
        """Load SmolVLM model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            quant_args = model_quant.create_config(module='LLM')
            self.model = transformers.AutoModelForVision2Seq.from_pretrained(
                repo,
                cache_dir=shared.opts.hfcache_dir,
                torch_dtype=devices.dtype,
                **quant_args,
            )
            self.processor = transformers.AutoProcessor.from_pretrained(repo, max_pixels=1024*1024, cache_dir=shared.opts.hfcache_dir)
            if 'LLM' in shared.opts.cuda_compile:
                self.model = sd_models_compile.compile_torch(self.model)
            self.loaded = repo
            devices.torch_gc()

    def _smol(self, question: str, image: Image.Image, repo: str, system_prompt: str = None, model_name: str = None, prefill: str = None, thinking_mode: bool = False):
        self._load_smol(repo)
        sd_models.move_model(self.model, devices.device)
        # Get model class name for logging
        cls_name = self.model.__class__.__name__
        debug(f'VQA caption: handler=smol model_name="{model_name}" model_class="{cls_name}" repo="{repo}" question="{question}" system_prompt="{system_prompt}" image_size={image.size if image else None}')

        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        system_prompt = system_prompt or shared.opts.caption_vlm_system
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": b64(image)},
                    {"type": "text", "text": question},
                ],
            }
        ]
        # Add prefill if provided
        prefill_value = vlm_prefill if prefill is None else prefill
        prefill_text = prefill_value.strip()
        use_prefill = len(prefill_text) > 0
        # Thinking models emit their own <think> tags via the chat template
        # Use manual toggle OR auto-detection based on model name
        use_thinking = thinking_mode or is_thinking_model(model_name)
        if use_prefill:
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": prefill_text}],
            })
            debug(f'VQA caption: handler=smol prefill="{prefill_text}"')
        else:
            debug('VQA caption: handler=smol prefill disabled (empty), relying on add_generation_prompt')
        if debug_enabled:
            debug(f'VQA caption: handler=smol conversation_roles={[msg["role"] for msg in conversation]}')
            debug(f'VQA caption: handler=smol full_conversation={truncate_b64_in_conversation(conversation)}')
            debug_prefill_mode = 'add_generation_prompt=False continue_final_message=True' if use_prefill else 'add_generation_prompt=True'
            debug(f'VQA caption: handler=smol template_mode={debug_prefill_mode}')
        try:
            if use_prefill:
                text_prompt = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            else:
                text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        except (TypeError, ValueError) as e:
            debug(f'VQA caption: handler=smol chat_template fallback add_generation_prompt=True: {e}')
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        if use_prefill and use_thinking:
            text_prompt = keep_think_block_open(text_prompt)
        if debug_enabled:
            debug(f'VQA caption: handler=smol text_prompt="{text_prompt}"')
        inputs = self.processor(text=text_prompt, images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(devices.device, devices.dtype)
        gen_kwargs = get_kwargs()
        debug(f'VQA caption: handler=smol generation_kwargs={gen_kwargs}')
        with devices.inference_context():
            output_ids = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        debug(f'VQA caption: handler=smol output_ids_shape={output_ids.shape}')
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        if debug_enabled:
            debug(f'VQA caption: handler=smol response_before_clean="{response}"')

        if len(response) > 0:
            response[0] = strip_think_xml_tags(response[0], keep=get_keep_thinking())
        return response

    def _load_git(self, repo: str):
        """Load Microsoft GIT model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            self.model = transformers.GitForCausalLM.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.processor = transformers.GitProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
            self.loaded = repo
            devices.torch_gc()

    def _git(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        self._load_git(repo)
        sd_models.move_model(self.model, devices.device)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        git_dict = {}
        git_dict['pixel_values'] = pixel_values.to(devices.device, devices.dtype)
        if len(question) > 0:
            input_ids = self.processor(text=question, add_special_tokens=False).input_ids
            input_ids = [self.processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            git_dict['input_ids'] = input_ids.to(devices.device)
        with devices.inference_context():
            generated_ids = self.model.generate(**git_dict)
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def _load_blip(self, repo: str):
        """Load Salesforce BLIP model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            self.model = transformers.BlipForQuestionAnswering.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.processor = transformers.BlipProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
            self.loaded = repo
            devices.torch_gc()

    def _blip(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        self._load_blip(repo)
        sd_models.move_model(self.model, devices.device)
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = inputs.to(devices.device, devices.dtype)
        with devices.inference_context():
            outputs = self.model.generate(**inputs)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

    def _load_vilt(self, repo: str):
        """Load ViLT model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            self.model = transformers.ViltForQuestionAnswering.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.processor = transformers.ViltProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
            self.loaded = repo
            devices.torch_gc()

    def _vilt(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        self._load_vilt(repo)
        sd_models.move_model(self.model, devices.device)
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = inputs.to(devices.device)
        with devices.inference_context():
            outputs = self.model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        response = self.model.config.id2label[idx]
        return response

    def _load_pix(self, repo: str):
        """Load Pix2Struct model and processor."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            self.model = transformers.Pix2StructForConditionalGeneration.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.processor = transformers.Pix2StructProcessor.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
            self.loaded = repo
            devices.torch_gc()

    def _pix(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        self._load_pix(repo)
        sd_models.move_model(self.model, devices.device)
        if len(question) > 0:
            inputs = self.processor(images=image, text=question, return_tensors="pt").to(devices.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(devices.device)
        with devices.inference_context():
            outputs = self.model.generate(**inputs)
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        return response

    def _load_moondream(self, repo: str):
        """Load Moondream 2 model and tokenizer."""
        if self.model is None or self.loaded != repo:
            shared.log.debug(f'Caption load: vlm="{repo}"')
            self.model = None
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                repo,
                revision="2025-06-21",
                trust_remote_code=True,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
            )
            self.processor = transformers.AutoTokenizer.from_pretrained(repo, cache_dir=shared.opts.hfcache_dir)
            self.loaded = repo
            self.model.eval()  # required: trust_remote_code model
            devices.torch_gc()

    def _moondream(self, question: str, image: Image.Image, repo: str, model_name: str = None, thinking_mode: bool = False):
        debug(f'VQA caption: handler=moondream model_name="{model_name}" repo="{repo}" question="{question}" thinking_mode={thinking_mode}')
        self._load_moondream(repo)
        sd_models.move_model(self.model, devices.device)
        question = question.replace('<', '').replace('>', '').replace('_', ' ')
        with devices.inference_context():
            if question == 'CAPTION':
                response = self.model.caption(image, length="short")['caption']
            elif question == 'DETAILED CAPTION':
                response = self.model.caption(image, length="normal")['caption']
            elif question == 'MORE DETAILED CAPTION':
                response = self.model.caption(image, length="long")['caption']
            elif question.lower().startswith('point at ') or question == 'POINT_MODE':
                target = question[9:].strip() if question.lower().startswith('point at ') else ''
                if not target:
                    return "Please specify an object to locate"
                debug(f'VQA caption: handler=moondream method=point target="{target}"')
                result = self.model.point(image, target)
                debug(f'VQA caption: handler=moondream point_raw_result={result}')
                points = vqa_detection.parse_points(result)
                if points:
                    self.last_detection_data = {'points': points}
                    return vqa_detection.format_points_text(points)
                return "Object not found"
            elif question == 'DETECT_GAZE' or question.lower() == 'detect gaze':
                # Must be checked before generic 'detect ' prefix to avoid matching as detect target="Gaze"
                debug('VQA caption: handler=moondream method=detect_gaze')
                faces = self.model.detect(image, "face")
                debug(f'VQA caption: handler=moondream detect_gaze faces={faces}')
                if faces.get('objects'):
                    eye_x, eye_y = vqa_detection.calculate_eye_position(faces['objects'][0])
                    result = self.model.detect_gaze(image, eye=(eye_x, eye_y))
                    debug(f'VQA caption: handler=moondream detect_gaze result={result}')
                    if result.get('gaze'):
                        gaze = result['gaze']
                        self.last_detection_data = {'points': [(gaze['x'], gaze['y'])]}
                        return f"Gaze direction: ({gaze['x']:.3f}, {gaze['y']:.3f})"
                return "No face/gaze detected"
            elif question.lower().startswith('detect ') or question == 'DETECT_MODE':
                target = question[7:].strip() if question.lower().startswith('detect ') else ''
                if not target:
                    return "Please specify an object to detect"
                debug(f'VQA caption: handler=moondream method=detect target="{target}"')
                result = self.model.detect(image, target)
                debug(f'VQA caption: handler=moondream detect_raw_result={result}')
                detections = vqa_detection.parse_detections(result, target)
                if detections:
                    self.last_detection_data = {'detections': detections}
                    return vqa_detection.format_detections_text(detections, include_confidence=False)
                return "No objects detected"
            else:
                debug(f'VQA caption: handler=moondream method=query question="{question}" reasoning={thinking_mode}')
                result = self.model.query(image, question, reasoning=thinking_mode)
                response = result['answer']
                debug(f'VQA caption: handler=moondream query_result keys={list(result.keys()) if isinstance(result, dict) else "not dict"}')
                if thinking_mode and 'reasoning' in result:
                    reasoning_text = result['reasoning'].get('text', '') if isinstance(result['reasoning'], dict) else str(result['reasoning'])
                    debug(f'VQA caption: handler=moondream reasoning_text="{reasoning_text[:100]}..."')
                    if get_keep_thinking():
                        response = f"Reasoning:\n{reasoning_text}\n\nAnswer:\n{response}"
                    # When keep_thinking is False, just use the answer (reasoning is discarded)
        return response

    def _load_florence(self, repo: str, revision: str = None):
        """Load Florence-2 model and processor."""
        _get_imports = transformers.dynamic_module_utils.get_imports

        def get_imports(f):
            R = _get_imports(f)
            if "flash_attn" in R:
                R.remove("flash_attn")  # flash_attn is optional
            return R

        # Handle revision splitting and caching
        cache_key = repo
        effective_revision = revision
        repo_name = repo

        if repo and '@' in repo:
            repo_name, revision_from_repo = repo.split('@')
            effective_revision = revision_from_repo

        if self.model is None or self.loaded != cache_key:
            shared.log.debug(f'Caption load: vlm="{repo_name}" revision="{effective_revision}" path="{shared.opts.hfcache_dir}"')
            transformers.dynamic_module_utils.get_imports = get_imports
            self.model = None
            quant_args = model_quant.create_config(module='LLM')
            self.model = transformers.Florence2ForConditionalGeneration.from_pretrained(
                repo_name,
                revision=effective_revision,
                torch_dtype=devices.dtype,
                cache_dir=shared.opts.hfcache_dir,
                **quant_args,
            )
            self.processor = transformers.AutoProcessor.from_pretrained(repo_name, max_pixels=1024*1024, trust_remote_code=True, revision=effective_revision, cache_dir=shared.opts.hfcache_dir)
            transformers.dynamic_module_utils.get_imports = _get_imports
            self.loaded = cache_key
            devices.torch_gc()

    def _florence(self, question: str, image: Image.Image, repo: str, revision: str = None, model_name: str = None): # pylint: disable=unused-argument
        self._load_florence(repo, revision)
        sd_models.move_model(self.model, devices.device)
        if question.startswith('<'):
            task = question.split('>', 1)[0] + '>'
        else:
            task = '<MORE_DETAILED_CAPTION>'
        debug(f'VQA caption: handler=florence model_name="{model_name}" repo="{repo}" task="{task}" question="{question}" image_size={image.size}')
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        input_ids = inputs['input_ids'].to(devices.device)
        pixel_values = inputs['pixel_values'].to(devices.device, devices.dtype)
        debug(f'VQA caption: handler=florence input_ids={input_ids.shape} pixel_values={pixel_values.shape} dtype={pixel_values.dtype}')
        # Florence-2 requires beam search, not sampling - sampling causes probability tensor errors
        overrides = _get_overrides()
        max_tokens = overrides.get('max_tokens') if overrides.get('max_tokens') is not None else shared.opts.caption_vlm_max_length
        gen_kwargs = {'max_new_tokens': max_tokens, 'num_beams': 3, 'do_sample': False}
        # Some Florence fine-tunes (e.g., CogFlorence) don't have decoder_start_token_id set
        if getattr(self.model.config, 'decoder_start_token_id', None) is None:
            bos_token_id = getattr(self.processor.tokenizer, 'bos_token_id', None) or 0
            gen_kwargs['decoder_start_token_id'] = bos_token_id
            debug(f'VQA caption: handler=florence setting decoder_start_token_id={bos_token_id}')
        debug(f'VQA caption: handler=florence generation_kwargs={gen_kwargs}')
        with devices.inference_context(), devices.bypass_sdpa_hijacks():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                **gen_kwargs,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            debug(f'VQA caption: handler=florence generated_text="{generated_text}"')
            # task="task" is intentional: produces {'task': text} which both parse_florence_detections and
            # format_florence_response handle via explicit 'task' key fallbacks, avoiding task-token-specific keys
            response = self.processor.post_process_generation(generated_text, task="task", image_size=(image.width, image.height))
            debug(f'VQA caption: handler=florence raw_response={response}')
        return response

    def _load_sa2(self, repo: str):
        """Load SA2VA model and tokenizer."""
        if self.model is None or self.loaded != repo:
            self.model = None
            self.model = transformers.AutoModel.from_pretrained(
                repo,
                torch_dtype=devices.dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True)
            self.model = self.model.eval()  # required: trust_remote_code model
            self.processor = transformers.AutoTokenizer.from_pretrained(
                repo,
                trust_remote_code=True,
                use_fast=False,
            )
            self.loaded = repo
            devices.torch_gc()

    def _sa2(self, question: str, image: Image.Image, repo: str, model_name: str = None): # pylint: disable=unused-argument
        self._load_sa2(repo)
        sd_models.move_model(self.model, devices.device)
        if question.startswith('<'):
            task = question.split('>', 1)[0] + '>'
        else:
            task = '<MORE_DETAILED_CAPTION>'
        input_dict = {
            'image': image,
            'text': f'<image>{task}',
            'past_text': '',
            'mask_prompts': None,
            'tokenizer': self.processor,
        }
        with devices.inference_context():
            return_dict = self.model.predict_forward(**input_dict)
        response = return_dict["prediction"]  # the text format answer
        return response

    def caption(self, question: str = '', system_prompt: str = None, prompt: str = None, image: Image.Image = None, model_name: str = None, prefill: str = None, thinking_mode: bool = None, quiet: bool = False, generation_kwargs: dict = None) -> str:
        """
        Main entry point for VQA captioning. Returns string answer.
        Detection data stored in self.last_detection_data for annotated image creation.

        Args:
            question: Question/task to perform
            system_prompt: System prompt for the model
            prompt: Additional prompt text
            image: PIL Image to process
            model_name: Model to use (defaults to settings)
            prefill: Text to prefill the response with
            thinking_mode: Enable thinking/reasoning mode (None = use settings)
            quiet: Suppress logging
            generation_kwargs: Optional dict with generation parameter overrides:
                max_tokens, temperature, top_k, top_p, num_beams, do_sample, keep_thinking, keep_prefill
        """
        self.last_annotated_image = None
        self.last_detection_data = None
        self._generation_overrides = generation_kwargs  # Set per-request overrides
        jobid = shared.state.begin('Caption LLM')
        t0 = time.time()
        model_name = model_name or shared.opts.caption_vlm_model
        prefill = vlm_prefill if prefill is None else prefill  # Use provided prefill when specified
        thinking_mode = shared.opts.caption_vlm_thinking_mode if thinking_mode is None else thinking_mode  # Resolve from settings if not specified
        if isinstance(image, list):
            image = image[0] if len(image) > 0 else None
        if isinstance(image, dict) and 'name' in image:
            image = Image.open(image['name'])
        if isinstance(image, Image.Image):
            if image.width > 768 or image.height > 768:
                image.thumbnail((768, 768), Image.Resampling.LANCZOS)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        if image is None:
            shared.log.error(f'VQA caption: model="{model_name}" error="No input image provided"')
            self._generation_overrides = None
            shared.state.end(jobid)
            return 'Error: No input image provided. Please upload or select an image.'

        # Convert friendly prompt names to internal tokens/commands
        if question == "Use Prompt":
            # Use content from Prompt field directly - requires user input
            if not prompt or len(prompt.strip()) < 2:
                shared.log.error(f'VQA caption: model="{model_name}" error="Please enter a prompt"')
                self._generation_overrides = None
                shared.state.end(jobid)
                return 'Error: Please enter a question or instruction in the Prompt field.'
            question = prompt
        elif question in vlm_prompt_mapping:
            # Check if this is a mode that requires user input (Point/Detect)
            raw_mapping = vlm_prompt_mapping.get(question)
            if raw_mapping in ("POINT_MODE", "DETECT_MODE"):
                # These modes require user input in the prompt field
                if not prompt or len(prompt.strip()) < 2:
                    shared.log.error(f'VQA caption: model="{model_name}" error="Please specify what to find in the prompt field"')
                    self._generation_overrides = None
                    shared.state.end(jobid)
                    return 'Error: Please specify what to find in the prompt field (e.g., "the red car" or "faces").'
            # Convert friendly name to internal token (handles Point/Detect prefix)
            question = get_internal_prompt(question, prompt)
        # else: question is already an internal token or custom text

        from modules import modelloader
        modelloader.hf_login()

        try:
            if model_name is None:
                shared.log.error(f'Caption: type=vlm model="{model_name}" no model selected')
                shared.state.end(jobid)
                return ''
            vqa_model = vlm_models.get(model_name, None)
            if vqa_model is None:
                shared.log.error(f'Caption: type=vlm model="{model_name}" unknown')
                shared.state.end(jobid)
                return ''

            handler = 'unknown'
            if 'git' in vqa_model.lower():
                handler = 'git'
                answer = self._git(question, image, vqa_model, model_name)
            elif 'vilt' in vqa_model.lower():
                handler = 'vilt'
                answer = self._vilt(question, image, vqa_model, model_name)
            elif 'blip' in vqa_model.lower():
                handler = 'blip'
                answer = self._blip(question, image, vqa_model, model_name)
            elif 'pix' in vqa_model.lower():
                handler = 'pix'
                answer = self._pix(question, image, vqa_model, model_name)
            elif 'moondream3' in vqa_model.lower():
                handler = 'moondream3'
                from modules.caption import moondream3
                answer = moondream3.predict(question, image, vqa_model, model_name, thinking_mode=thinking_mode)
            elif 'moondream2' in vqa_model.lower():
                handler = 'moondream'
                answer = self._moondream(question, image, vqa_model, model_name, thinking_mode)
            elif 'florence' in vqa_model.lower():
                handler = 'florence'
                answer = self._florence(question, image, vqa_model, None, model_name)
                # Parse Florence detection response for annotated image (handles both dict and string formats)
                florence_detections = vqa_detection.parse_florence_detections(answer, image.size if image else None)
                if florence_detections:
                    self.last_detection_data = {'detections': florence_detections}
                    debug(f'VQA caption: handler=florence parsed {len(florence_detections)} detections')
                # Format dict answer as readable string (string answers pass through unchanged)
                if isinstance(answer, dict):
                    answer = vqa_detection.format_florence_response(answer)
            elif 'qwen' in vqa_model.lower() or 'torii' in vqa_model.lower() or 'mimo' in vqa_model.lower():
                handler = 'qwen'
                answer = self._qwen(question, image, vqa_model, system_prompt, model_name, prefill, thinking_mode)
            elif 'smol' in vqa_model.lower():
                handler = 'smol'
                answer = self._smol(question, image, vqa_model, system_prompt, model_name, prefill, thinking_mode)
            elif 'joytag' in vqa_model.lower():
                handler = 'joytag'
                from modules.caption import joytag
                answer = joytag.predict(image)
            elif 'joycaption' in vqa_model.lower():
                handler = 'joycaption'
                from modules.caption import joycaption
                answer = joycaption.predict(question, image, vqa_model)
            elif 'deepseek' in vqa_model.lower():
                handler = 'deepseek'
                from modules.caption import deepseek
                answer = deepseek.predict(question, image, vqa_model)
            elif 'paligemma' in vqa_model.lower():
                handler = 'paligemma'
                answer = self._paligemma(question, image, vqa_model, model_name)
            elif 'gemma' in vqa_model.lower():
                handler = 'gemma'
                answer = self._gemma(question, image, vqa_model, system_prompt, model_name, prefill, thinking_mode)
            elif 'ovis' in vqa_model.lower():
                handler = 'ovis'
                answer = self._ovis(question, image, vqa_model, model_name)
            elif 'sa2' in vqa_model.lower():
                handler = 'sa2'
                answer = self._sa2(question, image, vqa_model, model_name)
            elif 'fastvlm' in vqa_model.lower():
                handler = 'fastvlm'
                answer = self._fastvlm(question, image, vqa_model, model_name)
            else:
                answer = 'unknown model'
        except Exception as e:
            errors.display(e, 'VQA')
            answer = 'error'

        if shared.opts.caption_offload and self.model is not None:
            sd_models.move_model(self.model, devices.cpu, force=True)
        devices.torch_gc(force=True, reason='vqa')

        # Clean the answer
        answer = clean(answer, question, prefill)

        # Create annotated image if detection data is available
        if self.last_detection_data and isinstance(self.last_detection_data, dict) and image:
            detections = self.last_detection_data.get('detections', None)
            points = self.last_detection_data.get('points', None)
            if detections or points:
                self.last_annotated_image = vqa_detection.draw_bounding_boxes(image, detections or [], points)
                debug(f'VQA caption: handler={handler} created annotated image detections={len(detections) if detections else 0} points={len(points) if points else 0}')

        debug(f'VQA caption: handler={handler} response_after_clean="{answer}" has_annotation={self.last_annotated_image is not None}')
        t1 = time.time()
        if not quiet:
            shared.log.debug(f'Caption: type=vlm model="{model_name}" repo="{vqa_model}" args={get_kwargs()} time={t1-t0:.2f}')
        self._generation_overrides = None  # Clear per-request overrides
        shared.state.end(jobid)
        return answer


    def batch(self, model_name, system_prompt, batch_files, batch_folder, batch_str, question, prompt, write, append, recursive, prefill=None, thinking_mode=False):
        class BatchWriter:
            def __init__(self, folder, mode='w'):
                self.folder = folder
                self.csv = None
                self.file = None
                self.mode = mode

            def add(self, file, prompt_text):
                txt_file = os.path.splitext(file)[0] + ".txt"
                if self.mode == 'a':
                    prompt_text = '\n' + prompt_text
                with open(os.path.join(self.folder, txt_file), self.mode, encoding='utf-8') as f:
                    f.write(prompt_text)

            def close(self):
                if self.file is not None:
                    self.file.close()

        files = []
        if batch_files is not None:
            files += [f.name for f in batch_files]
        if batch_folder is not None:
            files += [f.name for f in batch_folder]
        if batch_str is not None and len(batch_str) > 0 and os.path.exists(batch_str) and os.path.isdir(batch_str):
            from modules.files_cache import list_files
            files += list(list_files(batch_str, ext_filter=['.png', '.jpg', '.jpeg', '.webp', '.jxl'], recursive=recursive))
        if len(files) == 0:
            shared.log.warning('Caption batch: type=vlm no images')
            return ''
        jobid = shared.state.begin('Caption batch')
        prompts = []
        if write:
            mode = 'w' if not append else 'a'
            writer = BatchWriter(os.path.dirname(files[0]), mode=mode)
        orig_offload = shared.opts.caption_offload
        shared.opts.caption_offload = False
        try:
            import rich.progress as rp
            pbar = rp.Progress(rp.TextColumn('[cyan]Caption:'), rp.BarColumn(), rp.MofNCompleteColumn(), rp.TaskProgressColumn(), rp.TimeRemainingColumn(), rp.TimeElapsedColumn(), rp.TextColumn('[cyan]{task.description}'), console=shared.console)
            with pbar:
                task = pbar.add_task(total=len(files), description='starting...')
                for file in files:
                    pbar.update(task, advance=1, description=file)
                    try:
                        if shared.state.interrupted:
                            break
                        img = Image.open(file)
                        result = self.caption(question, system_prompt, prompt, img, model_name, prefill, thinking_mode, quiet=True)
                        # Save annotated image if available
                        if self.last_annotated_image and write:
                            annotated_path = os.path.splitext(file)[0] + "_annotated.png"
                            self.last_annotated_image.save(annotated_path)
                        prompts.append(result)
                        if write:
                            writer.add(file, result)
                    except Exception as e:
                        shared.log.error(f'Caption batch: {e}')
            if write:
                writer.close()
        finally:
            shared.opts.caption_offload = orig_offload
        shared.state.end(jobid)
        return '\n\n'.join(prompts)


# Module-level singleton instance
_instance = None


def get_instance() -> VQA:
    """Get or create the singleton VQA instance."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        _instance = VQA()
    return _instance


# Backwards-compatible module-level functions
def caption(*args, **kwargs):
    return get_instance().caption(*args, **kwargs)



def unload_model():
    return get_instance().unload()


def load_model(model_name: str = None):
    return get_instance().load(model_name)


def get_last_annotated_image():
    return get_instance().last_annotated_image


def batch(*args, **kwargs):
    return get_instance().batch(*args, **kwargs)
