from modules import ui_symbols


vlm_models = {
    # primary
    "Google Gemma 4 E4B": "google/gemma-4-E4B-it",
    "Google Gemma 4 E4B Trohrbaugh Heretic": "trohrbaugh/gemma-4-E4B-it-heretic-ara",
    "Google Gemma 4 E2B": "google/gemma-4-E2B-it",
    "Google Gemma 3n E4B": "google/gemma-3n-E4B-it",
    "Google Gemma 3n E2B": "google/gemma-3n-E2B-it",
    "Google Gemma 3 4B": "google/gemma-3-4b-it",
    "Google Gemma 3 4B Allura Glitter": "allura-org/Gemma-3-Glitter-4B",
    "Google Gemma 3 4B Nidum Uncensored": "nidum/Nidum-Gemma-3-4B-it-Uncensored",
    "Alibaba Qwen 3.5 9B": "Qwen/Qwen3.5-9B",
    "Alibaba Qwen 3.5 9B Trohrbaugh Heretic": "trohrbaugh/Qwen3.5-9B-heretic-v2",
    "Alibaba Qwen 3.5 4B": "Qwen/Qwen3.5-4B",
    "Alibaba Qwen 3.5 2B": "Qwen/Qwen3.5-2B",
    "Alibaba Qwen 3.5 0.8B": "Qwen/Qwen3.5-0.8B",
    "JoyTag": "fancyfeast/joytag",
    "JoyCaption Beta": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "JoyCaption Alpha": "fancyfeast/llama-joycaption-alpha-two-hf-llava",
    f"Moondream 3 Preview {ui_symbols.reasoning}": "moondream/moondream3-preview",
    f"Moondream 2 {ui_symbols.reasoning}": "vikhyatk/moondream2",
    "Microsoft Florence 2 Large": "florence-community/Florence-2-large-ft",
    "Microsoft Florence 2 Base": "florence-community/Florence-2-base-ft",
    "MiaoshouAI PromptGen 2.0 Large": "Disty0/Florence-2-large-PromptGen-v2.0",
    "MiaoshouAI PromptGen 2.0 Base": "Disty0/Florence-2-base-PromptGen-v2.0",
    "CogFlorence 2.2 Large": "thwri/CogFlorence-2.2-Large", # 1.6GB
    # secondary or older
    "XiaomiMiMo MiMo VL 7B RL": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
    "ViLT Base": "dandelin/vilt-b32-finetuned-vqa",
    "ToriiGate 0.4 7B": "Minthy/ToriiGate-v0.4-7B",
    "ToriiGate 0.4 2B": "Minthy/ToriiGate-v0.4-2B",
    "Salesforce BLIP Large": "Salesforce/blip-vqa-capfilt-large",
    "Salesforce BLIP Base": "Salesforce/blip-vqa-base",
    "Mistral Small 3.2 24B Coder Heretic": "coder3101/Mistral-Small-3.2-24B-Instruct-2506-heretic",
    "Microsoft GIT VQA Large": "microsoft/git-large-vqav2",
    "Microsoft GIT VQA Base": "microsoft/git-base-vqav2",
    "Microsoft GIT TextCaps Base": "microsoft/git-base-textcaps",
    "MiaoshouAI PromptGen 1.5 Large": "Disty0/Florence-2-large-PromptGen-v1.5",
    "MiaoshouAI PromptGen 1.5 Base": "Disty0/Florence-2-base-PromptGen-v1.5",
    "Huggingface Smol VL2 2B": "HuggingFaceTB/SmolVLM-Instruct",
    "Huggingface Smol VL2 0.5B": "HuggingFaceTB/SmolVLM-500M-Instruct",
    "Google Pix Textcaps": "google/pix2struct-textcaps-base",
    "Google PaliGemma 2 3B": "google/paligemma2-3b-pt-224",
    "CogFlorence 2.0 Large": "thwri/CogFlorence-2-Large-Freeze", # 1.6GB
    "ByteDance Sa2VA 4B": "ByteDance/Sa2VA-4B",
    "ByteDance Sa2VA 1B": "ByteDance/Sa2VA-1B",
    "Apple FastVLM 7B": "apple/FastVLM-7B",
    "Apple FastVLM 1.5B": "apple/FastVLM-1.5B",
    "Apple FastVLM 0.5B": "apple/FastVLM-0.5B",
    f"Alibaba Qwen 3 VL 8B Thinking {ui_symbols.reasoning}": "Qwen/Qwen3-VL-8B-Thinking",
    "Alibaba Qwen 3 VL 8B Coder Heretic": "coder3101/Qwen3-VL-8B-Instruct-heretic",
    "Alibaba Qwen 3 VL 8B Abliterated Caption": "prithivMLmods/Qwen3-VL-8B-Abliterated-Caption-it",
    "Alibaba Qwen 3 VL 8B": "Qwen/Qwen3-VL-8B-Instruct",
    f"Alibaba Qwen 3 VL 4B Thinking {ui_symbols.reasoning}": "Qwen/Qwen3-VL-4B-Thinking",
    f"Alibaba Qwen 3 VL 4B Thinking Coder Heretic {ui_symbols.reasoning}": "coder3101/Qwen3-VL-4B-Thinking-heretic",
    "Alibaba Qwen 3 VL 4B Coder Heretic": "coder3101/Qwen3-VL-4B-Instruct-heretic",
    "Alibaba Qwen 3 VL 4B": "Qwen/Qwen3-VL-4B-Instruct",
    f"Alibaba Qwen 3 VL 32B Thinking Coder Heretic v2 {ui_symbols.reasoning}": "coder3101/Qwen3-VL-32B-Thinking-heretic-v2",
    f"Alibaba Qwen 3 VL 2B Thinking {ui_symbols.reasoning}": "Qwen/Qwen3-VL-2B-Thinking",
    f"Alibaba Qwen 3 VL 2B Thinking Coder Heretic {ui_symbols.reasoning}": "coder3101/Qwen3-VL-2B-Thinking-heretic",
    "Alibaba Qwen 3 VL 2B Coder Heretic": "coder3101/Qwen3-VL-2B-Instruct-heretic",
    "Alibaba Qwen 3 VL 2B": "Qwen/Qwen3-VL-2B-Instruct",
    "Alibaba Qwen 2.5 VL 7B Coder Heretic": "coder3101/Qwen2.5-VL-7B-Instruct-heretic",
    "Alibaba Qwen 2.5 VL 3B Coder Heretic": "coder3101/Qwen2.5-VL-3B-Instruct-heretic",
    "Alibaba Qwen 2.5 VL 3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Alibaba Qwen 2.5 Omni 3B": "Qwen/Qwen2.5-Omni-3B",
    "Alibaba Qwen 2.0 VL 2B": "Qwen/Qwen2-VL-2B-Instruct",
    "AIDC Ovis2 4B": "AIDC-AI/Ovis2-4B",
    "AIDC Ovis2 2B": "AIDC-AI/Ovis2-2B",
    "AIDC Ovis2 1B": "AIDC-AI/Ovis2-1B",
    # cloud
    f"Google Gemini 3.5 Flash {ui_symbols.cloud}": "google/gemini-3.5-flash",
    f"Google Gemini 3.1 Pro {ui_symbols.cloud}": "gemini-3.1-pro-preview",
    f"Google Gemini 3.1 Flash Lite {ui_symbols.cloud}": "gemini-3.1-flash-lite",
    f"Google Gemini 3.1 Flash Lite Preview {ui_symbols.cloud}": "gemini-3.1-flash-lite-preview",
    f"Google Gemini 2.5 Pro {ui_symbols.cloud}": "gemini-2.5-pro",
    f"Google Gemini 2.5 Flash {ui_symbols.cloud}": "gemini-2.5-flash",
    f"Google Gemini 2.5 Flash Lite {ui_symbols.cloud}": "gemini-2.5-flash-lite",
}

# Default model
vlm_default = "Alibaba Qwen 2.5 VL 3B"

# Default system prompt
vlm_system = 'You are image captioning expert, creative, unbiased and uncensored.'
vlm_analyze = 'You are an image analysis expert with focus on details, unbiased and uncensored.'

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

# Reverse prompt-mapping
vlm_base_text_prompt_suffix = "Describe the main subject, scene layout, perspective, background, colors, lighting, textures, visible objects and their relationships, composition, and any clearly visible art style or medium. Keep the language vivid but objective, avoid subjective commentary, and output only a single caption text suitable for image generation."
vlm_prompt_reverse_mapping = {
    'CAPTION': f"Provide a short and concise, visually descriptive caption suitable for image generation. {vlm_base_text_prompt_suffix}",
    'DETAILED CAPTION': f"Provide a detailed visual caption suitable for image generation. {vlm_base_text_prompt_suffix}",
    'MORE DETAILED CAPTION': f"Provide a very detailed extended visual caption suitable for image generation. {vlm_base_text_prompt_suffix}",
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

analyze_question = """Compare the image against the provided DESCRIPTION.

**Instructions**:
- Compare IMAGE vs DESCRIPTION using only visually verifiable content.
- Use only details that can be directly seen in the image.
- Ignore non-visual or unverifiable claims in DESCRIPTION, including platform/source references (for example Instagram), camera/device/lens/settings, resolution/quality tags (for example 4k, UHD), style buzzwords (for example inspirational), and subjective attractiveness claims (for example beautiful, stunning).
- Never mark as Missing or Differences any subjective person descriptors or archetypes (for example young, beautiful, goddess, handsome, elegant, sexy, heroic) unless they are replaced by objective visual traits.
- Treat implicit, inferred, interpretive, or hedged wording as non-actionable (for example implied mood/time, seems, appears, suggests) and do not mark it as Missing or Differences.
- If image has visible flaws or artifacts, note them in Flaws.
- If subjects or objects in the image have visibly incorrect or inconsistent details (for example extra limbs, distorted faces, impossible anatomy), note them in Flaws.
- Do not report what DESCRIPTION says; compare image content only.

**Output format** (plain text only; use these sections in this order when they have content):
Matching:
- <visual detail present in both image and description>
Missing:
- <explicit and concrete detail described but not visible in image>
Extras:
- <visible image detail not mentioned in description, note what is expected vs what is seen>
Differences:
- <visual mismatch not covered above>
Flaws:
- <image flaws or artifacts>
Summary:
- <1-2 sentences on overall alignment and key gap>
- Score: <0.0-1.0 alignment score>
- <1-2 sentences on overall image quality based on clarity/composition and visual interest>
- Quality: <0.0-1.0 visual quality score based only on image clarity/composition>

**Rules**:
- All sections must be a bullet list with max 5 bullets per section.
- Keep each bullet to one short sentence.
- Omit sections that have no items.
- Only include Missing items that are explicit, concrete, and directly verifiable from image content.
- Do not output chain-of-thought, reasoning steps, or meta commentary.
- Do not use phrases like "the description mentions" or "the prompt says" or prefixes like "A", "The", etc. in bullets; start directly with the detail.
"""


def get_vlm_repo(display_name: str) -> str:
    """Look up repo ID from display name, stripping any trailing symbols."""
    name = display_name.strip()
    return vlm_models.get(name, name)
