from modules import ui_symbols


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
    f"Google Gemini 3.1 Pro {ui_symbols.cloud}": "gemini-3.1-pro-preview",
    f"Google Gemini 3.1 Flash Lite {ui_symbols.cloud}": "gemini-3.1-flash-lite-preview",
    f"Google Gemini 3.0 Flash {ui_symbols.cloud}": "gemini-3-flash-preview",
    f"Google Gemini 2.5 Pro {ui_symbols.cloud}": "gemini-2.5-pro",
    f"Google Gemini 2.5 Flash {ui_symbols.cloud}": "gemini-2.5-flash",
}

# Default model
vlm_default = "Alibaba Qwen 2.5 VL 3B"

# Default system prompt
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
