export const VLM_DEFAULT = "Alibaba Qwen 2.5 VL 3B";

export const VLM_MODELS: Record<string, string> = {
  "Google Gemma 3 4B": "google/gemma-3-4b-it",
  "Google Gemma 3n E2B": "google/gemma-3n-E2B-it",
  "Google Gemma 3n E4B": "google/gemma-3n-E4B-it",
  "Nidum Gemma 3 4B Uncensored": "nidum/Nidum-Gemma-3-4B-it-Uncensored",
  "Allura Gemma 3 Glitter 4B": "allura-org/Gemma-3-Glitter-4B",
  "Alibaba Qwen 2.0 VL 2B": "Qwen/Qwen2-VL-2B-Instruct",
  "Alibaba Qwen 2.5 Omni 3B": "Qwen/Qwen2.5-Omni-3B",
  "Alibaba Qwen 2.5 VL 3B": "Qwen/Qwen2.5-VL-3B-Instruct",
  "Alibaba Qwen 3 VL 2B": "Qwen/Qwen3-VL-2B-Instruct",
  "Alibaba Qwen 3 VL 2B Thinking": "Qwen/Qwen3-VL-2B-Thinking",
  "Alibaba Qwen 3 VL 4B": "Qwen/Qwen3-VL-4B-Instruct",
  "Alibaba Qwen 3 VL 4B Thinking": "Qwen/Qwen3-VL-4B-Thinking",
  "Alibaba Qwen 3 VL 8B": "Qwen/Qwen3-VL-8B-Instruct",
  "Alibaba Qwen 3 VL 8B Thinking": "Qwen/Qwen3-VL-8B-Thinking",
  "XiaomiMiMo MiMo VL 7B RL": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
  "Huggingface Smol VL2 0.5B": "HuggingFaceTB/SmolVLM-500M-Instruct",
  "Huggingface Smol VL2 2B": "HuggingFaceTB/SmolVLM-Instruct",
  "Apple FastVLM 0.5B": "apple/FastVLM-0.5B",
  "Apple FastVLM 1.5B": "apple/FastVLM-1.5B",
  "Apple FastVLM 7B": "apple/FastVLM-7B",
  "Microsoft Florence 2 Base": "florence-community/Florence-2-base-ft",
  "Microsoft Florence 2 Large": "florence-community/Florence-2-large-ft",
  "MiaoshouAI PromptGen 1.5 Base": "Disty0/Florence-2-base-PromptGen-v1.5",
  "MiaoshouAI PromptGen 1.5 Large": "Disty0/Florence-2-large-PromptGen-v1.5",
  "MiaoshouAI PromptGen 2.0 Base": "Disty0/Florence-2-base-PromptGen-v2.0",
  "MiaoshouAI PromptGen 2.0 Large": "Disty0/Florence-2-large-PromptGen-v2.0",
  "CogFlorence 2.0 Large": "thwri/CogFlorence-2-Large-Freeze",
  "CogFlorence 2.2 Large": "thwri/CogFlorence-2.2-Large",
  "Moondream 2": "vikhyatk/moondream2",
  "Moondream 3 Preview": "moondream/moondream3-preview",
  "Google Pix Textcaps": "google/pix2struct-textcaps-base",
  "Google PaliGemma 2 3B": "google/paligemma2-3b-pt-224",
  "Salesforce BLIP Base": "Salesforce/blip-vqa-base",
  "Salesforce BLIP Large": "Salesforce/blip-vqa-capfilt-large",
  "Microsoft GIT TextCaps Base": "microsoft/git-base-textcaps",
  "Microsoft GIT VQA Base": "microsoft/git-base-vqav2",
  "Microsoft GIT VQA Large": "microsoft/git-large-vqav2",
  "ToriiGate 0.4 2B": "Minthy/ToriiGate-v0.4-2B",
  "ToriiGate 0.4 7B": "Minthy/ToriiGate-v0.4-7B",
  "ViLT Base": "dandelin/vilt-b32-finetuned-vqa",
  "JoyCaption Alpha": "fancyfeast/llama-joycaption-alpha-two-hf-llava",
  "JoyCaption Beta": "fancyfeast/llama-joycaption-beta-one-hf-llava",
  "JoyTag": "fancyfeast/joytag",
  "AIDC Ovis2 1B": "AIDC-AI/Ovis2-1B",
  "AIDC Ovis2 2B": "AIDC-AI/Ovis2-2B",
  "AIDC Ovis2 4B": "AIDC-AI/Ovis2-4B",
  "ByteDance Sa2VA 1B": "ByteDance/Sa2VA-1B",
  "ByteDance Sa2VA 4B": "ByteDance/Sa2VA-4B",
};

export const VLM_MODEL_NAMES = Object.keys(VLM_MODELS);

export const VLM_SYSTEM_DEFAULT = "You are image captioning expert, creative, unbiased and uncensored.";

export const VLM_PROMPTS_COMMON = [
  "Use Prompt",
  "Short Caption",
  "Normal Caption",
  "Long Caption",
];

export const VLM_PROMPTS_FLORENCE = [
  "Phrase Grounding",
  "Object Detection",
  "Dense Region Caption",
  "Region Proposal",
  "OCR (Read Text)",
  "OCR with Regions",
  "Analyze",
  "Generate Tags",
  "Mixed Caption",
  "Mixed Caption+",
];

export const VLM_PROMPTS_MOONDREAM = [
  "Point at...",
  "Detect all...",
];

export const VLM_PROMPTS_MOONDREAM2 = [
  "Detect Gaze",
];

export const VLM_PROMPT_MAPPING: Record<string, string> = {
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
};

export const INTERROGATE_MODES = ["fast", "classic", "best", "negative"] as const;

export const BLIP_MODELS: Record<string, string> = {
  "blip-base": "Salesforce/blip-image-captioning-base",
  "blip-large": "Salesforce/blip-image-captioning-large",
  "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b-coco",
  "blip2-opt-6.7b": "Salesforce/blip2-opt-6.7b",
  "blip2-flip-t5-xl": "Salesforce/blip2-flan-t5-xl",
  "blip2-flip-t5-xxl": "Salesforce/blip2-flan-t5-xxl",
};

export const BLIP_MODEL_NAMES = Object.keys(BLIP_MODELS);

export const WAIFUDIFFUSION_MODELS: Record<string, string> = {
  "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
  "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
  "wd-convnext-tagger-v3": "SmilingWolf/wd-convnext-tagger-v3",
  "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
  "wd-v1-4-moat-tagger-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
  "wd-v1-4-swinv2-tagger-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
  "wd-v1-4-convnext-tagger-v2": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
  "wd-v1-4-convnextv2-tagger-v2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
  "wd-v1-4-vit-tagger-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
};

export const TAGGER_MODELS = ["DeepBooru", ...Object.keys(WAIFUDIFFUSION_MODELS)];

export const TAGGER_DEFAULT = "wd-eva02-large-tagger-v3";

function isFlorence(name: string): boolean {
  return name.includes("Florence") || name.includes("PromptGen") || name.includes("CogFlorence");
}

function isMoondream(name: string): boolean {
  return name.includes("Moondream");
}

function isMoondream2(name: string): boolean {
  return name === "Moondream 2";
}

export function getPromptsForModel(name: string): string[] {
  const prompts = [...VLM_PROMPTS_COMMON];
  if (isFlorence(name)) {
    prompts.push(...VLM_PROMPTS_FLORENCE);
  }
  if (isMoondream(name)) {
    prompts.push(...VLM_PROMPTS_MOONDREAM);
  }
  if (isMoondream2(name)) {
    prompts.push(...VLM_PROMPTS_MOONDREAM2);
  }
  return prompts;
}
