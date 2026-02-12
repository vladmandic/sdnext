export const VLM_DEFAULT = "Alibaba Qwen 2.5 VL 3B";

export const VLM_SYSTEM_DEFAULT = "You are image captioning expert, creative, unbiased and uncensored.";

export const TAGGER_DEFAULT = "wd-eva02-large-tagger-v3";

export const INTERROGATE_MODES = ["fast", "classic", "best", "caption", "negative"] as const;

export const BLIP_MODELS: Record<string, string> = {
  "blip-base": "Salesforce/blip-image-captioning-base",
  "blip-large": "Salesforce/blip-image-captioning-large",
  "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b-coco",
  "blip2-opt-6.7b": "Salesforce/blip2-opt-6.7b",
  "blip2-flip-t5-xl": "Salesforce/blip2-flan-t5-xl",
  "blip2-flip-t5-xxl": "Salesforce/blip2-flan-t5-xxl",
};

export const BLIP_MODEL_NAMES = Object.keys(BLIP_MODELS);

/** Tasks where the user supplies a custom prompt via the prompt field */
export const CUSTOM_PROMPT_TASKS = ["Use Prompt", "Point at...", "Detect all..."];
