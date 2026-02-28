export interface PromptEnhanceModel {
  name: string;
  group: string;
  vision: boolean;
  thinking: boolean;
  cached: boolean;
}

export interface PromptEnhanceRequest {
  prompt: string;
  type?: "text" | "image" | "video";
  model?: string;
  system_prompt?: string;
  image?: string;
  seed?: number;
  nsfw?: boolean;
  prefix?: string;
  suffix?: string;
  do_sample?: boolean;
  max_tokens?: number;
  temperature?: number;
  repetition_penalty?: number;
  top_k?: number;
  top_p?: number;
  thinking?: boolean;
  keep_thinking?: boolean;
  use_vision?: boolean;
  prefill?: string;
  keep_prefill?: boolean;
}

export interface PromptEnhanceResponse {
  ok: boolean;
  prompt: string;
  seed: number;
}
