export type CaptionMethod = "vlm" | "openclip" | "tagger";

// ---------------------------------------------------------------------------
// OpenCLIP / BLIP
// ---------------------------------------------------------------------------

export interface OpenClipRequest {
  image: string;
  model?: string;
  clip_model?: string;
  blip_model?: string;
  mode?: string;
  analyze?: boolean;
  max_length?: number | null;
  chunk_size?: number | null;
  min_flavors?: number | null;
  max_flavors?: number | null;
  flavor_count?: number | null;
  num_beams?: number | null;
}

export interface OpenClipResponse {
  ok: boolean;
  caption?: string;
  medium?: string;
  artist?: string;
  movement?: string;
  trending?: string;
  flavor?: string;
}

// ---------------------------------------------------------------------------
// Tagger (WaifuDiffusion / DeepBooru)
// ---------------------------------------------------------------------------

export interface TaggerRequest {
  image: string;
  model?: string;
  threshold?: number;
  character_threshold?: number;
  max_tags?: number;
  include_rating?: boolean;
  sort_alpha?: boolean;
  use_spaces?: boolean;
  escape_brackets?: boolean;
  exclude_tags?: string;
  show_scores?: boolean;
}

export interface TaggerResponse {
  ok: boolean;
  tags: string;
  scores?: Record<string, number> | null;
}

export interface TaggerModel {
  name: string;
  type: string;
}

// ---------------------------------------------------------------------------
// VLM / VQA
// ---------------------------------------------------------------------------

export interface VqaRequest {
  image: string;
  model?: string;
  question?: string;
  prompt?: string | null;
  system?: string;
  include_annotated?: boolean;
  max_tokens?: number | null;
  temperature?: number | null;
  top_k?: number | null;
  top_p?: number | null;
  num_beams?: number | null;
  do_sample?: boolean | null;
  thinking_mode?: boolean | null;
  prefill?: string | null;
  keep_thinking?: boolean | null;
  keep_prefill?: boolean | null;
}

export interface VqaResponse {
  ok: boolean;
  answer?: string;
  annotated_image?: string | null;
}

export interface VlmModel {
  name: string;
  repo: string;
  prompts: string[];
  capabilities: string[];
}
