export type CaptionMethod = "vlm" | "openclip" | "tagger";

export interface InterrogateRequest {
  image: string;
  model?: string;
  clip_model?: string;
  blip_model?: string;
  mode?: "fast" | "classic" | "best" | "negative";
  analyze?: boolean;
}

export interface InterrogateResponse {
  caption?: string;
  medium?: string;
  artist?: string;
  movement?: string;
  trending?: string;
  flavor?: string;
}

export interface VqaRequest {
  image: string;
  model?: string;
  question?: string;
  system?: string;
}

export interface VqaResponse {
  answer?: string;
}

/** Options set via /sdapi/v1/options before calling caption endpoints */
export interface VlmOptions {
  interrogate_vlm_max_length?: number;
  interrogate_vlm_num_beams?: number;
  interrogate_vlm_temperature?: number;
  interrogate_vlm_top_k?: number;
  interrogate_vlm_top_p?: number;
  interrogate_vlm_do_sample?: boolean;
  interrogate_vlm_thinking_mode?: boolean;
  interrogate_vlm_keep_thinking?: boolean;
  interrogate_vlm_keep_prefill?: boolean;
}

export interface OpenClipOptions {
  interrogate_blip_model?: string;
  interrogate_clip_num_beams?: number;
  interrogate_clip_min_length?: number;
  interrogate_clip_max_length?: number;
  interrogate_clip_min_flavors?: number;
  interrogate_clip_max_flavors?: number;
  interrogate_clip_flavor_count?: number;
  interrogate_clip_chunk_size?: number;
}

export interface TaggerOptions {
  waifudiffusion_model?: string;
  tagger_threshold?: number;
  waifudiffusion_character_threshold?: number;
  tagger_max_tags?: number;
  tagger_include_rating?: boolean;
  tagger_sort_alpha?: boolean;
  tagger_use_spaces?: boolean;
  tagger_escape_brackets?: boolean;
  tagger_exclude_tags?: string;
  tagger_show_scores?: boolean;
}
