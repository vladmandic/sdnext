// --- Job request types (discriminated union) ---

import type { ControlRequest } from "./generation";

export type GenerateJobRequest = ControlRequest & { type: "generate"; priority?: number };

export interface UpscaleJobParams {
  type: "upscale";
  image: string;
  upscaler?: string;
  scale?: number;
  priority?: number;
}

export interface CaptionJobParams {
  type: "caption";
  image: string;
  backend?: "vlm" | "openclip" | "tagger";
  model?: string;
  priority?: number;
}

export interface EnhanceJobParams {
  type: "enhance";
  prompt?: string;
  model?: string;
  enhance_type?: "text" | "image" | "video";
  seed?: number;
  image?: string;
  system_prompt?: string;
  prefix?: string;
  suffix?: string;
  priority?: number;
}

export interface DetectJobParams {
  type: "detect";
  image: string;
  model?: string;
  priority?: number;
}

export interface PreprocessJobParams {
  type: "preprocess";
  image: string;
  model: string;
  params?: Record<string, unknown>;
  priority?: number;
}

export interface VideoGenerateParams {
  type: "video";
  engine: string;
  model: string;
  prompt: string;
  negative?: string;
  width?: number;
  height?: number;
  frames?: number;
  steps?: number;
  sampler?: number;
  sampler_shift?: number;
  dynamic_shift?: boolean;
  seed?: number;
  guidance_scale?: number;
  guidance_true?: number;
  init_image?: string | null;
  init_strength?: number;
  last_image?: string | null;
  vae_type?: string;
  vae_tile_frames?: number;
  fps?: number;
  interpolate?: number;
  codec?: string;
  format?: string;
  codec_options?: string;
  save_video?: boolean;
  save_frames?: boolean;
  save_safetensors?: boolean;
  priority?: number;
}

export interface FramePackJobParams {
  type: "framepack";
  prompt: string;
  negative?: string;
  seed?: number;
  variant?: string;
  resolution?: number;
  duration?: number;
  latent_ws?: number;
  steps?: number;
  shift?: number;
  cfg_scale?: number;
  cfg_distilled?: number;
  cfg_rescale?: number;
  start_weight?: number;
  end_weight?: number;
  vision_weight?: number;
  section_prompt?: string;
  system_prompt?: string;
  use_teacache?: boolean;
  optimized_prompt?: boolean;
  use_cfgzero?: boolean;
  use_preview?: boolean;
  attention?: string;
  vae_type?: string;
  init_image?: string | null;
  end_image?: string | null;
  fps?: number;
  interpolate?: number;
  codec?: string;
  format?: string;
  codec_options?: string;
  save_video?: boolean;
  save_frames?: boolean;
  save_safetensors?: boolean;
  priority?: number;
}

export interface LtxJobParams {
  type: "ltx";
  model: string;
  prompt: string;
  negative?: string;
  seed?: number;
  width?: number;
  height?: number;
  frames?: number;
  steps?: number;
  decode_timestep?: number;
  image_cond_noise_scale?: number;
  upsample_enable?: boolean;
  upsample_ratio?: number;
  refine_enable?: boolean;
  refine_strength?: number;
  condition_strength?: number;
  condition_image?: string | null;
  condition_last?: string | null;
  audio_enable?: boolean;
  fps?: number;
  interpolate?: number;
  codec?: string;
  format?: string;
  codec_options?: string;
  save_video?: boolean;
  save_frames?: boolean;
  save_safetensors?: boolean;
  priority?: number;
}

export type JobRequest =
  | GenerateJobRequest
  | UpscaleJobParams
  | CaptionJobParams
  | EnhanceJobParams
  | DetectJobParams
  | PreprocessJobParams
  | VideoGenerateParams
  | FramePackJobParams
  | LtxJobParams;

// --- Job response types ---

export interface ImageRef {
  index: number;
  url: string;
  width: number;
  height: number;
  format: string;
  size: number;
}

export interface JobResult {
  images: ImageRef[];
  info: Record<string, unknown>;
  params: Record<string, unknown>;
}

export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: string;
  type: string;
  status: JobStatus;
  progress: number;
  step: number;
  steps: number;
  eta: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  result: JobResult | null;
}

export interface JobListResponse {
  items: Job[];
  total: number;
  offset: number;
  limit: number;
}

// --- WebSocket event types ---

export type JobWsEvent =
  | { type: "status"; status: JobStatus; progress: number }
  | { type: "progress"; step: number; steps: number; progress: number; eta: number | null }
  | { type: "completed"; result: JobResult }
  | { type: "error"; error: string }
  | { type: "cancelled" }
  | { type: "ping" }
  | { type: "ack"; command: string };
