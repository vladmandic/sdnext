// --- Job request types (discriminated union) ---

export interface GenerateJobParams {
  type: "generate";
  prompt?: string;
  negative_prompt?: string;
  steps?: number;
  width?: number;
  height?: number;
  cfg_scale?: number;
  seed?: number;
  batch_size?: number;
  batch_count?: number;
  sampler_name?: string;
  denoising_strength?: number;
  inputs?: string[];
  inits?: string[];
  mask?: string;
  control?: Record<string, unknown>[];
  ip_adapter?: Record<string, unknown>[];
  extra?: Record<string, unknown>;
  save_images?: boolean;
  clip_skip?: number;
  cfg_end?: number;
  script_name?: string;
  script_args?: unknown[];
  alwayson_scripts?: Record<string, unknown>;
  override_settings?: Record<string, unknown>;
  priority?: number;
}

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

export type JobRequest =
  | GenerateJobParams
  | UpscaleJobParams
  | CaptionJobParams
  | EnhanceJobParams
  | DetectJobParams
  | PreprocessJobParams;

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
