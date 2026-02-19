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

export type JobRequest =
  | GenerateJobRequest
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
