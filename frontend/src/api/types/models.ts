export interface SdModel {
  title: string;
  model_name: string;
  filename: string;
  type: string;
  sha256: string | null;
  hash: string | null;
  config: string | null;
}

export interface SdVae {
  model_name: string;
  filename: string;
}

export interface Sampler {
  name: string;
  options: Record<string, unknown>;
}

export interface Upscaler {
  name: string;
  model_name: string | null;
  model_path: string | null;
  model_url: string | null;
  scale: number | null;
}

export interface LoraNetwork {
  name: string;
  type: string;
  title: string | null;
  fullname: string | null;
  filename: string | null;
  hash: string | null;
  preview: string | null;
}

export interface Embedding {
  step: number | null;
  sd_checkpoint: string | null;
  sd_checkpoint_name: string | null;
  shape: number;
  vectors: number;
}

export interface EmbeddingsResponse {
  loaded: unknown[];
  skipped: unknown[];
}

export interface PromptStyle {
  name: string;
  prompt: string | null;
  negative_prompt: string | null;
  extra: string | null;
  filename: string | null;
  preview: string | null;
}

export interface Extension {
  name: string;
  remote: string;
  branch: string;
  commit_hash: string;
  version: string;
  commit_date: string | number;
  enabled: boolean;
}
