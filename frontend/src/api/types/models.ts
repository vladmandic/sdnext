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
  version: string | null;
  tags: string | null;
}

export interface NetworkDetail {
  name: string;
  type: string;
  title: string | null;
  filename: string | null;
  hash: string | null;
  alias: string | null;
  size: number | null;
  mtime: string | null;
  version: string | null;
  tags: string | null;
  description: string | null;
  info: Record<string, unknown> | null;
}

export interface NetworkDetailFull {
  name: string;
  type: string;
  title: string | null;
  fullname: string | null;
  filename: string | null;
  hash: string | null;
  preview: string | null;
  alias: string | null;
  size: number | null;
  mtime: string | null;
  version: string | null;
  tags: string | null;
  description: string | null;
  info: Record<string, unknown> | null;
}

export interface NetworkDetailsResponse {
  items: NetworkDetailFull[];
  total: number;
  offset: number;
  limit: number;
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

// --- V2 types ---

export interface ExtraNetworkV2 {
  name: string;
  type: string;
  title: string | null;
  fullname: string | null;
  filename: string | null;
  hash: string | null;
  preview: string | null;
  version: string | null;
  tags: string[];
  size: number | null;
  mtime: string | null;
}

export interface ExtraNetworksResponse {
  items: ExtraNetworkV2[];
  total: number;
  offset: number;
  limit: number;
}

export interface SdModelV2 {
  title: string;
  model_name: string;
  filename: string;
  type: string;
  hash: string | null;
  sha256: string | null;
  size: number | null;
  mtime: string | null;
  version: string | null;
  subfolder: string | null;
}

export interface SdModelsResponse {
  items: SdModelV2[];
  total: number;
  offset: number;
  limit: number;
}

export interface SamplerV2 {
  name: string;
  group: "Standard" | "FlowMatch" | "Res4Lyf";
  compatible: boolean | null;
  options: Record<string, unknown>;
}

export interface CheckpointInfoV2 {
  loaded: boolean;
  type: string | null;
  class_name: string | null;
  checkpoint: string | null;
  title: string | null;
  name: string | null;
  filename: string | null;
  hash: string | null;
}
