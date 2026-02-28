export interface VaeV2 {
  name: string;
  filename: string;
}

export interface UpscalerV2 {
  name: string;
  group: string;
  model_name: string | null;
  model_path: string | null;
  scale: number | null;
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

export interface PromptStyleV2 {
  name: string;
  prompt: string | null;
  negative_prompt: string | null;
  extra: string | null;
  description: string | null;
  wildcards: string | null;
  filename: string | null;
  preview: string | null;
  mtime: string | null;
}

export interface Extension {
  name: string;
  remote: string | null;
  branch: string;
  commit_hash: string | null;
  version: string | null;
  commit_date: string | null;
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
