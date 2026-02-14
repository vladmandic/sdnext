export interface ResMemory {
  ram: {
    free?: number;
    used?: number;
    total?: number;
    error?: string;
  };
  cuda: {
    system?: { free: number; used: number; total: number };
    active?: { current: number; peak: number };
    allocated?: { current: number; peak: number };
    reserved?: { current: number; peak: number };
    inactive?: { current: number; peak: number };
    events?: { retries: number; oom: number };
    error?: string;
  };
}

export interface ResGPU {
  name: string;
  data: Record<string, unknown>;
  chart: [number, number];
}

export interface ServerInfo {
  version: Record<string, string>;
  backend: string;
  platform: string;
  gpu: string;
  api_version: string;
  capabilities: {
    txt2img: boolean;
    img2img: boolean;
    control: boolean;
    video: boolean;
    websocket: boolean;
  };
  model: {
    name: string | null;
    type: string | null;
  };
}

export interface VersionInfo {
  app: string;
  updated: string;
  hash: string;
  url: string;
  branch: string;
  commit: string;
}

export interface LoadedModel {
  name: string;
  category: string;
  device?: string | null;
  size_bytes?: number | null;
  dtype?: string | null;
  extra?: Record<string, unknown> | null;
}
