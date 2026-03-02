export interface VersionInfo {
  app: string;
  updated: string;
  commit: string;
  branch: string;
  url: string;
}

export interface ServerCapabilities {
  txt2img: boolean;
  img2img: boolean;
  control: boolean;
  video: boolean;
  websocket: boolean;
}

export interface ServerModelInfo {
  name: string | null;
  type: string | null;
  supports_strength: boolean;
}

export interface ServerInfo {
  version: VersionInfo;
  backend: string;
  platform: string;
  api_version: string;
  capabilities: ServerCapabilities;
  model: ServerModelInfo;
}

export interface MemoryUsage {
  free?: number;
  used?: number;
  total?: number;
}

export interface MemoryPeakUsage {
  current?: number;
  peak?: number;
}

export interface MemoryWarnings {
  retries: number;
  oom: number;
}

export interface RamMemory {
  free?: number;
  used?: number;
  total?: number;
  error?: string;
}

export interface CudaMemory {
  system?: MemoryUsage;
  active?: MemoryPeakUsage;
  allocated?: MemoryPeakUsage;
  reserved?: MemoryPeakUsage;
  inactive?: MemoryPeakUsage;
  events?: MemoryWarnings;
  error?: string;
}

export interface ResMemory {
  ram: RamMemory;
  cuda: CudaMemory;
}

export interface GpuMetrics {
  load_gpu: number | null;
  load_vram: number | null;
  temperature: number | null;
  fan_speed: number | null;
  power_current: number | null;
  power_limit: number | null;
  vram_used: number | null;
  vram_total: number | null;
}

export interface ResGPU {
  name: string;
  metrics: GpuMetrics;
  details: Record<string, string>;
  chart_vram_pct: number | null;
  chart_gpu_pct: number | null;
}

export interface LoadedModel {
  name: string;
  category: string;
  device?: string | null;
  size_bytes?: number | null;
  dtype?: string | null;
  extra?: Record<string, unknown> | null;
}
