export interface UpdateCheckResult {
  url: string;
  branch: string;
  current_date: string;
  current_hash: string;
  latest_date: string;
  latest_hash: string;
  up_to_date: boolean;
  error?: string;
}

export interface UpdateApplyRequest {
  rebase?: boolean;
  submodules?: boolean;
  extensions?: boolean;
}

export interface UpdateApplyResult {
  status: string;
  changed: boolean;
  version?: UpdateCheckResult;
}

export interface BenchmarkRunRequest {
  level?: string;
  steps?: string;
  width?: number;
  height?: number;
}

export interface BenchmarkResult {
  batch: number;
  its: number | string;
}

export interface BenchmarkRunResult {
  results: BenchmarkResult[];
  error: string | null;
}

export interface BenchmarkHistory {
  headers: string[];
  data: (string | null)[][];
  error?: string;
}

export interface HistoryEntry {
  id: string | number | null;
  job: string;
  op: string;
  timestamp: number | null;
  duration: number | null;
  outputs: string[];
}

export interface StorageEntry {
  label: string;
  path: string;
  size: number;
}

export type StorageInfo = Record<string, StorageEntry[]>;

export interface HistoryEntryV2 {
  id: string | number | null;
  job: string;
  op: string;
  timestamp: number | null;
  duration: number | null;
  outputs: string[];
}

export interface HistoryResponse {
  items: HistoryEntryV2[];
  total: number;
  offset: number;
  limit: number;
}

export interface SystemInfoFull {
  version: Record<string, string>;
  uptime: string;
  timestamp: string;
  state: Record<string, unknown> | null;
  memory: Record<string, unknown> | null;
  platform: Record<string, string> | null;
  torch: string | null;
  gpu: Record<string, string> | null;
  flags: string[] | null;
  crossatention: string | null;
  device: Record<string, string> | null;
  backend: string | null;
  pipeline: string | null;
  libs?: Record<string, string> | null;
}
