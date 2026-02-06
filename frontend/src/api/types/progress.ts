export interface ResProgress {
  id: number | string | null;
  progress: number;
  eta_relative: number;
  state: Record<string, unknown>;
  current_image: string | null;
  textinfo: string | null;
}

export interface ResStatus {
  status: "idle" | "running" | "paused" | "interrupted" | "skipped" | string;
  task: string;
  timestamp: string | null;
  current: string;
  id: number | string | null;
  job: number;
  jobs: number;
  total: number;
  step: number;
  steps: number;
  queued: number;
  uptime: number;
  elapsed: number | null;
  eta: number | null;
  progress: number | null;
}

export interface ResHistory {
  id: number | string | null;
  job: string;
  op: string;
  timestamp: number | null;
  duration: number | null;
  outputs: string[];
}
