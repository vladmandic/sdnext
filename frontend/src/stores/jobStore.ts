import { create } from "zustand";
import type { JobResult, JobStatus } from "@/api/types/v2";
import type { MaskLine } from "@/stores/img2imgStore";
import type { ControlUnitSnapshot } from "@/api/types/control";

export type JobDomain = "generate" | "upscale" | "video" | "framepack" | "ltx";

export interface JobSnapshot {
  inputImage?: string;
  inputMask?: MaskLine[];
  controlUnits?: ControlUnitSnapshot[];
}

export interface TrackedJob {
  id: string;
  domain: JobDomain;
  status: JobStatus;
  progress: number;
  eta: number;
  step: number;
  steps: number;
  task: string;
  textinfo: string | null;
  previewUrl: string | null;
  result: JobResult | null;
  error: string | null;
  createdAt: number;
  snapshot: JobSnapshot;
}

const MAX_TRACKED_JOBS = 50;

function isTerminal(status: JobStatus) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

interface JobQueueState {
  jobs: Map<string, TrackedJob>;
  activeJobId: string | null;

  trackJob: (id: string, domain: JobDomain, snapshot: JobSnapshot) => void;
  updateStatus: (id: string, status: JobStatus) => void;
  updateProgress: (id: string, progress: number, eta: number, step: number, steps: number, task?: string, textinfo?: string | null) => void;
  updatePreview: (id: string, previewUrl: string) => void;
  completeJob: (id: string, result: JobResult) => void;
  failJob: (id: string, error: string) => void;
  removeJob: (id: string) => void;
  setActiveJob: (id: string | null) => void;
  clearTerminal: () => void;
}

function pruneOldTerminal(jobs: Map<string, TrackedJob>): Map<string, TrackedJob> {
  if (jobs.size <= MAX_TRACKED_JOBS) return jobs;
  const entries = Array.from(jobs.entries());
  const terminal = entries.filter(([, j]) => isTerminal(j.status)).sort((a, b) => a[1].createdAt - b[1].createdAt);
  const toRemove = jobs.size - MAX_TRACKED_JOBS;
  const next = new Map(jobs);
  for (let i = 0; i < Math.min(toRemove, terminal.length); i++) {
    const job = terminal[i][1];
    if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
    next.delete(terminal[i][0]);
  }
  return next;
}

export const useJobQueueStore = create<JobQueueState>()((set) => ({
  jobs: new Map(),
  activeJobId: null,

  trackJob: (id, domain, snapshot) =>
    set((state) => {
      const next = new Map(state.jobs);
      next.set(id, {
        id,
        domain,
        status: "pending",
        progress: 0,
        eta: 0,
        step: 0,
        steps: 0,
        task: "",
        textinfo: null,
        previewUrl: null,
        result: null,
        error: null,
        createdAt: Date.now(),
        snapshot,
      });
      return { jobs: pruneOldTerminal(next), activeJobId: state.activeJobId ?? id };
    }),

  updateStatus: (id, status) =>
    set((state) => {
      const job = state.jobs.get(id);
      if (!job) return state;
      const next = new Map(state.jobs);
      next.set(id, { ...job, status });
      const updates: Partial<JobQueueState> = { jobs: next };
      if (status === "running" && !state.activeJobId) {
        updates.activeJobId = id;
      }
      return updates;
    }),

  updateProgress: (id, progress, eta, step, steps, task?, textinfo?) =>
    set((state) => {
      const job = state.jobs.get(id);
      if (!job) return state;
      const next = new Map(state.jobs);
      const jobUpdates: Partial<TrackedJob> = { progress, eta, step, steps, status: job.status === "pending" ? "running" : job.status };
      if (task !== undefined) jobUpdates.task = task;
      if (textinfo !== undefined) jobUpdates.textinfo = textinfo;
      next.set(id, { ...job, ...jobUpdates });
      const stateUpdates: Partial<JobQueueState> = { jobs: next };
      if (state.activeJobId !== id && !isTerminal(job.status)) {
        const currentActive = state.activeJobId ? state.jobs.get(state.activeJobId) : null;
        if (!currentActive || isTerminal(currentActive.status)) {
          stateUpdates.activeJobId = id;
        }
      }
      return stateUpdates;
    }),

  updatePreview: (id, previewUrl) =>
    set((state) => {
      const job = state.jobs.get(id);
      if (!job) return state;
      if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
      const next = new Map(state.jobs);
      next.set(id, { ...job, previewUrl });
      return { jobs: next };
    }),

  completeJob: (id, result) =>
    set((state) => {
      const job = state.jobs.get(id);
      if (!job) return state;
      if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
      const next = new Map(state.jobs);
      next.set(id, { ...job, status: "completed", result, previewUrl: null, progress: 1 });
      const updates: Partial<JobQueueState> = { jobs: next };
      if (state.activeJobId === id) {
        const nextRunning = Array.from(next.values()).find((j) => j.status === "running" || j.status === "pending");
        updates.activeJobId = nextRunning?.id ?? null;
      }
      return updates;
    }),

  failJob: (id, error) =>
    set((state) => {
      const job = state.jobs.get(id);
      if (!job) return state;
      if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
      const next = new Map(state.jobs);
      next.set(id, { ...job, status: "failed", error, previewUrl: null });
      const updates: Partial<JobQueueState> = { jobs: next };
      if (state.activeJobId === id) {
        const nextRunning = Array.from(next.values()).find((j) => j.status === "running" || j.status === "pending");
        updates.activeJobId = nextRunning?.id ?? null;
      }
      return updates;
    }),

  removeJob: (id) =>
    set((state) => {
      const job = state.jobs.get(id);
      if (!job) return state;
      if (job.previewUrl) URL.revokeObjectURL(job.previewUrl);
      const next = new Map(state.jobs);
      next.delete(id);
      const updates: Partial<JobQueueState> = { jobs: next };
      if (state.activeJobId === id) {
        const nextRunning = Array.from(next.values()).find((j) => j.status === "running" || j.status === "pending");
        updates.activeJobId = nextRunning?.id ?? null;
      }
      return updates;
    }),

  setActiveJob: (id) => set({ activeJobId: id }),

  clearTerminal: () =>
    set((state) => {
      const next = new Map<string, TrackedJob>();
      for (const [k, v] of state.jobs) {
        if (isTerminal(v.status)) {
          if (v.previewUrl) URL.revokeObjectURL(v.previewUrl);
        } else {
          next.set(k, v);
        }
      }
      return { jobs: next };
    }),
}));

// --- Selectors ---

export function selectAllJobs(state: JobQueueState): TrackedJob[] {
  return Array.from(state.jobs.values()).sort((a, b) => b.createdAt - a.createdAt);
}

export function selectActiveJobs(state: JobQueueState): TrackedJob[] {
  return Array.from(state.jobs.values()).filter((j) => !isTerminal(j.status));
}

export function selectRunningJob(state: JobQueueState): TrackedJob | undefined {
  return Array.from(state.jobs.values()).find((j) => j.status === "running");
}

export function selectPendingCount(state: JobQueueState): number {
  let count = 0;
  for (const j of state.jobs.values()) {
    if (j.status === "pending") count++;
  }
  return count;
}

export function selectHasActiveJobs(state: JobQueueState): boolean {
  for (const j of state.jobs.values()) {
    if (!isTerminal(j.status)) return true;
  }
  return false;
}

export function selectViewedJob(state: JobQueueState): TrackedJob | undefined {
  if (state.activeJobId) {
    const active = state.jobs.get(state.activeJobId);
    if (active && !isTerminal(active.status)) return active;
  }
  const running = Array.from(state.jobs.values()).find((j) => j.status === "running");
  if (running) return running;
  return Array.from(state.jobs.values()).find((j) => j.status === "pending");
}

export function selectDomainJobs(domain: JobDomain) {
  return (state: JobQueueState): TrackedJob[] =>
    Array.from(state.jobs.values()).filter((j) => j.domain === domain);
}

export function selectDomainActive(domain: JobDomain) {
  return (state: JobQueueState): boolean => {
    for (const j of state.jobs.values()) {
      if (j.domain === domain && !isTerminal(j.status)) return true;
    }
    return false;
  };
}

export function selectDomainRunning(domain: JobDomain) {
  return (state: JobQueueState): TrackedJob | undefined =>
    Array.from(state.jobs.values()).find((j) => j.domain === domain && j.status === "running");
}

export function selectDomainProgress(domain: JobDomain) {
  return (state: JobQueueState): number => {
    const running = Array.from(state.jobs.values()).find((j) => j.domain === domain && j.status === "running");
    return running?.progress ?? 0;
  };
}
