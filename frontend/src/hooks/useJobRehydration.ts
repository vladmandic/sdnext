import { useEffect, useRef } from "react";
import { api } from "@/api/client";
import { useJobQueueStore, type JobDomain, type TrackedJob } from "@/stores/jobStore";
import { getAllJobPayloads, deleteJobPayload, type StoredJobPayload } from "@/lib/jobPayloadDb";
import type { Job, JobListResponse } from "@/api/types/v2";

const JOB_TYPE_TO_DOMAIN: Record<string, JobDomain> = {
  generate: "generate",
  upscale: "upscale",
  video: "video",
  framepack: "framepack",
  ltx: "ltx",
  "xyz-grid": "xyz-grid",
};

function jobToDomain(type: string): JobDomain {
  return JOB_TYPE_TO_DOMAIN[type] ?? "generate";
}

function buildTrackedJob(backendJob: Job, local: StoredJobPayload | undefined): TrackedJob {
  return {
    id: backendJob.id,
    domain: local?.domain ?? jobToDomain(backendJob.type),
    status: backendJob.status,
    progress: backendJob.progress ?? 0,
    eta: backendJob.eta ?? 0,
    step: backendJob.step ?? 0,
    steps: backendJob.steps ?? 0,
    task: "",
    textinfo: null,
    previewUrl: null,
    result: backendJob.result ?? null,
    error: backendJob.error ?? null,
    createdAt: backendJob.created_at ? new Date(backendJob.created_at).getTime() : Date.now(),
    snapshot: local?.snapshot ? { controlUnits: local.snapshot.controlUnits } : {},
    request: local?.request ?? null,
    priority: local?.priority ?? 0,
  };
}

export function useJobRehydration() {
  const ran = useRef(false);

  useEffect(() => {
    if (ran.current) return;
    ran.current = true;

    (async () => {
      try {
        const payloads = await getAllJobPayloads();
        const payloadMap = new Map<string, StoredJobPayload>();
        for (const p of payloads) payloadMap.set(p.id, p);

        const [pending, running] = await Promise.all([
          api.get<JobListResponse>("/sdapi/v2/jobs", { status: "pending", limit: "50" }),
          api.get<JobListResponse>("/sdapi/v2/jobs", { status: "running", limit: "10" }),
        ]);

        const backendJobs = [...pending.items, ...running.items];
        const backendIds = new Set(backendJobs.map((j) => j.id));
        const store = useJobQueueStore.getState();

        for (const bj of backendJobs) {
          if (store.jobs.has(bj.id)) continue;
          const local = payloadMap.get(bj.id);
          const tracked = buildTrackedJob(bj, local);
          store.rehydrateJob(tracked);
        }

        for (const [id] of payloadMap) {
          if (!backendIds.has(id)) {
            deleteJobPayload(id);
          }
        }
      } catch {
        // Rehydration is best-effort; silently skip on error
      }
    })();
  }, []);
}
