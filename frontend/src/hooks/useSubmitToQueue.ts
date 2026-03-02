import { useCallback, useState } from "react";
import { toast } from "sonner";
import { useSubmitJob } from "@/api/hooks/useJobs";
import { useJobQueueStore, type JobDomain, type JobSnapshot } from "@/stores/jobStore";
import { putJobPayload } from "@/lib/jobPayloadDb";
import type { JobRequest } from "@/api/types/v2";

interface SubmitOptions {
  domain: JobDomain;
  buildRequest: () => Promise<{ payload: JobRequest; snapshot: JobSnapshot }>;
}

export function useSubmitToQueue({ domain, buildRequest }: SubmitOptions) {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const submitJob = useSubmitJob();
  const trackJob = useJobQueueStore((s) => s.trackJob);

  const submit = useCallback(async () => {
    setIsSubmitting(true);
    try {
      const { payload, snapshot } = await buildRequest();
      const job = await submitJob.mutateAsync(payload);
      const priority = (payload as { priority?: number }).priority ?? 0;
      trackJob(job.id, domain, snapshot, payload, priority);
      putJobPayload({ id: job.id, domain, request: payload, priority, snapshot: { controlUnits: snapshot.controlUnits }, createdAt: Date.now() });
    } catch (err) {
      toast.error("Failed to submit job", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      setIsSubmitting(false);
    }
  }, [buildRequest, submitJob, trackJob, domain]);

  return { submit, isSubmitting };
}
