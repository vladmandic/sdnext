import { useCallback, useMemo } from "react";
import { Trash2, ListOrdered, Ban } from "lucide-react";
import { toast } from "sonner";
import { useJobQueueStore, selectPendingJobsSorted } from "@/stores/jobStore";
import type { TrackedJob } from "@/stores/jobStore";
import { useUiStore } from "@/stores/uiStore";
import { useSubmitJob, useCancelJob } from "@/api/hooks/useJobs";
import { putJobPayload } from "@/lib/jobPayloadDb";
import { QueueJobCard } from "./QueueJobCard";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ProgressRing } from "@/components/ui/progress-ring";

export function QueueTab() {
  const jobsMap = useJobQueueStore((s) => s.jobs);
  const clearTerminal = useJobQueueStore((s) => s.clearTerminal);
  const trackJob = useJobQueueStore((s) => s.trackJob);
  const removeJob = useJobQueueStore((s) => s.removeJob);
  const pendingJobsSorted = useMemo(() =>
    Array.from(jobsMap.values())
      .filter((j) => j.status === "pending")
      .sort((a, b) => b.priority - a.priority || a.createdAt - b.createdAt),
  [jobsMap]);
  const submitJob = useSubmitJob();
  const cancelJob = useCancelJob();

  const { runningJobs, terminalJobs, totalCount, avgProgress } = useMemo(() => {
    const all = Array.from(jobsMap.values()).sort((a, b) => b.createdAt - a.createdAt);
    const running = all.filter((j) => j.status === "running");
    const avg = running.length > 0 ? running.reduce((sum, j) => sum + j.progress, 0) / running.length : 0;
    return {
      runningJobs: running,
      terminalJobs: all.filter((j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled"),
      totalCount: all.length,
      avgProgress: avg,
    };
  }, [jobsMap]);

  const handleView = useCallback((job: TrackedJob) => {
    if (job.domain === "generate") {
      useUiStore.getState().setSidebarView("images");
    } else if (job.domain === "video" || job.domain === "framepack" || job.domain === "ltx") {
      useUiStore.getState().setSidebarView("video");
    }
  }, []);

  const handleDuplicate = useCallback(async (job: TrackedJob) => {
    if (!job.request) return;
    try {
      const newJob = await submitJob.mutateAsync(job.request);
      const priority = (job.request as { priority?: number }).priority ?? 0;
      trackJob(newJob.id, job.domain, job.snapshot, job.request, priority);
      putJobPayload({ id: newJob.id, domain: job.domain, request: job.request, priority, snapshot: { controlUnits: job.snapshot.controlUnits }, createdAt: Date.now() });
      toast.success("Job duplicated");
    } catch (err) {
      toast.error("Failed to duplicate job", { description: err instanceof Error ? err.message : String(err) });
    }
  }, [submitJob, trackJob]);

  const handleRetry = useCallback(async (job: TrackedJob) => {
    if (!job.request) return;
    try {
      const newJob = await submitJob.mutateAsync(job.request);
      const priority = (job.request as { priority?: number }).priority ?? 0;
      trackJob(newJob.id, job.domain, job.snapshot, job.request, priority);
      putJobPayload({ id: newJob.id, domain: job.domain, request: job.request, priority, snapshot: { controlUnits: job.snapshot.controlUnits }, createdAt: Date.now() });
      toast.success("Job retried");
    } catch (err) {
      toast.error("Failed to retry job", { description: err instanceof Error ? err.message : String(err) });
    }
  }, [submitJob, trackJob]);

  const handleMoveUp = useCallback(async (job: TrackedJob) => {
    if (job.status !== "pending" || !job.request) {
      toast.error("Cannot reorder: job is no longer pending");
      return;
    }
    const pending = useJobQueueStore.getState();
    const sorted = selectPendingJobsSorted(pending);
    const maxPriority = sorted.length > 0 ? Math.max(...sorted.map((j) => j.priority)) : 0;
    if (job.priority >= maxPriority && sorted[0]?.id === job.id) return;
    try {
      cancelJob.mutate(job.id);
      const newPriority = maxPriority + 1;
      const newRequest = { ...job.request, priority: newPriority } as typeof job.request;
      const newJob = await submitJob.mutateAsync(newRequest);
      trackJob(newJob.id, job.domain, job.snapshot, newRequest, newPriority);
      putJobPayload({ id: newJob.id, domain: job.domain, request: newRequest, priority: newPriority, snapshot: { controlUnits: job.snapshot.controlUnits }, createdAt: Date.now() });
    } catch (err) {
      toast.error("Failed to reorder job", { description: err instanceof Error ? err.message : String(err) });
    }
  }, [cancelJob, submitJob, trackJob]);

  const handleMoveDown = useCallback(async (job: TrackedJob) => {
    if (job.status !== "pending" || !job.request) {
      toast.error("Cannot reorder: job is no longer pending");
      return;
    }
    const pending = useJobQueueStore.getState();
    const sorted = selectPendingJobsSorted(pending);
    const minPriority = sorted.length > 0 ? Math.min(...sorted.map((j) => j.priority)) : 0;
    if (job.priority <= minPriority && sorted[sorted.length - 1]?.id === job.id) return;
    try {
      cancelJob.mutate(job.id);
      const newPriority = minPriority - 1;
      const newRequest = { ...job.request, priority: newPriority } as typeof job.request;
      const newJob = await submitJob.mutateAsync(newRequest);
      trackJob(newJob.id, job.domain, job.snapshot, newRequest, newPriority);
      putJobPayload({ id: newJob.id, domain: job.domain, request: newRequest, priority: newPriority, snapshot: { controlUnits: job.snapshot.controlUnits }, createdAt: Date.now() });
    } catch (err) {
      toast.error("Failed to reorder job", { description: err instanceof Error ? err.message : String(err) });
    }
  }, [cancelJob, submitJob, trackJob]);

  const handleRemove = useCallback((job: TrackedJob) => {
    removeJob(job.id);
  }, [removeJob]);

  const handleCancelAll = useCallback(() => {
    for (const job of pendingJobsSorted) {
      cancelJob.mutate(job.id);
    }
    toast.success(`Cancelling ${pendingJobsSorted.length} pending jobs`);
  }, [cancelJob, pendingJobsSorted]);

  if (totalCount === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-12 text-muted-foreground">
        <ListOrdered className="h-8 w-8" strokeWidth={1} />
        <p className="text-xs">No jobs in queue</p>
        <p className="text-2xs">Submit a generation to get started</p>
      </div>
    );
  }

  return (
    <div className="py-1">
      {/* Running */}
      {runningJobs.length > 0 && (
        <div>
          <div className="flex items-center gap-1.5 px-3 py-1">
            <p className="text-3xs font-medium text-muted-foreground uppercase tracking-wide flex-1">Running</p>
            {runningJobs.length > 1 && (
              <ProgressRing progress={avgProgress} size={14} strokeWidth={2} className="text-primary" />
            )}
          </div>
          {runningJobs.map((job) => (
            <QueueJobCard key={job.id} job={job} onView={handleView} />
          ))}
        </div>
      )}

      {/* Pending */}
      {pendingJobsSorted.length > 0 && (
        <div>
          {runningJobs.length > 0 && <Separator className="my-1" />}
          <div className="flex items-center justify-between px-3 py-1">
            <p className="text-3xs font-medium text-muted-foreground uppercase tracking-wide">
              Queued ({pendingJobsSorted.length})
            </p>
            <Button size="icon" variant="ghost" className="h-5 w-5" onClick={handleCancelAll} title="Cancel all pending">
              <Ban className="h-3 w-3" />
            </Button>
          </div>
          {pendingJobsSorted.map((job, i) => (
            <QueueJobCard
              key={job.id}
              job={job}
              onMoveUp={handleMoveUp}
              onMoveDown={handleMoveDown}
              canMoveUp={i > 0}
              canMoveDown={i < pendingJobsSorted.length - 1}
            />
          ))}
        </div>
      )}

      {/* Completed / Failed */}
      {terminalJobs.length > 0 && (
        <div>
          {(runningJobs.length > 0 || pendingJobsSorted.length > 0) && <Separator className="my-1" />}
          <div className="flex items-center justify-between px-3 py-1">
            <p className="text-3xs font-medium text-muted-foreground uppercase tracking-wide">
              History ({terminalJobs.length})
            </p>
            <Button size="icon" variant="ghost" className="h-5 w-5" onClick={clearTerminal} title="Clear history">
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
          {terminalJobs.map((job) => (
            <QueueJobCard
              key={job.id}
              job={job}
              onView={handleView}
              onRetry={handleRetry}
              onDuplicate={handleDuplicate}
              onRemove={handleRemove}
            />
          ))}
        </div>
      )}
    </div>
  );
}
