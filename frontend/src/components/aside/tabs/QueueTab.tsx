import { useCallback, useMemo } from "react";
import { Trash2, ListOrdered } from "lucide-react";
import { useJobQueueStore } from "@/stores/jobStore";
import type { TrackedJob } from "@/stores/jobStore";
import { useUiStore } from "@/stores/uiStore";
import { QueueJobCard } from "./QueueJobCard";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

export function QueueTab() {
  const jobsMap = useJobQueueStore((s) => s.jobs);
  const clearTerminal = useJobQueueStore((s) => s.clearTerminal);

  const { runningJobs, pendingJobs, terminalJobs, totalCount } = useMemo(() => {
    const all = Array.from(jobsMap.values()).sort((a, b) => b.createdAt - a.createdAt);
    return {
      runningJobs: all.filter((j) => j.status === "running"),
      pendingJobs: all.filter((j) => j.status === "pending"),
      terminalJobs: all.filter((j) => j.status === "completed" || j.status === "failed" || j.status === "cancelled"),
      totalCount: all.length,
    };
  }, [jobsMap]);

  const handleView = useCallback((job: TrackedJob) => {
    if (job.domain === "generate") {
      useUiStore.getState().setSidebarView("images");
    } else if (job.domain === "video" || job.domain === "framepack" || job.domain === "ltx") {
      useUiStore.getState().setSidebarView("video");
    }
  }, []);

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
          <p className="px-3 py-1 text-3xs font-medium text-muted-foreground uppercase tracking-wide">Running</p>
          {runningJobs.map((job) => (
            <QueueJobCard key={job.id} job={job} onView={handleView} />
          ))}
        </div>
      )}

      {/* Pending */}
      {pendingJobs.length > 0 && (
        <div>
          {runningJobs.length > 0 && <Separator className="my-1" />}
          <p className="px-3 py-1 text-3xs font-medium text-muted-foreground uppercase tracking-wide">
            Queued ({pendingJobs.length})
          </p>
          {pendingJobs.map((job) => (
            <QueueJobCard key={job.id} job={job} />
          ))}
        </div>
      )}

      {/* Completed / Failed */}
      {terminalJobs.length > 0 && (
        <div>
          {(runningJobs.length > 0 || pendingJobs.length > 0) && <Separator className="my-1" />}
          <div className="flex items-center justify-between px-3 py-1">
            <p className="text-3xs font-medium text-muted-foreground uppercase tracking-wide">
              History ({terminalJobs.length})
            </p>
            <Button size="icon" variant="ghost" className="h-5 w-5" onClick={clearTerminal} title="Clear history">
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
          {terminalJobs.map((job) => (
            <QueueJobCard key={job.id} job={job} onView={handleView} />
          ))}
        </div>
      )}
    </div>
  );
}
