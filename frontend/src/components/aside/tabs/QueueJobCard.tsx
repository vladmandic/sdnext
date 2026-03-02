import { useCallback, useEffect, useState } from "react";
import { X, RotateCcw, Eye, Image, Video, Sparkles, ChevronUp, ChevronDown, Copy, Trash2 } from "lucide-react";
import type { TrackedJob, JobDomain } from "@/stores/jobStore";
import { useCancelJob } from "@/api/hooks/useJobs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const DOMAIN_ICONS: Record<JobDomain, typeof Image> = {
  generate: Image,
  upscale: Sparkles,
  video: Video,
  framepack: Video,
  ltx: Video,
};

const DOMAIN_LABELS: Record<JobDomain, string> = {
  generate: "Image",
  upscale: "Upscale",
  video: "Video",
  framepack: "FramePack",
  ltx: "LTX",
};

function statusBadgeVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "running": return "default";
    case "completed": return "secondary";
    case "failed":
    case "cancelled": return "destructive";
    default: return "outline";
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function useElapsed(startTime: number, active: boolean): number {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    if (!active) return;
    const update = () => setElapsed(Math.floor((performance.now() / 1000) - (startTime / 1000) + (Date.now() - performance.now()) / 1000));
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [startTime, active]);
  return elapsed;
}

interface QueueJobCardProps {
  job: TrackedJob;
  onView?: (job: TrackedJob) => void;
  onRetry?: (job: TrackedJob) => void;
  onDuplicate?: (job: TrackedJob) => void;
  onRemove?: (job: TrackedJob) => void;
  onMoveUp?: (job: TrackedJob) => void;
  onMoveDown?: (job: TrackedJob) => void;
  canMoveUp?: boolean;
  canMoveDown?: boolean;
}

export function QueueJobCard({ job, onView, onRetry, onDuplicate, onRemove, onMoveUp, onMoveDown, canMoveUp, canMoveDown }: QueueJobCardProps) {
  const cancelJob = useCancelJob();
  const DomainIcon = DOMAIN_ICONS[job.domain] ?? Image;
  const isRunning = job.status === "running";
  const isPending = job.status === "pending";
  const isTerminal = job.status === "completed" || job.status === "failed" || job.status === "cancelled";
  const elapsed = useElapsed(job.createdAt, isRunning);

  const handleCancel = useCallback(() => {
    cancelJob.mutate(job.id);
  }, [cancelJob, job.id]);

  return (
    <div className="space-y-1 px-3 py-1.5">
      <div className="flex items-center gap-1.5 text-2xs min-w-0">
        <DomainIcon className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="truncate flex-1 min-w-0">
          {DOMAIN_LABELS[job.domain]}
          {job.task ? ` — ${job.task}` : ""}
        </span>
        <Badge variant={statusBadgeVariant(job.status)} className="text-4xs px-1 py-0 shrink-0">
          {job.status}
        </Badge>
        {/* Reorder buttons for pending */}
        {isPending && onMoveUp && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={() => onMoveUp(job)} disabled={!canMoveUp} title="Move up">
            <ChevronUp className="h-2.5 w-2.5" />
          </Button>
        )}
        {isPending && onMoveDown && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={() => onMoveDown(job)} disabled={!canMoveDown} title="Move down">
            <ChevronDown className="h-2.5 w-2.5" />
          </Button>
        )}
        {/* View result */}
        {isTerminal && onView && job.result && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={() => onView(job)} title="View result">
            <Eye className="h-2.5 w-2.5" />
          </Button>
        )}
        {/* Duplicate */}
        {isTerminal && onDuplicate && job.request && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={() => onDuplicate(job)} title="Duplicate">
            <Copy className="h-2.5 w-2.5" />
          </Button>
        )}
        {/* Retry failed */}
        {job.status === "failed" && onRetry && job.request && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={() => onRetry(job)} title="Retry">
            <RotateCcw className="h-2.5 w-2.5" />
          </Button>
        )}
        {/* Remove terminal */}
        {isTerminal && onRemove && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={() => onRemove(job)} title="Remove">
            <Trash2 className="h-2.5 w-2.5" />
          </Button>
        )}
        {/* Cancel running/pending */}
        {(isRunning || isPending) && (
          <Button size="icon" variant="ghost" className="h-5 w-5 shrink-0" onClick={handleCancel} title="Cancel">
            <X className="h-2.5 w-2.5" />
          </Button>
        )}
      </div>

      {/* Progress bar for running jobs */}
      {isRunning && (
        <div className="flex items-center gap-1.5">
          <div className="h-1.5 rounded bg-primary/20 overflow-hidden flex-1">
            <div className={cn("h-full bg-primary rounded transition-all")} style={{ width: `${(job.progress * 100).toFixed(1)}%` }} />
          </div>
          <span className="text-4xs text-muted-foreground tabular-nums w-8 text-right">
            {job.step > 0 ? `${job.step}/${job.steps}` : `${Math.round(job.progress * 100)}%`}
          </span>
        </div>
      )}

      {/* ETA and elapsed for running */}
      {isRunning && (
        <div className="flex items-center gap-2 text-4xs text-muted-foreground">
          <span>{formatDuration(elapsed)} elapsed</span>
          {job.eta > 0 && <span>ETA: ~{formatDuration(job.eta)}</span>}
        </div>
      )}

      {/* Preview thumbnail for running */}
      {isRunning && job.previewUrl && (
        <img src={job.previewUrl} alt="Preview" className="w-full rounded-sm max-h-24 object-cover" />
      )}

      {/* Error message */}
      {job.status === "failed" && job.error && (
        <p className="text-4xs text-destructive truncate" title={job.error}>{job.error}</p>
      )}

      {/* Timestamp for terminal */}
      {isTerminal && (
        <p className="text-4xs text-muted-foreground">
          {new Date(job.createdAt).toLocaleTimeString()}
        </p>
      )}
    </div>
  );
}
