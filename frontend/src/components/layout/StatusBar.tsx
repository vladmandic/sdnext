import { useJobQueueStore, selectRunningJob, selectPendingCount, selectHasActiveJobs } from "@/stores/jobStore";
import { useStatus } from "@/api/hooks/useGeneration";
import { useMemory } from "@/api/hooks/useServer";
import { formatBytes, formatDuration } from "@/lib/utils";
import { LoadedModelsPanel } from "@/components/layout/LoadedModelsPanel";

const DOMAIN_LABELS: Record<string, string> = {
  generate: "Generating",
  upscale: "Upscaling",
  video: "Video",
  framepack: "FramePack",
  ltx: "LTX Video",
};

export function StatusBar() {
  const runningJob = useJobQueueStore(selectRunningJob);
  const pendingCount = useJobQueueStore(selectPendingCount);
  const hasActive = useJobQueueStore(selectHasActiveJobs);
  const { data: status } = useStatus();
  const { data: memory } = useMemory();

  const backendIdle = !status || status.status === "idle" || status.status === "interrupted" || status.status === "skipped";
  const isIdle = !hasActive && backendIdle;

  const progress = runningJob?.progress || status?.progress || 0;
  const progressPct = !isIdle ? Math.round(progress * 100) : 0;
  const step = runningJob?.step || status?.step || 0;
  const steps = runningJob?.steps || status?.steps || 0;
  const eta = runningJob?.eta || status?.eta || 0;

  const phase = status?.current || status?.task || "";
  const domainLabel = runningJob ? (DOMAIN_LABELS[runningJob.domain] ?? "Working") : "Working";
  const taskName = phase || domainLabel;

  return (
    <footer className="flex items-center h-6 px-3 gap-4 border-t border-border bg-card text-[11px] text-muted-foreground flex-shrink-0">
      {/* Status */}
      <span className="flex items-center gap-1.5">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full ${isIdle ? "bg-emerald-500" : "bg-amber-500 animate-pulse"}`}
        />
        {isIdle ? "Idle" : taskName}
      </span>

      {/* Progress */}
      {!isIdle && (
        <>
          {steps > 0 && (
            <span>
              Step {step}/{steps}
            </span>
          )}
          <div className="flex items-center gap-1.5 min-w-[120px]">
            <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-[width] duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <span className="tabular-nums">{progressPct}%</span>
          </div>
          {eta > 0 && (
            <span>ETA {formatDuration(eta)}</span>
          )}
          {pendingCount > 0 && (
            <span>Queue: {pendingCount}</span>
          )}
        </>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Memory info */}
      {memory?.cuda?.allocated && (
        <LoadedModelsPanel>
          <button type="button" className="hover:text-foreground transition-colors cursor-pointer">
            VRAM {formatBytes(memory.cuda.allocated.current)} / {formatBytes(memory.cuda.system?.total ?? 0)}
          </button>
        </LoadedModelsPanel>
      )}

      {memory?.ram && !memory.ram.error && (
        <span>
          RAM {formatBytes(memory.ram.used ?? 0)} / {formatBytes(memory.ram.total ?? 0)}
        </span>
      )}

      {/* Uptime */}
      {status?.uptime != null && <span>{formatDuration(status.uptime)}</span>}
    </footer>
  );
}
