import { useJobQueueStore, selectRunningJob, selectPendingCount, selectHasActiveJobs } from "@/stores/jobStore";
import { useBackendStatusStore } from "@/stores/backendStatusStore";
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
  const { data: memory } = useMemory();

  const bsStatus = useBackendStatusStore((s) => s.status);
  const bsCurrent = useBackendStatusStore((s) => s.current);
  const bsTask = useBackendStatusStore((s) => s.task);
  const bsTextinfo = useBackendStatusStore((s) => s.textinfo);
  const bsStep = useBackendStatusStore((s) => s.step);
  const bsSteps = useBackendStatusStore((s) => s.steps);
  const bsProgress = useBackendStatusStore((s) => s.progress);
  const bsEta = useBackendStatusStore((s) => s.eta);
  const bsUptime = useBackendStatusStore((s) => s.uptime);
  const wsConnected = useBackendStatusStore((s) => s.connected);

  const backendIdle = !bsStatus || bsStatus === "idle" || bsStatus === "interrupted" || bsStatus === "skipped";
  const isIdle = !hasActive && backendIdle;

  const progress = runningJob?.progress ?? bsProgress ?? 0;
  const progressPct = !isIdle ? Math.round(progress * 100) : 0;
  const step = runningJob?.step ?? bsStep ?? 0;
  const steps = runningJob?.steps ?? bsSteps ?? 0;
  const eta = runningJob?.eta ?? bsEta ?? 0;

  const phase = bsCurrent || bsTask || "";
  const textinfo = runningJob?.textinfo || bsTextinfo || "";
  const domainLabel = runningJob ? (DOMAIN_LABELS[runningJob.domain] ?? "Working") : "Working";
  const taskName = phase || domainLabel;

  // Connection indicator: green=idle, amber=active, red=disconnected
  const dotClass = !wsConnected
    ? "bg-red-500"
    : isIdle
      ? "bg-emerald-500"
      : "bg-amber-500 animate-pulse";

  return (
    <footer className="flex items-center h-6 px-3 gap-4 border-t border-border bg-card text-2xs text-muted-foreground flex-shrink-0">
      {/* Status */}
      <span className="flex items-center gap-1.5">
        <span className={`inline-block w-1.5 h-1.5 rounded-full ${dotClass}`} />
        {isIdle ? "Idle" : taskName}
        {!isIdle && textinfo && (
          <span className="text-muted-foreground/70">— {textinfo}</span>
        )}
      </span>

      {/* Progress */}
      {!isIdle && (
        <>
          {steps > 0 && (
            <span>
              Step {step}/{steps}
            </span>
          )}
          <div className="flex items-center gap-1.5 min-w-[7.5rem]">
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
            VRAM {formatBytes(memory.cuda.allocated.current ?? 0)} / {formatBytes(memory.cuda.system?.total ?? 0)}
          </button>
        </LoadedModelsPanel>
      )}

      {memory?.ram && !memory.ram.error && (
        <span>
          RAM {formatBytes(memory.ram.used ?? 0)} / {formatBytes(memory.ram.total ?? 0)}
        </span>
      )}

      {/* Uptime */}
      {bsUptime > 0 && <span>{formatDuration(bsUptime)}</span>}
    </footer>
  );
}
