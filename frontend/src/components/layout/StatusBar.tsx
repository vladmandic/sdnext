import { useGenerationStore } from "@/stores/generationStore";
import { useProgress, useStatus } from "@/api/hooks/useGeneration";
import { useMemory } from "@/api/hooks/useServer";
import { formatBytes, formatDuration } from "@/lib/utils";

export function StatusBar() {
  const isGenerating = useGenerationStore((s) => s.isGenerating);
  const { data: status } = useStatus();
  const { data: progressData } = useProgress(isGenerating);
  const { data: memory } = useMemory();

  // Use client-side isGenerating as ground truth since backend status
  // flips to 'idle' between sub-phases (TE Encode → Base → Inference).
  const isIdle = !isGenerating && (!status || status.status === "idle");

  // Get step/steps from the progress endpoint state dict (raw values).
  const samplingStep = (progressData?.state?.sampling_step as number) ?? 0;
  const samplingSteps = (progressData?.state?.sampling_steps as number) ?? 0;

  // Use the progress endpoint's corrected progress (with fallback calculation).
  const progressPct = !isIdle ? Math.round((progressData?.progress ?? 0) * 100) : 0;

  // Task name from status endpoint when available, else generic label.
  const taskName = status?.current || status?.task || "Working";

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
          {samplingSteps > 0 && (
            <span>
              Step {samplingStep}/{samplingSteps}
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
          {progressData?.eta_relative != null && progressData.eta_relative > 0 && (
            <span>ETA {formatDuration(progressData.eta_relative)}</span>
          )}
        </>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Memory info */}
      {memory?.cuda?.allocated && (
        <span>
          VRAM {formatBytes(memory.cuda.allocated.current)} / {formatBytes(memory.cuda.system?.total ?? 0)}
        </span>
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
