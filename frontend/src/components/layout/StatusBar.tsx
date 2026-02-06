import { useStatus } from "@/api/hooks/useGeneration";
import { useMemory } from "@/api/hooks/useServer";
import { formatBytes, formatDuration } from "@/lib/utils";

export function StatusBar() {
  const { data: status } = useStatus();
  const { data: memory } = useMemory();

  const isIdle = !status || status.status === "idle";
  const progressPct = status?.progress != null ? Math.round(status.progress * 100) : 0;

  return (
    <footer className="flex items-center h-6 px-3 gap-4 border-t border-border bg-card text-[11px] text-muted-foreground flex-shrink-0">
      {/* Status */}
      <span className="flex items-center gap-1.5">
        <span
          className={`inline-block w-1.5 h-1.5 rounded-full ${isIdle ? "bg-emerald-500" : "bg-amber-500 animate-pulse"}`}
        />
        {isIdle ? "Idle" : status?.task || "Working"}
      </span>

      {/* Progress */}
      {!isIdle && status && (
        <>
          <span>
            Step {status.step}/{status.steps}
          </span>
          <div className="flex items-center gap-1.5 min-w-[120px]">
            <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-[width] duration-300"
                style={{ width: `${progressPct}%` }}
              />
            </div>
            <span className="tabular-nums">{progressPct}%</span>
          </div>
          {status.eta != null && status.eta > 0 && (
            <span>ETA {formatDuration(status.eta)}</span>
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
