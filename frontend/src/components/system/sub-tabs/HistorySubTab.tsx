import { useState } from "react";
import { useHistory } from "@/api/hooks/useSystem";
import { formatDuration } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { ChevronDown, ChevronRight } from "lucide-react";

export function HistorySubTab() {
  const { data: history } = useHistory();
  const [filter, setFilter] = useState("");
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const filtered = (history ?? []).filter((entry) => {
    if (!filter) return true;
    const q = filter.toLowerCase();
    return (
      entry.job?.toLowerCase().includes(q) ||
      entry.op?.toLowerCase().includes(q) ||
      entry.outputs?.some((o) => o.toLowerCase().includes(q))
    );
  });

  function toggle(idx: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) next.delete(idx);
      else next.add(idx);
      return next;
    });
  }

  return (
    <div className="space-y-3">
      <Input
        placeholder="Filter history..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        className="h-6 text-2xs"
      />

      {filtered.length === 0 ? (
        <p className="text-xs text-muted-foreground text-center py-4">No history entries</p>
      ) : (
        <div className="space-y-0.5">
          <div className="grid grid-cols-[1fr_auto_auto] gap-2 text-3xs font-medium text-muted-foreground px-2 pb-1 border-b border-border">
            <span>Job / Op</span>
            <span>Duration</span>
            <span>Time</span>
          </div>
          {filtered.map((entry, i) => {
            const hasOutputs = entry.outputs && entry.outputs.length > 0;
            const isExpanded = expanded.has(i);
            return (
              <div key={entry.id ?? i}>
                <button
                  type="button"
                  className="w-full grid grid-cols-[1fr_auto_auto] gap-2 items-center text-xs px-2 py-1 rounded hover:bg-muted/50 text-left"
                  onClick={() => hasOutputs && toggle(i)}
                >
                  <span className="truncate flex items-center gap-1">
                    {hasOutputs && (
                      isExpanded
                        ? <ChevronDown className="h-3 w-3 shrink-0" />
                        : <ChevronRight className="h-3 w-3 shrink-0" />
                    )}
                    <span className="font-medium">{entry.op}</span>
                    {entry.job && <span className="text-muted-foreground">({entry.job})</span>}
                  </span>
                  <span className="tabular-nums text-muted-foreground text-2xs">
                    {entry.duration != null ? formatDuration(entry.duration) : "-"}
                  </span>
                  <span className="tabular-nums text-muted-foreground text-2xs">
                    {entry.timestamp ? new Date(entry.timestamp * 1000).toLocaleTimeString() : "-"}
                  </span>
                </button>
                {isExpanded && hasOutputs && (
                  <div className="pl-7 pr-2 pb-1 space-y-0.5">
                    {entry.outputs.map((output, j) => (
                      <p key={j} className="text-3xs text-muted-foreground truncate">{output}</p>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
