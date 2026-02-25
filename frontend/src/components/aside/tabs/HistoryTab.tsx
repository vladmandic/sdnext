import { useHistory } from "@/api/hooks/useSystem";
import { formatDuration } from "@/lib/utils";

export function HistoryTab() {
  const { data: history } = useHistory();

  return (
    <div className="p-3 space-y-1">
      {history?.items.map((job) => (
        <div key={job.id} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50 text-xs">
          <span className="truncate min-w-0">{job.op || job.job}</span>
          <div className="flex items-center gap-2 shrink-0 text-muted-foreground">
            {job.duration != null && job.duration > 0 && <span>{formatDuration(job.duration)}</span>}
            {job.timestamp != null && <span>{new Date(job.timestamp * 1000).toLocaleTimeString()}</span>}
          </div>
        </div>
      ))}
      {(!history || history.items.length === 0) && (
        <p className="text-xs text-muted-foreground">No generation history yet.</p>
      )}
    </div>
  );
}
