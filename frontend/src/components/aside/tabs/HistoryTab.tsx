import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { formatDuration } from "@/lib/utils";

interface HistoryJob {
  id: string;
  op: string;
  job: string;
  timestamp: number;
  duration: number;
}

export function HistoryTab() {
  const { data: history } = useQuery({
    queryKey: ["history"],
    queryFn: () => api.get<HistoryJob[]>("/sdapi/v1/history"),
    staleTime: 10_000,
  });

  return (
    <div className="p-3 space-y-1">
      {history?.map((job) => (
        <div key={job.id} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-muted/50 text-xs">
          <span className="truncate min-w-0">{job.op || job.job}</span>
          <div className="flex items-center gap-2 shrink-0 text-muted-foreground">
            {job.duration > 0 && <span>{formatDuration(job.duration)}</span>}
            <span>{new Date(job.timestamp * 1000).toLocaleTimeString()}</span>
          </div>
        </div>
      ))}
      {(!history || history.length === 0) && (
        <p className="text-xs text-muted-foreground">No generation history yet.</p>
      )}
    </div>
  );
}
