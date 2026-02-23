import { X } from "lucide-react";
import { useCivitHistory, useCivitClearHistory } from "@/api/hooks/useCivitai";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface CivitSearchHistoryProps {
  onSelect: (query: string, tag: string) => void;
}

export function CivitSearchHistory({ onSelect }: CivitSearchHistoryProps) {
  const { data } = useCivitHistory();
  const clearHistory = useCivitClearHistory();
  const entries = data?.history ?? [];

  if (entries.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {entries.slice(0, 10).map((e) => (
        <Badge
          key={`${e.type}-${e.term}`}
          variant="secondary"
          className="text-2xs cursor-pointer hover:bg-muted"
          onClick={() => onSelect(e.type === "query" ? e.term : "", e.type === "tag" ? e.term : "")}
        >
          {e.type === "tag" ? `#${e.term}` : e.term}
        </Badge>
      ))}
      <Button variant="ghost" size="icon" className="h-5 w-5" onClick={() => clearHistory.mutate()}>
        <X className="h-3 w-3" />
      </Button>
    </div>
  );
}
