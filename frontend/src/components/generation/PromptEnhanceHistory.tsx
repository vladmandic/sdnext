import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import { useGenerationStore } from "@/stores/generationStore";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Trash2 } from "lucide-react";
import { toast } from "sonner";

function formatRelativeTime(timestamp: number): string {
  const seconds = Math.floor((Date.now() - timestamp) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function PromptEnhanceHistory() {
  const history = usePromptEnhanceStore((s) => s.history);
  const clearHistory = usePromptEnhanceStore((s) => s.clearHistory);
  const setParam = useGenerationStore((s) => s.setParam);

  if (history.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5 p-3">
      <div className="flex items-center justify-between">
        <span className="text-3xs uppercase tracking-wider text-muted-foreground/60">
          History ({history.length})
        </span>
        <button
          type="button"
          onClick={() => {
            clearHistory();
            toast.info("History cleared");
          }}
          className="p-0.5 rounded text-muted-foreground/50 hover:text-red-400 hover:bg-muted transition-colors"
          title="Clear history"
        >
          <Trash2 size={11} />
        </button>
      </div>
      <ScrollArea className="max-h-50">
        <div className="flex flex-col gap-0.5">
          {history.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setParam("prompt", item.prompt);
                toast.success("Prompt loaded from history");
              }}
              className="flex items-start gap-2 w-full text-left p-1.5 rounded hover:bg-muted/50 transition-colors group"
            >
              <span className="text-2xs leading-snug text-muted-foreground group-hover:text-foreground line-clamp-2 flex-1">
                {item.prompt.length > 80 ? `${item.prompt.slice(0, 80)}...` : item.prompt}
              </span>
              <span className="flex flex-col items-end gap-0.5 flex-shrink-0">
                <span className="text-4xs bg-muted px-1 rounded text-muted-foreground">{item.seed}</span>
                <span className="text-4xs text-muted-foreground/50">{formatRelativeTime(item.timestamp)}</span>
              </span>
            </button>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
