import { useState, useRef, useEffect } from "react";
import { Copy, Trash2, WrapText } from "lucide-react";
import { useServerLog, useClearLog } from "@/api/hooks/useLog";
import { cn } from "@/lib/utils";

export function ConsoleTab() {
  const { data: lines } = useServerLog(200);
  const clearLog = useClearLog();
  const [wrap, setWrap] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const autoScroll = useRef(true);

  useEffect(() => {
    const el = scrollRef.current;
    if (el && autoScroll.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [lines]);

  function handleScroll() {
    const el = scrollRef.current;
    if (!el) return;
    autoScroll.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  async function handleCopy() {
    if (lines) {
      await navigator.clipboard.writeText(lines.join("\n"));
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border shrink-0">
        <button
          type="button"
          onClick={() => setWrap(!wrap)}
          className={cn(
            "p-1.5 rounded text-muted-foreground hover:text-foreground transition-colors",
            wrap && "bg-muted text-foreground",
          )}
          title="Toggle line wrap"
        >
          <WrapText className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={handleCopy}
          className="p-1.5 rounded text-muted-foreground hover:text-foreground transition-colors"
          title="Copy all"
        >
          <Copy className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={() => clearLog.mutate()}
          className="p-1.5 rounded text-muted-foreground hover:text-foreground transition-colors"
          title="Clear log"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
        <span className="ml-auto text-3xs text-muted-foreground tabular-nums">
          {lines?.length ?? 0} lines
        </span>
      </div>

      {/* Log output */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-auto bg-muted/30 font-mono text-2xs leading-relaxed p-2"
      >
        {lines?.map((line, i) => (
          <div key={i} className={cn("px-1 hover:bg-muted/50", !wrap && "whitespace-nowrap")}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
