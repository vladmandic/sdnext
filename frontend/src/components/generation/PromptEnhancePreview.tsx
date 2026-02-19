import { useMemo } from "react";
import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import { useGenerationStore } from "@/stores/generationStore";
import { wordDiff } from "@/lib/wordDiff";
import { Check, RotateCcw, X, Loader2 } from "lucide-react";

interface PromptEnhancePreviewProps {
  onEnhance: () => void;
  isPending: boolean;
}

export function PromptEnhancePreview({ onEnhance, isPending }: PromptEnhancePreviewProps) {
  const pendingResult = usePromptEnhanceStore((s) => s.pendingResult);
  const setPendingResult = usePromptEnhanceStore((s) => s.setPendingResult);
  const addToHistory = usePromptEnhanceStore((s) => s.addToHistory);
  const setParam = useGenerationStore((s) => s.setParam);

  const segments = useMemo(() => {
    if (!pendingResult) return [];
    return wordDiff(pendingResult.originalPrompt, pendingResult.prompt);
  }, [pendingResult]);

  if (!pendingResult) return null;

  const handleAccept = () => {
    setParam("prompt", pendingResult.prompt);
    addToHistory({
      prompt: pendingResult.prompt,
      originalPrompt: pendingResult.originalPrompt,
      seed: pendingResult.seed,
    });
    setPendingResult(null);
  };

  return (
    <div className="flex flex-col gap-2 p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground/60">Preview</div>
      <div className="text-xs leading-relaxed p-2 rounded-md bg-muted/30 max-h-[200px] overflow-y-auto">
        {segments.map((seg, i) => {
          if (seg.type === "equal") return <span key={i}>{seg.text}</span>;
          if (seg.type === "added") return <span key={i} className="bg-green-500/20 text-green-300 rounded px-0.5">{seg.text}</span>;
          return <span key={i} className="bg-red-500/20 text-red-300 line-through rounded px-0.5">{seg.text}</span>;
        })}
      </div>
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          onClick={handleAccept}
          className="flex items-center gap-1 px-2 py-1 rounded text-[11px] bg-green-600/20 text-green-300 hover:bg-green-600/30 transition-colors"
        >
          <Check size={12} /> Accept
        </button>
        <button
          type="button"
          onClick={onEnhance}
          disabled={isPending}
          className="flex items-center gap-1 px-2 py-1 rounded text-[11px] bg-muted text-muted-foreground hover:bg-muted/80 transition-colors disabled:opacity-50"
        >
          {isPending ? <Loader2 size={12} className="animate-spin" /> : <RotateCcw size={12} />} Retry
        </button>
        <button
          type="button"
          onClick={() => setPendingResult(null)}
          className="ml-auto p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          title="Dismiss"
        >
          <X size={12} />
        </button>
      </div>
    </div>
  );
}
