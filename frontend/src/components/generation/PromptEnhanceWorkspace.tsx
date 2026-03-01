import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import { Sparkles, Pin, PinOff, X, Loader2 } from "lucide-react";
import { Separator } from "@/components/ui/separator";
import { PromptEnhancePreview } from "./PromptEnhancePreview";
import { PromptEnhancePanel } from "./PromptEnhancePanel";
import { PromptEnhanceHistory } from "./PromptEnhanceHistory";

interface PromptEnhanceWorkspaceProps {
  onEnhance: () => void;
  isPending: boolean;
  onClose: () => void;
  onAccept?: (prompt: string) => void;
  onSelectPrompt?: (prompt: string) => void;
}

export function PromptEnhanceWorkspace({ onEnhance, isPending, onClose, onAccept, onSelectPrompt }: PromptEnhanceWorkspaceProps) {
  const pinned = usePromptEnhanceStore((s) => s.pinned);
  const setPinned = usePromptEnhanceStore((s) => s.setPinned);
  const pendingResult = usePromptEnhanceStore((s) => s.pendingResult);
  const history = usePromptEnhanceStore((s) => s.history);

  return (
    <div className="flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border sticky top-0 bg-popover z-10">
        <span className="text-xs font-medium flex-1">Enhance</span>
        <button
          type="button"
          onClick={onEnhance}
          disabled={isPending}
          className="p-1 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
          title="Enhance prompt"
        >
          {isPending ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
        </button>
        <button
          type="button"
          onClick={() => setPinned(!pinned)}
          className={`p-1 rounded transition-colors ${pinned ? "text-foreground bg-muted" : "text-muted-foreground hover:text-foreground hover:bg-muted"}`}
          title={pinned ? "Unpin panel" : "Pin panel open"}
        >
          {pinned ? <Pin size={13} /> : <PinOff size={13} />}
        </button>
        <button
          type="button"
          onClick={onClose}
          className="p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          title="Close"
        >
          <X size={13} />
        </button>
      </div>

      {/* Preview */}
      {pendingResult && (
        <>
          <PromptEnhancePreview onEnhance={onEnhance} isPending={isPending} onAccept={onAccept} />
          <Separator />
        </>
      )}

      {/* Settings */}
      <div className="p-4">
        <PromptEnhancePanel />
      </div>

      {/* History */}
      {history.length > 0 && (
        <>
          <Separator />
          <PromptEnhanceHistory onSelectPrompt={onSelectPrompt} />
        </>
      )}
    </div>
  );
}
