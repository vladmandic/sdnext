import { useGenerationStore } from "@/stores/generationStore";
import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight } from "lucide-react";

export function PromptEditor() {
  const prompt = useGenerationStore((s) => s.prompt);
  const negativePrompt = useGenerationStore((s) => s.negativePrompt);
  const setParam = useGenerationStore((s) => s.setParam);
  const [showNegative, setShowNegative] = useState(false);

  return (
    <div className="flex flex-col gap-2">
      {/* Positive prompt */}
      <div>
        <Label className="text-[11px] text-muted-foreground mb-1 block">Prompt</Label>
        <Textarea
          value={prompt}
          onChange={(e) => setParam("prompt", e.target.value)}
          placeholder="Describe what you want to generate..."
          className="min-h-[80px] max-h-[200px] resize-y text-sm"
          onKeyDown={(e) => {
            if (e.ctrlKey && (e.key === "ArrowUp" || e.key === "ArrowDown")) {
              e.preventDefault();
              adjustAttentionWeight(e.currentTarget, e.key === "ArrowUp" ? 0.1 : -0.1);
              setParam("prompt", e.currentTarget.value);
            }
          }}
        />
      </div>

      {/* Negative prompt */}
      <Collapsible open={showNegative} onOpenChange={setShowNegative}>
        <CollapsibleTrigger className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
          {showNegative ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
          Negative prompt
        </CollapsibleTrigger>
        <CollapsibleContent>
          <Textarea
            value={negativePrompt}
            onChange={(e) => setParam("negativePrompt", e.target.value)}
            placeholder="What to avoid..."
            className="min-h-[50px] max-h-[120px] resize-y text-sm mt-1.5"
          />
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

function adjustAttentionWeight(textarea: HTMLTextAreaElement, delta: number) {
  const { selectionStart, selectionEnd, value } = textarea;
  if (selectionStart === selectionEnd) return;

  const selected = value.slice(selectionStart, selectionEnd);

  const match = selected.match(/^\((.+):([0-9.]+)\)$/);
  if (match) {
    const newWeight = Math.max(0, Math.min(2, parseFloat(match[2]) + delta));
    const replacement = `(${match[1]}:${newWeight.toFixed(1)})`;
    textarea.value = value.slice(0, selectionStart) + replacement + value.slice(selectionEnd);
    textarea.selectionStart = selectionStart;
    textarea.selectionEnd = selectionStart + replacement.length;
  } else {
    const weight = Math.max(0, Math.min(2, 1.0 + delta));
    const replacement = `(${selected}:${weight.toFixed(1)})`;
    textarea.value = value.slice(0, selectionStart) + replacement + value.slice(selectionEnd);
    textarea.selectionStart = selectionStart;
    textarea.selectionEnd = selectionStart + replacement.length;
  }
}
