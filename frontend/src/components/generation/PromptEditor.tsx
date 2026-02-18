import { useGenerationStore } from "@/stores/generationStore";
import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import { usePromptEnhance } from "@/api/hooks/usePromptEnhance";
import { useState, useCallback } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ChevronDown, ChevronRight, Sparkles, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { PromptEnhancePanel } from "./PromptEnhancePanel";
import type { PromptEnhanceRequest } from "@/api/types/promptEnhance";

export function PromptEditor() {
  const prompt = useGenerationStore((s) => s.prompt);
  const negativePrompt = useGenerationStore((s) => s.negativePrompt);
  const setParam = useGenerationStore((s) => s.setParam);
  const [showNegative, setShowNegative] = useState(false);
  const [showEnhancePanel, setShowEnhancePanel] = useState(false);

  const enhanceStore = usePromptEnhanceStore();
  const enhanceMutation = usePromptEnhance();

  const handleEnhance = useCallback(() => {
    if (!prompt.trim()) {
      toast.warning("Enter a prompt first");
      return;
    }
    const req: PromptEnhanceRequest = {
      prompt,
      type: "text",
      model: enhanceStore.model || undefined,
      system_prompt: enhanceStore.systemPrompt || undefined,
      prefix: enhanceStore.prefix || undefined,
      suffix: enhanceStore.suffix || undefined,
      nsfw: enhanceStore.nsfw,
      seed: enhanceStore.seed,
      do_sample: enhanceStore.doSample,
      max_tokens: enhanceStore.maxTokens,
      temperature: enhanceStore.temperature,
      repetition_penalty: enhanceStore.repetitionPenalty,
      top_k: enhanceStore.topK || undefined,
      top_p: enhanceStore.topP || undefined,
      thinking: enhanceStore.thinking,
      keep_thinking: enhanceStore.keepThinking,
      use_vision: enhanceStore.useVision,
      prefill: enhanceStore.prefill || undefined,
      keep_prefill: enhanceStore.keepPrefill,
    };
    enhanceMutation.mutate(req, {
      onSuccess: (res) => {
        setParam("prompt", res.prompt);
        toast.success(`Prompt enhanced (seed: ${res.seed})`);
      },
      onError: (err) => {
        toast.error(`Enhance failed: ${err instanceof Error ? err.message : "Unknown error"}`);
      },
    });
  }, [prompt, enhanceStore, enhanceMutation, setParam]);

  return (
    <div className="flex flex-col gap-2">
      {/* Positive prompt */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <Label className="text-[11px] text-muted-foreground">Prompt</Label>
          <div className="flex items-center gap-0.5">
            <button
              type="button"
              onClick={handleEnhance}
              disabled={enhanceMutation.isPending}
              className="p-0.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
              title="Enhance prompt"
            >
              {enhanceMutation.isPending
                ? <Loader2 size={14} className="animate-spin" />
                : <Sparkles size={14} />}
            </button>
            <button
              type="button"
              onClick={() => setShowEnhancePanel((v) => !v)}
              className="p-0.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              title="Enhance settings"
            >
              {showEnhancePanel ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            </button>
          </div>
        </div>
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
        {showEnhancePanel && <PromptEnhancePanel />}
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
