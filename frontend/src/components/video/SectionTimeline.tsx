import { useState, useMemo, useCallback } from "react";
import { Copy, ClipboardPaste } from "lucide-react";
import { toast } from "sonner";
import { computeSectionCount } from "@/lib/framepackSections";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface SectionTimelineProps {
  fps: number;
  duration: number;
  latentWindowSize: number;
  variant: string;
  interpolate: number;
  value: string;
  onChange: (value: string) => void;
}

function splitPrompts(raw: string): string[] {
  if (!raw.trim()) return [];
  return raw.split(/[,\n]+/).map((s) => s.trim());
}

function joinPrompts(prompts: string[]): string {
  return prompts.join(", ");
}

export function SectionTimeline({ fps, duration, latentWindowSize, variant, interpolate, value, onChange }: SectionTimelineProps) {
  const [activeSection, setActiveSection] = useState<number | null>(null);

  const sectionCount = useMemo(
    () => computeSectionCount(fps, duration, latentWindowSize, variant, interpolate),
    [fps, duration, latentWindowSize, variant, interpolate],
  );

  const prompts = useMemo(() => {
    const parsed = splitPrompts(value);
    const arr: string[] = [];
    for (let i = 0; i < sectionCount; i++) {
      arr.push(parsed[i] ?? "");
    }
    return arr;
  }, [value, sectionCount]);

  const allSame = useMemo(() => {
    if (prompts.length === 0) return true;
    const first = prompts[0];
    return prompts.every((p) => p === first);
  }, [prompts]);

  const handleSectionChange = useCallback((index: number, text: string) => {
    const next = [...prompts];
    next[index] = text;
    onChange(joinPrompts(next));
  }, [prompts, onChange]);

  const handleGlobalChange = useCallback((text: string) => {
    const filled = new Array(sectionCount).fill(text) as string[];
    onChange(joinPrompts(filled));
  }, [sectionCount, onChange]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(value);
    toast.success("Copied section prompts");
  }, [value]);

  const handlePaste = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      onChange(text);
      toast.success("Pasted section prompts");
    } catch {
      toast.error("Failed to read clipboard");
    }
  }, [onChange]);

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-2xs text-muted-foreground">{sectionCount} section{sectionCount !== 1 ? "s" : ""}</span>
        <div className="flex gap-0.5">
          <Button size="icon" variant="ghost" className="h-5 w-5" onClick={handleCopy} title="Copy as text">
            <Copy className="h-3 w-3" />
          </Button>
          <Button size="icon" variant="ghost" className="h-5 w-5" onClick={handlePaste} title="Paste from text">
            <ClipboardPaste className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Timeline bar */}
      <div className="flex gap-0.5">
        {Array.from({ length: sectionCount }, (_, i) => {
          const hasPrompt = !!prompts[i];
          const isActive = activeSection === i;
          return (
            <button
              key={i}
              type="button"
              onClick={() => setActiveSection(isActive ? null : i)}
              className={cn(
                "flex-1 h-7 rounded-sm text-3xs font-medium transition-all flex items-center justify-center",
                isActive
                  ? "bg-primary/20 ring-1 ring-primary text-foreground"
                  : hasPrompt
                    ? "bg-muted text-foreground hover:bg-muted/80"
                    : "bg-muted/40 text-muted-foreground hover:bg-muted/60",
              )}
              title={prompts[i] || `Section ${i + 1}`}
            >
              {i + 1}
            </button>
          );
        })}
      </div>

      {/* Editor */}
      {activeSection === null ? (
        /* Global prompt — shown when no section is selected */
        <div className="space-y-1">
          <label className="text-2xs text-muted-foreground">
            {allSame ? "All sections" : "Click a section to edit individually"}
          </label>
          {allSame && (
            <Input
              value={prompts[0] ?? ""}
              onChange={(e) => handleGlobalChange(e.target.value)}
              placeholder="Shared prompt for all sections"
              className="h-7 text-xs"
            />
          )}
        </div>
      ) : (
        /* Per-section prompt */
        <div className="space-y-1">
          <label className="text-2xs text-muted-foreground">Section {activeSection + 1}</label>
          <Input
            value={prompts[activeSection]}
            onChange={(e) => handleSectionChange(activeSection, e.target.value)}
            placeholder={`Prompt for section ${activeSection + 1}`}
            className="h-7 text-xs"
            autoFocus
          />
        </div>
      )}
    </div>
  );
}
