import { useMemo, useState, useCallback } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { usePromptStyles } from "@/api/hooks/useNetworks";
import { Link2, Link2Off, ArrowLeftRight, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { PromptEditor } from "../PromptEditor";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function PromptsTab() {
  const state = useGenerationStore(useShallow((s) => ({
    width: s.width,
    height: s.height,
    batchCount: s.batchCount,
    batchSize: s.batchSize,
  })));
  const setParam = useGenerationStore((s) => s.setParam);
  const { data: styles } = usePromptStyles();
  const [aspectLocked, setAspectLocked] = useState(false);
  const [selectedStyles, setSelectedStyles] = useState<string[]>([]);

  const aspectRatio = state.width / state.height;

  const setWidth = useCallback((w: number) => {
    const rounded = Math.round(w / 8) * 8;
    setParam("width", rounded);
    if (aspectLocked) setParam("height", Math.round(rounded / aspectRatio / 8) * 8);
  }, [setParam, aspectLocked, aspectRatio]);

  const setHeight = useCallback((h: number) => {
    const rounded = Math.round(h / 8) * 8;
    setParam("height", rounded);
    if (aspectLocked) setParam("width", Math.round(rounded * aspectRatio / 8) * 8);
  }, [setParam, aspectLocked, aspectRatio]);

  const swapDimensions = useCallback(() => {
    const w = state.width;
    setParam("width", state.height);
    setParam("height", w);
  }, [setParam, state.width, state.height]);

  const set = useMemo(() => ({
    batchCount: (v: number) => setParam("batchCount", v),
    batchSize: (v: number) => setParam("batchSize", v),
  }), [setParam]);

  function addStyle(name: string) {
    if (!selectedStyles.includes(name)) {
      setSelectedStyles([...selectedStyles, name]);
    }
  }

  function removeStyle(name: string) {
    setSelectedStyles(selectedStyles.filter((s) => s !== name));
  }

  return (
    <div className="flex flex-col gap-3 text-sm">
      <PromptEditor />

      {styles && styles.length > 0 && (
        <ParamSection title="Styles" defaultOpen={false}>
          <div className="flex flex-wrap gap-1 mb-1">
            {selectedStyles.map((name) => (
              <span key={name} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] bg-muted rounded">
                {name}
                <button onClick={() => removeStyle(name)} className="text-muted-foreground hover:text-foreground">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
          <Select value="_placeholder_" onValueChange={addStyle}>
            <SelectTrigger size="sm" className="h-6 text-[11px]">
              <SelectValue placeholder="Add style..." />
            </SelectTrigger>
            <SelectContent>
              {styles.filter((s) => !selectedStyles.includes(s.name)).map((s) => (
                <SelectItem key={s.name} value={s.name} className="text-xs">{s.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </ParamSection>
      )}

      <ParamSection title="Size">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Width</Label>
          <NumberInput
            value={state.width}
            onChange={setWidth}
            step={8} min={64} max={4096} fallback={512}
            className="flex-1 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Button
            variant="ghost" size="icon-xs"
            onClick={() => setAspectLocked(!aspectLocked)}
            className={cn(aspectLocked ? "text-primary" : "text-muted-foreground")}
            title={aspectLocked ? "Unlock aspect ratio" : "Lock aspect ratio"}
          >
            {aspectLocked ? <Link2 size={12} /> : <Link2Off size={12} />}
          </Button>
          <Button
            variant="ghost" size="icon-xs"
            onClick={swapDimensions}
            className="text-muted-foreground"
            title="Swap width/height"
          >
            <ArrowLeftRight size={12} />
          </Button>
          <NumberInput
            value={state.height}
            onChange={setHeight}
            step={8} min={64} max={4096} fallback={512}
            className="flex-1 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Label className="text-[11px] text-muted-foreground w-16">Height</Label>
        </div>
      </ParamSection>

      <ParamSection title="Batch">
        <ParamSlider label="Count" value={state.batchCount} onChange={set.batchCount} min={1} max={100} />
        <ParamSlider label="Size" value={state.batchSize} onChange={set.batchSize} min={1} max={16} />
      </ParamSection>
    </div>
  );
}
