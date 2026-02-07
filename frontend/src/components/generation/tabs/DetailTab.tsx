import { useMemo, useCallback } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useDetailerModels } from "@/api/hooks/useDetailer";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { X } from "lucide-react";

export function DetailTab() {
  const state = useGenerationStore(useShallow((s) => ({
    detailerEnabled: s.detailerEnabled,
    detailerModels: s.detailerModels,
    detailerPrompt: s.detailerPrompt,
    detailerNegative: s.detailerNegative,
    detailerSteps: s.detailerSteps,
    detailerStrength: s.detailerStrength,
    detailerResolution: s.detailerResolution,
    detailerMaxDetected: s.detailerMaxDetected,
    detailerPadding: s.detailerPadding,
    detailerBlur: s.detailerBlur,
    detailerConfidence: s.detailerConfidence,
    detailerIou: s.detailerIou,
    detailerMinSize: s.detailerMinSize,
    detailerMaxSize: s.detailerMaxSize,
    detailerRenoise: s.detailerRenoise,
    detailerRenoiseEnd: s.detailerRenoiseEnd,
    detailerSegmentation: s.detailerSegmentation,
    detailerIncludeDetections: s.detailerIncludeDetections,
    detailerMerge: s.detailerMerge,
    detailerSort: s.detailerSort,
    detailerClasses: s.detailerClasses,
  })));
  const setParam = useGenerationStore((s) => s.setParam);
  const { data: models } = useDetailerModels();

  const set = useMemo(() => ({
    detailerEnabled: (checked: boolean) => setParam("detailerEnabled", checked),
    detailerPrompt: (e: React.ChangeEvent<HTMLTextAreaElement>) => setParam("detailerPrompt", e.target.value),
    detailerNegative: (e: React.ChangeEvent<HTMLTextAreaElement>) => setParam("detailerNegative", e.target.value),
    detailerSteps: (v: number) => setParam("detailerSteps", v),
    detailerStrength: (v: number) => setParam("detailerStrength", v),
    detailerResolution: (v: number) => setParam("detailerResolution", v),
    detailerMaxDetected: (v: number) => setParam("detailerMaxDetected", v),
    detailerPadding: (v: number) => setParam("detailerPadding", v),
    detailerBlur: (v: number) => setParam("detailerBlur", v),
    detailerConfidence: (v: number) => setParam("detailerConfidence", v),
    detailerIou: (v: number) => setParam("detailerIou", v),
    detailerMinSize: (v: number) => setParam("detailerMinSize", v),
    detailerMaxSize: (v: number) => setParam("detailerMaxSize", v),
    detailerRenoise: (v: number) => setParam("detailerRenoise", v),
    detailerRenoiseEnd: (v: number) => setParam("detailerRenoiseEnd", v),
    detailerSegmentation: (c: boolean | "indeterminate") => setParam("detailerSegmentation", !!c),
    detailerIncludeDetections: (c: boolean | "indeterminate") => setParam("detailerIncludeDetections", !!c),
    detailerMerge: (c: boolean | "indeterminate") => setParam("detailerMerge", !!c),
    detailerSort: (c: boolean | "indeterminate") => setParam("detailerSort", !!c),
    detailerClasses: (e: React.ChangeEvent<HTMLInputElement>) => setParam("detailerClasses", e.target.value),
  }), [setParam]);

  const addModel = useCallback((name: string) => {
    const current = useGenerationStore.getState().detailerModels;
    if (!current.includes(name)) {
      setParam("detailerModels", [...current, name]);
    }
  }, [setParam]);

  const removeModel = useCallback((name: string) => {
    const current = useGenerationStore.getState().detailerModels;
    setParam("detailerModels", current.filter((m) => m !== name));
  }, [setParam]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Detailer">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Enabled</Label>
          <Switch checked={state.detailerEnabled} onCheckedChange={set.detailerEnabled} />
        </div>

        {state.detailerEnabled && (
          <>
            <div className="flex flex-col gap-1">
              <Label className="text-[11px] text-muted-foreground">Models</Label>
              <div className="flex flex-wrap gap-1 mb-1">
                {state.detailerModels.map((m) => (
                  <span key={m} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] bg-muted rounded">
                    {m}
                    <button onClick={() => removeModel(m)} className="text-muted-foreground hover:text-foreground">
                      <X size={10} />
                    </button>
                  </span>
                ))}
              </div>
              <Select value="_placeholder_" onValueChange={addModel}>
                <SelectTrigger className="h-7 text-xs flex-1">
                  <SelectValue placeholder="Add model..." />
                </SelectTrigger>
                <SelectContent>
                  {models?.filter((m) => !state.detailerModels.includes(m.name)).map((m) => (
                    <SelectItem key={m.name} value={m.name}>{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1">
              <Label className="text-[11px] text-muted-foreground">Prompt</Label>
              <Textarea
                value={state.detailerPrompt}
                onChange={set.detailerPrompt}
                placeholder="Detailer prompt (optional)"
                className="min-h-[48px] text-xs resize-none"
              />
            </div>

            <div className="flex flex-col gap-1">
              <Label className="text-[11px] text-muted-foreground">Negative</Label>
              <Textarea
                value={state.detailerNegative}
                onChange={set.detailerNegative}
                placeholder="Detailer negative prompt (optional)"
                className="min-h-[36px] text-xs resize-none"
              />
            </div>
          </>
        )}
      </ParamSection>

      {state.detailerEnabled && (
        <>
          <ParamSection title="Generation">
            <ParamSlider label="Steps" value={state.detailerSteps} onChange={set.detailerSteps} min={0} max={99} />
            <ParamSlider label="Strength" value={state.detailerStrength} onChange={set.detailerStrength} min={0} max={1} step={0.01} />
            <ParamSlider label="Resolution" value={state.detailerResolution} onChange={set.detailerResolution} min={256} max={4096} step={8} />
          </ParamSection>

          <ParamSection title="Detection" defaultOpen={false}>
            <ParamSlider label="Max detect" value={state.detailerMaxDetected} onChange={set.detailerMaxDetected} min={1} max={10} />
            <ParamSlider label="Padding" value={state.detailerPadding} onChange={set.detailerPadding} min={0} max={100} />
            <ParamSlider label="Blur" value={state.detailerBlur} onChange={set.detailerBlur} min={0} max={100} />
            <ParamSlider label="Confidence" value={state.detailerConfidence} onChange={set.detailerConfidence} min={0} max={1} step={0.01} />
            <ParamSlider label="IoU" value={state.detailerIou} onChange={set.detailerIou} min={0} max={1} step={0.01} />
            <ParamSlider label="Min size" value={state.detailerMinSize} onChange={set.detailerMinSize} min={0} max={1} step={0.01} />
            <ParamSlider label="Max size" value={state.detailerMaxSize} onChange={set.detailerMaxSize} min={0} max={1} step={0.01} />
            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Classes</Label>
              <Input
                value={state.detailerClasses}
                onChange={set.detailerClasses}
                placeholder="e.g. person, face"
                className="flex-1 h-6 text-[11px] px-2"
              />
            </div>
          </ParamSection>

          <ParamSection title="Options" defaultOpen={false}>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={state.detailerSegmentation} onCheckedChange={set.detailerSegmentation} />
                Segmentation
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={state.detailerIncludeDetections} onCheckedChange={set.detailerIncludeDetections} />
                Include detections
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={state.detailerMerge} onCheckedChange={set.detailerMerge} />
                Merge
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={state.detailerSort} onCheckedChange={set.detailerSort} />
                Sort
              </label>
            </div>
          </ParamSection>

          <ParamSection title="Noise" defaultOpen={false}>
            <ParamSlider label="Renoise" value={state.detailerRenoise} onChange={set.detailerRenoise} min={0.5} max={1.5} step={0.01} />
            <ParamSlider label="End" value={state.detailerRenoiseEnd} onChange={set.detailerRenoiseEnd} min={0} max={1} step={0.01} />
          </ParamSection>
        </>
      )}
    </div>
  );
}
