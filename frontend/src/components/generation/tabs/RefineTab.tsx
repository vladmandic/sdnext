import { useMemo } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useSamplerList, useUpscalerList } from "@/api/hooks/useModels";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function RefineTab() {
  const state = useGenerationStore(useShallow((s) => ({
    hiresEnabled: s.hiresEnabled,
    hiresUpscaler: s.hiresUpscaler,
    hiresScale: s.hiresScale,
    hiresSteps: s.hiresSteps,
    hiresDenoising: s.hiresDenoising,
    hiresResizeMode: s.hiresResizeMode,
    hiresSampler: s.hiresSampler,
    hiresForce: s.hiresForce,
    hiresResizeX: s.hiresResizeX,
    hiresResizeY: s.hiresResizeY,
    refinerStart: s.refinerStart,
    refinerSteps: s.refinerSteps,
    refinerPrompt: s.refinerPrompt,
    refinerNegative: s.refinerNegative,
  })));
  const setParam = useGenerationStore((s) => s.setParam);
  const { data: upscalers } = useUpscalerList();
  const { data: samplers } = useSamplerList();

  const set = useMemo(() => ({
    hiresEnabled: (checked: boolean) => setParam("hiresEnabled", checked),
    hiresResizeMode: (v: string) => setParam("hiresResizeMode", Number(v)),
    hiresScale: (v: number) => setParam("hiresScale", v),
    hiresResizeX: (v: number) => setParam("hiresResizeX", v),
    hiresResizeY: (v: number) => setParam("hiresResizeY", v),
    hiresUpscaler: (v: string) => setParam("hiresUpscaler", v),
    hiresSampler: (v: string) => setParam("hiresSampler", v === "_same_" ? "" : v),
    hiresDenoising: (v: number) => setParam("hiresDenoising", v),
    hiresSteps: (v: number) => setParam("hiresSteps", v),
    hiresForce: (c: boolean | "indeterminate") => setParam("hiresForce", !!c),
    refinerStart: (v: number) => setParam("refinerStart", v),
    refinerSteps: (v: number) => setParam("refinerSteps", v),
    refinerPrompt: (e: React.ChangeEvent<HTMLTextAreaElement>) => setParam("refinerPrompt", e.target.value),
    refinerNegative: (e: React.ChangeEvent<HTMLTextAreaElement>) => setParam("refinerNegative", e.target.value),
  }), [setParam]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Hires Fix">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Enable</Label>
          <Switch
            checked={state.hiresEnabled}
            onCheckedChange={set.hiresEnabled}
          />
        </div>
        {state.hiresEnabled && (
          <>
            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Mode</Label>
              <Select value={String(state.hiresResizeMode)} onValueChange={set.hiresResizeMode}>
                <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0" className="text-xs">Scale</SelectItem>
                  <SelectItem value="1" className="text-xs">Fixed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {state.hiresResizeMode === 0 ? (
              <ParamSlider label="Scale" value={state.hiresScale} onChange={set.hiresScale} min={1} max={4} step={0.1} />
            ) : (
              <div className="flex items-center gap-2">
                <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Size</Label>
                <NumberInput
                  value={state.hiresResizeX}
                  onChange={set.hiresResizeX}
                  placeholder="Width"
                  step={8} min={0} max={8192} fallback={0}
                  className="flex-1 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                <span className="text-[10px] text-muted-foreground">x</span>
                <NumberInput
                  value={state.hiresResizeY}
                  onChange={set.hiresResizeY}
                  placeholder="Height"
                  step={8} min={0} max={8192} fallback={0}
                  className="flex-1 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
              </div>
            )}

            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Upscaler</Label>
              <Select value={state.hiresUpscaler} onValueChange={set.hiresUpscaler}>
                <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {upscalers?.map((u) => (
                    <SelectItem key={u.name} value={u.name} className="text-xs">{u.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Sampler</Label>
              <Select value={state.hiresSampler || "_same_"} onValueChange={set.hiresSampler}>
                <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
                  <SelectValue placeholder="Same as primary" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="_same_" className="text-xs">Same as primary</SelectItem>
                  {samplers?.map((s) => (
                    <SelectItem key={s.name} value={s.name} className="text-xs">{s.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <ParamSlider label="Denoise" value={state.hiresDenoising} onChange={set.hiresDenoising} min={0} max={1} step={0.05} />
            <ParamSlider label="Steps" value={state.hiresSteps} onChange={set.hiresSteps} min={0} max={150} />

            <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
              <Checkbox checked={state.hiresForce} onCheckedChange={set.hiresForce} />
              Force hires
            </label>
          </>
        )}
      </ParamSection>

      <ParamSection title="Refiner" defaultOpen={false}>
        <ParamSlider label="Start" value={state.refinerStart} onChange={set.refinerStart} min={0} max={1} step={0.01} />
        <ParamSlider label="Steps" value={state.refinerSteps} onChange={set.refinerSteps} min={0} max={150} />

        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Refiner prompt</Label>
          <Textarea
            value={state.refinerPrompt}
            onChange={set.refinerPrompt}
            placeholder="Refiner prompt (optional)"
            className="min-h-[48px] text-xs resize-none"
          />
        </div>
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Refiner negative</Label>
          <Textarea
            value={state.refinerNegative}
            onChange={set.refinerNegative}
            placeholder="Refiner negative prompt (optional)"
            className="min-h-[36px] text-xs resize-none"
          />
        </div>
      </ParamSection>
    </div>
  );
}
