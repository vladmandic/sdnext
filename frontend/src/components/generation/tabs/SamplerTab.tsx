import { useMemo, useCallback } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useSamplerList } from "@/api/hooks/useModels";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";

export function SamplerTab() {
  const state = useGenerationStore(useShallow((s) => ({
    sampler: s.sampler,
    steps: s.steps,
    sigmaMethod: s.sigmaMethod,
    timestepSpacing: s.timestepSpacing,
    betaSchedule: s.betaSchedule,
    predictionMethod: s.predictionMethod,
    timestepsPreset: s.timestepsPreset,
    timestepsOverride: s.timestepsOverride,
    sigmaAdjust: s.sigmaAdjust,
    sigmaAdjustStart: s.sigmaAdjustStart,
    sigmaAdjustEnd: s.sigmaAdjustEnd,
    flowShift: s.flowShift,
    baseShift: s.baseShift,
    maxShift: s.maxShift,
    lowOrder: s.lowOrder,
    thresholding: s.thresholding,
    dynamic: s.dynamic,
    rescale: s.rescale,
    seed: s.seed,
    subseed: s.subseed,
    subseedStrength: s.subseedStrength,
  })));
  const setParam = useGenerationStore((s) => s.setParam);
  const { data: samplers } = useSamplerList();

  const set = useMemo(() => ({
    sampler: (v: string) => setParam("sampler", v),
    steps: (v: number) => setParam("steps", v),
    sigmaMethod: (v: string) => setParam("sigmaMethod", v),
    timestepSpacing: (v: string) => setParam("timestepSpacing", v),
    betaSchedule: (v: string) => setParam("betaSchedule", v),
    predictionMethod: (v: string) => setParam("predictionMethod", v),
    timestepsPreset: (v: string) => setParam("timestepsPreset", v),
    timestepsOverride: (e: React.ChangeEvent<HTMLInputElement>) => setParam("timestepsOverride", e.target.value),
    sigmaAdjust: (v: number) => setParam("sigmaAdjust", v),
    sigmaAdjustStart: (v: number) => setParam("sigmaAdjustStart", v),
    sigmaAdjustEnd: (v: number) => setParam("sigmaAdjustEnd", v),
    flowShift: (v: number) => setParam("flowShift", v),
    baseShift: (v: number) => setParam("baseShift", v),
    maxShift: (v: number) => setParam("maxShift", v),
    lowOrder: (c: boolean | "indeterminate") => setParam("lowOrder", !!c),
    thresholding: (c: boolean | "indeterminate") => setParam("thresholding", !!c),
    dynamic: (c: boolean | "indeterminate") => setParam("dynamic", !!c),
    rescale: (c: boolean | "indeterminate") => setParam("rescale", !!c),
    seed: (v: number) => setParam("seed", v),
    seedRandom: () => setParam("seed", -1),
    subseed: (v: number) => setParam("subseed", v),
    subseedRandom: () => setParam("subseed", -1),
    subseedStrength: (v: number) => setParam("subseedStrength", v),
  }), [setParam]);

  const handleSamplerChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setParam("sampler", e.target.value);
  }, [setParam]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Sampler">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Method</Label>
          <select
            value={state.sampler}
            onChange={handleSamplerChange}
            className="flex-1 h-6 text-[11px] bg-transparent border border-input rounded-md px-1.5 text-foreground outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] transition-[color,box-shadow]"
          >
            {samplers?.map((s) => (
              <option key={s.name} value={s.name}>{s.name}</option>
            ))}
          </select>
        </div>
        <ParamSlider label="Steps" value={state.steps} onChange={set.steps} min={1} max={150} />
      </ParamSection>

      <ParamSection title="Scheduler" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Sigma</Label>
          <Select value={state.sigmaMethod} onValueChange={set.sigmaMethod}>
            <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["default", "karras", "exponential", "polyexponential"].map((v) => (
                <SelectItem key={v} value={v} className="text-xs">{v}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Spacing</Label>
          <Select value={state.timestepSpacing} onValueChange={set.timestepSpacing}>
            <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["default", "linspace", "leading", "trailing"].map((v) => (
                <SelectItem key={v} value={v} className="text-xs">{v}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Beta</Label>
          <Select value={state.betaSchedule} onValueChange={set.betaSchedule}>
            <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["default", "linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"].map((v) => (
                <SelectItem key={v} value={v} className="text-xs">{v}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Prediction</Label>
          <Select value={state.predictionMethod} onValueChange={set.predictionMethod}>
            <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {["default", "epsilon", "sample", "v_prediction", "flow_prediction"].map((v) => (
                <SelectItem key={v} value={v} className="text-xs">{v}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </ParamSection>

      <ParamSection title="Timesteps" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Preset</Label>
          <Select value={state.timestepsPreset} onValueChange={set.timestepsPreset}>
            <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="None" className="text-xs">None</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Override</Label>
          <Input
            value={state.timestepsOverride}
            onChange={set.timestepsOverride}
            placeholder="e.g. 999,850,700,550,400,250,100"
            className="flex-1 h-6 text-[11px] px-2"
          />
        </div>
      </ParamSection>

      <ParamSection title="Sigma" defaultOpen={false}>
        <ParamSlider label="Adjust" value={state.sigmaAdjust} onChange={set.sigmaAdjust} min={0.5} max={1.5} step={0.01} />
        <ParamSlider label="Start" value={state.sigmaAdjustStart} onChange={set.sigmaAdjustStart} min={0} max={1} step={0.01} />
        <ParamSlider label="End" value={state.sigmaAdjustEnd} onChange={set.sigmaAdjustEnd} min={0} max={1} step={0.01} />
      </ParamSection>

      <ParamSection title="Shifts" defaultOpen={false}>
        <ParamSlider label="Flow shift" value={state.flowShift} onChange={set.flowShift} min={0.1} max={10} step={0.1} />
        <ParamSlider label="Base shift" value={state.baseShift} onChange={set.baseShift} min={0} max={1} step={0.01} />
        <ParamSlider label="Max shift" value={state.maxShift} onChange={set.maxShift} min={0} max={4} step={0.01} />
      </ParamSection>

      <ParamSection title="Options" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={state.lowOrder} onCheckedChange={set.lowOrder} />
            Low order
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={state.thresholding} onCheckedChange={set.thresholding} />
            Thresholding
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={state.dynamic} onCheckedChange={set.dynamic} />
            Dynamic
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={state.rescale} onCheckedChange={set.rescale} />
            Rescale
          </label>
        </div>
      </ParamSection>

      <ParamSection title="Seed">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Seed</Label>
          <NumberInput
            value={state.seed}
            onChange={set.seed}
            fallback={-1}
            className="flex-1 h-6 text-[11px] px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Button
            variant="secondary" size="xs"
            onClick={set.seedRandom}
            className="text-muted-foreground"
          >
            Random
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Variation</Label>
          <NumberInput
            value={state.subseed}
            onChange={set.subseed}
            fallback={-1}
            className="flex-1 h-6 text-[11px] px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Button
            variant="secondary" size="xs"
            onClick={set.subseedRandom}
            className="text-muted-foreground"
          >
            Random
          </Button>
        </div>

        <ParamSlider label="Var. str." value={state.subseedStrength} onChange={set.subseedStrength} min={0} max={1} step={0.01} />
      </ParamSection>
    </div>
  );
}
