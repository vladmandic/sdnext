import { useMemo } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useSamplerList } from "@/api/hooks/useModels";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Combobox } from "@/components/ui/combobox";
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

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Sampler">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Method</Label>
          <Combobox
            value={state.sampler}
            onValueChange={set.sampler}
            options={samplers?.map((s) => s.name) ?? []}
            className="flex-1 h-6 text-[11px]"
          />
        </div>
        <ParamSlider label="Steps" value={state.steps} onChange={set.steps} min={1} max={150} />
      </ParamSection>

      <ParamSection title="Scheduler" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Sigma</Label>
          <Combobox
            value={state.sigmaMethod}
            onValueChange={set.sigmaMethod}
            options={["default", "karras", "exponential", "polyexponential"]}
            className="flex-1 h-6 text-[11px]"
          />
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Spacing</Label>
          <Combobox
            value={state.timestepSpacing}
            onValueChange={set.timestepSpacing}
            options={["default", "linspace", "leading", "trailing"]}
            className="flex-1 h-6 text-[11px]"
          />
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Beta</Label>
          <Combobox
            value={state.betaSchedule}
            onValueChange={set.betaSchedule}
            options={["default", "linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]}
            className="flex-1 h-6 text-[11px]"
          />
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Prediction</Label>
          <Combobox
            value={state.predictionMethod}
            onValueChange={set.predictionMethod}
            options={["default", "epsilon", "sample", "v_prediction", "flow_prediction"]}
            className="flex-1 h-6 text-[11px]"
          />
        </div>
      </ParamSection>

      <ParamSection title="Timesteps" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Preset</Label>
          <Combobox
            value={state.timestepsPreset}
            onValueChange={set.timestepsPreset}
            options={["None"]}
            className="flex-1 h-6 text-[11px]"
          />
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
