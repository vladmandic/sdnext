import { useMemo } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useSamplerList, useCurrentCheckpoint } from "@/api/hooks/useModels";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamRow, ParamGrid } from "../ParamRow";
import { Combobox, type ComboboxGroup } from "@/components/ui/combobox";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ParamLabel } from "../ParamLabel";
import type { GenerationInfo } from "@/api/types/generation";

const SAMPLER_GROUP_ORDER = ["Standard", "FlowMatch", "Res4Lyf"] as const;

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
  const lastResult = useGenerationStore((s) => s.results[0]);
  const { data: checkpoint } = useCurrentCheckpoint();
  const { data: samplers } = useSamplerList(checkpoint?.type);

  const samplerGroups = useMemo<ComboboxGroup[]>(() => {
    if (!samplers) return [];
    const buckets: Record<string, string[]> = {};
    for (const s of samplers) {
      (buckets[s.group] ??= []).push(s.name);
    }
    for (const names of Object.values(buckets)) {
      names.sort((a, b) => a.localeCompare(b));
    }
    return SAMPLER_GROUP_ORDER
      .filter((g) => buckets[g]?.length)
      .map((g) => ({ heading: g, options: buckets[g] }));
  }, [samplers]);

  const lastInfo = useMemo<GenerationInfo | null>(() => {
    if (!lastResult?.info) return null;
    try { return JSON.parse(lastResult.info) as GenerationInfo; }
    catch { return null; }
  }, [lastResult]);

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
    seedReuse: () => { if (lastInfo?.seed != null) setParam("seed", lastInfo.seed); },
    subseed: (v: number) => setParam("subseed", v),
    subseedRandom: () => setParam("subseed", -1),
    subseedReuse: () => { if (lastInfo?.subseed != null) setParam("subseed", lastInfo.subseed); },
    subseedStrength: (v: number) => setParam("subseedStrength", v),
  }), [setParam, lastInfo]);

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Sampler">
        <ParamRow label="Method">
          <Combobox
            value={state.sampler}
            onValueChange={set.sampler}
            groups={samplerGroups}
            className="h-6 text-2xs"
          />
        </ParamRow>
        <ParamSlider label="Steps" value={state.steps} onChange={set.steps} min={1} max={150} />
      </ParamSection>

      <ParamSection title="Scheduler" defaultOpen={false}>
        <ParamGrid>
          <ParamRow label="Sigma">
            <Combobox
              value={state.sigmaMethod}
              onValueChange={set.sigmaMethod}
              options={["default", "karras", "exponential", "polyexponential"]}
              className="h-6 text-2xs"
            />
          </ParamRow>
          <ParamRow label="Spacing">
            <Combobox
              value={state.timestepSpacing}
              onValueChange={set.timestepSpacing}
              options={["default", "linspace", "leading", "trailing"]}
              className="h-6 text-2xs"
            />
          </ParamRow>
          <ParamRow label="Beta">
            <Combobox
              value={state.betaSchedule}
              onValueChange={set.betaSchedule}
              options={["default", "linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"]}
              className="h-6 text-2xs"
            />
          </ParamRow>
          <ParamRow label="Prediction">
            <Combobox
              value={state.predictionMethod}
              onValueChange={set.predictionMethod}
              options={["default", "epsilon", "sample", "v_prediction", "flow_prediction"]}
              className="h-6 text-2xs"
            />
          </ParamRow>
        </ParamGrid>
      </ParamSection>

      <ParamSection title="Timesteps" defaultOpen={false}>
        <ParamGrid>
          <ParamRow label="Preset">
            <Combobox
              value={state.timestepsPreset}
              onValueChange={set.timestepsPreset}
              options={["None"]}
              className="h-6 text-2xs"
            />
          </ParamRow>
          <ParamRow label="Override">
            <Input
              value={state.timestepsOverride}
              onChange={set.timestepsOverride}
              placeholder="e.g. 999,850,700,..."
              className="h-6 text-2xs px-2"
            />
          </ParamRow>
        </ParamGrid>
      </ParamSection>

      <ParamSection title="Sigma" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Start" value={state.sigmaAdjustStart} onChange={set.sigmaAdjustStart} min={0} max={1} step={0.01} />
          <ParamSlider label="End" value={state.sigmaAdjustEnd} onChange={set.sigmaAdjustEnd} min={0} max={1} step={0.01} />
        </ParamGrid>
        <ParamSlider label="Adjust" value={state.sigmaAdjust} onChange={set.sigmaAdjust} min={0.5} max={1.5} step={0.01} />
      </ParamSection>

      <ParamSection title="Shifts" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Base shift" value={state.baseShift} onChange={set.baseShift} min={0} max={1} step={0.01} />
          <ParamSlider label="Max shift" value={state.maxShift} onChange={set.maxShift} min={0} max={4} step={0.01} />
        </ParamGrid>
        <ParamSlider label="Flow shift" value={state.flowShift} onChange={set.flowShift} min={0.1} max={10} step={0.1} />
      </ParamSection>

      <ParamSection title="Options" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
            <Checkbox checked={state.lowOrder} onCheckedChange={set.lowOrder} />
            <ParamLabel className="text-2xs text-muted-foreground">Low order</ParamLabel>
          </label>
          <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
            <Checkbox checked={state.thresholding} onCheckedChange={set.thresholding} />
            <ParamLabel className="text-2xs text-muted-foreground">Thresholding</ParamLabel>
          </label>
          <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
            <Checkbox checked={state.dynamic} onCheckedChange={set.dynamic} />
            <ParamLabel className="text-2xs text-muted-foreground">Dynamic</ParamLabel>
          </label>
          <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
            <Checkbox checked={state.rescale} onCheckedChange={set.rescale} />
            <ParamLabel className="text-2xs text-muted-foreground">Rescale</ParamLabel>
          </label>
        </div>
      </ParamSection>

      <ParamSection title="Seed">
        <ParamRow label="Seed">
          <div className="flex items-center gap-1">
            <NumberInput
              value={state.seed}
              onChange={set.seed}
              fallback={-1}
              className="flex-1 h-6 text-2xs px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            />
            <Button variant="secondary" size="xs" onClick={set.seedRandom} className="text-muted-foreground">Random</Button>
            <Button
              variant="secondary" size="xs"
              onClick={set.seedReuse}
              disabled={lastInfo?.seed == null}
              className="text-muted-foreground"
              title={lastInfo?.seed != null ? `Reuse seed ${lastInfo.seed}` : "No previous generation"}
            >
              Reuse
            </Button>
          </div>
        </ParamRow>

        <ParamRow label="Variation">
          <div className="flex items-center gap-1">
            <NumberInput
              value={state.subseed}
              onChange={set.subseed}
              fallback={-1}
              className="flex-1 h-6 text-2xs px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
            />
            <Button variant="secondary" size="xs" onClick={set.subseedRandom} className="text-muted-foreground">Random</Button>
            <Button
              variant="secondary" size="xs"
              onClick={set.subseedReuse}
              disabled={lastInfo?.subseed == null}
              className="text-muted-foreground"
              title={lastInfo?.subseed != null ? `Reuse subseed ${lastInfo.subseed}` : "No previous generation"}
            >
              Reuse
            </Button>
          </div>
        </ParamRow>

        <ParamSlider label="Var. str." value={state.subseedStrength} onChange={set.subseedStrength} min={0} max={1} step={0.01} />
      </ParamSection>
    </div>
  );
}
