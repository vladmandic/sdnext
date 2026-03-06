import { useMemo, useCallback } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useSamplerList, useUpscalerGroups } from "@/api/hooks/useModels";
import { HIRES_SIZE_MODES, HIRES_FIT_MODES, HIRES_CONTEXT_MODES } from "@/lib/constants";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamRow, ParamGrid } from "../ParamRow";
import { getParamHelp } from "@/data/parameterHelp";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Textarea } from "@/components/ui/textarea";
import { Combobox } from "@/components/ui/combobox";

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
    hiresResizeContext: s.hiresResizeContext,
    refinerStart: s.refinerStart,
    refinerSteps: s.refinerSteps,
    refinerPrompt: s.refinerPrompt,
    refinerNegative: s.refinerNegative,
  })));
  const setParam = useGenerationStore((s) => s.setParam);
  const upscalerGroups = useUpscalerGroups();
  const { data: samplers } = useSamplerList();

  // Derive size mode from store: fixed dims set = "fixed", otherwise "scale"
  const sizeMode = state.hiresResizeX > 0 || state.hiresResizeY > 0 ? "fixed" : "scale";

  const set = useMemo(() => ({
    hiresEnabled: (checked: boolean) => setParam("hiresEnabled", checked),
    hiresResizeMode: (v: string) => setParam("hiresResizeMode", Number(v)),
    hiresScale: (v: number) => setParam("hiresScale", v),
    hiresResizeX: (v: number) => setParam("hiresResizeX", v),
    hiresResizeY: (v: number) => setParam("hiresResizeY", v),
    hiresResizeContext: (v: string) => setParam("hiresResizeContext", v),
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

  const handleSizeMode = useCallback((v: string) => {
    if (v === "scale") {
      // Zero out fixed dims so backend uses scale multiplier
      setParam("hiresResizeX", 0);
      setParam("hiresResizeY", 0);
      setParam("hiresResizeMode", 0);
    } else {
      // Switch to fixed: seed dims from base resolution × scale
      const s = useGenerationStore.getState();
      const scale = s.hiresScale > 1 ? s.hiresScale : 2;
      setParam("hiresResizeX", Math.round(s.width * scale / 8) * 8);
      setParam("hiresResizeY", Math.round(s.height * scale / 8) * 8);
      if (s.hiresResizeMode === 0) {
        setParam("hiresResizeMode", 2); // default to Crop
      }
    }
  }, [setParam]);

  const showContextDropdown = state.hiresResizeMode === 5;

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Hires Fix">
        {/* Enable toggle */}
        <div className="flex items-center justify-between">
          <Label className="text-2xs text-muted-foreground">Enable</Label>
          <Switch
            checked={state.hiresEnabled}
            onCheckedChange={set.hiresEnabled}
          />
        </div>

        <div className={state.hiresEnabled ? "" : "opacity-40 pointer-events-none"}>
          <div className="flex flex-col gap-2">
            <ParamRow label="Upscaler">
              <Combobox
                value={state.hiresUpscaler}
                onValueChange={set.hiresUpscaler}
                groups={upscalerGroups}
                className="h-6 text-2xs"
                disabled={!state.hiresEnabled}
              />
            </ParamRow>

            <ParamRow label="Size" tooltip={getParamHelp("hires size")}>
              <Combobox
                value={sizeMode}
                onValueChange={handleSizeMode}
                options={HIRES_SIZE_MODES}
                className="h-6 text-2xs"
                disabled={!state.hiresEnabled}
              />
            </ParamRow>

            {sizeMode === "scale" ? (
              <ParamSlider label="Scale" value={state.hiresScale} onChange={set.hiresScale} min={1} max={4} step={0.1} disabled={!state.hiresEnabled} />
            ) : (
              <>
                <ParamRow label="Dims">
                  <div className="flex items-center gap-2">
                    <NumberInput
                      value={state.hiresResizeX}
                      onChange={set.hiresResizeX}
                      placeholder="Width"
                      step={8} min={0} max={8192} fallback={0}
                      disabled={!state.hiresEnabled}
                      className="flex-1 h-6 text-2xs text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    />
                    <span className="text-3xs text-muted-foreground">x</span>
                    <NumberInput
                      value={state.hiresResizeY}
                      onChange={set.hiresResizeY}
                      placeholder="Height"
                      step={8} min={0} max={8192} fallback={0}
                      disabled={!state.hiresEnabled}
                      className="flex-1 h-6 text-2xs text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                    />
                  </div>
                </ParamRow>
                <ParamRow label="Fit" tooltip={getParamHelp("hires fit")}>
                  <Combobox
                    value={String(state.hiresResizeMode)}
                    onValueChange={set.hiresResizeMode}
                    options={HIRES_FIT_MODES}
                    className="h-6 text-2xs"
                    disabled={!state.hiresEnabled}
                  />
                </ParamRow>
                {showContextDropdown && (
                  <ParamRow label="Context">
                    <Combobox
                      value={state.hiresResizeContext}
                      onValueChange={set.hiresResizeContext}
                      options={HIRES_CONTEXT_MODES}
                      className="h-6 text-2xs"
                      disabled={!state.hiresEnabled}
                    />
                  </ParamRow>
                )}
              </>
            )}

            <ParamRow label="Sampler">
              <Combobox
                value={state.hiresSampler || "_same_"}
                onValueChange={set.hiresSampler}
                options={[
                  { value: "_same_", label: "Same as primary" },
                  ...(samplers?.map((s) => ({ value: s.name, label: s.name })) ?? []),
                ]}
                placeholder="Same as primary"
                className="h-6 text-2xs"
                disabled={!state.hiresEnabled}
              />
            </ParamRow>

            <ParamGrid>
              <ParamSlider label="Denoise" value={state.hiresDenoising} onChange={set.hiresDenoising} min={0} max={1} step={0.05} disabled={!state.hiresEnabled} />
              <ParamSlider label="Steps" tooltip={getParamHelp("hires steps")} value={state.hiresSteps} onChange={set.hiresSteps} min={0} max={150} disabled={!state.hiresEnabled} />
            </ParamGrid>

            <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
              <Checkbox checked={state.hiresForce} onCheckedChange={set.hiresForce} disabled={!state.hiresEnabled} />
              Force hires
            </label>
          </div>
        </div>
      </ParamSection>

      <ParamSection title="Refiner" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Start" tooltip={getParamHelp("refiner start")} value={state.refinerStart} onChange={set.refinerStart} min={0} max={1} step={0.01} />
          <ParamSlider label="Steps" tooltip={getParamHelp("refiner steps")} value={state.refinerSteps} onChange={set.refinerSteps} min={0} max={150} />
        </ParamGrid>

        <div className="flex flex-col gap-1">
          <Label className="text-2xs text-muted-foreground">Refiner prompt</Label>
          <Textarea
            value={state.refinerPrompt}
            onChange={set.refinerPrompt}
            placeholder="Refiner prompt (optional)"
            className="min-h-12 text-xs resize-none"
          />
        </div>
        <div className="flex flex-col gap-1">
          <Label className="text-2xs text-muted-foreground">Refiner negative</Label>
          <Textarea
            value={state.refinerNegative}
            onChange={set.refinerNegative}
            placeholder="Refiner negative prompt (optional)"
            className="min-h-9 text-xs resize-none"
          />
        </div>
      </ParamSection>
    </div>
  );
}
