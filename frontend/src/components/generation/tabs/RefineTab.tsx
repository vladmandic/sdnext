import { useMemo, useState } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { useSamplerList, useUpscalerList } from "@/api/hooks/useModels";
import { HIRES_RESIZE_MODES, HIRES_CONTEXT_MODES } from "@/lib/constants";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { ParamRow, ParamGrid } from "../ParamRow";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import { Textarea } from "@/components/ui/textarea";
import { Combobox } from "@/components/ui/combobox";
import { cn } from "@/lib/utils";

type SizeMode = "scale" | "fixed";

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
  const { data: upscalers } = useUpscalerList();
  const { data: samplers } = useSamplerList();

  // Local state for size mode toggle (scale vs fixed)
  const [sizeMode, setSizeMode] = useState<SizeMode>(() => {
    // Initialize based on whether fixed dimensions are set
    return state.hiresResizeX > 0 || state.hiresResizeY > 0 ? "fixed" : "scale";
  });

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

  // Context dropdown only for mode 5 (Context aware)
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

        {state.hiresEnabled && (
          <>
            <ParamGrid>
              <ParamRow label="Mode">
                <Combobox
                  value={String(state.hiresResizeMode)}
                  onValueChange={set.hiresResizeMode}
                  options={HIRES_RESIZE_MODES}
                  className="h-6 text-2xs"
                />
              </ParamRow>
              <ParamRow label="Upscaler">
                <Combobox
                  value={state.hiresUpscaler}
                  onValueChange={set.hiresUpscaler}
                  options={upscalers?.map((u) => u.name) ?? []}
                  className="h-6 text-2xs"
                />
              </ParamRow>
            </ParamGrid>

            {showContextDropdown && (
              <ParamRow label="Context">
                <Combobox
                  value={state.hiresResizeContext}
                  onValueChange={set.hiresResizeContext}
                  options={HIRES_CONTEXT_MODES}
                  className="h-6 text-2xs"
                />
              </ParamRow>
            )}

            {/* Size controls */}
            {state.hiresResizeMode === 0 ? (
              <ParamSlider label="Scale" value={state.hiresScale} onChange={set.hiresScale} min={1} max={4} step={0.1} />
            ) : (
              <div className="flex flex-col gap-2">
                <ParamRow label="Size">
                  <div className="flex h-6 rounded-md border border-border overflow-hidden w-fit">
                    <button
                      type="button"
                      onClick={() => setSizeMode("scale")}
                      className={cn(
                        "px-3 text-3xs font-medium transition-colors",
                        sizeMode === "scale"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted/50 text-muted-foreground hover:bg-muted"
                      )}
                    >
                      Scale
                    </button>
                    <button
                      type="button"
                      onClick={() => setSizeMode("fixed")}
                      className={cn(
                        "px-3 text-3xs font-medium transition-colors border-l border-border",
                        sizeMode === "fixed"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted/50 text-muted-foreground hover:bg-muted"
                      )}
                    >
                      Fixed
                    </button>
                  </div>
                </ParamRow>

                {sizeMode === "scale" ? (
                  <ParamSlider label="Scale" value={state.hiresScale} onChange={set.hiresScale} min={1} max={4} step={0.1} />
                ) : (
                  <ParamRow label="Dims">
                    <div className="flex items-center gap-2">
                      <NumberInput
                        value={state.hiresResizeX}
                        onChange={set.hiresResizeX}
                        placeholder="Width"
                        step={8} min={0} max={8192} fallback={0}
                        className="flex-1 h-6 text-2xs text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      />
                      <span className="text-3xs text-muted-foreground">x</span>
                      <NumberInput
                        value={state.hiresResizeY}
                        onChange={set.hiresResizeY}
                        placeholder="Height"
                        step={8} min={0} max={8192} fallback={0}
                        className="flex-1 h-6 text-2xs text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                      />
                    </div>
                  </ParamRow>
                )}
              </div>
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
              />
            </ParamRow>

            <ParamGrid>
              <ParamSlider label="Denoise" value={state.hiresDenoising} onChange={set.hiresDenoising} min={0} max={1} step={0.05} />
              <ParamSlider label="Steps" value={state.hiresSteps} onChange={set.hiresSteps} min={0} max={150} />
            </ParamGrid>

            <label className="flex items-center gap-1.5 text-2xs text-muted-foreground cursor-pointer">
              <Checkbox checked={state.hiresForce} onCheckedChange={set.hiresForce} />
              Force hires
            </label>
          </>
        )}
      </ParamSection>

      <ParamSection title="Refiner" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Start" value={state.refinerStart} onChange={set.refinerStart} min={0} max={1} step={0.01} />
          <ParamSlider label="Steps" value={state.refinerSteps} onChange={set.refinerSteps} min={0} max={150} />
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
