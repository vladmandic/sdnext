import { useMemo } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Combobox } from "@/components/ui/combobox";

export function AdvancedTab() {
  const state = useGenerationStore(useShallow((s) => ({
    clipSkip: s.clipSkip,
    vaeType: s.vaeType,
    tiling: s.tiling,
    hidiffusion: s.hidiffusion,
    hdrMode: s.hdrMode,
    hdrBrightness: s.hdrBrightness,
    hdrSharpen: s.hdrSharpen,
    hdrColor: s.hdrColor,
    hdrClamp: s.hdrClamp,
    hdrBoundary: s.hdrBoundary,
    hdrThreshold: s.hdrThreshold,
    hdrMaximize: s.hdrMaximize,
    hdrMaxCenter: s.hdrMaxCenter,
    hdrMaxBoundary: s.hdrMaxBoundary,
    hdrTintRatio: s.hdrTintRatio,
    overrideSettings: s.overrideSettings,
  })));
  const setParam = useGenerationStore((s) => s.setParam);

  const set = useMemo(() => ({
    clipSkip: (v: number) => setParam("clipSkip", v),
    vaeType: (v: string) => setParam("vaeType", v),
    tiling: (checked: boolean) => setParam("tiling", checked),
    hidiffusion: (checked: boolean) => setParam("hidiffusion", checked),
    hdrMode: (v: string) => setParam("hdrMode", Number(v)),
    hdrBrightness: (v: number) => setParam("hdrBrightness", v),
    hdrSharpen: (v: number) => setParam("hdrSharpen", v),
    hdrColor: (v: number) => setParam("hdrColor", v),
    hdrClamp: (checked: boolean) => setParam("hdrClamp", checked),
    hdrBoundary: (v: number) => setParam("hdrBoundary", v),
    hdrThreshold: (v: number) => setParam("hdrThreshold", v),
    hdrMaximize: (checked: boolean) => setParam("hdrMaximize", checked),
    hdrMaxCenter: (v: number) => setParam("hdrMaxCenter", v),
    hdrMaxBoundary: (v: number) => setParam("hdrMaxBoundary", v),
    hdrTintRatio: (v: number) => setParam("hdrTintRatio", v),
  }), [setParam]);

  function parseOverrides(text: string): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.includes(":")) continue;
      const idx = trimmed.indexOf(":");
      const key = trimmed.slice(0, idx).trim();
      const rawVal = trimmed.slice(idx + 1).trim();
      if (!key) continue;
      if (rawVal === "true") result[key] = true;
      else if (rawVal === "false") result[key] = false;
      else if (rawVal !== "" && !isNaN(Number(rawVal))) result[key] = Number(rawVal);
      else result[key] = rawVal;
    }
    return result;
  }

  function overridesToText(obj: Record<string, unknown>): string {
    return Object.entries(obj).map(([k, v]) => `${k}: ${v}`).join("\n");
  }

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Advanced">
        <ParamSlider label="CLIP skip" value={state.clipSkip} onChange={set.clipSkip} min={0} max={12} step={0.1} />

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">VAE type</Label>
          <Combobox
            value={state.vaeType}
            onValueChange={set.vaeType}
            options={["Full", "Tiny", "Remote"]}
            className="h-7 text-xs flex-1"
          />
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground flex-shrink-0">Tiling</Label>
            <Switch checked={state.tiling} onCheckedChange={set.tiling} />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground flex-shrink-0">HiDiffusion</Label>
            <Switch checked={state.hidiffusion} onCheckedChange={set.hidiffusion} />
          </div>
        </div>
      </ParamSection>

      <ParamSection title="Corrections">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Mode</Label>
          <Combobox
            value={String(state.hdrMode)}
            onValueChange={set.hdrMode}
            options={[{ value: "0", label: "Relative values" }, { value: "1", label: "Absolute values" }]}
            className="h-7 text-xs flex-1"
          />
        </div>

        <ParamSlider label="Brightness" value={state.hdrBrightness} onChange={set.hdrBrightness} min={-1} max={1} step={0.1} />
        <ParamSlider label="Sharpen" value={state.hdrSharpen} onChange={set.hdrSharpen} min={-1} max={1} step={0.1} />
        <ParamSlider label="Color" value={state.hdrColor} onChange={set.hdrColor} min={0} max={4} step={0.1} />

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">HDR clamp</Label>
          <Switch checked={state.hdrClamp} onCheckedChange={set.hdrClamp} />
        </div>
        {state.hdrClamp && (
          <>
            <ParamSlider label="Range" value={state.hdrBoundary} onChange={set.hdrBoundary} min={0} max={10} step={0.1} />
            <ParamSlider label="Threshold" value={state.hdrThreshold} onChange={set.hdrThreshold} min={0} max={1} step={0.01} />
          </>
        )}

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Maximize</Label>
          <Switch checked={state.hdrMaximize} onCheckedChange={set.hdrMaximize} />
        </div>
        {state.hdrMaximize && (
          <>
            <ParamSlider label="Center" value={state.hdrMaxCenter} onChange={set.hdrMaxCenter} min={0} max={2} step={0.1} />
            <ParamSlider label="Max range" value={state.hdrMaxBoundary} onChange={set.hdrMaxBoundary} min={0.5} max={2} step={0.1} />
          </>
        )}

        <ParamSlider label="Color grade" value={state.hdrTintRatio} onChange={set.hdrTintRatio} min={-1} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Override Settings" defaultOpen={false}>
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Key-value overrides (one per line, key: value)</Label>
          <Textarea
            value={overridesToText(state.overrideSettings)}
            onChange={(e) => setParam("overrideSettings", parseOverrides(e.target.value))}
            placeholder={"scheduler: Euler a\nsd_model_checkpoint: model.safetensors"}
            className="min-h-[60px] text-xs resize-y font-mono"
          />
        </div>
      </ParamSection>
    </div>
  );
}
