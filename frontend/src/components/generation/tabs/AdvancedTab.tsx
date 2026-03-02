import { useMemo } from "react";
import { useGenerationStore } from "@/stores/generationStore";
import { useShallow } from "zustand/react/shallow";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Switch } from "@/components/ui/switch";
import { ParamLabel } from "../ParamLabel";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Combobox } from "@/components/ui/combobox";

export function AdvancedTab() {
  const state = useGenerationStore(useShallow((s) => ({
    clipSkip: s.clipSkip,
    vaeType: s.vaeType,
    tiling: s.tiling,
    hidiffusion: s.hidiffusion,
    overrideSettings: s.overrideSettings,
  })));
  const setParam = useGenerationStore((s) => s.setParam);

  const set = useMemo(() => ({
    clipSkip: (v: number) => setParam("clipSkip", v),
    vaeType: (v: string) => setParam("vaeType", v),
    tiling: (checked: boolean) => setParam("tiling", checked),
    hidiffusion: (checked: boolean) => setParam("hidiffusion", checked),
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
          <ParamLabel className="text-2xs text-muted-foreground w-16 flex-shrink-0">VAE type</ParamLabel>
          <Combobox
            value={state.vaeType}
            onValueChange={set.vaeType}
            options={["Full", "Tiny", "Remote"]}
            className="h-6 text-2xs flex-1"
          />
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <ParamLabel className="text-2xs text-muted-foreground flex-shrink-0">Tiling</ParamLabel>
            <Switch checked={state.tiling} onCheckedChange={set.tiling} />
          </div>
          <div className="flex items-center gap-2">
            <ParamLabel className="text-2xs text-muted-foreground flex-shrink-0">HiDiffusion</ParamLabel>
            <Switch checked={state.hidiffusion} onCheckedChange={set.hidiffusion} />
          </div>
        </div>
      </ParamSection>

      <ParamSection title="Override Settings" defaultOpen={false}>
        <div className="flex flex-col gap-1">
          <Label className="text-2xs text-muted-foreground">Key-value overrides (one per line, key: value)</Label>
          <Textarea
            value={overridesToText(state.overrideSettings)}
            onChange={(e) => setParam("overrideSettings", parseOverrides(e.target.value))}
            placeholder={"scheduler: Euler a\nsd_model_checkpoint: model.safetensors"}
            className="min-h-15 text-xs resize-y font-mono"
          />
        </div>
      </ParamSection>
    </div>
  );
}
