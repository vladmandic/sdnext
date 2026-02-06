import { useGenerationStore } from "@/stores/generationStore";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export function AdvancedTab() {
  const store = useGenerationStore();

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
        <ParamSlider label="CLIP skip" value={store.clipSkip} onChange={(v) => store.setParam("clipSkip", v)} min={0} max={12} step={0.1} />

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">VAE type</Label>
          <Select value={store.vaeType} onValueChange={(v) => store.setParam("vaeType", v)}>
            <SelectTrigger className="h-7 text-xs flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Full">Full</SelectItem>
              <SelectItem value="Tiny">Tiny</SelectItem>
              <SelectItem value="Remote">Remote</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground flex-shrink-0">Tiling</Label>
            <Switch checked={store.tiling} onCheckedChange={(checked) => store.setParam("tiling", checked)} />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground flex-shrink-0">HiDiffusion</Label>
            <Switch checked={store.hidiffusion} onCheckedChange={(checked) => store.setParam("hidiffusion", checked)} />
          </div>
        </div>
      </ParamSection>

      <ParamSection title="Corrections">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Mode</Label>
          <Select value={String(store.hdrMode)} onValueChange={(v) => store.setParam("hdrMode", Number(v))}>
            <SelectTrigger className="h-7 text-xs flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="0">Relative values</SelectItem>
              <SelectItem value="1">Absolute values</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <ParamSlider label="Brightness" value={store.hdrBrightness} onChange={(v) => store.setParam("hdrBrightness", v)} min={-1} max={1} step={0.1} />
        <ParamSlider label="Sharpen" value={store.hdrSharpen} onChange={(v) => store.setParam("hdrSharpen", v)} min={-1} max={1} step={0.1} />
        <ParamSlider label="Color" value={store.hdrColor} onChange={(v) => store.setParam("hdrColor", v)} min={0} max={4} step={0.1} />

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">HDR clamp</Label>
          <Switch checked={store.hdrClamp} onCheckedChange={(checked) => store.setParam("hdrClamp", checked)} />
        </div>
        {store.hdrClamp && (
          <>
            <ParamSlider label="Range" value={store.hdrBoundary} onChange={(v) => store.setParam("hdrBoundary", v)} min={0} max={10} step={0.1} />
            <ParamSlider label="Threshold" value={store.hdrThreshold} onChange={(v) => store.setParam("hdrThreshold", v)} min={0} max={1} step={0.01} />
          </>
        )}

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Maximize</Label>
          <Switch checked={store.hdrMaximize} onCheckedChange={(checked) => store.setParam("hdrMaximize", checked)} />
        </div>
        {store.hdrMaximize && (
          <>
            <ParamSlider label="Center" value={store.hdrMaxCenter} onChange={(v) => store.setParam("hdrMaxCenter", v)} min={0} max={2} step={0.1} />
            <ParamSlider label="Max range" value={store.hdrMaxBoundary} onChange={(v) => store.setParam("hdrMaxBoundary", v)} min={0.5} max={2} step={0.1} />
          </>
        )}

        <ParamSlider label="Color grade" value={store.hdrTintRatio} onChange={(v) => store.setParam("hdrTintRatio", v)} min={-1} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Override Settings" defaultOpen={false}>
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Key-value overrides (one per line, key: value)</Label>
          <Textarea
            value={overridesToText(store.overrideSettings)}
            onChange={(e) => store.setParam("overrideSettings", parseOverrides(e.target.value))}
            placeholder={"scheduler: Euler a\nsd_model_checkpoint: model.safetensors"}
            className="min-h-[60px] text-xs resize-y font-mono"
          />
        </div>
      </ParamSection>
    </div>
  );
}
