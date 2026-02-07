import { useGenerationStore } from "@/stores/generationStore";
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
  const store = useGenerationStore();
  const { data: upscalers } = useUpscalerList();
  const { data: samplers } = useSamplerList();

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Hires Fix">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Enable</Label>
          <Switch
            checked={store.hiresEnabled}
            onCheckedChange={(checked) => store.setParam("hiresEnabled", checked)}
          />
        </div>
        {store.hiresEnabled && (
          <>
            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Mode</Label>
              <Select value={String(store.hiresResizeMode)} onValueChange={(v) => store.setParam("hiresResizeMode", Number(v))}>
                <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0" className="text-xs">Scale</SelectItem>
                  <SelectItem value="1" className="text-xs">Fixed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {store.hiresResizeMode === 0 ? (
              <ParamSlider label="Scale" value={store.hiresScale} onChange={(v) => store.setParam("hiresScale", v)} min={1} max={4} step={0.1} />
            ) : (
              <div className="flex items-center gap-2">
                <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Size</Label>
                <NumberInput
                  value={store.hiresResizeX}
                  onChange={(v) => store.setParam("hiresResizeX", v)}
                  placeholder="Width"
                  step={8} min={0} max={8192} fallback={0}
                  className="flex-1 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
                <span className="text-[10px] text-muted-foreground">x</span>
                <NumberInput
                  value={store.hiresResizeY}
                  onChange={(v) => store.setParam("hiresResizeY", v)}
                  placeholder="Height"
                  step={8} min={0} max={8192} fallback={0}
                  className="flex-1 h-6 text-[11px] text-center px-1 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
                />
              </div>
            )}

            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Upscaler</Label>
              <Select value={store.hiresUpscaler} onValueChange={(v) => store.setParam("hiresUpscaler", v)}>
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
              <Select value={store.hiresSampler || "_same_"} onValueChange={(v) => store.setParam("hiresSampler", v === "_same_" ? "" : v)}>
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

            <ParamSlider label="Denoise" value={store.hiresDenoising} onChange={(v) => store.setParam("hiresDenoising", v)} min={0} max={1} step={0.05} />
            <ParamSlider label="Steps" value={store.hiresSteps} onChange={(v) => store.setParam("hiresSteps", v)} min={0} max={150} />

            <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
              <Checkbox checked={store.hiresForce} onCheckedChange={(c) => store.setParam("hiresForce", !!c)} />
              Force hires
            </label>
          </>
        )}
      </ParamSection>

      <ParamSection title="Refiner" defaultOpen={false}>
        <ParamSlider label="Start" value={store.refinerStart} onChange={(v) => store.setParam("refinerStart", v)} min={0} max={1} step={0.01} />
        <ParamSlider label="Steps" value={store.refinerSteps} onChange={(v) => store.setParam("refinerSteps", v)} min={0} max={150} />

        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Refiner prompt</Label>
          <Textarea
            value={store.refinerPrompt}
            onChange={(e) => store.setParam("refinerPrompt", e.target.value)}
            placeholder="Refiner prompt (optional)"
            className="min-h-[48px] text-xs resize-none"
          />
        </div>
        <div className="flex flex-col gap-1">
          <Label className="text-[11px] text-muted-foreground">Refiner negative</Label>
          <Textarea
            value={store.refinerNegative}
            onChange={(e) => store.setParam("refinerNegative", e.target.value)}
            placeholder="Refiner negative prompt (optional)"
            className="min-h-[36px] text-xs resize-none"
          />
        </div>
      </ParamSection>
    </div>
  );
}
