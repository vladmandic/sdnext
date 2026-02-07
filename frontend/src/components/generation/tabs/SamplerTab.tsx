import { useGenerationStore } from "@/stores/generationStore";
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
  const store = useGenerationStore();
  const { data: samplers } = useSamplerList();

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Sampler">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Method</Label>
          <Select value={store.sampler} onValueChange={(v) => store.setParam("sampler", v)}>
            <SelectTrigger size="sm" className="flex-1 h-6 text-[11px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {samplers?.map((s) => (
                <SelectItem key={s.name} value={s.name} className="text-xs">{s.name}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <ParamSlider label="Steps" value={store.steps} onChange={(v) => store.setParam("steps", v)} min={1} max={150} />
      </ParamSection>

      <ParamSection title="Scheduler" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Sigma</Label>
          <Select value={store.sigmaMethod} onValueChange={(v) => store.setParam("sigmaMethod", v)}>
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
          <Select value={store.timestepSpacing} onValueChange={(v) => store.setParam("timestepSpacing", v)}>
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
          <Select value={store.betaSchedule} onValueChange={(v) => store.setParam("betaSchedule", v)}>
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
          <Select value={store.predictionMethod} onValueChange={(v) => store.setParam("predictionMethod", v)}>
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
          <Select value={store.timestepsPreset} onValueChange={(v) => store.setParam("timestepsPreset", v)}>
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
            value={store.timestepsOverride}
            onChange={(e) => store.setParam("timestepsOverride", e.target.value)}
            placeholder="e.g. 999,850,700,550,400,250,100"
            className="flex-1 h-6 text-[11px] px-2"
          />
        </div>
      </ParamSection>

      <ParamSection title="Sigma" defaultOpen={false}>
        <ParamSlider label="Adjust" value={store.sigmaAdjust} onChange={(v) => store.setParam("sigmaAdjust", v)} min={0.5} max={1.5} step={0.01} />
        <ParamSlider label="Start" value={store.sigmaAdjustStart} onChange={(v) => store.setParam("sigmaAdjustStart", v)} min={0} max={1} step={0.01} />
        <ParamSlider label="End" value={store.sigmaAdjustEnd} onChange={(v) => store.setParam("sigmaAdjustEnd", v)} min={0} max={1} step={0.01} />
      </ParamSection>

      <ParamSection title="Shifts" defaultOpen={false}>
        <ParamSlider label="Flow shift" value={store.flowShift} onChange={(v) => store.setParam("flowShift", v)} min={0.1} max={10} step={0.1} />
        <ParamSlider label="Base shift" value={store.baseShift} onChange={(v) => store.setParam("baseShift", v)} min={0} max={1} step={0.01} />
        <ParamSlider label="Max shift" value={store.maxShift} onChange={(v) => store.setParam("maxShift", v)} min={0} max={4} step={0.01} />
      </ParamSection>

      <ParamSection title="Options" defaultOpen={false}>
        <div className="grid grid-cols-2 gap-2">
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={store.lowOrder} onCheckedChange={(c) => store.setParam("lowOrder", !!c)} />
            Low order
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={store.thresholding} onCheckedChange={(c) => store.setParam("thresholding", !!c)} />
            Thresholding
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={store.dynamic} onCheckedChange={(c) => store.setParam("dynamic", !!c)} />
            Dynamic
          </label>
          <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
            <Checkbox checked={store.rescale} onCheckedChange={(c) => store.setParam("rescale", !!c)} />
            Rescale
          </label>
        </div>
      </ParamSection>

      <ParamSection title="Seed">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Seed</Label>
          <NumberInput
            value={store.seed}
            onChange={(v) => store.setParam("seed", v)}
            fallback={-1}
            className="flex-1 h-6 text-[11px] px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Button
            variant="secondary" size="xs"
            onClick={() => store.setParam("seed", -1)}
            className="text-muted-foreground"
          >
            Random
          </Button>
        </div>

        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Variation</Label>
          <NumberInput
            value={store.subseed}
            onChange={(v) => store.setParam("subseed", v)}
            fallback={-1}
            className="flex-1 h-6 text-[11px] px-2 [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <Button
            variant="secondary" size="xs"
            onClick={() => store.setParam("subseed", -1)}
            className="text-muted-foreground"
          >
            Random
          </Button>
        </div>

        <ParamSlider label="Var. str." value={store.subseedStrength} onChange={(v) => store.setParam("subseedStrength", v)} min={0} max={1} step={0.01} />
      </ParamSection>
    </div>
  );
}
