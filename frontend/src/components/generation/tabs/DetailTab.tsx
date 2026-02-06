import { useGenerationStore } from "@/stores/generationStore";
import { useDetailerModels } from "@/api/hooks/useDetailer";
import { ParamSlider } from "../ParamSlider";
import { ParamSection } from "../ParamSection";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { X } from "lucide-react";

export function DetailTab() {
  const store = useGenerationStore();
  const { data: models } = useDetailerModels();

  function addModel(name: string) {
    if (!store.detailerModels.includes(name)) {
      store.setParam("detailerModels", [...store.detailerModels, name]);
    }
  }

  function removeModel(name: string) {
    store.setParam("detailerModels", store.detailerModels.filter((m) => m !== name));
  }

  return (
    <div className="flex flex-col gap-3 text-sm">
      <ParamSection title="Detailer">
        <div className="flex items-center gap-2">
          <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Enabled</Label>
          <Switch checked={store.detailerEnabled} onCheckedChange={(checked) => store.setParam("detailerEnabled", checked)} />
        </div>

        {store.detailerEnabled && (
          <>
            <div className="flex flex-col gap-1">
              <Label className="text-[11px] text-muted-foreground">Models</Label>
              <div className="flex flex-wrap gap-1 mb-1">
                {store.detailerModels.map((m) => (
                  <span key={m} className="inline-flex items-center gap-0.5 px-1.5 py-0.5 text-[10px] bg-muted rounded">
                    {m}
                    <button onClick={() => removeModel(m)} className="text-muted-foreground hover:text-foreground">
                      <X size={10} />
                    </button>
                  </span>
                ))}
              </div>
              <Select value="_placeholder_" onValueChange={addModel}>
                <SelectTrigger className="h-7 text-xs flex-1">
                  <SelectValue placeholder="Add model..." />
                </SelectTrigger>
                <SelectContent>
                  {models?.filter((m) => !store.detailerModels.includes(m.name)).map((m) => (
                    <SelectItem key={m.name} value={m.name}>{m.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex flex-col gap-1">
              <Label className="text-[11px] text-muted-foreground">Prompt</Label>
              <Textarea
                value={store.detailerPrompt}
                onChange={(e) => store.setParam("detailerPrompt", e.target.value)}
                placeholder="Detailer prompt (optional)"
                className="min-h-[48px] text-xs resize-none"
              />
            </div>

            <div className="flex flex-col gap-1">
              <Label className="text-[11px] text-muted-foreground">Negative</Label>
              <Textarea
                value={store.detailerNegative}
                onChange={(e) => store.setParam("detailerNegative", e.target.value)}
                placeholder="Detailer negative prompt (optional)"
                className="min-h-[36px] text-xs resize-none"
              />
            </div>
          </>
        )}
      </ParamSection>

      {store.detailerEnabled && (
        <>
          <ParamSection title="Generation">
            <ParamSlider label="Steps" value={store.detailerSteps} onChange={(v) => store.setParam("detailerSteps", v)} min={0} max={99} />
            <ParamSlider label="Strength" value={store.detailerStrength} onChange={(v) => store.setParam("detailerStrength", v)} min={0} max={1} step={0.01} />
            <ParamSlider label="Resolution" value={store.detailerResolution} onChange={(v) => store.setParam("detailerResolution", v)} min={256} max={4096} step={8} />
          </ParamSection>

          <ParamSection title="Detection" defaultOpen={false}>
            <ParamSlider label="Max detect" value={store.detailerMaxDetected} onChange={(v) => store.setParam("detailerMaxDetected", v)} min={1} max={10} />
            <ParamSlider label="Padding" value={store.detailerPadding} onChange={(v) => store.setParam("detailerPadding", v)} min={0} max={100} />
            <ParamSlider label="Blur" value={store.detailerBlur} onChange={(v) => store.setParam("detailerBlur", v)} min={0} max={100} />
            <ParamSlider label="Confidence" value={store.detailerConfidence} onChange={(v) => store.setParam("detailerConfidence", v)} min={0} max={1} step={0.01} />
            <ParamSlider label="IoU" value={store.detailerIou} onChange={(v) => store.setParam("detailerIou", v)} min={0} max={1} step={0.01} />
            <ParamSlider label="Min size" value={store.detailerMinSize} onChange={(v) => store.setParam("detailerMinSize", v)} min={0} max={1} step={0.01} />
            <ParamSlider label="Max size" value={store.detailerMaxSize} onChange={(v) => store.setParam("detailerMaxSize", v)} min={0} max={1} step={0.01} />
            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 flex-shrink-0">Classes</Label>
              <Input
                value={store.detailerClasses}
                onChange={(e) => store.setParam("detailerClasses", e.target.value)}
                placeholder="e.g. person, face"
                className="flex-1 h-6 text-[11px] px-2"
              />
            </div>
          </ParamSection>

          <ParamSection title="Options" defaultOpen={false}>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={store.detailerSegmentation} onCheckedChange={(c) => store.setParam("detailerSegmentation", !!c)} />
                Segmentation
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={store.detailerIncludeDetections} onCheckedChange={(c) => store.setParam("detailerIncludeDetections", !!c)} />
                Include detections
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={store.detailerMerge} onCheckedChange={(c) => store.setParam("detailerMerge", !!c)} />
                Merge
              </label>
              <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground cursor-pointer">
                <Checkbox checked={store.detailerSort} onCheckedChange={(c) => store.setParam("detailerSort", !!c)} />
                Sort
              </label>
            </div>
          </ParamSection>

          <ParamSection title="Noise" defaultOpen={false}>
            <ParamSlider label="Renoise" value={store.detailerRenoise} onChange={(v) => store.setParam("detailerRenoise", v)} min={0.5} max={1.5} step={0.01} />
            <ParamSlider label="End" value={store.detailerRenoiseEnd} onChange={(v) => store.setParam("detailerRenoiseEnd", v)} min={0} max={1} step={0.01} />
          </ParamSection>
        </>
      )}
    </div>
  );
}
