import { useCallback, useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useVideoStore } from "@/stores/videoStore";
import { useFramePackVariants, useLoadFramePack, useUnloadFramePack } from "@/api/hooks/useVideo";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { ParamGrid } from "@/components/generation/ParamRow";
import { SectionTimeline } from "@/components/video/SectionTimeline";
import { Combobox } from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { VideoOutputSection } from "./VideoOutputSection";

export function FramePackTab() {
  const fpVariant = useVideoStore((s) => s.fpVariant);
  const fpResolution = useVideoStore((s) => s.fpResolution);
  const fpDuration = useVideoStore((s) => s.fpDuration);
  const fpLatentWindowSize = useVideoStore((s) => s.fpLatentWindowSize);
  const fpSteps = useVideoStore((s) => s.fpSteps);
  const fpShift = useVideoStore((s) => s.fpShift);
  const fpCfgScale = useVideoStore((s) => s.fpCfgScale);
  const fpCfgDistilled = useVideoStore((s) => s.fpCfgDistilled);
  const fpCfgRescale = useVideoStore((s) => s.fpCfgRescale);
  const fpStartWeight = useVideoStore((s) => s.fpStartWeight);
  const fpEndWeight = useVideoStore((s) => s.fpEndWeight);
  const fpVisionWeight = useVideoStore((s) => s.fpVisionWeight);
  const fpSectionPrompt = useVideoStore((s) => s.fpSectionPrompt);
  const fpSystemPrompt = useVideoStore((s) => s.fpSystemPrompt);
  const fpTeacache = useVideoStore((s) => s.fpTeacache);
  const fpOptimizedPrompt = useVideoStore((s) => s.fpOptimizedPrompt);
  const fpCfgZero = useVideoStore((s) => s.fpCfgZero);
  const fpPreview = useVideoStore((s) => s.fpPreview);
  const fpAttention = useVideoStore((s) => s.fpAttention);
  const fpVaeType = useVideoStore((s) => s.fpVaeType);
  const fps = useVideoStore((s) => s.fps);
  const interpolate = useVideoStore((s) => s.interpolate);
  const seed = useVideoStore((s) => s.seed);
  const setParam = useVideoStore((s) => s.setParam);

  const [rawEdit, setRawEdit] = useState(false);

  const { data: variants } = useFramePackVariants();
  const loadFP = useLoadFramePack();
  const unloadFP = useUnloadFramePack();

  const handleLoad = useCallback(() => {
    loadFP.mutate({ variant: fpVariant, attention: fpAttention }, {
      onSuccess: () => toast.success(`Loaded FramePack ${fpVariant}`),
      onError: (err) => toast.error("Failed to load FramePack", { description: err.message }),
    });
  }, [fpVariant, fpAttention, loadFP]);

  const handleUnload = useCallback(() => {
    unloadFP.mutate(undefined, {
      onSuccess: () => toast.success("FramePack unloaded"),
      onError: (err) => toast.error("Failed to unload", { description: err.message }),
    });
  }, [unloadFP]);

  return (
    <div className="space-y-1">
      <ParamSection title="Model">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Variant</Label>
            <Combobox value={fpVariant} onValueChange={(v) => setParam("fpVariant", v)} options={variants ?? ["bi-directional", "forward-only"]} className="h-6 text-2xs flex-1" />
          </div>
          <div className="flex gap-1.5">
            <Button size="sm" variant="secondary" onClick={handleLoad} disabled={loadFP.isPending} className="flex-1">
              {loadFP.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
              Load
            </Button>
            <Button size="sm" variant="outline" onClick={handleUnload} disabled={unloadFP.isPending} className="flex-1">
              Unload
            </Button>
          </div>
        </div>
      </ParamSection>

      <ParamSection title="Size" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Resolution" value={fpResolution} onChange={(v) => setParam("fpResolution", v)} min={240} max={1088} step={16} />
          <ParamSlider label="Duration" value={fpDuration} onChange={(v) => setParam("fpDuration", v)} min={1} max={120} step={1} />
        </ParamGrid>
      </ParamSection>

      <ParamSection title="Inputs" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Start wt" value={fpStartWeight} onChange={(v) => setParam("fpStartWeight", v)} min={0} max={2} step={0.05} />
          <ParamSlider label="End wt" value={fpEndWeight} onChange={(v) => setParam("fpEndWeight", v)} min={0} max={2} step={0.05} />
        </ParamGrid>
        <ParamSlider label="Vision wt" value={fpVisionWeight} onChange={(v) => setParam("fpVisionWeight", v)} min={0} max={2} step={0.05} />
      </ParamSection>

      <ParamSection title="Sections" defaultOpen={false}>
        <div className="flex items-center justify-between mb-1">
          <Label className="text-2xs text-muted-foreground">Raw edit</Label>
          <Switch checked={rawEdit} onCheckedChange={setRawEdit} />
        </div>
        {rawEdit ? (
          <Textarea
            value={fpSectionPrompt}
            onChange={(e) => setParam("fpSectionPrompt", e.target.value)}
            placeholder="Section prompts (comma or newline separated)"
            className="text-xs min-h-12 resize-y"
          />
        ) : (
          <SectionTimeline
            fps={fps}
            duration={fpDuration}
            latentWindowSize={fpLatentWindowSize}
            variant={fpVariant}
            interpolate={interpolate}
            value={fpSectionPrompt}
            onChange={(v) => setParam("fpSectionPrompt", v)}
          />
        )}
      </ParamSection>

      <ParamSection title="Advanced" defaultOpen={false}>
        <ParamSlider label="Seed" value={seed} onChange={(v) => setParam("seed", v)} min={-1} max={999999999} step={1} />
        <ParamSlider label="Window" value={fpLatentWindowSize} onChange={(v) => setParam("fpLatentWindowSize", v)} min={1} max={33} step={4} />
        <ParamGrid>
          <ParamSlider label="Steps" value={fpSteps} onChange={(v) => setParam("fpSteps", v)} min={1} max={100} step={1} />
          <ParamSlider label="Shift" value={fpShift} onChange={(v) => setParam("fpShift", v)} min={0} max={20} step={0.5} />
          <ParamSlider label="CFG" value={fpCfgScale} onChange={(v) => setParam("fpCfgScale", v)} min={0} max={20} step={0.5} />
          <ParamSlider label="Distilled" value={fpCfgDistilled} onChange={(v) => setParam("fpCfgDistilled", v)} min={0} max={20} step={0.5} />
        </ParamGrid>
        <ParamSlider label="Rescale" value={fpCfgRescale} onChange={(v) => setParam("fpCfgRescale", v)} min={0} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Model Options" defaultOpen={false}>
        <Textarea
          value={fpSystemPrompt}
          onChange={(e) => setParam("fpSystemPrompt", e.target.value)}
          placeholder="System prompt (optional)"
          className="text-xs min-h-9 resize-y"
        />
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">TeaCache</Label>
          <Switch checked={fpTeacache} onCheckedChange={(v) => setParam("fpTeacache", v)} />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Optimized</Label>
          <Switch checked={fpOptimizedPrompt} onCheckedChange={(v) => setParam("fpOptimizedPrompt", v)} />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">CFG Zero</Label>
          <Switch checked={fpCfgZero} onCheckedChange={(v) => setParam("fpCfgZero", v)} />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Preview</Label>
          <Switch checked={fpPreview} onCheckedChange={(v) => setParam("fpPreview", v)} />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Attention</Label>
          <Combobox value={fpAttention} onValueChange={(v) => setParam("fpAttention", v)} options={["Default", "sdpa", "flash", "sage", "xformers"]} className="h-6 text-2xs flex-1" />
        </div>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">VAE</Label>
          <Combobox value={fpVaeType} onValueChange={(v) => setParam("fpVaeType", v)} options={["Full", "Tiny"]} className="h-6 text-2xs flex-1" />
        </div>
      </ParamSection>

      <VideoOutputSection />
    </div>
  );
}
