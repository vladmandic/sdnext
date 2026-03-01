import { useMemo, useCallback } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useVideoStore } from "@/stores/videoStore";
import { useVideoEngines, useLoadVideoModel } from "@/api/hooks/useVideo";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { ParamGrid } from "@/components/generation/ParamRow";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { Switch } from "@/components/ui/switch";
import { VideoOutputSection } from "./VideoOutputSection";

export function LtxVideoTab() {
  const ltxModel = useVideoStore((s) => s.ltxModel);
  const width = useVideoStore((s) => s.width);
  const height = useVideoStore((s) => s.height);
  const frames = useVideoStore((s) => s.frames);
  const ltxSteps = useVideoStore((s) => s.ltxSteps);
  const seed = useVideoStore((s) => s.seed);
  const ltxConditionStrength = useVideoStore((s) => s.ltxConditionStrength);
  const ltxUpsampleEnable = useVideoStore((s) => s.ltxUpsampleEnable);
  const ltxUpsampleRatio = useVideoStore((s) => s.ltxUpsampleRatio);
  const ltxRefineEnable = useVideoStore((s) => s.ltxRefineEnable);
  const ltxRefineStrength = useVideoStore((s) => s.ltxRefineStrength);
  const ltxDecodeTimestep = useVideoStore((s) => s.ltxDecodeTimestep);
  const ltxNoiseScale = useVideoStore((s) => s.ltxNoiseScale);
  const ltxAudioEnable = useVideoStore((s) => s.ltxAudioEnable);
  const setParam = useVideoStore((s) => s.setParam);

  const { data: engines } = useVideoEngines();
  const loadModel = useLoadVideoModel();

  const ltxModels = useMemo(() => {
    if (!engines) return [];
    const eng = engines.find((e) => e.engine === "LTX Video");
    return eng?.models ?? [];
  }, [engines]);

  const handleLoad = useCallback(() => {
    if (!ltxModel) return;
    loadModel.mutate({ engine: "LTX Video", model: ltxModel }, {
      onSuccess: () => toast.success(`Loaded ${ltxModel}`),
      onError: (err) => toast.error("Failed to load LTX model", { description: err.message }),
    });
  }, [ltxModel, loadModel]);

  return (
    <div className="space-y-1">
      <ParamSection title="Model">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Model</Label>
            <Combobox value={ltxModel} onValueChange={(v) => setParam("ltxModel", v)} options={ltxModels} placeholder="Select LTX model..." className="h-6 text-2xs flex-1" />
          </div>
          <Button size="sm" variant="secondary" onClick={handleLoad} disabled={!ltxModel || loadModel.isPending} className="w-full">
            {loadModel.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
            Load Model
          </Button>
        </div>
      </ParamSection>

      <ParamSection title="Size" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Width" value={width} onChange={(v) => setParam("width", v)} min={256} max={1920} step={16} />
          <ParamSlider label="Height" value={height} onChange={(v) => setParam("height", v)} min={256} max={1920} step={16} />
        </ParamGrid>
        <ParamSlider label="Frames" value={frames} onChange={(v) => setParam("frames", v)} min={1} max={257} step={8} />
      </ParamSection>

      <ParamSection title="Sampling" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Steps" value={ltxSteps} onChange={(v) => setParam("ltxSteps", v)} min={1} max={100} step={1} />
          <ParamSlider label="Seed" value={seed} onChange={(v) => setParam("seed", v)} min={-1} max={999999999} step={1} />
        </ParamGrid>
      </ParamSection>

      <ParamSection title="Condition" defaultOpen={false}>
        <ParamSlider label="Strength" value={ltxConditionStrength} onChange={(v) => setParam("ltxConditionStrength", v)} min={0} max={1} step={0.05} />
      </ParamSection>

      <ParamSection title="Upsample" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Enable</Label>
          <Switch checked={ltxUpsampleEnable} onCheckedChange={(v) => setParam("ltxUpsampleEnable", v)} />
        </div>
        <ParamSlider label="Ratio" value={ltxUpsampleRatio} onChange={(v) => setParam("ltxUpsampleRatio", v)} min={1} max={4} step={0.5} disabled={!ltxUpsampleEnable} />
      </ParamSection>

      <ParamSection title="Refine" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Enable</Label>
          <Switch checked={ltxRefineEnable} onCheckedChange={(v) => setParam("ltxRefineEnable", v)} />
        </div>
        <ParamSlider label="Strength" value={ltxRefineStrength} onChange={(v) => setParam("ltxRefineStrength", v)} min={0.1} max={1} step={0.05} disabled={!ltxRefineEnable} />
      </ParamSection>

      <ParamSection title="Advanced" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Decode dt" value={ltxDecodeTimestep} onChange={(v) => setParam("ltxDecodeTimestep", v)} min={0} max={1} step={0.005} />
          <ParamSlider label="Noise scale" value={ltxNoiseScale} onChange={(v) => setParam("ltxNoiseScale", v)} min={0} max={1} step={0.005} />
        </ParamGrid>
      </ParamSection>

      <ParamSection title="Audio" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Enable</Label>
          <Switch checked={ltxAudioEnable} onCheckedChange={(v) => setParam("ltxAudioEnable", v)} />
        </div>
      </ParamSection>

      <VideoOutputSection />
    </div>
  );
}
