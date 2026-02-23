import { useMemo, useCallback } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useVideoStore } from "@/stores/videoStore";
import { useVideoEngines, useLoadVideoModel } from "@/api/hooks/useVideo";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { ParamGrid } from "@/components/generation/ParamRow";
import { Combobox } from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ImageDropInput } from "@/components/video/ImageDropInput";
import { VideoOutputSection } from "./VideoOutputSection";

export function ModelsVideoTab() {
  const engine = useVideoStore((s) => s.engine);
  const model = useVideoStore((s) => s.model);
  const width = useVideoStore((s) => s.width);
  const height = useVideoStore((s) => s.height);
  const frames = useVideoStore((s) => s.frames);
  const steps = useVideoStore((s) => s.steps);
  const seed = useVideoStore((s) => s.seed);
  const guidanceScale = useVideoStore((s) => s.guidanceScale);
  const guidanceTrue = useVideoStore((s) => s.guidanceTrue);
  const samplerShift = useVideoStore((s) => s.samplerShift);
  const dynamicShift = useVideoStore((s) => s.dynamicShift);
  const initStrength = useVideoStore((s) => s.initStrength);
  const initImage = useVideoStore((s) => s.initImage);
  const lastImage = useVideoStore((s) => s.lastImage);
  const vaeType = useVideoStore((s) => s.vaeType);
  const vaeTileFrames = useVideoStore((s) => s.vaeTileFrames);
  const setParam = useVideoStore((s) => s.setParam);

  const { data: engines } = useVideoEngines();
  const loadModel = useLoadVideoModel();

  const engineNames = useMemo(() => engines?.map((e) => e.engine) ?? [], [engines]);
  const modelNames = useMemo(() => {
    if (!engines || !engine) return [];
    const eng = engines.find((e) => e.engine === engine);
    return eng?.models ?? [];
  }, [engines, engine]);

  const handleLoad = useCallback(() => {
    if (!engine || !model) return;
    loadModel.mutate({ engine, model }, {
      onSuccess: () => toast.success(`Loaded ${model}`),
      onError: (err) => toast.error("Failed to load model", { description: err.message }),
    });
  }, [engine, model, loadModel]);

  return (
    <div className="space-y-1">
      <ParamSection title="Model">
        <div className="space-y-1.5">
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Engine</Label>
            <Combobox value={engine} onValueChange={(v) => { setParam("engine", v); setParam("model", ""); }} options={engineNames} placeholder="Select engine..." className="h-6 text-2xs flex-1" />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-2xs text-muted-foreground w-16 shrink-0">Model</Label>
            <Combobox value={model} onValueChange={(v) => setParam("model", v)} options={modelNames} placeholder={engine ? "Select model..." : "Select engine first"} className="h-6 text-2xs flex-1" />
          </div>
          <Button size="sm" variant="secondary" onClick={handleLoad} disabled={!engine || !model || loadModel.isPending} className="w-full">
            {loadModel.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
            Load Model
          </Button>
        </div>
      </ParamSection>

      <ParamSection title="Parameters" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Steps" value={steps} onChange={(v) => setParam("steps", v)} min={1} max={100} step={1} />
          <ParamSlider label="Guidance" value={guidanceScale} onChange={(v) => setParam("guidanceScale", v)} min={0} max={20} step={0.5} />
          <ParamSlider label="True CFG" value={guidanceTrue} onChange={(v) => setParam("guidanceTrue", v)} min={-1} max={20} step={0.5} />
          <ParamSlider label="Shift" value={samplerShift} onChange={(v) => setParam("samplerShift", v)} min={-1} max={20} step={0.5} />
        </ParamGrid>
        <ParamSlider label="Seed" value={seed} onChange={(v) => setParam("seed", v)} min={-1} max={999999999} step={1} />
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">Dynamic</Label>
          <Switch checked={dynamicShift} onCheckedChange={(v) => setParam("dynamicShift", v)} />
        </div>
      </ParamSection>

      <ParamSection title="Size" defaultOpen={false}>
        <ParamGrid>
          <ParamSlider label="Width" value={width} onChange={(v) => setParam("width", v)} min={256} max={1920} step={16} />
          <ParamSlider label="Height" value={height} onChange={(v) => setParam("height", v)} min={256} max={1920} step={16} />
        </ParamGrid>
        <ParamSlider label="Frames" value={frames} onChange={(v) => setParam("frames", v)} min={1} max={256} step={1} />
      </ParamSection>

      <ParamSection title="Inputs" defaultOpen={false}>
        <ParamSlider label="Strength" value={initStrength} onChange={(v) => setParam("initStrength", v)} min={0} max={1} step={0.05} />
        <ImageDropInput label="Init image" value={initImage} onChange={(v) => setParam("initImage", v)} />
        <ImageDropInput label="Last image" value={lastImage} onChange={(v) => setParam("lastImage", v)} />
      </ParamSection>

      <ParamSection title="Decode" defaultOpen={false}>
        <div className="flex items-center gap-2">
          <Label className="text-2xs text-muted-foreground w-16 shrink-0">VAE type</Label>
          <Combobox value={vaeType} onValueChange={(v) => setParam("vaeType", v)} options={["Default", "Tiny", "Remote", "Upscale"]} className="h-6 text-2xs flex-1" />
        </div>
        <ParamSlider label="Tile frames" value={vaeTileFrames} onChange={(v) => setParam("vaeTileFrames", v)} min={0} max={64} step={1} />
      </ParamSection>

      <VideoOutputSection />
    </div>
  );
}
