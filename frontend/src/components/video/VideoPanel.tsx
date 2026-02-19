import { useMemo, useEffect, useCallback } from "react";
import { Play, Square, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useVideoStore } from "@/stores/videoStore";
import { useVideoEngines, useLoadVideoModel } from "@/api/hooks/useVideo";
import { useSubmitJob, useCancelJob } from "@/api/hooks/useJobs";
import { useJobWebSocket } from "@/api/hooks/useJobWebSocket";
import { ParamSection } from "@/components/generation/ParamSection";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { Combobox } from "@/components/ui/combobox";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/api/client";

export function VideoPanel() {
  const engine = useVideoStore((s) => s.engine);
  const model = useVideoStore((s) => s.model);
  const prompt = useVideoStore((s) => s.prompt);
  const negative = useVideoStore((s) => s.negative);
  const width = useVideoStore((s) => s.width);
  const height = useVideoStore((s) => s.height);
  const frames = useVideoStore((s) => s.frames);
  const steps = useVideoStore((s) => s.steps);
  const seed = useVideoStore((s) => s.seed);
  const guidanceScale = useVideoStore((s) => s.guidanceScale);
  const samplerShift = useVideoStore((s) => s.samplerShift);
  const dynamicShift = useVideoStore((s) => s.dynamicShift);
  const initStrength = useVideoStore((s) => s.initStrength);
  const fps = useVideoStore((s) => s.fps);
  const format = useVideoStore((s) => s.format);
  const saveVideo = useVideoStore((s) => s.saveVideo);
  const saveFrames = useVideoStore((s) => s.saveFrames);
  const isGenerating = useVideoStore((s) => s.isGenerating);
  const jobId = useVideoStore((s) => s.jobId);
  const progress = useVideoStore((s) => s.progress);
  const setParam = useVideoStore((s) => s.setParam);
  const setGenerating = useVideoStore((s) => s.setGenerating);
  const setJobId = useVideoStore((s) => s.setJobId);
  const setProgress = useVideoStore((s) => s.setProgress);
  const setResultVideo = useVideoStore((s) => s.setResultVideo);

  const { data: engines } = useVideoEngines();
  const loadModel = useLoadVideoModel();
  const submitJob = useSubmitJob();
  const cancelJob = useCancelJob();
  const { progress: wsProgress, result: wsResult, status: wsStatus, error: wsError } = useJobWebSocket(jobId);

  const engineNames = useMemo(() => engines?.map((e) => e.engine) ?? [], [engines]);
  const modelNames = useMemo(() => {
    if (!engines || !engine) return [];
    const eng = engines.find((e) => e.engine === engine);
    return eng?.models ?? [];
  }, [engines, engine]);

  useEffect(() => {
    if (jobId && wsProgress.progress > 0) {
      setProgress(wsProgress.progress);
    }
  }, [jobId, wsProgress, setProgress]);

  useEffect(() => {
    if (!jobId) return;
    if (wsStatus === "completed" && wsResult) {
      const vid = wsResult.images[0];
      if (vid) {
        setResultVideo(`${api.getBaseUrl()}${vid.url}`);
      }
      setJobId(null);
      setGenerating(false);
    } else if (wsStatus === "failed") {
      toast.error("Video generation failed", { description: wsError ?? "Unknown error" });
      setJobId(null);
      setGenerating(false);
    } else if (wsStatus === "cancelled") {
      setJobId(null);
      setGenerating(false);
    }
  }, [wsStatus, wsResult, wsError, jobId, setResultVideo, setJobId, setGenerating]);

  const handleLoad = useCallback(() => {
    if (!engine || !model) return;
    loadModel.mutate({ engine, model }, {
      onSuccess: () => toast.success(`Loaded ${model}`),
      onError: (err) => toast.error("Failed to load model", { description: err.message }),
    });
  }, [engine, model, loadModel]);

  const handleGenerate = useCallback(async () => {
    if (isGenerating || !engine || !model || !prompt) return;
    setGenerating(true);
    setResultVideo(null);
    try {
      const state = useVideoStore.getState();
      const job = await submitJob.mutateAsync({
        type: "video",
        engine: state.engine,
        model: state.model,
        prompt: state.prompt,
        negative: state.negative,
        width: state.width,
        height: state.height,
        frames: state.frames,
        steps: state.steps,
        seed: state.seed,
        guidance_scale: state.guidanceScale,
        sampler_shift: state.samplerShift,
        dynamic_shift: state.dynamicShift,
        init_strength: state.initStrength,
        fps: state.fps,
        format: state.format,
        save_video: state.saveVideo,
        save_frames: state.saveFrames,
      });
      setJobId(job.id);
    } catch (err) {
      toast.error("Failed to start video generation", { description: err instanceof Error ? err.message : String(err) });
      setGenerating(false);
    }
  }, [isGenerating, engine, model, prompt, setGenerating, setResultVideo, submitJob, setJobId]);

  const handleCancel = useCallback(() => {
    if (jobId) cancelJob.mutate(jobId);
    setJobId(null);
    setGenerating(false);
  }, [jobId, cancelJob, setJobId, setGenerating]);

  const progressPct = Math.round(progress * 100);

  return (
    <ScrollArea className="h-full">
      <div className="p-3 space-y-1">
        {/* Prompt */}
        <ParamSection title="Prompt">
          <Textarea
            value={prompt}
            onChange={(e) => setParam("prompt", e.target.value)}
            placeholder="Describe the video..."
            className="text-xs min-h-[60px] resize-y"
          />
          <Textarea
            value={negative}
            onChange={(e) => setParam("negative", e.target.value)}
            placeholder="Negative prompt (optional)"
            className="text-xs min-h-[36px] resize-y"
          />
        </ParamSection>

        {/* Engine / Model */}
        <ParamSection title="Model">
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 shrink-0">Engine</Label>
              <Combobox value={engine} onValueChange={(v) => { setParam("engine", v); setParam("model", ""); }} options={engineNames} placeholder="Select engine..." className="h-7 text-xs flex-1" />
            </div>
            <div className="flex items-center gap-2">
              <Label className="text-[11px] text-muted-foreground w-16 shrink-0">Model</Label>
              <Combobox value={model} onValueChange={(v) => setParam("model", v)} options={modelNames} placeholder={engine ? "Select model..." : "Select engine first"} className="h-7 text-xs flex-1" />
            </div>
            <Button size="sm" variant="secondary" onClick={handleLoad} disabled={!engine || !model || loadModel.isPending} className="w-full">
              {loadModel.isPending ? <Loader2 size={14} className="animate-spin" /> : null}
              Load Model
            </Button>
          </div>
        </ParamSection>

        {/* Size */}
        <ParamSection title="Size" defaultOpen={false}>
          <ParamSlider label="Width" value={width} onChange={(v) => setParam("width", v)} min={256} max={1920} step={16} />
          <ParamSlider label="Height" value={height} onChange={(v) => setParam("height", v)} min={256} max={1920} step={16} />
          <ParamSlider label="Frames" value={frames} onChange={(v) => setParam("frames", v)} min={1} max={256} step={1} />
        </ParamSection>

        {/* Sampling */}
        <ParamSection title="Sampling" defaultOpen={false}>
          <ParamSlider label="Steps" value={steps} onChange={(v) => setParam("steps", v)} min={1} max={100} step={1} />
          <ParamSlider label="Guidance" value={guidanceScale} onChange={(v) => setParam("guidanceScale", v)} min={0} max={20} step={0.5} />
          <ParamSlider label="Shift" value={samplerShift} onChange={(v) => setParam("samplerShift", v)} min={-1} max={20} step={0.5} />
          <ParamSlider label="Strength" value={initStrength} onChange={(v) => setParam("initStrength", v)} min={0} max={1} step={0.05} />
          <ParamSlider label="Seed" value={seed} onChange={(v) => setParam("seed", v)} min={-1} max={999999999} step={1} />
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 shrink-0">Dynamic</Label>
            <Switch checked={dynamicShift} onCheckedChange={(v) => setParam("dynamicShift", v)} />
          </div>
        </ParamSection>

        {/* Output */}
        <ParamSection title="Output" defaultOpen={false}>
          <ParamSlider label="FPS" value={fps} onChange={(v) => setParam("fps", v)} min={1} max={60} step={1} />
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 shrink-0">Format</Label>
            <Combobox value={format} onValueChange={(v) => setParam("format", v)} options={["mp4", "webm", "gif"]} className="h-7 text-xs flex-1" />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 shrink-0">Save video</Label>
            <Switch checked={saveVideo} onCheckedChange={(v) => setParam("saveVideo", v)} />
          </div>
          <div className="flex items-center gap-2">
            <Label className="text-[11px] text-muted-foreground w-16 shrink-0">Frames</Label>
            <Switch checked={saveFrames} onCheckedChange={(v) => setParam("saveFrames", v)} />
          </div>
        </ParamSection>

        {/* Generate button */}
        <div className="pt-2 space-y-2">
          <Button
            type="button"
            onClick={isGenerating ? handleCancel : handleGenerate}
            disabled={!isGenerating && (!engine || !model || !prompt)}
            variant={isGenerating ? "destructive" : "default"}
            size="sm"
            className="w-full"
          >
            {isGenerating ? (
              <>
                <Square size={14} />
                Stop
              </>
            ) : (
              <>
                <Play size={14} />
                Generate
              </>
            )}
          </Button>
          {isGenerating && progressPct > 0 && (
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full transition-[width] duration-300" style={{ width: `${progressPct}%` }} />
              </div>
              <span className="text-xs text-muted-foreground tabular-nums">{progressPct}%</span>
            </div>
          )}
        </div>
      </div>
    </ScrollArea>
  );
}
