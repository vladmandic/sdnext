import { useCallback, useMemo, useState } from "react";
import { Play, Square, Sparkles, Settings2, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { useVideoStore } from "@/stores/videoStore";
import { usePromptEnhanceStore } from "@/stores/promptEnhanceStore";
import { useJobQueueStore, selectDomainActive, selectDomainProgress, selectDomainRunning } from "@/stores/jobStore";
import { useSubmitToQueue } from "@/hooks/useSubmitToQueue";
import { sendToJob } from "@/hooks/useJobTracker";
import { useCancelJob } from "@/api/hooks/useJobs";
import { usePromptEnhance } from "@/api/hooks/usePromptEnhance";
import { uploadFile, uploadBlob } from "@/lib/upload";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Label } from "@/components/ui/label";
import { Popover, PopoverTrigger, PopoverContent } from "@/components/ui/popover";
import { PromptEnhanceWorkspace } from "@/components/generation/PromptEnhanceWorkspace";
import { ModelsVideoTab } from "./tabs/ModelsVideoTab";
import { FramePackTab } from "./tabs/FramePackTab";
import { LtxVideoTab } from "./tabs/LtxVideoTab";
import type { JobDomain } from "@/stores/jobStore";
import type { PromptEnhanceRequest } from "@/api/types/promptEnhance";

const tabs = [
  { id: "models", label: "Models" },
  { id: "framepack", label: "FramePack" },
  { id: "ltx", label: "LTX" },
] as const;

function tabToDomain(tab: string): JobDomain {
  if (tab === "framepack") return "framepack";
  if (tab === "ltx") return "ltx";
  return "video";
}

async function buildJobPayload(tab: string) {
  const s = useVideoStore.getState();
  const output = {
    fps: s.fps,
    interpolate: s.interpolate,
    codec: s.codec,
    format: s.format,
    codec_options: s.codecOptions,
    save_video: s.saveVideo,
    save_frames: s.saveFrames,
    save_safetensors: s.saveSafetensors,
  };

  const initRef = s.initImage ? await uploadFile(s.initImage) : null;
  const lastRef = s.lastImage ? await uploadFile(s.lastImage) : null;

  if (tab === "framepack") {
    return {
      type: "framepack" as const,
      prompt: s.prompt,
      negative: s.negative,
      seed: s.seed,
      variant: s.fpVariant,
      resolution: s.fpResolution,
      duration: s.fpDuration,
      latent_ws: s.fpLatentWindowSize,
      steps: s.fpSteps,
      shift: s.fpShift,
      cfg_scale: s.fpCfgScale,
      cfg_distilled: s.fpCfgDistilled,
      cfg_rescale: s.fpCfgRescale,
      start_weight: s.fpStartWeight,
      end_weight: s.fpEndWeight,
      vision_weight: s.fpVisionWeight,
      section_prompt: s.fpSectionPrompt,
      system_prompt: s.fpSystemPrompt,
      use_teacache: s.fpTeacache,
      optimized_prompt: s.fpOptimizedPrompt,
      use_cfgzero: s.fpCfgZero,
      use_preview: s.fpPreview,
      attention: s.fpAttention,
      vae_type: s.fpVaeType,
      init_image: initRef,
      end_image: lastRef,
      ...output,
    };
  }

  if (tab === "ltx") {
    return {
      type: "ltx" as const,
      model: s.ltxModel,
      prompt: s.prompt,
      negative: s.negative,
      seed: s.seed,
      width: s.width,
      height: s.height,
      frames: s.frames,
      steps: s.ltxSteps,
      decode_timestep: s.ltxDecodeTimestep,
      image_cond_noise_scale: s.ltxNoiseScale,
      upsample_enable: s.ltxUpsampleEnable,
      upsample_ratio: s.ltxUpsampleRatio,
      refine_enable: s.ltxRefineEnable,
      refine_strength: s.ltxRefineStrength,
      condition_strength: s.ltxConditionStrength,
      condition_image: initRef,
      condition_last: lastRef,
      audio_enable: s.ltxAudioEnable,
      ...output,
    };
  }

  // models (generic video)
  return {
    type: "video" as const,
    engine: s.engine,
    model: s.model,
    prompt: s.prompt,
    negative: s.negative,
    seed: s.seed,
    width: s.width,
    height: s.height,
    frames: s.frames,
    steps: s.steps,
    guidance_scale: s.guidanceScale,
    guidance_true: s.guidanceTrue,
    sampler_shift: s.samplerShift,
    dynamic_shift: s.dynamicShift,
    init_strength: s.initStrength,
    init_image: initRef,
    last_image: lastRef,
    vae_type: s.vaeType,
    vae_tile_frames: s.vaeTileFrames,
    ...output,
  };
}

function canGenerate(tab: string) {
  const s = useVideoStore.getState();
  if (!s.prompt) return false;
  if (tab === "models") return !!(s.engine && s.model);
  if (tab === "ltx") return !!s.ltxModel;
  return true; // framepack only needs prompt
}

export function VideoPanel() {
  const activeVideoTab = useVideoStore((s) => s.activeVideoTab);
  const prompt = useVideoStore((s) => s.prompt);
  const negative = useVideoStore((s) => s.negative);
  const setParam = useVideoStore((s) => s.setParam);
  const domain = tabToDomain(activeVideoTab);
  const isVideoActive = useJobQueueStore(selectDomainActive("video"));
  const isFramepackActive = useJobQueueStore(selectDomainActive("framepack"));
  const isLtxActive = useJobQueueStore(selectDomainActive("ltx"));
  const isGenerating = isVideoActive || isFramepackActive || isLtxActive;
  const progress = useJobQueueStore(selectDomainProgress(domain));
  const runningVideoJob = useJobQueueStore(selectDomainRunning(domain));

  const cancelJob = useCancelJob();

  // Prompt enhance
  const [enhanceOpen, setEnhanceOpen] = useState(false);
  const enhanceStore = usePromptEnhanceStore();
  const pinned = usePromptEnhanceStore((s) => s.pinned);
  const setPendingResult = usePromptEnhanceStore((s) => s.setPendingResult);
  const enhanceMutation = usePromptEnhance();

  const handleEnhance = useCallback(async () => {
    if (!prompt.trim()) {
      toast.warning("Enter a prompt first");
      return;
    }
    let image: string | undefined;
    if (enhanceStore.useVision) {
      const initImg = useVideoStore.getState().initImage;
      if (initImg) image = await uploadBlob(initImg, "vision.png");
    }
    const req: PromptEnhanceRequest = {
      prompt,
      type: "video",
      model: enhanceStore.model || undefined,
      system_prompt: enhanceStore.systemPrompt || undefined,
      prefix: enhanceStore.prefix || undefined,
      suffix: enhanceStore.suffix || undefined,
      nsfw: enhanceStore.nsfw,
      seed: enhanceStore.seed,
      do_sample: enhanceStore.doSample,
      max_tokens: enhanceStore.maxTokens,
      temperature: enhanceStore.temperature,
      repetition_penalty: enhanceStore.repetitionPenalty,
      top_k: enhanceStore.topK || undefined,
      top_p: enhanceStore.topP || undefined,
      thinking: enhanceStore.thinking,
      keep_thinking: enhanceStore.keepThinking,
      use_vision: enhanceStore.useVision,
      prefill: enhanceStore.prefill || undefined,
      keep_prefill: enhanceStore.keepPrefill,
      image,
    };
    enhanceMutation.mutate(req, {
      onSuccess: (res) => {
        setPendingResult({ prompt: res.prompt, seed: res.seed, originalPrompt: prompt });
        setEnhanceOpen(true);
        toast.success(`Prompt enhanced (seed: ${res.seed})`);
      },
      onError: (err) => {
        toast.error(`Enhance failed: ${err instanceof Error ? err.message : "Unknown error"}`);
      },
    });
  }, [prompt, enhanceStore, enhanceMutation, setPendingResult]);

  const handleAcceptEnhanced = useCallback((p: string) => setParam("prompt", p), [setParam]);

  const buildRequest = useCallback(async () => {
    const payload = await buildJobPayload(activeVideoTab);
    return { payload, snapshot: {} };
  }, [activeVideoTab]);

  const { submit, isSubmitting } = useSubmitToQueue(useMemo(() => ({ domain, buildRequest }), [domain, buildRequest]));

  const handleCancel = useCallback(() => {
    if (runningVideoJob) {
      sendToJob(runningVideoJob.id, { type: "interrupt" });
      cancelJob.mutate(runningVideoJob.id);
    }
  }, [runningVideoJob, cancelJob]);

  const progressPct = Math.round(progress * 100);

  return (
    <ScrollArea className="h-full">
      <div className="p-3 space-y-1">
        {/* Prompt - always visible */}
        <div className="space-y-1.5 mb-3">
          <div className="flex items-center justify-between mb-1">
            <Label className="text-2xs text-muted-foreground">Prompt</Label>
            <div className="flex items-center gap-0.5">
              <button
                type="button"
                onClick={handleEnhance}
                disabled={enhanceMutation.isPending}
                className="p-0.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                title="Enhance prompt"
              >
                {enhanceMutation.isPending ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
              </button>
              <Popover open={enhanceOpen} onOpenChange={setEnhanceOpen}>
                <PopoverTrigger asChild>
                  <button
                    type="button"
                    className="p-0.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                    title="Enhance settings"
                  >
                    <Settings2 size={13} />
                  </button>
                </PopoverTrigger>
                <PopoverContent
                  side="right"
                  align="start"
                  className="w-96 p-0"
                  onInteractOutside={(e) => { if (pinned) e.preventDefault(); }}
                  onEscapeKeyDown={(e) => { if (pinned) e.preventDefault(); }}
                >
                  <ScrollArea className="max-h-[80vh]">
                    <PromptEnhanceWorkspace
                      onEnhance={handleEnhance}
                      isPending={enhanceMutation.isPending}
                      onClose={() => setEnhanceOpen(false)}
                      onAccept={handleAcceptEnhanced}
                      onSelectPrompt={handleAcceptEnhanced}
                    />
                  </ScrollArea>
                </PopoverContent>
              </Popover>
            </div>
          </div>
          <Textarea
            value={prompt}
            onChange={(e) => setParam("prompt", e.target.value)}
            placeholder="Describe the video..."
            className="text-xs min-h-15 resize-y"
          />
          <Textarea
            value={negative}
            onChange={(e) => setParam("negative", e.target.value)}
            placeholder="Negative prompt (optional)"
            className="text-xs min-h-9 resize-y"
          />
        </div>

        {/* Generate / Stop */}
        <div className="space-y-2 mb-3">
          <div className="flex gap-2">
            <Button
              type="button"
              onClick={submit}
              disabled={isSubmitting || !canGenerate(activeVideoTab)}
              variant="default"
              size="sm"
              className="flex-1"
            >
              <Play size={14} />
              Generate
            </Button>
            {isGenerating && (
              <Button
                type="button"
                onClick={handleCancel}
                variant="destructive"
                size="sm"
              >
                <Square size={14} />
                Stop
              </Button>
            )}
          </div>
          {isGenerating && progressPct > 0 && (
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full transition-[width] duration-300" style={{ width: `${progressPct}%` }} />
              </div>
              <span className="text-xs text-muted-foreground tabular-nums">{progressPct}%</span>
            </div>
          )}
        </div>

        {/* Sub-tab bar */}
        <div className="flex border-b border-border mb-3">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setParam("activeVideoTab", tab.id)}
              className={`px-3 py-1.5 text-xs font-medium transition-colors relative ${
                activeVideoTab === tab.id
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground/70"
              }`}
            >
              {tab.label}
              {activeVideoTab === tab.id && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-t" />
              )}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeVideoTab === "models" && <ModelsVideoTab />}
        {activeVideoTab === "framepack" && <FramePackTab />}
        {activeVideoTab === "ltx" && <LtxVideoTab />}
      </div>
    </ScrollArea>
  );
}
