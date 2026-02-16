import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useTxt2Img, useImg2Img, useProgress, useInterrupt, useSkip } from "@/api/hooks/useGeneration";
import { buildTxt2ImgRequest, buildImg2ImgRequest, restoreFromResult } from "@/lib/requestBuilder";
import { snapshotUnits, useControlStore } from "@/stores/controlStore";
import type { Img2ImgRequest } from "@/api/types/generation";
import { Play, Square, SkipForward, Loader2, History } from "lucide-react";
import { useEffect, useRef, useCallback, memo } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ws, ensureWs, isWsConnected } from "@/api/wsManager";

export const ActionBar = memo(function ActionBar() {
  const isGenerating = useGenerationStore((s) => s.isGenerating);
  const prompt = useGenerationStore((s) => s.prompt);
  const progress = useGenerationStore((s) => s.progress);
  const setGenerating = useGenerationStore((s) => s.setGenerating);
  const setProgress = useGenerationStore((s) => s.setProgress);
  const setPreview = useGenerationStore((s) => s.setPreview);
  const addResult = useGenerationStore((s) => s.addResult);
  const clearSelection = useGenerationStore((s) => s.clearSelection);
  const lastResult = useGenerationStore((s) => s.results[0]);
  const generationMode = useUiStore((s) => s.generationMode);
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);
  const txt2img = useTxt2Img();
  const img2img = useImg2Img();
  const interrupt = useInterrupt();
  const skip = useSkip();
  const generatingRef = useRef(false);

  // WebSocket connection for live previews (module-level singleton)
  useEffect(() => {
    ensureWs();
    const unsubMsg = ws.on("message", (data) => {
      const msg = data as { type: string; data?: Record<string, unknown> };
      if (msg.type === "progress" && msg.data) {
        const d = msg.data;
        if (generatingRef.current && typeof d.progress === "number") {
          setProgress(d.progress, typeof d.eta === "number" ? d.eta : 0);
        }
      }
    });
    const unsubBin = ws.on("binary", (buf) => {
      if (generatingRef.current) {
        const blob = new Blob([buf], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setPreview(url);
      }
    });
    return () => { unsubMsg(); unsubBin(); };
  }, [setProgress, setPreview]);

  // REST polling: always active during generation for preview images.
  // WS provides faster progress updates but may not send binary previews
  // depending on the backend's live-preview decoder configuration.
  const { data: progressData } = useProgress(isGenerating);

  useEffect(() => {
    if (progressData && generatingRef.current) {
      // Use REST progress when WS is unavailable
      if (!isWsConnected()) {
        setProgress(progressData.progress, progressData.eta_relative);
      }
      // Always use REST for preview images as fallback
      if (progressData.current_image) {
        setPreview(`data:image/jpeg;base64,${progressData.current_image}`);
      }
    }
  }, [progressData, setProgress, setPreview]);

  const handleGenerate = useCallback(async () => {
    if (generatingRef.current) return;
    generatingRef.current = true;
    setGenerating(true);
    setPreview(null);
    clearSelection();
    try {
      const isImg2Img = generationMode === "img2img";
      const request = isImg2Img
        ? await buildImg2ImgRequest()
        : await buildTxt2ImgRequest();

      // Snapshot input state before the (possibly slow) API call
      const inputImage = isImg2Img ? (request as Img2ImgRequest).init_images?.[0] : undefined;
      const maskLines = useImg2ImgStore.getState().maskLines;
      const inputMask = isImg2Img && maskLines.length > 0 ? maskLines.slice() : undefined;
      const controlUnits = await snapshotUnits();

      const result = isImg2Img
        ? await img2img.mutateAsync(request as Img2ImgRequest)
        : await txt2img.mutateAsync(request);
      // Don't add result if generation was interrupted while awaiting
      if (!generatingRef.current) return;
      // Store preprocessed composite from control pipeline (if returned)
      if (result.processed_images && result.processed_images.length > 0) {
        useControlStore.getState().setCompositeProcessed(`data:image/png;base64,${result.processed_images[0]}`);
      }
      addResult({
        id: crypto.randomUUID(),
        images: result.images,
        parameters: result.parameters,
        info: result.info,
        timestamp: Date.now(),
        inputImage,
        inputMask,
        controlUnits,
      });
    } catch (err) {
      toast.error("Generation failed", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      generatingRef.current = false;
      setGenerating(false);
    }
  }, [txt2img, img2img, generationMode, setGenerating, setPreview, addResult, clearSelection]);

  const handleInterrupt = useCallback(() => {
    interrupt.mutate();
    generatingRef.current = false;
    setGenerating(false);
  }, [interrupt, setGenerating]);

  const handleSkip = useCallback(() => {
    skip.mutate();
  }, [skip]);

  const handleRestore = useCallback(() => {
    if (!lastResult) return;
    restoreFromResult(lastResult);
    toast.success("Settings restored from last generation");
  }, [lastResult]);

  const progressPct = Math.round(progress * 100);

  return (
    <div className="flex items-center gap-2">
      {/* Generate / Stop button */}
      <Button
        type="button"
        onClick={isGenerating ? handleInterrupt : handleGenerate}
        disabled={!isGenerating && (generationMode === "img2img" ? !hasLayers : !prompt)}
        variant={isGenerating ? "destructive" : "default"}
        size="sm"
        className="flex-1"
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

      {/* Restore last settings */}
      {!isGenerating && (
        <Button
          type="button"
          onClick={handleRestore}
          disabled={!lastResult}
          variant="secondary"
          size="icon-sm"
          title="Restore settings from last generation"
        >
          <History size={14} />
        </Button>
      )}

      {/* Skip button */}
      {isGenerating && (
        <Button
          type="button"
          onClick={handleSkip}
          variant="secondary"
          size="icon-sm"
          title="Skip current"
        >
          <SkipForward size={14} />
        </Button>
      )}

      {/* Progress indicator */}
      {isGenerating && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground tabular-nums">
          <Loader2 size={12} className="animate-spin" />
          {progressPct > 0 && <span>{progressPct}%</span>}
        </div>
      )}
    </div>
  );
});
