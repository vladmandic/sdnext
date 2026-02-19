import { useGenerationStore } from "@/stores/generationStore";
import { useUiStore } from "@/stores/uiStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { buildControlRequest } from "@/lib/requestBuilder";
import { restoreFromResult } from "@/lib/requestBuilder";
import { snapshotUnits } from "@/stores/controlStore";
import type { ControlRequest } from "@/api/types/generation";
import { useSubmitJob, useCancelJob } from "@/api/hooks/useJobs";
import { useJobWebSocket } from "@/api/hooks/useJobWebSocket";
import { Play, Square, SkipForward, Loader2, History, FileSearch } from "lucide-react";
import { useState, useEffect, useRef, useCallback, memo } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { PngInfoDialog } from "@/components/generation/PngInfoDialog";

export const ActionBar = memo(function ActionBar() {
  const isGenerating = useGenerationStore((s) => s.isGenerating);
  const prompt = useGenerationStore((s) => s.prompt);
  const progress = useGenerationStore((s) => s.progress);
  const setGenerating = useGenerationStore((s) => s.setGenerating);
  const setProgress = useGenerationStore((s) => s.setProgress);
  const setPreview = useGenerationStore((s) => s.setPreview);
  const addResult = useGenerationStore((s) => s.addResult);
  const clearSelection = useGenerationStore((s) => s.clearSelection);
  const setTaskId = useGenerationStore((s) => s.setTaskId);
  const currentJobId = useGenerationStore((s) => s.currentTaskId);
  const lastResult = useGenerationStore((s) => s.results[0]);
  const generationMode = useUiStore((s) => s.generationMode);
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);

  const [pngInfoOpen, setPngInfoOpen] = useState(false);
  const submitJob = useSubmitJob();
  const cancelJob = useCancelJob();
  const { progress: wsProgress, preview, result: wsResult, status: wsStatus, error: wsError, send } = useJobWebSocket(currentJobId);

  // Snapshot refs: capture input state at generation time so it survives async completion
  const snapshotRef = useRef<{
    inputImage?: string;
    inputMask?: import("@/stores/img2imgStore").MaskLine[];
    controlUnits?: import("@/api/types/control").ControlUnitSnapshot[];
  }>({});

  // Update progress from per-job WebSocket
  useEffect(() => {
    if (currentJobId && wsProgress.progress > 0) {
      setProgress(wsProgress.progress, wsProgress.eta ?? 0, wsProgress.step, wsProgress.steps);
    }
  }, [currentJobId, wsProgress, setProgress]);

  // Update preview from per-job WebSocket binary frames
  useEffect(() => {
    if (currentJobId && preview) {
      setPreview(preview);
    }
  }, [currentJobId, preview, setPreview]);

  // Handle terminal job states
  useEffect(() => {
    if (!currentJobId) return;

    if (wsStatus === "completed" && wsResult) {
      // Update processed previews if present
      if (wsResult.images.length > 0) {
        addResult({
          id: crypto.randomUUID(),
          images: wsResult.images.map((img) => img.url),
          parameters: wsResult.params,
          info: JSON.stringify(wsResult.info),
          timestamp: Date.now(),
          inputImage: snapshotRef.current.inputImage,
          inputMask: snapshotRef.current.inputMask,
          controlUnits: snapshotRef.current.controlUnits,
        });
      }
      snapshotRef.current = {};
      setTaskId(null);
      setGenerating(false);
    } else if (wsStatus === "failed") {
      toast.error("Generation failed", { description: wsError ?? "Unknown error" });
      snapshotRef.current = {};
      setTaskId(null);
      setGenerating(false);
    } else if (wsStatus === "cancelled") {
      snapshotRef.current = {};
      setTaskId(null);
      setGenerating(false);
    }
  }, [wsStatus, wsResult, wsError, currentJobId, addResult, setTaskId, setGenerating]);

  const handleGenerate = useCallback(async () => {
    if (isGenerating) return;
    setGenerating(true);
    setPreview(null);
    clearSelection();
    try {
      const isImg2Img = generationMode === "img2img";
      const request = await buildControlRequest();

      // Snapshot input state before the async call
      const inputImage = isImg2Img ? (request as ControlRequest).inputs?.[0] : undefined;
      const maskLines = useImg2ImgStore.getState().maskLines;
      const inputMask = isImg2Img && maskLines.length > 0 ? maskLines.slice() : undefined;
      const controlUnits = await snapshotUnits();
      snapshotRef.current = { inputImage, inputMask, controlUnits };

      const job = await submitJob.mutateAsync({ type: "generate", ...request });
      setTaskId(job.id);
    } catch (err) {
      toast.error("Generation failed", { description: err instanceof Error ? err.message : String(err) });
      snapshotRef.current = {};
      setGenerating(false);
    }
  }, [isGenerating, generationMode, setGenerating, setPreview, clearSelection, submitJob, setTaskId]);

  const handleInterrupt = useCallback(() => {
    send({ type: "interrupt" });
    if (currentJobId) cancelJob.mutate(currentJobId);
    snapshotRef.current = {};
    setTaskId(null);
    setGenerating(false);
  }, [send, currentJobId, cancelJob, setTaskId, setGenerating]);

  const handleSkip = useCallback(() => {
    send({ type: "skip" });
  }, [send]);

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
        <>
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
          <Button
            type="button"
            onClick={() => setPngInfoOpen(true)}
            variant="secondary"
            size="icon-sm"
            title="Extract PNG info"
          >
            <FileSearch size={14} />
          </Button>
          <PngInfoDialog open={pngInfoOpen} onOpenChange={setPngInfoOpen} />
        </>
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
