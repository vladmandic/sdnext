import { useGenerationStore } from "@/stores/generationStore";
import { useJobQueueStore, selectRunningJob, selectDomainActive, selectPendingCount } from "@/stores/jobStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { buildControlRequest, restoreFromResult } from "@/lib/requestBuilder";
import { blobToBase64 } from "@/lib/image";
import { snapshotUnits } from "@/stores/controlStore";
import { useSubmitToQueue } from "@/hooks/useSubmitToQueue";
import { sendToJob } from "@/hooks/useJobTracker";
import { useCancelJob } from "@/api/hooks/useJobs";
import { Play, Square, SkipForward, Loader2, History, FileSearch } from "lucide-react";
import { useState, useCallback, useMemo, memo } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { PngInfoDialog } from "@/components/generation/PngInfoDialog";

export const ActionBar = memo(function ActionBar() {
  const prompt = useGenerationStore((s) => s.prompt);
  const clearSelection = useGenerationStore((s) => s.clearSelection);
  const lastResult = useGenerationStore((s) => s.results[0]);
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);

  const isActive = useJobQueueStore(selectDomainActive("generate"));
  const runningJob = useJobQueueStore(selectRunningJob);
  const pendingCount = useJobQueueStore(selectPendingCount);

  const [pngInfoOpen, setPngInfoOpen] = useState(false);
  const cancelJob = useCancelJob();

  const buildRequest = useCallback(async () => {
    const isImg2Img = useCanvasStore.getState().layers.length > 0;
    const { request, inputBlob } = await buildControlRequest();
    const inputImage = isImg2Img && inputBlob ? await blobToBase64(inputBlob) : undefined;
    const maskLines = useImg2ImgStore.getState().maskLines;
    const inputMask = isImg2Img && maskLines.length > 0 ? maskLines.slice() : undefined;
    const controlUnits = await snapshotUnits();
    clearSelection();
    return {
      payload: { type: "generate" as const, ...request },
      snapshot: { inputImage, inputMask, controlUnits },
    };
  }, [clearSelection]);

  const { submit, isSubmitting } = useSubmitToQueue(useMemo(() => ({ domain: "generate" as const, buildRequest }), [buildRequest]));

  const isGenerating = isActive || isSubmitting;
  const runningGenJob = useJobQueueStore(useMemo(() => selectDomainActive("generate"), []));
  const progress = runningJob?.domain === "generate" ? runningJob.progress : 0;

  const handleInterrupt = useCallback(() => {
    if (runningJob && runningJob.domain === "generate") {
      sendToJob(runningJob.id, { type: "interrupt" });
      cancelJob.mutate(runningJob.id);
    }
  }, [runningJob, cancelJob]);

  const handleSkip = useCallback(() => {
    if (runningJob && runningJob.domain === "generate") {
      sendToJob(runningJob.id, { type: "skip" });
    }
  }, [runningJob]);

  const handleRestore = useCallback(() => {
    if (!lastResult) return;
    restoreFromResult(lastResult);
    toast.success("Settings restored from last generation");
  }, [lastResult]);

  const progressPct = Math.round(progress * 100);
  const canSubmit = hasLayers || !!prompt;

  return (
    <div className="flex items-center gap-2">
      {/* Generate button */}
      <Button
        type="button"
        onClick={submit}
        disabled={!canSubmit || isSubmitting}
        variant="default"
        size="sm"
        className="flex-1"
      >
        <Play size={14} />
        Generate{pendingCount > 0 ? ` [${pendingCount}]` : ""}
      </Button>

      {/* Stop button */}
      {isGenerating && (
        <Button
          type="button"
          onClick={handleInterrupt}
          variant="destructive"
          size="icon-sm"
          title="Stop generation"
        >
          <Square size={14} />
        </Button>
      )}

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
      {runningGenJob && (
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
