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
import { Play, Square, SkipForward, History, FileSearch, ChevronDown } from "lucide-react";
import { ProgressRing } from "@/components/ui/progress-ring";
import { formatDuration } from "@/lib/utils";
import { useState, useCallback, useMemo, memo } from "react";
import { toast } from "sonner";
import { useShortcut } from "@/hooks/useShortcut";
import { Button } from "@/components/ui/button";
import { PngInfoDialog } from "@/components/generation/PngInfoDialog";
import { BatchDialog } from "@/components/generation/BatchDialog";
import { GenerationDiffDialog } from "@/components/generation/GenerationDiffDialog";

export const ActionBar = memo(function ActionBar() {
  const prompt = useGenerationStore((s) => s.prompt);
  const clearSelection = useGenerationStore((s) => s.clearSelection);
  const lastResult = useGenerationStore((s) => s.results[0]);
  const hasLayers = useCanvasStore((s) => s.layers.length > 0);

  const isActive = useJobQueueStore(selectDomainActive("generate"));
  const runningJob = useJobQueueStore(selectRunningJob);
  const pendingCount = useJobQueueStore(selectPendingCount);

  const [pngInfoOpen, setPngInfoOpen] = useState(false);
  const [batchOpen, setBatchOpen] = useState(false);
  const [diffOpen, setDiffOpen] = useState(false);
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

  const handleHistoryClick = useCallback((e: React.MouseEvent) => {
    if (!lastResult) return;
    if (e.shiftKey) {
      setDiffOpen(true);
    } else {
      restoreFromResult(lastResult);
      toast.success("Settings restored from last generation");
    }
  }, [lastResult]);

  const progressPct = Math.round(progress * 100);
  const eta = runningJob?.domain === "generate" ? runningJob.eta ?? 0 : 0;
  const canSubmit = hasLayers || !!prompt;
  const phase = runningJob?.domain === "generate" ? runningJob.task : "";
  const phaseLabel = phase || "Generating";

  // Global keyboard shortcuts for generation
  useShortcut("generate", () => { if (canSubmit && !isSubmitting) submit(); });
  useShortcut("skip", handleSkip);

  return (
    <div className="flex items-center gap-2">
      {/* Generate button group */}
      <div className="flex flex-1 min-w-0">
        <Button
          type="button"
          onClick={submit}
          disabled={!canSubmit || isSubmitting}
          variant="default"
          size="sm"
          className="flex-1 rounded-r-none"
        >
          {isGenerating ? <ProgressRing progress={progress} size={14} strokeWidth={2} /> : <Play size={14} />}
          {isGenerating
            ? `${phaseLabel}${progressPct > 0 ? ` ${progressPct}%` : ""}${pendingCount > 0 ? ` [+${pendingCount}]` : ""}`
            : `Generate${pendingCount > 0 ? ` [${pendingCount}]` : ""}`}
        </Button>
        {!isGenerating && (
          <Button
            type="button"
            variant="default"
            size="sm"
            className="px-1.5 rounded-l-none border-l border-primary-foreground/20"
            onClick={() => setBatchOpen(true)}
            disabled={!canSubmit}
            title="Batch generation"
          >
            <ChevronDown size={14} />
          </Button>
        )}
      </div>
      <BatchDialog open={batchOpen} onOpenChange={setBatchOpen} buildRequest={buildRequest} />

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
            onClick={handleHistoryClick}
            disabled={!lastResult}
            variant="secondary"
            size="icon-sm"
            title="Restore settings (Shift+click to compare)"
          >
            <History size={14} />
          </Button>
          <GenerationDiffDialog open={diffOpen} onOpenChange={setDiffOpen} result={lastResult ?? null} />
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
          <ProgressRing progress={progress} size={16} strokeWidth={2} />
          {progressPct > 0 && <span>{progressPct}%</span>}
          {eta > 0 && <span className="text-3xs text-muted-foreground/70">~{formatDuration(eta)}</span>}
        </div>
      )}
    </div>
  );
});
