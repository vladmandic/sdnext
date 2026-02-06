import { useGenerationStore } from "@/stores/generationStore";
import { useTxt2Img, useProgress, useInterrupt, useSkip } from "@/api/hooks/useGeneration";
import { buildTxt2ImgRequest } from "@/lib/requestBuilder";
import { Play, Square, SkipForward, Loader2 } from "lucide-react";
import { useEffect } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";

export function ActionBar() {
  const store = useGenerationStore();
  const isGenerating = useGenerationStore((s) => s.isGenerating);
  const setProgress = useGenerationStore((s) => s.setProgress);
  const setPreview = useGenerationStore((s) => s.setPreview);
  const txt2img = useTxt2Img();
  const interrupt = useInterrupt();
  const skip = useSkip();
  const { data: progressData } = useProgress(isGenerating);

  useEffect(() => {
    if (progressData && isGenerating) {
      setProgress(progressData.progress, progressData.eta_relative);
      if (progressData.current_image) {
        setPreview(`data:image/jpeg;base64,${progressData.current_image}`);
      }
    }
  }, [progressData, isGenerating, setProgress, setPreview]);

  async function handleGenerate() {
    store.setGenerating(true);
    store.setPreview(null);
    try {
      const request = await buildTxt2ImgRequest();
      const result = await txt2img.mutateAsync(request);
      store.addResult({
        id: crypto.randomUUID(),
        images: result.images,
        parameters: result.parameters,
        info: result.info,
        timestamp: Date.now(),
      });
    } catch (err) {
      toast.error("Generation failed", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      store.setGenerating(false);
    }
  }

  function handleInterrupt() {
    interrupt.mutate();
    store.setGenerating(false);
  }

  function handleSkip() {
    skip.mutate();
  }

  const progressPct = Math.round(store.progress * 100);

  return (
    <div className="flex items-center gap-2">
      {/* Generate / Stop button */}
      <Button
        onClick={store.isGenerating ? handleInterrupt : handleGenerate}
        disabled={!store.prompt && !store.isGenerating}
        variant={store.isGenerating ? "destructive" : "default"}
        size="sm"
        className="flex-1"
      >
        {store.isGenerating ? (
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

      {/* Skip button */}
      {store.isGenerating && (
        <Button
          onClick={handleSkip}
          variant="secondary"
          size="icon-sm"
          title="Skip current"
        >
          <SkipForward size={14} />
        </Button>
      )}

      {/* Progress indicator */}
      {store.isGenerating && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground tabular-nums">
          <Loader2 size={12} className="animate-spin" />
          <span>{progressPct}%</span>
        </div>
      )}
    </div>
  );
}
