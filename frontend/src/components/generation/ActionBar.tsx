import { useGenerationStore } from "@/stores/generationStore";
import { useTxt2Img, useProgress, useInterrupt, useSkip } from "@/api/hooks/useGeneration";
import { buildTxt2ImgRequest } from "@/lib/requestBuilder";
import { Play, Square, SkipForward, Loader2 } from "lucide-react";
import { useEffect, useRef, useCallback, memo } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api } from "@/api/client";
import { WebSocketManager } from "@/api/websocket";

export const ActionBar = memo(function ActionBar() {
  const isGenerating = useGenerationStore((s) => s.isGenerating);
  const prompt = useGenerationStore((s) => s.prompt);
  const progress = useGenerationStore((s) => s.progress);
  const setGenerating = useGenerationStore((s) => s.setGenerating);
  const setProgress = useGenerationStore((s) => s.setProgress);
  const setPreview = useGenerationStore((s) => s.setPreview);
  const addResult = useGenerationStore((s) => s.addResult);
  const txt2img = useTxt2Img();
  const interrupt = useInterrupt();
  const skip = useSkip();
  const generatingRef = useRef(false);
  const wsConnected = useRef(false);

  // WebSocket connection for live previews (preferred)
  useEffect(() => {
    const wsUrl = api.getWebSocketUrl("/sdapi/v1/ws");
    const ws = new WebSocketManager(wsUrl);

    ws.on("open", () => { wsConnected.current = true; });
    ws.on("close", () => { wsConnected.current = false; });

    ws.on("message", (data) => {
      const msg = data as { type: string; data?: Record<string, unknown> };
      if (msg.type === "progress" && msg.data) {
        const d = msg.data;
        if (generatingRef.current && typeof d.progress === "number") {
          setProgress(d.progress, typeof d.eta === "number" ? d.eta : 0);
        }
      }
    });

    ws.on("binary", (buf) => {
      if (generatingRef.current) {
        const blob = new Blob([buf], { type: "image/jpeg" });
        const url = URL.createObjectURL(blob);
        setPreview(url);
      }
    });

    ws.connect();
    return () => ws.disconnect();
  }, [setProgress, setPreview]);

  // REST polling fallback when WebSocket is unavailable
  const { data: progressData } = useProgress(isGenerating && !wsConnected.current);

  useEffect(() => {
    if (progressData && generatingRef.current && !wsConnected.current) {
      setProgress(progressData.progress, progressData.eta_relative);
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
    try {
      const request = await buildTxt2ImgRequest();
      const result = await txt2img.mutateAsync(request);
      addResult({
        id: crypto.randomUUID(),
        images: result.images,
        parameters: result.parameters,
        info: result.info,
        timestamp: Date.now(),
      });
    } catch (err) {
      toast.error("Generation failed", { description: err instanceof Error ? err.message : String(err) });
    } finally {
      generatingRef.current = false;
      setGenerating(false);
    }
  }, [txt2img, setGenerating, setPreview, addResult]);

  const handleInterrupt = useCallback(() => {
    interrupt.mutate();
    generatingRef.current = false;
    setGenerating(false);
  }, [interrupt, setGenerating]);

  const handleSkip = useCallback(() => {
    skip.mutate();
  }, [skip]);

  const progressPct = Math.round(progress * 100);

  return (
    <div className="flex items-center gap-2">
      {/* Generate / Stop button */}
      <Button
        type="button"
        onClick={isGenerating ? handleInterrupt : handleGenerate}
        disabled={!prompt && !isGenerating}
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
