import { useCallback, useMemo, useRef, useState } from "react";
import { Download, Trash2, Film, Columns2, X, ImagePlus } from "lucide-react";
import { useVideoStore } from "@/stores/videoStore";
import { useVideoCanvasStore } from "@/stores/videoCanvasStore";
import { useJobQueueStore, selectVideoActive, selectFramepackActive, selectLtxActive, selectVideoProgress, selectFramepackProgress, selectLtxProgress, selectVideoDomainActiveJob } from "@/stores/jobStore";
import { useUiStore } from "@/stores/uiStore";
import { useVideoFrameLayout } from "@/canvas/useVideoFrameLayout";
import { VideoCanvasStage } from "@/canvas/VideoCanvasStage";
import { FrameHeader, INPUT_COLOR_ACTIVE, INPUT_COLOR_INACTIVE, OUTPUT_COLOR } from "@/canvas/ControlFramePanel";
import { VideoPlayer } from "@/components/video/VideoPlayer";
import { VideoCompare } from "@/components/video/VideoCompare";
import { VideoResultActions } from "@/components/video/VideoResultActions";
import { useDropTarget } from "@/hooks/useDropTarget";
import { payloadToFile } from "@/lib/sendTo";
import type { DragPayload } from "@/stores/dragStore";
import { Button } from "@/components/ui/button";
import { ParamSlider } from "@/components/generation/ParamSlider";
import { fileToBase64 } from "@/lib/image";
import { contrastText, cn } from "@/lib/utils";

const DOMAIN_LABELS: Record<string, string> = {
  video: "Models",
  framepack: "FP",
  ltx: "LTX",
};

export function VideoCanvasView() {
  const layout = useVideoFrameLayout();
  const viewport = useVideoCanvasStore((s) => s.viewport);
  const initFrame = useVideoCanvasStore((s) => s.initFrame);
  const lastFrame = useVideoCanvasStore((s) => s.lastFrame);
  const setFrame = useVideoCanvasStore((s) => s.setFrame);
  const clearFrame = useVideoCanvasStore((s) => s.clearFrame);
  const labelScale = useUiStore((s) => s.canvasLabelScale);

  const results = useVideoStore((s) => s.results);
  const selectedResultId = useVideoStore((s) => s.selectedResultId);
  const selectResult = useVideoStore((s) => s.selectResult);
  const clearResults = useVideoStore((s) => s.clearResults);
  const initStrength = useVideoStore((s) => s.initStrength);
  const videoWidth = useVideoStore((s) => s.width);
  const videoHeight = useVideoStore((s) => s.height);
  const setParam = useVideoStore((s) => s.setParam);
  const sizeText = `${videoWidth}\u00d7${videoHeight}`;

  const selectedResult = useMemo(() => results.find((r) => r.id === selectedResultId) ?? null, [results, selectedResultId]);

  const isVideoActive = useJobQueueStore(selectVideoActive);
  const isFramepackActive = useJobQueueStore(selectFramepackActive);
  const isLtxActive = useJobQueueStore(selectLtxActive);
  const isGenerating = isVideoActive || isFramepackActive || isLtxActive;
  const videoProgress = useJobQueueStore(selectVideoProgress);
  const fpProgress = useJobQueueStore(selectFramepackProgress);
  const ltxProgress = useJobQueueStore(selectLtxProgress);
  const progress = Math.max(videoProgress, fpProgress, ltxProgress);
  const progressPct = Math.round(progress * 100);

  const activeVideoJob = useJobQueueStore(selectVideoDomainActiveJob);
  const stepInfo = activeVideoJob ? { step: activeVideoJob.step, steps: activeVideoJob.steps, textinfo: activeVideoJob.textinfo } : null;

  // Compare mode
  const [compareMode, setCompareMode] = useState(false);
  const [compareIds, setCompareIds] = useState<[string | null, string | null]>([null, null]);

  const handleCompareToggle = useCallback(() => {
    if (compareMode) {
      setCompareMode(false);
      setCompareIds([null, null]);
    } else if (results.length >= 2) {
      setCompareMode(true);
      setCompareIds([results[0]?.id ?? null, results[1]?.id ?? null]);
    }
  }, [compareMode, results]);

  const handleCompareSelect = useCallback(
    (id: string) => {
      if (!compareMode) {
        selectResult(id);
        return;
      }
      setCompareIds((prev) => {
        if (prev[0] === id) return prev;
        if (prev[1] === id) return prev;
        return [prev[1], id];
      });
    },
    [compareMode, selectResult],
  );

  const compareLeft = useMemo(() => results.find((r) => r.id === compareIds[0]) ?? null, [results, compareIds]);
  const compareRight = useMemo(() => results.find((r) => r.id === compareIds[1]) ?? null, [results, compareIds]);

  // File input refs for click-to-pick
  const initInputRef = useRef<HTMLInputElement>(null);
  const lastInputRef = useRef<HTMLInputElement>(null);

  const handlePickImage = useCallback((which: "init" | "last") => {
    if (which === "init") initInputRef.current?.click();
    else lastInputRef.current?.click();
  }, []);

  const handleFileSelected = useCallback(async (which: "init" | "last", file: File) => {
    if (!file.type.startsWith("image/")) return;
    const base64 = await fileToBase64(file);
    const objectUrl = URL.createObjectURL(file);
    const img = new window.Image();
    img.src = objectUrl;
    await new Promise<void>((r) => { img.onload = () => r(); });
    setFrame(which, file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  }, [setFrame]);

  const handleInputChange = useCallback((which: "init" | "last") => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelected(which, file);
    e.target.value = "";
  }, [handleFileSelected]);

  // Hit-test: determine which video frame a drop lands on based on screen coords
  const hitTestTarget = useCallback((e: React.DragEvent): "init" | "last" => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const screenX = e.clientX - rect.left;
    const canvasX = (screenX - viewport.x) / viewport.scale;
    const { lastX, displayW } = layout;
    if (canvasX >= lastX && canvasX < lastX + displayW) return "last";
    return "init";
  }, [viewport, layout]);

  const handleDropFile = useCallback(async (file: File, e: React.DragEvent) => {
    await handleFileSelected(hitTestTarget(e), file);
  }, [handleFileSelected, hitTestTarget]);

  const { isOver, ...dropHandlers } = useDropTarget({
    onDropPayload: useCallback((payload: DragPayload, e: React.DragEvent) => {
      const target = hitTestTarget(e);
      payloadToFile(payload).then((f: File) => handleFileSelected(target, f)).catch(() => {});
    }, [hitTestTarget, handleFileSelected]),
    onFileDrop: handleDropFile,
  });

  // Paste handler
  const handlePaste = useCallback(async (e: React.ClipboardEvent) => {
    const item = Array.from(e.clipboardData.items).find((i) => i.type.startsWith("image/"));
    if (!item) return;
    const file = item.getAsFile();
    if (file) await handleFileSelected("init", file);
  }, [handleFileSelected]);

  // Compute output frame overlay position
  const { outputX, displayW, displayH } = layout;
  const outputScreenX = outputX * viewport.scale + viewport.x;
  const outputScreenY = viewport.y;
  const outputScreenW = displayW * viewport.scale;
  const outputScreenH = displayH * viewport.scale;
  const showVideoOverlay = outputScreenW > 0 && outputScreenH > 0;

  const initColor = initFrame ? INPUT_COLOR_ACTIVE : INPUT_COLOR_INACTIVE;
  const lastColor = lastFrame ? INPUT_COLOR_ACTIVE : INPUT_COLOR_INACTIVE;
  const initTextColor = contrastText(initColor);
  const lastTextColor = contrastText(lastColor);
  const outputTextColor = contrastText(OUTPUT_COLOR);

  return (
    <div
      className={cn("h-full flex flex-col", isOver && "ring-2 ring-primary ring-inset")}
      {...dropHandlers}
      onPaste={handlePaste}
      tabIndex={-1}
    >
      {/* Canvas + overlays */}
      <div className="flex-1 relative min-h-0">
        <VideoCanvasStage layout={layout} onPickImage={handlePickImage} />

        {/* Floating header: Init frame */}
        <FrameHeader
          mode="panel"
          color={initColor}
          label="Init"
          canvasX={layout.initX}
          frameW={displayW}
          viewport={viewport}
          labelScale={labelScale}
          actions={
            <>
              <Button variant="ghost" size="icon-xs" onClick={() => handlePickImage("init")} title="Add image" className="hover:bg-black/10">
                <ImagePlus size={16} style={{ color: initTextColor }} />
              </Button>
              {initFrame && (
                <Button variant="ghost" size="icon-xs" onClick={() => clearFrame("init")} title="Clear" className="hover:bg-black/10">
                  <Trash2 size={16} style={{ color: initTextColor }} />
                </Button>
              )}
            </>
          }
          drawer={
            <ParamSlider label="Strength" value={initStrength} onChange={(v) => setParam("initStrength", v)} min={0} max={1} step={0.05} />
          }
          collapsed={!initFrame}
          onToggleCollapsed={() => {/* drawer auto-shows when frame present */}}
        />

        {/* Floating header: Last frame */}
        <FrameHeader
          mode="panel"
          color={lastColor}
          label="Last"
          canvasX={layout.lastX}
          frameW={displayW}
          viewport={viewport}
          labelScale={labelScale}
          actions={
            <>
              <Button variant="ghost" size="icon-xs" onClick={() => handlePickImage("last")} title="Add image" className="hover:bg-black/10">
                <ImagePlus size={16} style={{ color: lastTextColor }} />
              </Button>
              {lastFrame && (
                <Button variant="ghost" size="icon-xs" onClick={() => clearFrame("last")} title="Clear" className="hover:bg-black/10">
                  <Trash2 size={16} style={{ color: lastTextColor }} />
                </Button>
              )}
            </>
          }
        />

        {/* Floating header: Output frame */}
        <FrameHeader
          mode="hat"
          color={OUTPUT_COLOR}
          label="Output"
          sizeText={sizeText}
          canvasX={outputX}
          frameW={displayW}
          viewport={viewport}
          labelScale={labelScale}
          actions={
            <>
              {selectedResult?.videoUrl && !isGenerating && (
                <>
                  <VideoResultActions result={selectedResult} />
                  <a href={selectedResult.videoUrl} download>
                    <Button variant="ghost" size="icon-xs" title="Download" className="hover:bg-black/10">
                      <Download size={16} style={{ color: outputTextColor }} />
                    </Button>
                  </a>
                </>
              )}
            </>
          }
        />

        {/* Video player overlay positioned at output frame */}
        {showVideoOverlay && (
          <div
            className="absolute overflow-hidden"
            style={{
              left: `${outputScreenX}px`,
              top: `${outputScreenY}px`,
              width: `${outputScreenW}px`,
              height: `${outputScreenH}px`,
              pointerEvents: (selectedResult?.videoUrl || (compareMode && compareLeft?.videoUrl)) ? "auto" : "none",
            }}
          >
            {compareMode && compareLeft?.videoUrl && compareRight?.videoUrl ? (
              <VideoCompare
                leftSrc={compareLeft.videoUrl}
                rightSrc={compareRight.videoUrl}
                leftLabel={DOMAIN_LABELS[compareLeft.domain] ?? compareLeft.domain}
                rightLabel={DOMAIN_LABELS[compareRight.domain] ?? compareRight.domain}
              />
            ) : !isGenerating && selectedResult?.videoUrl ? (
              <VideoPlayer src={selectedResult.videoUrl} />
            ) : null}
          </div>
        )}

        {/* Progress overlay during generation */}
        {isGenerating && (
          <div className="absolute inset-x-0 bottom-0 p-4 pointer-events-none">
            <div className="flex flex-col gap-1 bg-background/80 backdrop-blur-sm rounded-lg px-3 py-2">
              {stepInfo && (stepInfo.step > 0 || stepInfo.textinfo) && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  {stepInfo.steps > 0 && (
                    <span className="tabular-nums">Step {stepInfo.step}/{stepInfo.steps}</span>
                  )}
                  {stepInfo.textinfo && (
                    <>
                      <span className="text-muted-foreground/40">|</span>
                      <span className="truncate">{stepInfo.textinfo}</span>
                    </>
                  )}
                </div>
              )}
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-[width] duration-300"
                    style={{ width: `${progressPct}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground tabular-nums min-w-[3ch]">{progressPct}%</span>
              </div>
            </div>
          </div>
        )}

        {/* Hidden file inputs */}
        <input ref={initInputRef} type="file" accept="image/*" className="hidden" onChange={handleInputChange("init")} />
        <input ref={lastInputRef} type="file" accept="image/*" className="hidden" onChange={handleInputChange("last")} />
      </div>

      {/* Result strip */}
      {results.length > 0 && (
        <div className="flex-shrink-0 border-t border-border bg-muted/30 px-2 py-1.5">
          <div className="flex items-center gap-1.5">
            <div className="flex-1 flex items-center gap-1 overflow-x-auto scrollbar-thin">
              {results.map((r) => (
                <button
                  key={r.id}
                  onClick={() => handleCompareSelect(r.id)}
                  className={cn(
                    "flex-shrink-0 w-16 h-10 rounded border overflow-hidden relative group transition-all",
                    !compareMode && r.id === selectedResultId
                      ? "border-primary ring-1 ring-primary/30"
                      : compareMode && (r.id === compareIds[0] || r.id === compareIds[1])
                        ? "border-primary ring-1 ring-primary/30"
                        : "border-border hover:border-primary/40",
                  )}
                >
                  {r.thumbnailUrl ? (
                    <img src={r.thumbnailUrl} alt="" className="w-full h-full object-cover" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-muted">
                      <Film size={12} className="text-muted-foreground/50" />
                    </div>
                  )}
                  <div className="absolute bottom-0 left-0 right-0 px-0.5 bg-black/60">
                    <span className="text-5xs text-white/80 font-medium">{DOMAIN_LABELS[r.domain] ?? r.domain}</span>
                  </div>
                  {compareMode && r.id === compareIds[0] && (
                    <div className="absolute top-0 left-0 px-1 bg-primary text-primary-foreground text-5xs font-bold rounded-br">A</div>
                  )}
                  {compareMode && r.id === compareIds[1] && (
                    <div className="absolute top-0 left-0 px-1 bg-primary text-primary-foreground text-5xs font-bold rounded-br">B</div>
                  )}
                </button>
              ))}
            </div>
            {results.length >= 2 && (
              <Button
                variant={compareMode ? "default" : "ghost"}
                size="icon-sm"
                onClick={handleCompareToggle}
                title={compareMode ? "Exit compare" : "Compare two results"}
              >
                {compareMode ? <X size={12} /> : <Columns2 size={12} />}
              </Button>
            )}
            <Button variant="ghost" size="icon-sm" onClick={clearResults} title="Clear history">
              <Trash2 size={12} />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
