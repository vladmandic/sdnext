import { useCallback, useEffect } from "react";
import { Upload, X, Download, Loader2, GitCompareArrows, Maximize2 } from "lucide-react";
import { useProcessStore } from "@/stores/processStore";
import { useComparisonStore } from "@/stores/comparisonStore";
import { useJobQueueStore, selectDomainActive } from "@/stores/jobStore";
import { useDropTarget } from "@/hooks/useDropTarget";
import { payloadToFile } from "@/lib/sendTo";
import type { DragPayload } from "@/stores/dragStore";
import { Button } from "@/components/ui/button";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { SwipeMode } from "@/components/comparison/SwipeMode";

export function ProcessView() {
  const imagePreviewUrl = useProcessStore((s) => s.imagePreviewUrl);
  const isProcessing = useJobQueueStore(selectDomainActive("upscale"));
  const resultImageUrl = useProcessStore((s) => s.resultImageUrl);
  const resultWidth = useProcessStore((s) => s.resultWidth);
  const resultHeight = useProcessStore((s) => s.resultHeight);
  const setImage = useProcessStore((s) => s.setImage);
  const compareMode = useProcessStore((s) => s.compareMode);
  const setCompareMode = useProcessStore((s) => s.setCompareMode);

  const canCompare = !!imagePreviewUrl && !!resultImageUrl;

  const dropTarget = useDropTarget({
    onDropPayload: useCallback((payload: DragPayload) => { payloadToFile(payload).then((f: File) => setImage(f)).catch(() => {}); }, [setImage]),
    onFileDrop: useCallback((file: File) => setImage(file), [setImage]),
  });

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (file) setImage(file);
    e.target.value = "";
  }, [setImage]);

  const handleFullscreen = useCallback(() => {
    if (!imagePreviewUrl || !resultImageUrl) return;
    useComparisonStore.getState().openComparison(
      { src: imagePreviewUrl, label: "Original" },
      { src: resultImageUrl, label: "Upscaled" },
    );
  }, [imagePreviewUrl, resultImageUrl]);

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable) return;
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) setImage(file);
          break;
        }
      }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [setImage]);

  // Inline swipe comparison mode
  if (compareMode && canCompare) {
    return (
      <div className="h-full flex flex-col">
        <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border flex-shrink-0">
          <Button variant="secondary" size="sm" onClick={() => setCompareMode(false)}>
            Exit Compare
          </Button>
          <Button variant="ghost" size="icon-sm" onClick={handleFullscreen} title="Fullscreen comparison">
            <Maximize2 size={14} />
          </Button>
        </div>
        <div className="flex-1 min-h-0">
          <SwipeMode
            imageA={{ src: imagePreviewUrl!, label: "Original" }}
            imageB={{ src: resultImageUrl!, label: "Upscaled" }}
          />
        </div>
      </div>
    );
  }

  return (
    <ResizablePanelGroup orientation="horizontal" className="h-full">
      <ResizablePanel defaultSize={50} minSize={30}>
        <div className={`relative h-full group${dropTarget.isOver ? " ring-2 ring-primary ring-inset" : ""}`} {...dropTarget}>
          {imagePreviewUrl ? (
            <>
              <img src={imagePreviewUrl} alt="Input" className="w-full h-full object-contain" />
              {isProcessing && (
                <div className="absolute inset-0 bg-background/60 flex items-center justify-center">
                  <Loader2 size={32} className="animate-spin text-primary" />
                </div>
              )}
              <Button
                variant="destructive"
                size="icon-sm"
                className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={() => setImage(null)}
              >
                <X size={14} />
              </Button>
            </>
          ) : (
            <label className="flex flex-col items-center justify-center h-full cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
              <Upload size={48} className="mb-3 opacity-40" />
              <p className="text-sm font-medium">Drop Image Here</p>
              <p className="text-xs mt-1 opacity-60">or click to browse, or paste from clipboard</p>
              <input type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
            </label>
          )}
        </div>
      </ResizablePanel>

      <ResizableHandle />

      <ResizablePanel defaultSize={50} minSize={30}>
        <div className="h-full flex items-center justify-center">
          {resultImageUrl ? (
            <div className="relative h-full w-full group">
              <img src={resultImageUrl} alt="Result" className="w-full h-full object-contain" />
              <div className="absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity flex gap-2">
                {canCompare && (
                  <Button variant="secondary" size="sm" onClick={() => setCompareMode(true)}>
                    <GitCompareArrows size={14} />
                    Compare
                  </Button>
                )}
                {resultWidth && resultHeight && (
                  <span className="text-xs bg-background/80 px-2 py-1 rounded text-foreground">{resultWidth} x {resultHeight}</span>
                )}
                <a href={resultImageUrl} download className="inline-flex">
                  <Button variant="secondary" size="icon-sm">
                    <Download size={14} />
                  </Button>
                </a>
              </div>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground opacity-50">Result will appear here</p>
          )}
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
