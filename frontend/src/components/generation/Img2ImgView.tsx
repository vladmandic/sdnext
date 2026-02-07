import { useCallback, memo } from "react";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useCanvasStore } from "@/stores/canvasStore";
import { CanvasStage } from "@/canvas/CanvasStage";
import { CanvasToolbar } from "@/canvas/CanvasToolbar";
import { Upload, RotateCcw, X } from "lucide-react";
import { Button } from "@/components/ui/button";

function extractImageFile(e: React.DragEvent | ClipboardEvent): File | null {
  if (e instanceof DragEvent || "dataTransfer" in e) {
    const dt = (e as React.DragEvent).dataTransfer;
    if (dt?.files?.[0]?.type.startsWith("image/")) return dt.files[0];
  }
  if ("clipboardData" in e) {
    const items = (e as ClipboardEvent).clipboardData?.items;
    if (items) {
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) return file;
        }
      }
    }
  }
  return null;
}

export const Img2ImgView = memo(function Img2ImgView() {
  const initImageData = useImg2ImgStore((s) => s.initImageData);
  const setInitImage = useImg2ImgStore((s) => s.setInitImage);
  const clearInitImage = useImg2ImgStore((s) => s.clearInitImage);
  const setViewport = useCanvasStore((s) => s.setViewport);
  const handleFile = useCallback((file: File) => {
    if (file.type.startsWith("image/")) {
      setInitImage(file);
    }
  }, [setInitImage]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = extractImageFile(e);
    if (file) handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }, [handleFile]);

  const handleResetZoom = useCallback(() => {
    // Reset triggers re-fit via CanvasStage's fitToView effect
    setViewport({ x: 0, y: 0, scale: 1 });
    // Force a re-fit by toggling — the CanvasStage effect will recalculate
    setTimeout(() => setViewport({ x: 0, y: 0, scale: 0.999 }), 0);
  }, [setViewport]);

  const handleClear = useCallback(() => {
    clearInitImage();
  }, [clearInitImage]);

  // Paste handler
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          handleFile(file);
          break;
        }
      }
    }
  }, [handleFile]);

  // No image loaded — show drop zone
  if (!initImageData) {
    return (
      <div
        className="flex items-center justify-center h-full p-8"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onPaste={handlePaste}
        tabIndex={0}
      >
        <label className="flex flex-col items-center justify-center gap-3 w-full max-w-md aspect-[4/3] rounded-xl border-2 border-dashed border-muted-foreground/30 hover:border-muted-foreground/50 transition-colors cursor-pointer text-muted-foreground">
          <Upload size={32} strokeWidth={1.5} />
          <div className="text-center">
            <p className="text-sm font-medium">Drop an image here or click to browse</p>
            <p className="text-xs mt-1 opacity-60">Supports PNG, JPEG, WebP</p>
          </div>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="sr-only"
          />
        </label>
      </div>
    );
  }

  // Image loaded — show canvas
  return (
    <div
      className="relative w-full h-full"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onPaste={handlePaste}
      tabIndex={0}
    >
      <CanvasStage />

      {/* Top-right utility buttons */}
      <div className="absolute top-2 right-2 flex items-center gap-1">
        <Button
          variant="secondary"
          size="icon-xs"
          onClick={handleResetZoom}
          title="Reset zoom"
          className="bg-background/80 backdrop-blur-sm"
        >
          <RotateCcw size={12} />
        </Button>
        <Button
          variant="secondary"
          size="icon-xs"
          onClick={handleClear}
          title="Clear image"
          className="bg-background/80 backdrop-blur-sm"
        >
          <X size={12} />
        </Button>
      </div>

      {/* Canvas toolbar for mask painting */}
      <CanvasToolbar />
    </div>
  );
});
