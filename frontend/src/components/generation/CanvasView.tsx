import { useCallback, useRef, memo } from "react";
import { useCanvasStore } from "@/stores/canvasStore";
import { useControlStore } from "@/stores/controlStore";
import { useImg2ImgStore } from "@/stores/img2imgStore";
import { useUiStore } from "@/stores/uiStore";
import { fileToBase64 } from "@/lib/image";
import { CanvasStage } from "@/canvas/CanvasStage";
import { CanvasToolbar } from "@/canvas/CanvasToolbar";
import { ControlFramePanel } from "@/canvas/ControlFramePanel";
import { useControlFrameLayout } from "@/canvas/useControlFrameLayout";
import { RotateCcw, X, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export const CanvasView = memo(function CanvasView() {
  const setViewport = useCanvasStore((s) => s.setViewport);
  const addImageLayer = useCanvasStore((s) => s.addImageLayer);
  const clearLayers = useCanvasStore((s) => s.clearLayers);
  const clearMask = useImg2ImgStore((s) => s.clearMask);
  const viewport = useCanvasStore((s) => s.viewport);
  const setUnitImage = useControlStore((s) => s.setUnitImage);
  const generationMode = useUiStore((s) => s.generationMode);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const layout = useControlFrameLayout();
  const isImg2Img = generationMode === "img2img";

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const base64 = await fileToBase64(file);
    const objectUrl = URL.createObjectURL(file);
    const img = new window.Image();
    img.src = objectUrl;
    await new Promise<void>((r) => { img.onload = () => r(); });
    addImageLayer(file, base64, objectUrl, img.naturalWidth, img.naturalHeight);
  }, [addImageLayer]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const dt = e.dataTransfer;
    if (!dt?.files) return;

    // Hit-test: convert screen coords to canvas coords
    const container = containerRef.current;
    if (container) {
      const rect = container.getBoundingClientRect();
      const screenX = e.clientX - rect.left;
      const screenY = e.clientY - rect.top;
      const canvasX = (screenX - viewport.x) / viewport.scale;
      const canvasY = (screenY - viewport.y) / viewport.scale;

      // Check if drop hits a control frame
      for (const frame of layout.controlFrames) {
        if (canvasX >= frame.x && canvasX <= frame.x + frame.width && canvasY >= frame.y && canvasY <= frame.y + frame.height) {
          const file = dt.files[0];
          if (file?.type.startsWith("image/")) {
            setUnitImage(frame.unitIndex, file);
          }
          return;
        }
      }
    }

    // Fall through to canvas image layer (img2img only)
    if (isImg2Img) {
      for (const file of dt.files) {
        if (file.type.startsWith("image/")) handleFile(file);
      }
    }
  }, [handleFile, isImg2Img, viewport, layout.controlFrames, setUnitImage]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files) {
      for (const file of files) handleFile(file);
    }
    e.target.value = "";
  }, [handleFile]);

  const handleResetZoom = useCallback(() => {
    setViewport({ x: 0, y: 0, scale: 1 });
    setTimeout(() => setViewport({ x: 0, y: 0, scale: 0.999 }), 0);
  }, [setViewport]);

  const handleClearAll = useCallback(() => {
    clearLayers();
    clearMask();
  }, [clearLayers, clearMask]);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    if (!isImg2Img) return;
    const items = e.clipboardData.items;
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) handleFile(file);
      }
    }
  }, [handleFile, isImg2Img]);

  const openPicker = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full"
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onPaste={handlePaste}
      tabIndex={0}
    >
      <CanvasStage layout={layout} />

      {/* Top-right utility buttons */}
      <div className="absolute top-2 right-2 flex items-center gap-1">
        {isImg2Img && (
          <Button
            variant="secondary"
            size="icon-xs"
            onClick={openPicker}
            title="Add image"
            className="bg-background/80 backdrop-blur-sm"
          >
            <Plus size={12} />
          </Button>
        )}
        <Button
          variant="secondary"
          size="icon-xs"
          onClick={handleResetZoom}
          title="Reset zoom"
          className="bg-background/80 backdrop-blur-sm"
        >
          <RotateCcw size={12} />
        </Button>
        {isImg2Img && (
          <Button
            variant="secondary"
            size="icon-xs"
            onClick={handleClearAll}
            title="Clear all"
            className="bg-background/80 backdrop-blur-sm"
          >
            <X size={12} />
          </Button>
        )}
      </div>

      {/* Canvas toolbar for mask painting — img2img only */}
      {isImg2Img && <CanvasToolbar />}

      {/* Floating control panel */}
      <ControlFramePanel layout={layout} />

      {isImg2Img && (
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          onChange={handleFileInput}
          className="hidden"
        />
      )}
    </div>
  );
});
