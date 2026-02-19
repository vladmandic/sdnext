import { useCallback, useEffect } from "react";
import { Upload, X, Download, Loader2 } from "lucide-react";
import { useProcessStore } from "@/stores/processStore";
import { Button } from "@/components/ui/button";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";

export function ProcessView() {
  const imagePreviewUrl = useProcessStore((s) => s.imagePreviewUrl);
  const isProcessing = useProcessStore((s) => s.isProcessing);
  const resultImageUrl = useProcessStore((s) => s.resultImageUrl);
  const resultWidth = useProcessStore((s) => s.resultWidth);
  const resultHeight = useProcessStore((s) => s.resultHeight);
  const setImage = useProcessStore((s) => s.setImage);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file?.type.startsWith("image/")) setImage(file);
  }, [setImage]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] ?? null;
    if (file) setImage(file);
    e.target.value = "";
  }, [setImage]);

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

  return (
    <ResizablePanelGroup orientation="horizontal" className="h-full">
      <ResizablePanel defaultSize={50} minSize={30}>
        <div className="relative h-full group" onDrop={handleDrop} onDragOver={handleDragOver}>
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
