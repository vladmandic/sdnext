import { useCallback, useEffect } from "react";
import { Upload, X, Loader2, ImageOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useCaptionStore } from "@/stores/captionStore";

export function CaptionView() {
  const image = useCaptionStore((s) => s.image);
  const imagePreviewUrl = useCaptionStore((s) => s.imagePreviewUrl);
  const isProcessing = useCaptionStore((s) => s.isProcessing);
  const result = useCaptionStore((s) => s.result);
  const method = useCaptionStore((s) => s.method);
  const setImage = useCaptionStore((s) => s.setImage);

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

  // Global paste listener
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

  const answerText = result?.type === "vqa" ? (result.answer ?? "") : result?.type === "interrogate" ? (result.caption ?? "") : "";
  const hasAnalysis = result?.type === "interrogate" && (result.medium || result.artist || result.movement || result.trending || result.flavor);

  return (
    <div className="flex h-full gap-0">
      {/* Input Image */}
      <div className="flex-1 flex flex-col border-r border-border min-w-0">
        <div className="px-4 py-2 border-b border-border">
          <h3 className="text-sm font-medium">Input Image</h3>
        </div>
        <div className="flex-1 relative" onDrop={handleDrop} onDragOver={handleDragOver}>
          {!image || !imagePreviewUrl ? (
            <label className="flex flex-col items-center justify-center h-full cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
              <Upload size={48} className="mb-3 opacity-40" />
              <p className="text-sm font-medium">Drop Image Here</p>
              <p className="text-xs mt-1 opacity-60">or click to browse, or paste from clipboard</p>
              <input type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
            </label>
          ) : (
            <div className="relative flex items-center justify-center h-full p-4 group">
              <img
                src={imagePreviewUrl}
                alt="Image to caption"
                className="max-w-full max-h-full object-contain rounded-lg"
              />
              <Button
                variant="destructive"
                size="icon-sm"
                className="absolute top-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={() => setImage(null)}
              >
                <X size={14} />
              </Button>
              {isProcessing && (
                <div className="absolute inset-0 flex items-center justify-center bg-background/60 rounded-lg">
                  <Loader2 size={32} className="animate-spin text-primary" />
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Output */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="px-4 py-2 border-b border-border">
          <h3 className="text-sm font-medium">Output</h3>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-4 flex flex-col gap-4">
            {/* Answer - always shown */}
            <div className="flex flex-col gap-1.5">
              <Label className="text-xs text-muted-foreground">Answer</Label>
              <Textarea
                value={answerText}
                readOnly
                placeholder="ai generated image description"
                className="min-h-[120px] text-sm"
                rows={5}
              />
            </div>

            {/* VLM: Output Image (annotated) */}
            {method === "vlm" && (
              <div className="flex flex-col gap-1.5">
                <Label className="text-xs text-muted-foreground">Annotated Image</Label>
                <div className="rounded-md border border-border flex items-center justify-center min-h-[200px] bg-muted/20">
                  <div className="flex flex-col items-center gap-2 text-muted-foreground opacity-50">
                    <ImageOff size={32} />
                    <p className="text-xs">
                      {result ? "No annotated image for this result" : "Run VLM captioning with detection tasks to see annotated output"}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* OpenCLiP: CLIP Analysis */}
            {method === "openclip" && (
              <div className="flex flex-col gap-1.5">
                <Label className="text-xs text-muted-foreground">CLIP Analysis</Label>
                {hasAnalysis ? (
                  <div className="rounded-md border border-border p-3 flex flex-col gap-2 text-sm">
                    {result.medium && <AnalysisField label="Medium" value={result.medium} />}
                    {result.artist && <AnalysisField label="Artist" value={result.artist} />}
                    {result.movement && <AnalysisField label="Movement" value={result.movement} />}
                    {result.trending && <AnalysisField label="Trending" value={result.trending} />}
                    {result.flavor && <AnalysisField label="Flavor" value={result.flavor} />}
                  </div>
                ) : (
                  <div className="rounded-md border border-border p-6 flex items-center justify-center min-h-[120px]">
                    <p className="text-xs text-muted-foreground opacity-50">
                      {result ? "Enable Analyze to see CLIP breakdown" : "Run OpenCLiP captioning with Analyze enabled"}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Tagger: answer only, no extra sections */}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

function AnalysisField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-muted-foreground text-xs">{label}: </span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}
