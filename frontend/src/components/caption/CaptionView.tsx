import { useCallback, useEffect } from "react";
import { Upload, X, Loader2, ImageOff, Copy, ArrowRight } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { useCaptionStore } from "@/stores/captionStore";
import { useGenerationStore } from "@/stores/generationStore";
import { useDropTarget } from "@/hooks/useDropTarget";
import { payloadToFile } from "@/lib/sendTo";
import type { DragPayload } from "@/stores/dragStore";

export function CaptionView() {
  const image = useCaptionStore((s) => s.image);
  const imagePreviewUrl = useCaptionStore((s) => s.imagePreviewUrl);
  const isProcessing = useCaptionStore((s) => s.isProcessing);
  const result = useCaptionStore((s) => s.result);
  const method = useCaptionStore((s) => s.method);
  const setImage = useCaptionStore((s) => s.setImage);

  const { isOver, ...dropHandlers } = useDropTarget({
    onDropPayload: useCallback((payload: DragPayload) => { payloadToFile(payload).then((f: File) => setImage(f)).catch(() => {}); }, [setImage]),
    onFileDrop: useCallback((file: File) => setImage(file), [setImage]),
  });

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

  const answerText = result?.type === "vqa"
    ? (result.answer ?? "")
    : result?.type === "openclip"
      ? (result.caption ?? "")
      : result?.type === "tagger"
        ? (result.tags ?? "")
        : "";

  const hasAnalysis = result?.type === "openclip" && (result.medium || result.artist || result.movement || result.trending || result.flavor);
  const hasAnnotatedImage = result?.type === "vqa" && result.annotated_image;
  const hasScores = result?.type === "tagger" && result.scores;

  const handleCopy = useCallback(() => {
    if (answerText) {
      navigator.clipboard.writeText(answerText);
      toast.success("Copied to clipboard");
    }
  }, [answerText]);

  const handleToPrompt = useCallback(() => {
    if (!answerText) return;
    const current = useGenerationStore.getState().prompt;
    useGenerationStore.getState().setParam("prompt", current ? `${current}, ${answerText}` : answerText);
    toast.success("Appended to prompt");
  }, [answerText]);

  const handleToNegative = useCallback(() => {
    if (!answerText) return;
    const current = useGenerationStore.getState().negativePrompt;
    useGenerationStore.getState().setParam("negativePrompt", current ? `${current}, ${answerText}` : answerText);
    toast.success("Appended to negative prompt");
  }, [answerText]);

  return (
    <ResizablePanelGroup orientation="horizontal" id="caption-columns">
      {/* Input Image */}
      <ResizablePanel defaultSize="50%" minSize="20%">
        <div className="flex flex-col h-full">
          <div className="px-4 py-2 border-b border-border">
            <h3 className="text-sm font-medium">Input Image</h3>
          </div>
          <div className={`flex-1 relative min-h-0${isOver ? " ring-2 ring-primary ring-inset" : ""}`} {...dropHandlers}>
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
      </ResizablePanel>

      <ResizableHandle />

      {/* Output */}
      <ResizablePanel defaultSize="50%" minSize="20%">
        <div className="flex flex-col h-full">
          <div className="px-4 py-2 border-b border-border">
            <h3 className="text-sm font-medium">Output</h3>
          </div>
          <div className="flex-1 min-h-0">
            <ResizablePanelGroup orientation="vertical" id="caption-output-rows">
              {/* Answer */}
              <ResizablePanel defaultSize="50%" minSize="15%">
                <div className="flex flex-col h-full p-4 gap-1.5 min-h-0">
                  <Label className="text-xs text-muted-foreground shrink-0">
                    {result?.type === "tagger" ? "Tags" : "Answer"}
                  </Label>
                  <Textarea
                    value={answerText}
                    readOnly
                    placeholder="ai generated image description"
                    className="flex-1 min-h-0 text-sm resize-none"
                  />
                  {answerText && (
                    <div className="flex items-center gap-1.5 shrink-0">
                      <Button variant="secondary" size="sm" className="text-xs h-7 gap-1" onClick={handleCopy}>
                        <Copy size={10} />
                        Copy
                      </Button>
                      <Button variant="secondary" size="sm" className="text-xs h-7 gap-1" onClick={handleToPrompt}>
                        <ArrowRight size={10} />
                        Prompt
                      </Button>
                      <Button variant="secondary" size="sm" className="text-xs h-7 gap-1" onClick={handleToNegative}>
                        <ArrowRight size={10} />
                        Negative
                      </Button>
                    </div>
                  )}
                </div>
              </ResizablePanel>

              <ResizableHandle />

              {/* Secondary content */}
              <ResizablePanel defaultSize="50%" minSize="15%">
                <ScrollArea className="h-full">
                  <div className="p-4">
                    {/* VLM: Annotated Image */}
                    {method === "vlm" && (
                      <div className="flex flex-col gap-1.5">
                        <Label className="text-xs text-muted-foreground">Annotated Image</Label>
                        {hasAnnotatedImage ? (
                          <div className="rounded-md border border-border overflow-hidden">
                            <img
                              src={`data:image/png;base64,${result.annotated_image}`}
                              alt="Annotated detection result"
                              className="max-w-full object-contain"
                            />
                          </div>
                        ) : (
                          <div className="rounded-md border border-border flex items-center justify-center min-h-50 bg-muted/20">
                            <div className="flex flex-col items-center gap-2 text-muted-foreground opacity-50">
                              <ImageOff size={32} />
                              <p className="text-xs">
                                {result ? "No annotated image for this result" : "Run VLM captioning with detection tasks to see annotated output"}
                              </p>
                            </div>
                          </div>
                        )}
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
                          <div className="rounded-md border border-border p-6 flex items-center justify-center min-h-30">
                            <p className="text-xs text-muted-foreground opacity-50">
                              {result ? "Enable Analyze to see CLIP breakdown" : "Run OpenCLiP captioning with Analyze enabled"}
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Tagger: Confidence Scores */}
                    {method === "tagger" && (
                      <div className="flex flex-col gap-1.5">
                        <Label className="text-xs text-muted-foreground">Confidence Scores</Label>
                        {hasScores ? (
                          <div className="rounded-md border border-border p-3 flex flex-col gap-1 text-sm">
                            {Object.entries(result.scores!).map(([tag, score]) => (
                              <div key={tag} className="flex items-center justify-between">
                                <span className="text-foreground">{tag}</span>
                                <span className="text-muted-foreground text-xs tabular-nums">{score.toFixed(3)}</span>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="rounded-md border border-border p-6 flex items-center justify-center min-h-30">
                            <p className="text-xs text-muted-foreground opacity-50">
                              {result ? "Enable Show Confidence Scores to see per-tag scores" : "Run tagger to see results"}
                            </p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </ResizablePanel>
            </ResizablePanelGroup>
          </div>
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
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
