import { useState, useCallback, useRef } from "react";
import { Upload, Copy, ChevronDown, Send, Wand2, Image as ImageIcon } from "lucide-react";
import { Dialog, DialogContent, DialogDescription } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { usePngInfo } from "@/api/hooks/usePngInfo";
import { restoreFromPngInfo } from "@/lib/pngInfoRestore";
import { sendImageToCanvas, sendFrameToVideoInit, sendPromptToGeneration } from "@/lib/sendTo";
import { uploadFile } from "@/lib/upload";
import { toast } from "sonner";

const PARAM_GROUPS: { label: string; keys: string[] }[] = [
  { label: "Core", keys: ["Steps", "Sampler", "Scheduler", "Seed", "CFG scale", "CFG end", "CFG rescale", "Clip skip", "Denoising strength", "Size-1", "Size-2", "Batch count", "Batch size"] },
  { label: "Hires Fix", keys: ["Hires upscaler", "Hires scale", "Hires steps", "Hires strength", "Hires sampler", "Hires force", "HiRes mode", "HiRes context", "Hires fixed-1", "Hires fixed-2", "Hires size-1", "Hires size-2", "Hires CFG scale"] },
  { label: "Refiner", keys: ["Refiner start", "Refiner steps", "Refiner prompt", "Refiner negative"] },
  { label: "Variation", keys: ["Variation seed", "Variation strength"] },
  { label: "Guidance", keys: ["CFG true", "CFG adaptive", "Image CFG scale"] },
  { label: "Scheduler", keys: ["Sampler sigma", "Sampler spacing", "Sampler beta schedule", "Sampler type", "Sampler shift", "Sampler dynamic shift", "Sampler low order", "Sampler dynamic", "Sampler rescale", "Sampler order", "Sampler range"] },
  { label: "Token Merging", keys: ["ToMe", "ToDo"] },
  { label: "Model", keys: ["Model", "Model hash", "VAE", "VAE type", "Pipeline", "TE", "UNet"] },
];

const ALL_GROUPED_KEYS = new Set(PARAM_GROUPS.flatMap((g) => g.keys));

const SIZE_PAIRS: Record<string, [string, string]> = {
  "Size-1": ["Size-1", "Size-2"],
  "Hires fixed-1": ["Hires fixed-1", "Hires fixed-2"],
  "Hires size-1": ["Hires size-1", "Hires size-2"],
};

const SIZE_LABELS: Record<string, string> = {
  "Size-1": "Size",
  "Hires fixed-1": "Hires fixed",
  "Hires size-1": "Hires size",
};

const SIZE_SECOND_KEYS = new Set(["Size-2", "Hires fixed-2", "Hires size-2"]);

interface PngInfoDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function PromptSection({ label, text, onCopy, onUse }: { label: string; text: string; onCopy: () => void; onUse: () => void }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider">{label}</span>
        <div className="flex gap-1">
          <Button size="sm" variant="ghost" className="h-5 px-1.5 text-3xs" onClick={onCopy}><Copy size={10} /> Copy</Button>
          <Button size="sm" variant="ghost" className="h-5 px-1.5 text-3xs" onClick={onUse}><Send size={10} /> Use</Button>
        </div>
      </div>
      <div className="bg-muted/30 rounded p-2 text-2xs whitespace-pre-wrap break-words max-h-32 overflow-y-auto">{text}</div>
    </div>
  );
}

function ParamGroup({ label, entries }: { label: string; entries: [string, string][] }) {
  if (entries.length === 0) return null;
  return (
    <div className="space-y-1.5">
      <span className="text-3xs font-semibold text-muted-foreground uppercase tracking-wider">{label}</span>
      <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-2xs">
        {entries.map(([key, value]) => (
          <div key={key} className="contents">
            <span className="text-muted-foreground truncate">{key}</span>
            <span className="truncate font-medium">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function resolveGroupEntries(items: Record<string, string>, keys: string[]): [string, string][] {
  const entries: [string, string][] = [];
  const seen = new Set<string>();
  for (const key of keys) {
    if (seen.has(key)) continue;
    if (SIZE_SECOND_KEYS.has(key)) continue;
    const pair = SIZE_PAIRS[key];
    if (pair) {
      const v1 = items[pair[0]];
      const v2 = items[pair[1]];
      if (v1 && v2) {
        entries.push([SIZE_LABELS[key] ?? key, `${v1} \u00d7 ${v2}`]);
        seen.add(pair[0]);
        seen.add(pair[1]);
      }
    } else {
      const val = items[key];
      if (val) entries.push([key, val]);
    }
    seen.add(key);
  }
  return entries;
}

export function PngInfoDialog({ open, onOpenChange }: PngInfoDialogProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileRef, setFileRef] = useState<File | null>(null);
  const [infoText, setInfoText] = useState("");
  const [items, setItems] = useState<Record<string, string>>({});
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);
  const pngInfo = usePngInfo();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(async (file: File) => {
    setPreviewUrl(URL.createObjectURL(file));
    setFileRef(file);
    try {
      const ref = await uploadFile(file);
      pngInfo.mutate({ image: ref }, {
        onSuccess: (data) => {
          setInfoText(data.info ?? "");
          setItems(data.items ?? {});
          setParameters(data.parameters ?? {});
        },
        onError: () => toast.error("Failed to extract PNG info"),
      });
    } catch {
      toast.error("Failed to upload image");
    }
  }, [pngInfo]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file?.type.startsWith("image/")) processFile(file);
  }, [processFile]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current += 1;
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setDragging(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => { e.preventDefault(); }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = "";
  }, [processFile]);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const file = Array.from(e.clipboardData.items)
      .find((item) => item.type.startsWith("image/"))
      ?.getAsFile();
    if (file) processFile(file);
  }, [processFile]);

  const handleApply = useCallback(() => {
    restoreFromPngInfo(parameters);
    toast.success("Generation settings applied from PNG info");
    onOpenChange(false);
  }, [parameters, onOpenChange]);

  const handleClose = (nextOpen: boolean) => {
    if (!nextOpen) {
      setPreviewUrl(null);
      setFileRef(null);
      setInfoText("");
      setItems({});
      setParameters({});
      dragCounter.current = 0;
      setDragging(false);
    }
    onOpenChange(nextOpen);
  };

  const copyText = useCallback((text: string, label: string) => {
    navigator.clipboard.writeText(text);
    toast.success(`${label} copied to clipboard`);
  }, []);

  // parameters has generation info (Steps, Prompt, Size-1, etc.); items has image metadata (width, height, mode)
  const paramStrings: Record<string, string> = {};
  for (const [k, v] of Object.entries(parameters)) {
    if (v != null && v !== "") paramStrings[k] = String(v);
  }

  const prompt = paramStrings.Prompt ?? "";
  const negativePrompt = paramStrings["Negative prompt"] ?? "";

  // Build param lines (everything after prompt/negative, starting from Steps:)
  const paramLine = infoText ? (() => {
    const stepsIdx = infoText.search(/\nSteps:\s/);
    return stepsIdx >= 0 ? infoText.slice(stepsIdx).trim() : "";
  })() : "";

  const hasParams = Object.keys(parameters).length > 0;

  // Build grouped entries from parameters (generation info), not items (image metadata)
  const grouped = PARAM_GROUPS.map((g) => ({ label: g.label, entries: resolveGroupEntries(paramStrings, g.keys) }));
  const otherParamEntries: [string, string][] = Object.entries(paramStrings)
    .filter(([key, val]) => !ALL_GROUPED_KEYS.has(key) && key !== "Prompt" && key !== "Negative prompt" && val)
    .map(([key, val]) => [key, val]);
  // Also include image-level metadata from items (width, height, mode) if not already covered
  const otherEntries: [string, string][] = [
    ...otherParamEntries,
    ...Object.entries(items).filter(([key, val]) => !paramStrings[key] && val).map(([key, val]): [string, string] => [key, val]),
  ];

  const handleSendToCanvas = useCallback(async () => {
    if (!fileRef) return;
    await sendImageToCanvas(fileRef);
    toast.success("Image sent to canvas");
  }, [fileRef]);

  const handleSendToVideo = useCallback(async () => {
    if (!fileRef) return;
    const blob = new Blob([await fileRef.arrayBuffer()], { type: fileRef.type });
    await sendFrameToVideoInit(blob);
    toast.success("Image sent to video init");
  }, [fileRef]);

  const handleSendImageAndPrompt = async () => {
    if (!fileRef) return;
    await sendImageToCanvas(fileRef);
    if (prompt) sendPromptToGeneration(prompt, negativePrompt || undefined);
    toast.success("Image and prompt sent");
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent
        className="sm:max-w-3xl max-h-[85vh] p-0 gap-0 overflow-hidden flex flex-col"
        showCloseButton
        onPaste={handlePaste}
        onDrop={handleDrop}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
      >
        <DialogDescription className="sr-only">View and restore generation parameters from PNG image metadata</DialogDescription>

        {/* Title bar */}
        <div className="flex items-center px-4 py-2.5 border-b">
          <h2 className="text-sm font-semibold">PNG Info</h2>
        </div>

        {/* No image — full-body drop zone */}
        {!previewUrl ? (
          <div className="flex items-center justify-center min-h-72 p-6">
            <label className="flex flex-col items-center justify-center w-full h-full cursor-pointer border-2 border-dashed border-border rounded-lg p-8 text-muted-foreground hover:text-foreground hover:border-foreground/30 transition-colors">
              <Upload size={40} className="mb-3 opacity-40" />
              <p className="text-sm font-medium">Drop image, paste, or click to browse</p>
              <p className="text-xs mt-1.5 opacity-60">Ctrl+V to paste from clipboard</p>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
            </label>
          </div>
        ) : (
          <div className="flex flex-1 min-h-0 overflow-hidden relative">
            {/* Drag overlay */}
            {dragging && (
              <div className="absolute inset-0 z-10 flex items-center justify-center bg-background/80 backdrop-blur-sm border-2 border-dashed border-primary rounded-b-lg">
                <div className="text-center">
                  <Upload size={32} className="mx-auto mb-2 text-primary" />
                  <p className="text-sm font-medium text-primary">Drop to replace</p>
                </div>
              </div>
            )}

            {/* Left pane — image preview */}
            <div className="w-2/5 flex flex-col items-center justify-center gap-3 p-4 border-r bg-muted/10">
              <div className="flex-1 flex items-center justify-center min-h-0 w-full">
                <img src={previewUrl} alt="Preview" className="max-w-full max-h-[60vh] rounded object-contain" />
              </div>
              <Button size="sm" variant="ghost" className="text-xs" onClick={() => fileInputRef.current?.click()}>
                <ImageIcon size={14} />
                Change image
              </Button>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
            </div>

            {/* Right pane — metadata */}
            <div className="flex-1 flex flex-col min-h-0 min-w-0 overflow-hidden">
              <ScrollArea className="flex-1 min-h-0">
                <div className="p-4 space-y-3">
                  {pngInfo.isPending ? (
                    <p className="text-xs text-muted-foreground animate-pulse">Extracting info...</p>
                  ) : !infoText ? (
                    <p className="text-xs text-muted-foreground">No generation info found in this image.</p>
                  ) : (
                    <>
                      {/* Prompt */}
                      {prompt && (
                        <PromptSection
                          label="Prompt"
                          text={prompt}
                          onCopy={() => copyText(prompt, "Prompt")}
                          onUse={() => { sendPromptToGeneration(prompt); toast.success("Prompt applied"); }}
                        />
                      )}

                      {/* Negative prompt */}
                      {negativePrompt && (
                        <PromptSection
                          label="Negative Prompt"
                          text={negativePrompt}
                          onCopy={() => copyText(negativePrompt, "Negative prompt")}
                          onUse={() => { sendPromptToGeneration(prompt, negativePrompt); toast.success("Prompts applied"); }}
                        />
                      )}

                      {(prompt || negativePrompt) && <Separator />}

                      {/* Grouped parameters */}
                      {grouped.map((g) => g.entries.length > 0 && (
                        <ParamGroup key={g.label} label={g.label} entries={g.entries} />
                      ))}

                      {/* Other / uncategorized */}
                      {otherEntries.length > 0 && <ParamGroup label="Other" entries={otherEntries} />}
                    </>
                  )}
                </div>
              </ScrollArea>

              {/* Action bar */}
              {infoText && (
                <div className="flex items-center gap-2 px-4 py-2.5 border-t bg-muted/10">
                  <Button size="sm" onClick={handleApply} disabled={!hasParams}>
                    <Wand2 size={14} />
                    Apply All
                  </Button>

                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button size="sm" variant="secondary">
                        <Copy size={14} />
                        Copy
                        <ChevronDown size={12} />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      {prompt && <DropdownMenuItem onClick={() => copyText(prompt, "Prompt")}>Copy Prompt</DropdownMenuItem>}
                      {negativePrompt && <DropdownMenuItem onClick={() => copyText(negativePrompt, "Negative prompt")}>Copy Negative</DropdownMenuItem>}
                      <DropdownMenuItem onClick={() => copyText(infoText, "Raw info")}>Copy Raw Info</DropdownMenuItem>
                      {paramLine && <DropdownMenuItem onClick={() => copyText(paramLine, "Parameters")}>Copy Parameters</DropdownMenuItem>}
                    </DropdownMenuContent>
                  </DropdownMenu>

                  {fileRef && (
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button size="sm" variant="outline">
                          <Send size={14} />
                          Send
                          <ChevronDown size={12} />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="start">
                        <DropdownMenuItem onClick={handleSendToCanvas}>Send Image to Canvas</DropdownMenuItem>
                        <DropdownMenuItem onClick={handleSendToVideo}>Send to Video Init</DropdownMenuItem>
                        <DropdownMenuItem onClick={handleSendImageAndPrompt}>Image + Prompt to Canvas</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
