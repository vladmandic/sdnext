import { useState, useCallback } from "react";
import { Upload, Copy, ArrowRight } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { usePngInfo } from "@/api/hooks/usePngInfo";
import { restoreFromPngInfo } from "@/lib/pngInfoRestore";
import { toast } from "sonner";

interface PngInfoDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function PngInfoDialog({ open, onOpenChange }: PngInfoDialogProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [infoText, setInfoText] = useState("");
  const [items, setItems] = useState<Record<string, string>>({});
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const pngInfo = usePngInfo();

  const processFile = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setPreviewUrl(dataUrl);
      const base64 = dataUrl.split(",")[1];
      pngInfo.mutate({ image: base64 }, {
        onSuccess: (data) => {
          setInfoText(data.info ?? "");
          setItems(data.items ?? {});
          setParameters(data.parameters ?? {});
        },
        onError: () => toast.error("Failed to extract PNG info"),
      });
    };
    reader.readAsDataURL(file);
  }, [pngInfo]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file?.type.startsWith("image/")) processFile(file);
  }, [processFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) processFile(file);
    e.target.value = "";
  }, [processFile]);

  const handleApply = useCallback(() => {
    restoreFromPngInfo(parameters);
    toast.success("Generation settings applied from PNG info");
    onOpenChange(false);
  }, [parameters, onOpenChange]);

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(infoText);
    toast.success("Info copied to clipboard");
  }, [infoText]);

  const handleClose = (nextOpen: boolean) => {
    if (!nextOpen) {
      setPreviewUrl(null);
      setInfoText("");
      setItems({});
      setParameters({});
    }
    onOpenChange(nextOpen);
  };

  const itemEntries = Object.entries(items);

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-lg max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-sm">PNG Info</DialogTitle>
        </DialogHeader>

        {!previewUrl ? (
          <div
            className="flex-1 min-h-[200px] border-2 border-dashed border-border rounded-lg"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <label className="flex flex-col items-center justify-center h-full cursor-pointer text-muted-foreground hover:text-foreground transition-colors">
              <Upload size={36} className="mb-2 opacity-40" />
              <p className="text-sm font-medium">Drop PNG Here</p>
              <p className="text-xs mt-1 opacity-60">or click to browse</p>
              <input type="file" accept="image/*" className="hidden" onChange={handleFileInput} />
            </label>
          </div>
        ) : (
          <div className="flex-1 min-h-0 space-y-3 overflow-y-auto">
            <div className="flex gap-3">
              <img src={previewUrl} alt="Preview" className="w-24 h-24 rounded object-cover shrink-0" />
              <div className="flex-1 min-w-0">
                {pngInfo.isPending ? (
                  <p className="text-xs text-muted-foreground animate-pulse">Extracting info...</p>
                ) : infoText ? (
                  <textarea
                    readOnly
                    value={infoText}
                    className="w-full h-24 text-[11px] bg-muted/30 rounded p-2 resize-none font-mono border border-border"
                  />
                ) : (
                  <p className="text-xs text-muted-foreground">No generation info found in this image.</p>
                )}
              </div>
            </div>

            {itemEntries.length > 0 && (
              <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
                {itemEntries.map(([key, value]) => (
                  <div key={key} className="contents">
                    <span className="text-muted-foreground truncate">{key}</span>
                    <span className="truncate font-medium">{value}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="flex gap-2">
              <Button size="sm" onClick={handleApply} disabled={Object.keys(parameters).length === 0}>
                <ArrowRight size={14} />
                Apply to Generation
              </Button>
              <Button size="sm" variant="secondary" onClick={handleCopy} disabled={!infoText}>
                <Copy size={14} />
                Copy Info
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
