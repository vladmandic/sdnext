import { useCallback } from "react";
import { History, ImagePlus, ArrowUpCircle, GitCompareArrows, Download } from "lucide-react";
import { restoreFromResult } from "@/lib/requestBuilder";
import { sendResultToCanvas, sendResultToUpscale } from "@/lib/sendTo";
import { downloadImage, generateImageFilename } from "@/lib/utils";
import type { GenerationResult } from "@/stores/generationStore";
import { toast } from "sonner";

interface ResultThumbActionsProps {
  result: GenerationResult;
  imageIndex: number;
  onCompare: () => void;
}

export function ResultThumbActions({ result, imageIndex, onCompare }: ResultThumbActionsProps) {
  const handleRestore = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    restoreFromResult(result);
    toast.success("Settings restored");
  }, [result]);

  const handleSendToCanvas = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    sendResultToCanvas(result, imageIndex).catch(() => toast.error("Failed to send to canvas"));
  }, [result, imageIndex]);

  const handleSendToUpscale = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    sendResultToUpscale(result, imageIndex).catch(() => toast.error("Failed to send to upscale"));
  }, [result, imageIndex]);

  const handleDownload = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    const filename = generateImageFilename(result.info, imageIndex);
    downloadImage(result.images[imageIndex], filename);
  }, [result, imageIndex]);

  return (
    <div
      className="absolute bottom-0 left-0 right-0 flex justify-center gap-0.5 bg-gradient-to-t from-black/80 to-transparent pt-3 pb-0.5 px-0.5"
      onClick={(e) => e.stopPropagation()}
    >
      <ActionBtn onClick={handleRestore} title="Restore params"><History size={10} /></ActionBtn>
      <ActionBtn onClick={handleSendToCanvas} title="Send to canvas"><ImagePlus size={10} /></ActionBtn>
      <ActionBtn onClick={handleSendToUpscale} title="Send to upscale"><ArrowUpCircle size={10} /></ActionBtn>
      <ActionBtn onClick={(e) => { e.stopPropagation(); onCompare(); }} title="Compare"><GitCompareArrows size={10} /></ActionBtn>
      <ActionBtn onClick={handleDownload} title="Download"><Download size={10} /></ActionBtn>
    </div>
  );
}

function ActionBtn({ children, onClick, title }: { children: React.ReactNode; onClick: (e: React.MouseEvent) => void; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="w-5 h-5 flex items-center justify-center rounded text-white/80 hover:text-white hover:bg-white/20 transition-colors"
    >
      {children}
    </button>
  );
}
