import { useGenerationStore } from "@/stores/generationStore";
import { restoreFromResult } from "@/lib/requestBuilder";
import { base64ToObjectUrl, cn, downloadBase64Image, generateImageFilename } from "@/lib/utils";
import { memo, useCallback } from "react";
import { toast } from "sonner";
import { Download } from "lucide-react";

export const ResultGallery = memo(function ResultGallery() {
  const results = useGenerationStore((s) => s.results);
  const selectedResultId = useGenerationStore((s) => s.selectedResultId);
  const selectedImageIndex = useGenerationStore((s) => s.selectedImageIndex);
  const selectImage = useGenerationStore((s) => s.selectImage);

  const handleDoubleClick = useCallback(
    (resultId: string) => {
      const result = useGenerationStore.getState().results.find((r) => r.id === resultId);
      if (result) {
        restoreFromResult(result);
        toast.success("Settings restored from selected generation");
      }
    },
    [],
  );

  const handleDownload = useCallback(() => {
    if (!selectedResultId || selectedImageIndex === null) return;
    const result = results.find((r) => r.id === selectedResultId);
    if (!result || !result.images[selectedImageIndex]) return;
    const image = result.images[selectedImageIndex];
    const base64 = image.startsWith("data:") ? image.split(",")[1] : image;
    const filename = generateImageFilename(result.info, selectedImageIndex);
    downloadBase64Image(base64, filename);
  }, [results, selectedResultId, selectedImageIndex]);

  if (results.length === 0) {
    return (
      <div className="text-[11px] text-muted-foreground text-center py-2">
        No results yet
      </div>
    );
  }

  const allImages = results.flatMap((r) =>
    r.images.map((img, ii) => ({
      resultId: r.id,
      imageIndex: ii,
      image: img,
      key: `${r.id}-${ii}`,
    })),
  );

  return (
    <div className="flex flex-col gap-1">
      {selectedResultId && selectedImageIndex !== null && (
        <div className="flex items-center gap-1 justify-end">
          <button
            onClick={handleDownload}
            title="Download image"
            className="p-1 rounded hover:bg-accent transition-colors"
          >
            <Download size={14} />
          </button>
        </div>
      )}
      <div className="flex gap-1.5 overflow-x-auto" style={{ scrollbarWidth: "thin" }}>
        {allImages.map((item) => {
          const isSelected = item.resultId === selectedResultId && item.imageIndex === selectedImageIndex;
          return (
            <button
              key={item.key}
              onClick={() => selectImage(item.resultId, item.imageIndex)}
              onDoubleClick={() => handleDoubleClick(item.resultId)}
              className={cn(
                "w-14 h-14 flex-shrink-0 rounded overflow-hidden border transition-colors",
                isSelected ? "border-primary" : "border-transparent hover:border-muted-foreground/30",
              )}
            >
              <img
                src={
                  item.image.startsWith("data:") || item.image.startsWith("blob:")
                    ? item.image
                    : base64ToObjectUrl(item.image)
                }
                alt="Result"
                className="w-full h-full object-cover"
              />
            </button>
          );
        })}
      </div>
    </div>
  );
});
