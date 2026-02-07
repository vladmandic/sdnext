import { useGenerationStore } from "@/stores/generationStore";
import { restoreFromResult } from "@/lib/requestBuilder";
import { base64ToObjectUrl, cn } from "@/lib/utils";
import { memo, useCallback } from "react";
import { toast } from "sonner";

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
  );
});
