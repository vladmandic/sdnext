import { useGenerationStore } from "@/stores/generationStore";
import { base64ToObjectUrl } from "@/lib/utils";

export function GenerateView() {
  const results = useGenerationStore((s) => s.results);
  const isGenerating = useGenerationStore((s) => s.isGenerating);
  const previewImage = useGenerationStore((s) => s.previewImage);
  const progress = useGenerationStore((s) => s.progress);
  const selectedResultId = useGenerationStore((s) => s.selectedResultId);
  const selectedImageIndex = useGenerationStore((s) => s.selectedImageIndex);

  let displayImage: string | undefined;
  if (previewImage) {
    displayImage = previewImage;
  } else if (selectedResultId) {
    const selected = results.find((r) => r.id === selectedResultId);
    displayImage = selected?.images[selectedImageIndex ?? 0];
  } else {
    displayImage = results[0]?.images[0];
  }

  return (
    <div className="flex items-center justify-center h-full p-4">
      {displayImage ? (
        <div className="relative max-w-full max-h-full">
          <img
            src={
              displayImage.startsWith("data:") || displayImage.startsWith("blob:")
                ? displayImage
                : base64ToObjectUrl(displayImage)
            }
            alt="Generated"
            className="max-w-full max-h-[calc(100vh-8rem)] object-contain rounded-lg"
          />
          {isGenerating && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/30 rounded-lg">
              <div className="text-white text-sm font-medium">
                {Math.round(progress * 100)}%
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center text-muted-foreground">
          <p className="text-lg">No images yet</p>
          <p className="text-sm mt-1">Configure parameters and click Generate</p>
        </div>
      )}
    </div>
  );
}
