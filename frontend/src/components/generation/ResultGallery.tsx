import { useGenerationStore } from "@/stores/generationStore";
import { base64ToObjectUrl } from "@/lib/utils";
import { cn } from "@/lib/utils";
import { useState } from "react";

export function ResultGallery() {
  const results = useGenerationStore((s) => s.results);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  if (results.length === 0) {
    return (
      <div className="text-[11px] text-muted-foreground text-center py-2">
        No results yet
      </div>
    );
  }

  // Flatten all images from all results
  const allImages = results.flatMap((r, ri) =>
    r.images.map((img, ii) => ({ resultIdx: ri, imageIdx: ii, image: img, id: `${r.id}-${ii}` })),
  );

  return (
    <div className="grid grid-cols-4 gap-1">
      {allImages.slice(0, 16).map((item, idx) => (
        <button
          key={item.id}
          onClick={() => setSelectedIdx(idx === selectedIdx ? null : idx)}
          className={cn(
            "aspect-square rounded overflow-hidden border transition-colors",
            idx === selectedIdx ? "border-primary" : "border-transparent hover:border-muted-foreground/30",
          )}
        >
          <img
            src={
              item.image.startsWith("data:") || item.image.startsWith("blob:")
                ? item.image
                : base64ToObjectUrl(item.image)
            }
            alt={`Result ${idx + 1}`}
            className="w-full h-full object-cover"
          />
        </button>
      ))}
    </div>
  );
}
