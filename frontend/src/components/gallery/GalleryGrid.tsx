import type { ResHistory } from "@/api/types/progress";
import { GalleryCard } from "./GalleryCard";

interface GalleryGridProps {
  items: ResHistory[];
  onImageClick: (item: ResHistory, imageUrl: string) => void;
}

export function GalleryGrid({ items, onImageClick }: GalleryGridProps) {
  if (items.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
        <p className="text-lg">No images yet</p>
        <p className="text-sm mt-1">Generated images will appear here</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-[repeat(auto-fill,minmax(180px,1fr))] gap-2 p-4">
      {items.flatMap((item) =>
        item.outputs.map((output, idx) => {
          const imageUrl = `/file=${output}`;
          return (
            <GalleryCard
              key={`${item.id}-${idx}`}
              item={item}
              imageUrl={imageUrl}
              onClick={() => onImageClick(item, imageUrl)}
            />
          );
        }),
      )}
    </div>
  );
}
