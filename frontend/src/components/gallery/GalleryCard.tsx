import { memo } from "react";
import type { GalleryFile, CachedThumb } from "@/api/types/gallery";
import { cn } from "@/lib/utils";

interface GalleryCardProps {
  file: GalleryFile;
  thumb: CachedThumb | undefined;
  size: number;
  selected: boolean;
  onClick: () => void;
  onDoubleClick: () => void;
}

export const GalleryCard = memo(function GalleryCard({ file, thumb, size, selected, onClick, onDoubleClick }: GalleryCardProps) {
  const filename = file.relativePath.split("/").pop() ?? file.relativePath;

  return (
    <button
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      className={cn(
        "group relative rounded-md overflow-hidden bg-muted border transition-all cursor-pointer",
        selected ? "border-primary ring-1 ring-primary/30" : "border-border hover:border-primary/40",
      )}
      style={{ width: size, height: size }}
    >
      {thumb ? (
        <img
          src={thumb.data}
          alt={filename}
          className="w-full h-full object-cover"
          draggable={false}
        />
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <div className="w-6 h-6 rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground animate-spin" />
        </div>
      )}
      {/* Hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
        <div className="absolute bottom-0 left-0 right-0 p-1.5">
          <p className="text-[9px] text-white/90 font-medium truncate">{filename}</p>
          {thumb && (
            <p className="text-[8px] text-white/60">{thumb.width}x{thumb.height}</p>
          )}
        </div>
      </div>
    </button>
  );
});
