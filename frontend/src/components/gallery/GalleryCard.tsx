import { memo, useCallback, useRef } from "react";
import type { GalleryFile, CachedThumb } from "@/api/types/gallery";
import { useDragSource } from "@/hooks/useDragSource";
import { isVideoFile } from "@/lib/mediaType";
import { cn } from "@/lib/utils";

interface GalleryCardProps {
  file: GalleryFile;
  thumb: CachedThumb | undefined;
  size: number;
  selected: boolean;
  onClick: () => void;
  onDoubleClick: () => void;
}

const HOVER_DELAY = 300;

export const GalleryCard = memo(function GalleryCard({ file, thumb, size, selected, onClick, onDoubleClick }: GalleryCardProps) {
  const filename = file.relativePath.split("/").pop() ?? file.relativePath;
  const isVideo = isVideoFile(file.relativePath);
  const videoRef = useRef<HTMLVideoElement>(null);
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dragProps = useDragSource({ type: "gallery-image", fileId: file.id, filePath: file.fullPath, src: thumb?.data });

  const handleMouseEnter = useCallback(() => {
    if (!isVideo) return;
    hoverTimer.current = setTimeout(() => {
      const el = videoRef.current;
      if (!el) return;
      el.src = `/file=${file.fullPath}`;
      el.play().catch(() => {});
    }, HOVER_DELAY);
  }, [isVideo, file.fullPath]);

  const handleMouseLeave = useCallback(() => {
    if (!isVideo) return;
    if (hoverTimer.current) {
      clearTimeout(hoverTimer.current);
      hoverTimer.current = null;
    }
    const el = videoRef.current;
    if (el) {
      el.pause();
      el.removeAttribute("src");
      el.load();
    }
  }, [isVideo]);

  return (
    <button
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      {...dragProps}
      className={cn(
        "group relative rounded-md overflow-hidden bg-muted border transition-all cursor-pointer",
        selected ? "border-primary ring-1 ring-primary/30" : "border-border hover:border-primary/40",
      )}
      style={{ width: size, height: size }}
    >
      {thumb ? (
        isVideo ? (
          <>
            <video
              ref={videoRef}
              poster={thumb.data}
              muted
              preload="none"
              className="w-full h-full object-cover"
            />
            {/* Play icon overlay */}
            <div className="absolute bottom-1.5 left-1.5 w-5 h-5 flex items-center justify-center rounded-full bg-black/60 pointer-events-none">
              <svg width="10" height="10" viewBox="0 0 10 10" className="ml-0.5 text-white" fill="currentColor">
                <polygon points="0,0 10,5 0,10" />
              </svg>
            </div>
          </>
        ) : (
          <img
            src={thumb.data}
            alt={filename}
            className="w-full h-full object-cover"
            draggable={false}
          />
        )
      ) : (
        <div className="w-full h-full flex items-center justify-center">
          <div className="w-6 h-6 rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground animate-spin" />
        </div>
      )}
      {/* Hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
        <div className="absolute bottom-0 left-0 right-0 p-1.5">
          <p className="text-4xs text-white/90 font-medium truncate">{filename}</p>
          {thumb && (
            <p className="text-5xs text-white/60">{thumb.width}x{thumb.height}</p>
          )}
        </div>
      </div>
    </button>
  );
});
