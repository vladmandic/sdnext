import { memo, useCallback, useRef } from "react";
import type { GalleryFile, CachedThumb } from "@/api/types/gallery";
import { useGalleryStore } from "@/stores/galleryStore";
import { useDragSource } from "@/hooks/useDragSource";
import { isVideoFile } from "@/lib/mediaType";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";

interface GalleryCardProps {
  file: GalleryFile;
  thumb: CachedThumb | undefined;
  size: number;
  height?: number;
  selected: boolean;
  isSelected: boolean;
  isSelectMode: boolean;
  onClick: (e: React.MouseEvent) => void;
  onDoubleClick: () => void;
  onContextMenu?: () => void;
}

const HOVER_DELAY = 300;

export const GalleryCard = memo(function GalleryCard({ file, thumb, size, height, selected, isSelected, isSelectMode, onClick, onDoubleClick, onContextMenu }: GalleryCardProps) {
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

  const cardHeight = height ?? size;

  return (
    <button
      onClick={onClick}
      onDoubleClick={onDoubleClick}
      onContextMenu={onContextMenu}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      {...dragProps}
      className={cn(
        "group relative rounded-md overflow-hidden bg-muted border transition-all cursor-pointer",
        selected ? "border-primary ring-1 ring-primary/30" : isSelected ? "border-primary ring-1 ring-primary/20" : "border-border hover:border-primary/40",
      )}
      style={{ width: size, height: cardHeight }}
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
          <div className="w-6 h-6 rounded-full border-2 border-muted-foreground/20 animate-pulse" />
        </div>
      )}

      {/* Selection tint overlay */}
      {isSelected && (
        <div className="absolute inset-0 bg-primary/10 pointer-events-none" />
      )}

      {/* Selection checkbox */}
      {(isSelectMode || isSelected) && (
        <div className={cn(
          "absolute top-1 left-1 w-5 h-5 rounded-sm flex items-center justify-center transition-colors",
          isSelected ? "bg-primary text-primary-foreground" : "bg-black/40 text-white/60",
        )}>
          {isSelected && <Check size={12} strokeWidth={3} />}
        </div>
      )}

      {/* Hover checkbox (when not in select mode) */}
      {!isSelectMode && !isSelected && (
        <div className="absolute top-1 left-1 w-5 h-5 rounded-sm flex items-center justify-center bg-black/40 text-white/60 opacity-0 group-hover:opacity-100 transition-opacity">
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

/** Self-subscribing card that reads its own thumb/selection from the store. */
interface ConnectedGalleryCardProps {
  file: GalleryFile;
  index: number;
  size: number;
  height?: number;
  onClick: (file: GalleryFile, index: number, e: React.MouseEvent) => void;
  onDoubleClick: (index: number) => void;
  onContextMenu?: (file: GalleryFile, index: number) => void;
}

export const ConnectedGalleryCard = memo(function ConnectedGalleryCard({ file, index, size, height, onClick, onDoubleClick, onContextMenu }: ConnectedGalleryCardProps) {
  const thumb = useGalleryStore((s) => s.thumbs.get(file.id));
  const selected = useGalleryStore((s) => s.selectedFile?.id === file.id);
  const isSelected = useGalleryStore((s) => s.selectedIds.has(file.id));
  const isSelectMode = useGalleryStore((s) => s.selectedIds.size > 0);

  const handleClick = useCallback((e: React.MouseEvent) => onClick(file, index, e), [file, index, onClick]);
  const handleDoubleClick = useCallback(() => onDoubleClick(index), [index, onDoubleClick]);
  const handleContextMenu = useCallback(() => onContextMenu?.(file, index), [file, index, onContextMenu]);

  return (
    <GalleryCard
      file={file}
      thumb={thumb}
      size={size}
      height={height}
      selected={selected}
      isSelected={isSelected}
      isSelectMode={isSelectMode}
      onClick={handleClick}
      onDoubleClick={handleDoubleClick}
      onContextMenu={handleContextMenu}
    />
  );
});
