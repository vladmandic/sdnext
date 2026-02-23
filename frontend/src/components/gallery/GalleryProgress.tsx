import { useGalleryStore } from "@/stores/galleryStore";

export function GalleryProgress() {
  const isLoadingFiles = useGalleryStore((s) => s.isLoadingFiles);
  const fileCount = useGalleryStore((s) => s.files.length);
  const thumbCount = useGalleryStore((s) => s.thumbs.size);

  // Phase 1: file list streaming via WebSocket
  if (isLoadingFiles) {
    return (
      <div className="flex items-center gap-2 px-3 py-1 border-b border-border/50 bg-muted/30 flex-shrink-0">
        <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
          <div className="h-full bg-primary rounded-full animate-[indeterminate_1.5s_ease-in-out_infinite] origin-left" />
        </div>
        <span className="text-3xs text-muted-foreground tabular-nums whitespace-nowrap">
          Loading files...
        </span>
      </div>
    );
  }

  // Phase 2: thumbnail generation
  if (fileCount > 0 && thumbCount < fileCount) {
    const pct = Math.min(100, (thumbCount / fileCount) * 100);
    return (
      <div className="flex items-center gap-2 px-3 py-1 border-b border-border/50 bg-muted/30 flex-shrink-0">
        <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-[width] duration-300 ease-linear"
            style={{ width: `${pct}%` }}
          />
        </div>
        <span className="text-3xs text-muted-foreground tabular-nums whitespace-nowrap">
          {thumbCount} / {fileCount} thumbnails
        </span>
      </div>
    );
  }

  return null;
}
