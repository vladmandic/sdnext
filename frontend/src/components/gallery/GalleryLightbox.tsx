import { useCallback, useEffect, useMemo, useState } from "react";
import { useGalleryStore } from "@/stores/galleryStore";
import { useShortcut } from "@/hooks/useShortcut";
import { useShortcutScope } from "@/hooks/useShortcutScope";
import { useDragSource } from "@/hooks/useDragSource";
import { useImageZoomPan } from "@/hooks/useImageZoomPan";
import { isVideoFile } from "@/lib/mediaType";
import { VideoPlayer } from "@/components/video/VideoPlayer";
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw, GitCompareArrows } from "lucide-react";
import { useComparisonStore } from "@/stores/comparisonStore";

export function GalleryLightbox() {
  const lightboxIndex = useGalleryStore((s) => s.lightboxIndex);
  const file = useGalleryStore((s) => (s.lightboxIndex !== null ? s.sortedFiles[s.lightboxIndex] ?? null : null));
  const thumb = useGalleryStore((s) => {
    if (s.lightboxIndex === null) return undefined;
    const f = s.sortedFiles[s.lightboxIndex];
    return f ? s.thumbs.get(f.id) : undefined;
  });
  const fileCount = useGalleryStore((s) => s.sortedFiles.length);
  const closeLightbox = useGalleryStore((s) => s.closeLightbox);
  const navigateLightbox = useGalleryStore((s) => s.navigateLightbox);
  const selectFile = useGalleryStore((s) => s.selectFile);

  const zoom = useImageZoomPan();
  const [prevIndex, setPrevIndex] = useState(lightboxIndex);

  const isOpen = lightboxIndex !== null;
  const maxIndex = fileCount - 1;

  // Full-size image URL
  const fullUrl = useMemo(() => {
    if (!file) return null;
    return `/file=${file.fullPath}`;
  }, [file]);

  const lightboxDrag = useDragSource({
    type: "gallery-image",
    fileId: file?.id,
    filePath: file?.fullPath,
    src: thumb?.data,
  });

  // Reset transform on navigation (adjust state during render pattern)
  if (prevIndex !== lightboxIndex) {
    setPrevIndex(lightboxIndex);
    zoom.resetTransform();
  }

  // Sync selection
  useEffect(() => {
    if (file && thumb) selectFile(file, thumb);
  }, [file, thumb, selectFile]);

  const navigate = useCallback((delta: number) => {
    navigateLightbox(delta, maxIndex);
  }, [navigateLightbox, maxIndex]);

  const handleCompare = useCallback(() => {
    if (!fullUrl || lightboxIndex === null) return;
    const nextIndex = lightboxIndex < maxIndex ? lightboxIndex + 1 : lightboxIndex - 1;
    if (nextIndex < 0 || nextIndex > maxIndex) return;
    const sortedFiles = useGalleryStore.getState().sortedFiles;
    const nextFile = sortedFiles[nextIndex];
    if (!nextFile) return;
    const nextUrl = `/file=${nextFile.fullPath}`;
    const aName = file?.relativePath.split("/").pop() ?? "Image A";
    const bName = nextFile.relativePath.split("/").pop() ?? "Image B";
    useComparisonStore.getState().openComparison({ src: fullUrl, label: aName }, { src: nextUrl, label: bName });
  }, [fullUrl, lightboxIndex, maxIndex, file]);

  // Keyboard shortcuts (scoped to "lightbox", only active when open)
  useShortcutScope("lightbox", isOpen);
  useShortcut("lightbox-close", () => closeLightbox(), isOpen);
  useShortcut("lightbox-prev", () => navigate(-1), isOpen);
  useShortcut("lightbox-next", () => navigate(1), isOpen);
  useShortcut("lightbox-zoom-in", () => zoom.setScale((s) => s * 1.25), isOpen);
  useShortcut("lightbox-zoom-in-eq", () => zoom.setScale((s) => s * 1.25), isOpen);
  useShortcut("lightbox-zoom-out", () => zoom.setScale((s) => s / 1.25), isOpen);
  useShortcut("lightbox-zoom-reset", () => zoom.resetTransform(), isOpen);

  if (!isOpen || !file) return null;

  const filename = file.relativePath.split("/").pop() ?? file.relativePath;
  const isVideo = isVideoFile(file.relativePath);

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex flex-col" onClick={closeLightbox}>
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
        <span className="text-xs text-white/70 truncate max-w-[50%]">{filename}</span>
        <div className="flex items-center gap-1">
          {!isVideo && (
            <>
              <LightboxButton onClick={() => zoom.setScale((s) => s * 1.25)}><ZoomIn size={16} /></LightboxButton>
              <LightboxButton onClick={() => zoom.setScale((s) => s / 1.25)}><ZoomOut size={16} /></LightboxButton>
              <LightboxButton onClick={zoom.resetTransform}><RotateCcw size={16} /></LightboxButton>
              <span className="text-3xs text-white/50 tabular-nums w-10 text-center">{Math.round(zoom.scale * 100)}%</span>
              {fileCount > 1 && (
                <LightboxButton onClick={handleCompare}><GitCompareArrows size={16} /></LightboxButton>
              )}
            </>
          )}
          <LightboxButton onClick={closeLightbox}><X size={16} /></LightboxButton>
        </div>
      </div>

      {/* Content area */}
      {isVideo ? (
        <div
          className="flex-1 min-h-0 overflow-hidden select-none"
          onClick={(e) => e.stopPropagation()}
        >
          <VideoPlayer src={fullUrl} />
        </div>
      ) : (
        <div
          className="flex-1 flex items-center justify-center overflow-hidden select-none"
          style={{ cursor: zoom.style.cursor }}
          onWheel={zoom.handlers.onWheel}
          onMouseDown={(e) => { e.stopPropagation(); zoom.handlers.onMouseDown(e); }}
          onMouseMove={zoom.handlers.onMouseMove}
          onMouseUp={zoom.handlers.onMouseUp}
          onMouseLeave={zoom.handlers.onMouseLeave}
          onClick={(e) => e.stopPropagation()}
        >
          {fullUrl && (
            <img
              src={fullUrl}
              alt={filename}
              className="max-w-full max-h-full object-contain transition-transform duration-100"
              style={{ transform: zoom.style.transform }}
              {...lightboxDrag}
            />
          )}
        </div>
      )}

      {/* Navigation arrows */}
      {lightboxIndex > 0 && (
        <button
          className="absolute left-2 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-black/50 text-white/70 hover:text-white hover:bg-black/70 transition-colors"
          onClick={(e) => { e.stopPropagation(); navigate(-1); }}
        >
          <ChevronLeft size={24} />
        </button>
      )}
      {lightboxIndex < maxIndex && (
        <button
          className="absolute right-2 top-1/2 -translate-y-1/2 w-10 h-10 flex items-center justify-center rounded-full bg-black/50 text-white/70 hover:text-white hover:bg-black/70 transition-colors"
          onClick={(e) => { e.stopPropagation(); navigate(1); }}
        >
          <ChevronRight size={24} />
        </button>
      )}

      {/* Bottom bar */}
      <div className="flex items-center justify-center px-4 py-1.5 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
        <span className="text-3xs text-white/40 tabular-nums">
          {lightboxIndex + 1} / {fileCount}
          {thumb && ` | ${thumb.width}x${thumb.height}`}
        </span>
      </div>
    </div>
  );
}

function LightboxButton({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-8 h-8 flex items-center justify-center rounded text-white/70 hover:text-white hover:bg-white/10 transition-colors"
    >
      {children}
    </button>
  );
}
