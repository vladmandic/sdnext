import { useMemo, useRef, useCallback, useEffect, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useGalleryStore } from "@/stores/galleryStore";
import { useThumbnailLoader, useBackgroundPreloader } from "@/api/hooks/useGallery";
import type { GalleryFile, CachedThumb } from "@/api/types/gallery";
import { GalleryCard } from "./GalleryCard";

const GAP = 6;

export function GalleryGrid() {
  const files = useGalleryStore((s) => s.files);
  const thumbs = useGalleryStore((s) => s.thumbs);
  const sort = useGalleryStore((s) => s.sort);
  const searchQuery = useGalleryStore((s) => s.searchQuery);
  const thumbSize = useGalleryStore((s) => s.thumbSize);
  const selectedFile = useGalleryStore((s) => s.selectedFile);
  const selectFile = useGalleryStore((s) => s.selectFile);
  const setSortedFiles = useGalleryStore((s) => s.setSortedFiles);
  const openLightbox = useGalleryStore((s) => s.openLightbox);

  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800);

  // ResizeObserver for responsive columns
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w && w > 0) setContainerWidth(w);
    });
    observer.observe(el);
    setContainerWidth(el.clientWidth || 800);
    return () => observer.disconnect();
  }, []);

  // Keep a non-reactive ref to thumbs for sort comparisons so that
  // individual thumb loads don't trigger re-sort (which would cascade
  // into visibleFileIds change → thumbnail loader abort → stall).
  const thumbsRef = useRef(thumbs);
  thumbsRef.current = thumbs;

  // Filter + sort — only recompute when files/search/sort change, NOT on every thumb load
  const sorted = useMemo(() => {
    let result = files;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      result = result.filter((f) => f.relativePath.toLowerCase().includes(q));
    }
    const currentThumbs = thumbsRef.current;
    result = [...result].sort((a, b) => {
      const ta = currentThumbs.get(a.id);
      const tb = currentThumbs.get(b.id);
      let cmp = 0;
      switch (sort.field) {
        case "name":
          cmp = a.relativePath.localeCompare(b.relativePath);
          break;
        case "mtime":
          cmp = (ta?.mtime ?? 0) - (tb?.mtime ?? 0);
          break;
        case "size":
          cmp = (ta?.size ?? 0) - (tb?.size ?? 0);
          break;
        case "width":
          cmp = (ta?.width ?? 0) - (tb?.width ?? 0);
          break;
      }
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return result;
  }, [files, searchQuery, sort]);

  // Sync sorted list to store so lightbox can use it
  useEffect(() => {
    setSortedFiles(sorted);
  }, [sorted, setSortedFiles]);

  const cols = Math.max(1, Math.floor((containerWidth + GAP) / (thumbSize + GAP)));
  const rowCount = Math.ceil(sorted.length / cols);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => containerRef.current,
    estimateSize: () => thumbSize + GAP,
    overscan: 3,
  });

  // Compute visible file IDs for thumbnail loading
  const visibleRange = virtualizer.getVirtualItems();
  const visibleFileIds = useMemo(() => {
    const ids: string[] = [];
    for (const vItem of visibleRange) {
      const rowStart = vItem.index * cols;
      for (let c = 0; c < cols && rowStart + c < sorted.length; c++) {
        ids.push(sorted[rowStart + c].id);
      }
    }
    return ids;
  }, [visibleRange, cols, sorted]);

  useThumbnailLoader(visibleFileIds, sorted);
  useBackgroundPreloader(sorted);

  const handleSelect = useCallback((file: GalleryFile, thumb: CachedThumb | undefined) => {
    selectFile(file, thumb ?? null);
  }, [selectFile]);

  const handleDoubleClick = useCallback((index: number) => {
    openLightbox(index);
  }, [openLightbox]);

  if (files.length === 0) {
    return (
      <div ref={containerRef} className="flex flex-col items-center justify-center h-full text-muted-foreground overflow-auto">
        <p className="text-sm">No images</p>
        <p className="text-xs mt-1 opacity-60">Select a folder to browse</p>
      </div>
    );
  }

  if (sorted.length === 0) {
    return (
      <div ref={containerRef} className="flex flex-col items-center justify-center h-full text-muted-foreground overflow-auto">
        <p className="text-sm">No matches</p>
        <p className="text-xs mt-1 opacity-60">Try a different search term</p>
      </div>
    );
  }

  return (
    <div ref={containerRef} className="flex-1 overflow-auto">
      <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
        {virtualizer.getVirtualItems().map((vRow) => {
          const rowStart = vRow.index * cols;
          return (
            <div
              key={vRow.index}
              className="absolute left-0 w-full flex justify-start"
              style={{ top: vRow.start, height: vRow.size, gap: GAP, padding: `0 ${GAP}px` }}
            >
              {Array.from({ length: cols }, (_, c) => {
                const idx = rowStart + c;
                if (idx >= sorted.length) return null;
                const file = sorted[idx];
                const thumb = thumbs.get(file.id);
                return (
                  <GalleryCard
                    key={file.id}
                    file={file}
                    thumb={thumb}
                    size={thumbSize}
                    selected={selectedFile?.id === file.id}
                    onClick={() => handleSelect(file, thumb)}
                    onDoubleClick={() => handleDoubleClick(idx)}
                  />
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
