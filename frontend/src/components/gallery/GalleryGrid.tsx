import { useMemo, useRef, useCallback, useEffect, useState } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useGalleryStore } from "@/stores/galleryStore";
import { useThumbnailLoader, useBackgroundPreloader } from "@/api/hooks/useGallery";
import type { GalleryFile } from "@/api/types/gallery";
import { ConnectedGalleryCard } from "./GalleryCard";
import { MasonryGrid } from "./MasonryGrid";
import { sendImageToCanvas, fetchRemoteImage } from "@/lib/sendTo";
import { ContextMenu, ContextMenuTrigger, ContextMenuContent, ContextMenuItem, ContextMenuSeparator } from "@/components/ui/context-menu";
import { Trash2, FolderInput, Download, Maximize2, Copy, Paintbrush, CheckSquare, XSquare, CheckCheck } from "lucide-react";

const GAP = 6;

interface GalleryGridProps {
  onDeleteRequest?: () => void;
  onMoveRequest?: () => void;
  onDownloadRequest?: () => void;
}

export function GalleryGrid({ onDeleteRequest, onMoveRequest, onDownloadRequest }: GalleryGridProps) {
  const files = useGalleryStore((s) => s.files);
  const sort = useGalleryStore((s) => s.sort);
  const searchQuery = useGalleryStore((s) => s.searchQuery);
  const thumbSize = useGalleryStore((s) => s.thumbSize);
  const setSortedFiles = useGalleryStore((s) => s.setSortedFiles);
  const layoutMode = useGalleryStore((s) => s.layoutMode);

  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800);

  // Track which card was right-clicked for context menu
  const contextFileRef = useRef<GalleryFile | null>(null);
  // Force-update only for context menu content rendering
  const [, setContextTick] = useState(0);

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
  const thumbsRef = useRef(useGalleryStore.getState().thumbs);
  useEffect(() => useGalleryStore.subscribe((s) => { thumbsRef.current = s.thumbs; }), []);

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

  // Compute visible file IDs for thumbnail loading — memoize on the actual
  // row range (two numbers) rather than the virtualizer array reference,
  // which is a new array on every render and would re-fire the thumb loader.
  const visibleRange = virtualizer.getVirtualItems();
  const rangeStart = visibleRange[0]?.index ?? 0;
  const rangeEnd = visibleRange.length > 0 ? visibleRange[visibleRange.length - 1].index : -1;
  const visibleFileIds = useMemo(() => {
    const ids: string[] = [];
    for (let row = rangeStart; row <= rangeEnd; row++) {
      const rowStart = row * cols;
      for (let c = 0; c < cols && rowStart + c < sorted.length; c++) {
        ids.push(sorted[rowStart + c].id);
      }
    }
    return ids;
  }, [rangeStart, rangeEnd, cols, sorted]);

  useThumbnailLoader(visibleFileIds, sorted);
  useBackgroundPreloader(sorted);

  const handleClick = useCallback((file: GalleryFile, index: number, e: React.MouseEvent) => {
    const s = useGalleryStore.getState();
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      s.toggleSelect(file.id, index, e.shiftKey, e.ctrlKey || e.metaKey);
    } else if (s.selectedIds.size > 0) {
      s.toggleSelect(file.id, index, false, false);
    } else {
      s.selectFile(file, s.thumbs.get(file.id) ?? null);
    }
  }, []);

  const handleDoubleClick = useCallback((index: number) => {
    useGalleryStore.getState().openLightbox(index);
  }, []);

  const handleContextMenu = useCallback((file: GalleryFile, index: number) => {
    contextFileRef.current = file;
    setContextTick((t) => t + 1);
    const s = useGalleryStore.getState();
    if (!s.selectedIds.has(file.id)) {
      s.toggleSelect(file.id, index, false, false);
    }
  }, []);

  const handleCopyPath = useCallback(() => {
    const f = contextFileRef.current;
    if (f) navigator.clipboard.writeText(f.fullPath).catch(() => {});
  }, []);

  const handleOpenLightbox = useCallback(() => {
    const f = contextFileRef.current;
    if (f) {
      const s = useGalleryStore.getState();
      const idx = s.sortedFiles.findIndex((sf) => sf.id === f.id);
      if (idx >= 0) s.openLightbox(idx);
    }
  }, []);

  const handleSelectCtx = useCallback(() => {
    const f = contextFileRef.current;
    if (f) {
      const s = useGalleryStore.getState();
      const idx = s.sortedFiles.indexOf(f);
      s.toggleSelect(f.id, idx >= 0 ? idx : 0, false, false);
    }
  }, []);

  const handleDeselectAll = useCallback(() => useGalleryStore.getState().deselectAll(), []);
  const handleSelectAll = useCallback(() => useGalleryStore.getState().selectAll(), []);

  const handleSendToCanvas = useCallback(() => {
    const f = contextFileRef.current;
    if (f) {
      fetchRemoteImage(`/file=${f.fullPath}`, f.relativePath.split("/").pop() ?? "image.png")
        .then((blob) => sendImageToCanvas(blob));
    }
  }, []);

  // Read selection count only for context menu labels — use primitive
  const selectionCount = useGalleryStore((s) => s.selectedIds.size);

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
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <div ref={containerRef} className="h-full overflow-auto">
          {layoutMode === "masonry" ? (
            <MasonryGrid sorted={sorted} containerRef={containerRef} containerWidth={containerWidth} />
          ) : (
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
                      return (
                        <ConnectedGalleryCard
                          key={file.id}
                          file={file}
                          index={idx}
                          size={thumbSize}
                          onClick={handleClick}
                          onDoubleClick={handleDoubleClick}
                          onContextMenu={handleContextMenu}
                        />
                      );
                    })}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent className="w-48">
        {selectionCount > 0 ? (
          <ContextMenuItem onClick={handleDeselectAll}>
            <XSquare size={14} /> Deselect all
          </ContextMenuItem>
        ) : (
          <ContextMenuItem onClick={handleSelectCtx}>
            <CheckSquare size={14} /> Select
          </ContextMenuItem>
        )}
        <ContextMenuItem onClick={handleSelectAll}>
          <CheckCheck size={14} /> Select all
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem variant="destructive" onClick={onDeleteRequest}>
          <Trash2 size={14} /> Delete{selectionCount > 1 ? ` (${selectionCount})` : ""}
        </ContextMenuItem>
        <ContextMenuItem onClick={onMoveRequest}>
          <FolderInput size={14} /> Move to...
        </ContextMenuItem>
        <ContextMenuItem onClick={onDownloadRequest}>
          <Download size={14} /> Download{selectionCount > 1 ? ` (${selectionCount})` : ""}
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onClick={handleOpenLightbox}>
          <Maximize2 size={14} /> Open in lightbox
        </ContextMenuItem>
        <ContextMenuItem onClick={handleCopyPath}>
          <Copy size={14} /> Copy path
        </ContextMenuItem>
        <ContextMenuItem onClick={handleSendToCanvas}>
          <Paintbrush size={14} /> Send to canvas
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
}
