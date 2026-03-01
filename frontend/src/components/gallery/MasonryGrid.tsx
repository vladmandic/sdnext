import { useMemo, useRef, useCallback } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useGalleryStore } from "@/stores/galleryStore";
import type { GalleryFile, CachedThumb } from "@/api/types/gallery";
import { GalleryCard } from "./GalleryCard";

const GAP = 6;

interface MasonryItem {
  file: GalleryFile;
  col: number;
  y: number;
  height: number;
  globalIndex: number;
}

interface MasonryRow {
  items: MasonryItem[];
  y: number;
  height: number;
}

interface MasonryGridProps {
  sorted: GalleryFile[];
  containerRef: React.RefObject<HTMLDivElement | null>;
  containerWidth: number;
}

export function MasonryGrid({ sorted, containerRef, containerWidth }: MasonryGridProps) {
  const thumbs = useGalleryStore((s) => s.thumbs);
  const thumbSize = useGalleryStore((s) => s.thumbSize);
  const selectedFile = useGalleryStore((s) => s.selectedFile);
  const selectFile = useGalleryStore((s) => s.selectFile);
  const openLightbox = useGalleryStore((s) => s.openLightbox);
  const selectedIds = useGalleryStore((s) => s.selectedIds);
  const toggleSelect = useGalleryStore((s) => s.toggleSelect);

  const isSelectMode = selectedIds.size > 0;

  // Snapshot thumbs to avoid layout thrash on individual thumb loads
  const thumbsRef = useRef(thumbs);
  thumbsRef.current = thumbs;

  // Compute masonry layout: bin-pack items into columns
  const { rows, totalHeight } = useMemo(() => {
    const cols = Math.max(2, Math.floor((containerWidth + GAP) / (thumbSize + GAP)));
    const colWidth = (containerWidth - (cols + 1) * GAP) / cols;
    const colHeights = new Array(cols).fill(GAP);
    const currentThumbs = thumbsRef.current;
    const items: MasonryItem[] = [];

    for (let i = 0; i < sorted.length; i++) {
      const file = sorted[i];
      const thumb = currentThumbs.get(file.id);
      const aspect = thumb && thumb.width > 0 ? thumb.height / thumb.width : 1;
      const itemHeight = Math.round(colWidth * aspect);

      // Find shortest column
      let minCol = 0;
      for (let c = 1; c < cols; c++) {
        if (colHeights[c] < colHeights[minCol]) minCol = c;
      }

      const y = colHeights[minCol];
      items.push({ file, col: minCol, y, height: itemHeight, globalIndex: i });
      colHeights[minCol] = y + itemHeight + GAP;
    }

    // Group items into rows by y bands (for virtualization)
    const bandSize = thumbSize;
    const itemsByBand = new Map<number, MasonryItem[]>();
    for (const item of items) {
      const band = Math.floor(item.y / bandSize);
      if (!itemsByBand.has(band)) itemsByBand.set(band, []);
      itemsByBand.get(band)!.push(item);
    }

    const bandKeys = [...itemsByBand.keys()].sort((a, b) => a - b);
    const masonryRows: MasonryRow[] = [];
    for (const band of bandKeys) {
      const bandItems = itemsByBand.get(band)!;
      const minY = Math.min(...bandItems.map((it) => it.y));
      const maxBottom = Math.max(...bandItems.map((it) => it.y + it.height));
      masonryRows.push({ items: bandItems, y: minY, height: maxBottom - minY + GAP });
    }

    const total = Math.max(...colHeights);
    return { rows: masonryRows, totalHeight: total, cols, colWidth };
  }, [sorted, containerWidth, thumbSize]);

  // eslint-disable-next-line react-hooks/incompatible-library -- @tanstack/react-virtual is compatible; compiler limitation
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => containerRef.current,
    estimateSize: (i) => rows[i]?.height ?? thumbSize,
    overscan: 3,
  });

  const handleClick = useCallback((file: GalleryFile, thumb: CachedThumb | undefined, index: number, e: React.MouseEvent) => {
    if (e.shiftKey || e.ctrlKey || e.metaKey) {
      toggleSelect(file.id, index, e.shiftKey, e.ctrlKey || e.metaKey);
    } else if (isSelectMode) {
      toggleSelect(file.id, index, false, false);
    } else {
      selectFile(file, thumb ?? null);
    }
  }, [selectFile, toggleSelect, isSelectMode]);

  const handleDoubleClick = useCallback((index: number) => {
    openLightbox(index);
  }, [openLightbox]);

  const cols = Math.max(2, Math.floor((containerWidth + GAP) / (thumbSize + GAP)));
  const colWidth = (containerWidth - (cols + 1) * GAP) / cols;

  return (
    <div className="relative w-full" style={{ height: totalHeight }}>
      {virtualizer.getVirtualItems().map((vRow) => {
        const row = rows[vRow.index];
        if (!row) return null;
        return (
          <div key={vRow.index} className="absolute left-0 w-full" style={{ top: row.y }}>
            {row.items.map((item) => {
              const thumb = thumbs.get(item.file.id);
              const x = GAP + item.col * (colWidth + GAP);
              return (
                <div
                  key={item.file.id}
                  className="absolute"
                  style={{ left: x, top: item.y - row.y, width: colWidth }}
                >
                  <GalleryCard
                    file={item.file}
                    thumb={thumb}
                    size={Math.round(colWidth)}
                    height={item.height}
                    selected={selectedFile?.id === item.file.id}
                    isSelected={selectedIds.has(item.file.id)}
                    isSelectMode={isSelectMode}
                    onClick={(e) => handleClick(item.file, thumb, item.globalIndex, e)}
                    onDoubleClick={() => handleDoubleClick(item.globalIndex)}
                  />
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
