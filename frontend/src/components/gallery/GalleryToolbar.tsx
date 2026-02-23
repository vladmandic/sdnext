import { useCallback, useRef, useState, useEffect, useMemo } from "react";
import { useGalleryStore } from "@/stores/galleryStore";
import type { GallerySortField, GallerySortDir } from "@/api/types/gallery";
import { Input } from "@/components/ui/input";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";
import { Search, FolderOpen } from "lucide-react";

const SORT_OPTIONS: { value: string; label: string }[] = [
  { value: "name-asc", label: "Name A-Z" },
  { value: "name-desc", label: "Name Z-A" },
  { value: "mtime-desc", label: "Newest" },
  { value: "mtime-asc", label: "Oldest" },
  { value: "size-desc", label: "Largest" },
  { value: "size-asc", label: "Smallest" },
  { value: "width-desc", label: "Widest" },
  { value: "width-asc", label: "Narrowest" },
];

interface GalleryToolbarProps {
  totalCount: number;
  filteredCount: number;
}

export function GalleryToolbar({ totalCount, filteredCount }: GalleryToolbarProps) {
  const sort = useGalleryStore((s) => s.sort);
  const setSort = useGalleryStore((s) => s.setSort);
  const setSearchQuery = useGalleryStore((s) => s.setSearchQuery);
  const thumbSize = useGalleryStore((s) => s.thumbSize);
  const setThumbSize = useGalleryStore((s) => s.setThumbSize);
  const activeFolder = useGalleryStore((s) => s.activeFolder);
  const folders = useGalleryStore((s) => s.folders);

  const [localSearch, setLocalSearch] = useState("");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleSearch = useCallback((value: string) => {
    setLocalSearch(value);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setSearchQuery(value), 300);
  }, [setSearchQuery]);

  useEffect(() => () => { if (debounceRef.current) clearTimeout(debounceRef.current); }, []);

  const sortValue = `${sort.field}-${sort.dir}`;
  const handleSortChange = (v: string) => {
    const [field, dir] = v.split("-") as [GallerySortField, GallerySortDir];
    setSort({ field, dir });
  };

  // Build a short display path: use label if available, otherwise derive from path
  const folderDisplay = useMemo(() => {
    if (!activeFolder) return null;
    const match = folders.find((f) => f.path === activeFolder);
    if (!match) return activeFolder;
    // Show the label, and if path differs significantly show the trailing path segments
    const label = match.label;
    const pathParts = activeFolder.replace(/\/+$/, "").split("/");
    // If label already captures the meaning, just show it
    // Otherwise show last 2-3 path segments for context
    if (label && label !== activeFolder) return label;
    return pathParts.slice(-2).join("/");
  }, [activeFolder, folders]);

  return (
    <div className="flex flex-col border-b border-border flex-shrink-0">
      {/* Path breadcrumb */}
      {folderDisplay && (
        <div className="flex items-center gap-1.5 px-3 py-1 border-b border-border/50 bg-muted/30">
          <FolderOpen size={11} className="text-muted-foreground flex-shrink-0" />
          <span className="text-3xs text-muted-foreground truncate" title={activeFolder ?? ""}>
            {folderDisplay}
          </span>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-2 px-3 py-1.5">
        <div className="relative flex-1 max-w-xs">
          <Search size={13} className="absolute left-2 top-1/2 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search files..."
            value={localSearch}
            onChange={(e) => handleSearch(e.target.value)}
            className="h-6 text-2xs pl-7"
          />
        </div>

        <Combobox
          value={sortValue}
          onValueChange={handleSortChange}
          options={SORT_OPTIONS}
          className="w-28 h-6 text-2xs"
        />

        <div className="flex items-center gap-1.5 min-w-[6.25rem]">
          <span className="text-3xs text-muted-foreground">Size</span>
          <Slider
            value={[thumbSize]}
            min={120}
            max={320}
            step={20}
            onValueChange={([v]) => setThumbSize(v)}
            className="w-16"
          />
        </div>

        <span className="text-3xs text-muted-foreground tabular-nums ml-auto whitespace-nowrap">
          {filteredCount === totalCount
            ? `${totalCount} file${totalCount !== 1 ? "s" : ""}`
            : `${filteredCount} / ${totalCount}`}
        </span>
      </div>
    </div>
  );
}
