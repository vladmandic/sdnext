import { useCallback, useRef, useState, useEffect, useMemo } from "react";
import { useGalleryStore } from "@/stores/galleryStore";
import type { GallerySortField, GallerySortDir } from "@/api/types/gallery";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Combobox } from "@/components/ui/combobox";
import { Slider } from "@/components/ui/slider";
import { Search, FolderOpen, PanelRight, Trash2, FolderInput, Download, CheckCheck, XSquare, LayoutGrid, Rows3 } from "lucide-react";
import { cn } from "@/lib/utils";

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
  onDeleteRequest?: () => void;
  onMoveRequest?: () => void;
  onDownloadRequest?: () => void;
}

export function GalleryToolbar({ totalCount, filteredCount, onDeleteRequest, onMoveRequest, onDownloadRequest }: GalleryToolbarProps) {
  const sort = useGalleryStore((s) => s.sort);
  const setSort = useGalleryStore((s) => s.setSort);
  const setSearchQuery = useGalleryStore((s) => s.setSearchQuery);
  const thumbSize = useGalleryStore((s) => s.thumbSize);
  const setThumbSize = useGalleryStore((s) => s.setThumbSize);
  const activeFolder = useGalleryStore((s) => s.activeFolder);
  const folders = useGalleryStore((s) => s.folders);
  const layoutMode = useGalleryStore((s) => s.layoutMode);
  const setLayoutMode = useGalleryStore((s) => s.setLayoutMode);
  const metadataPanelOpen = useGalleryStore((s) => s.metadataPanelOpen);
  const toggleMetadataPanel = useGalleryStore((s) => s.toggleMetadataPanel);
  const selectionCount = useGalleryStore((s) => s.selectedIds.size);
  const selectAll = useGalleryStore((s) => s.selectAll);
  const deselectAll = useGalleryStore((s) => s.deselectAll);

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
    const label = match.label;
    const pathParts = activeFolder.replace(/\/+$/, "").split("/");
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

      {/* Selection action bar */}
      {selectionCount > 0 && (
        <div className="flex items-center gap-1.5 px-3 py-1 bg-primary/5 border-b border-primary/20">
          <span className="text-2xs font-medium text-primary tabular-nums">{selectionCount} selected</span>
          <Button variant="ghost" size="sm" className="h-5 px-1.5 text-3xs gap-1" onClick={selectAll}>
            <CheckCheck size={12} /> All
          </Button>
          <Button variant="ghost" size="sm" className="h-5 px-1.5 text-3xs gap-1" onClick={deselectAll}>
            <XSquare size={12} /> None
          </Button>
          <div className="ml-auto flex items-center gap-1">
            <Button variant="ghost" size="sm" className="h-5 px-1.5 text-3xs gap-1 text-destructive hover:text-destructive" onClick={onDeleteRequest}>
              <Trash2 size={12} /> Delete
            </Button>
            <Button variant="ghost" size="sm" className="h-5 px-1.5 text-3xs gap-1" onClick={onMoveRequest}>
              <FolderInput size={12} /> Move
            </Button>
            <Button variant="ghost" size="sm" className="h-5 px-1.5 text-3xs gap-1" onClick={onDownloadRequest}>
              <Download size={12} /> Download
            </Button>
          </div>
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

        <div className="flex items-center border border-border rounded-md overflow-hidden">
          <button
            onClick={() => setLayoutMode("grid")}
            title="Grid layout"
            className={cn(
              "p-1 transition-colors",
              layoutMode === "grid" ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground hover:bg-muted",
            )}
          >
            <LayoutGrid size={13} />
          </button>
          <button
            onClick={() => setLayoutMode("masonry")}
            title="Masonry layout"
            className={cn(
              "p-1 transition-colors",
              layoutMode === "masonry" ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground hover:bg-muted",
            )}
          >
            <Rows3 size={13} />
          </button>
        </div>

        <button
          onClick={toggleMetadataPanel}
          title="Toggle metadata panel (I)"
          className={cn(
            "p-1 rounded-md transition-colors",
            metadataPanelOpen ? "bg-primary/15 text-primary" : "text-muted-foreground hover:text-foreground hover:bg-muted",
          )}
        >
          <PanelRight size={14} />
        </button>
      </div>
    </div>
  );
}
