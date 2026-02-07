import { useMemo } from "react";
import { useBrowserFiles } from "@/api/hooks/useGallery";
import { useGalleryStore } from "@/stores/galleryStore";
import { GalleryToolbar } from "./GalleryToolbar";
import { GalleryProgress } from "./GalleryProgress";
import { GalleryGrid } from "./GalleryGrid";
import { GalleryLightbox } from "./GalleryLightbox";
import { FolderOpen } from "lucide-react";

export function GalleryView() {
  const activeFolder = useGalleryStore((s) => s.activeFolder);
  const files = useGalleryStore((s) => s.files);
  const searchQuery = useGalleryStore((s) => s.searchQuery);

  useBrowserFiles(activeFolder);

  const filteredCount = useMemo(() => {
    if (!searchQuery) return files.length;
    const q = searchQuery.toLowerCase();
    return files.filter((f) => f.relativePath.toLowerCase().includes(q)).length;
  }, [files, searchQuery]);

  if (!activeFolder) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-2">
        <FolderOpen size={32} className="opacity-30" />
        <p className="text-sm">Select a folder to browse</p>
        <p className="text-xs opacity-60">Choose a folder from the left panel</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <GalleryToolbar totalCount={files.length} filteredCount={filteredCount} />
      <GalleryProgress />
      <GalleryGrid />
      <GalleryLightbox />
    </div>
  );
}
