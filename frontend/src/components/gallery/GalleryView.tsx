import { useCallback, useState } from "react";
import { useBrowserFiles, useDeleteFiles, useMoveFiles, useDownloadFiles } from "@/api/hooks/useGallery";
import { useGalleryStore } from "@/stores/galleryStore";
import { useShortcut } from "@/hooks/useShortcut";
import { useShortcutScope } from "@/hooks/useShortcutScope";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";
import { GalleryToolbar } from "./GalleryToolbar";
import { GalleryProgress } from "./GalleryProgress";
import { GalleryGrid } from "./GalleryGrid";
import { GalleryLightbox } from "./GalleryLightbox";
import { GalleryMetadata } from "./GalleryMetadata";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";
import { MoveToDialog } from "./MoveToDialog";
import { FolderOpen } from "lucide-react";

export function GalleryView() {
  const activeFolder = useGalleryStore((s) => s.activeFolder);
  const fileCount = useGalleryStore((s) => s.files.length);
  const metadataPanelOpen = useGalleryStore((s) => s.metadataPanelOpen);
  const selectionCount = useGalleryStore((s) => s.selectedIds.size);

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [moveDialogOpen, setMoveDialogOpen] = useState(false);

  const deleteMutation = useDeleteFiles();
  const moveMutation = useMoveFiles();
  const downloadMutation = useDownloadFiles();

  useBrowserFiles(activeFolder);
  useShortcutScope("gallery");
  useShortcut("gallery-toggle-info", () => useGalleryStore.getState().toggleMetadataPanel());
  useShortcut("gallery-select-all", (e) => { e.preventDefault(); useGalleryStore.getState().selectAll(); });
  useShortcut("gallery-deselect", () => useGalleryStore.getState().deselectAll(), selectionCount > 0);
  useShortcut("gallery-delete", () => { if (useGalleryStore.getState().selectedIds.size > 0) setDeleteDialogOpen(true); }, selectionCount > 0);

  const getSelectedPaths = useCallback(() => {
    const store = useGalleryStore.getState();
    return store.files.filter((f) => store.selectedIds.has(f.id)).map((f) => f.fullPath);
  }, []);

  const handleDeleteRequest = useCallback(() => {
    if (useGalleryStore.getState().selectedIds.size > 0) setDeleteDialogOpen(true);
  }, []);

  const handleDeleteConfirm = useCallback(() => {
    const paths = getSelectedPaths();
    if (paths.length === 0) return;
    deleteMutation.mutate(paths, { onSettled: () => setDeleteDialogOpen(false) });
  }, [getSelectedPaths, deleteMutation]);

  const handleMoveRequest = useCallback(() => {
    if (useGalleryStore.getState().selectedIds.size > 0) setMoveDialogOpen(true);
  }, []);

  const handleMoveConfirm = useCallback((destination: string) => {
    const paths = getSelectedPaths();
    if (paths.length === 0) return;
    moveMutation.mutate({ files: paths, destination }, { onSettled: () => setMoveDialogOpen(false) });
  }, [getSelectedPaths, moveMutation]);

  const handleDownloadRequest = useCallback(() => {
    const paths = getSelectedPaths();
    if (paths.length === 0) return;
    downloadMutation.mutate(paths);
  }, [getSelectedPaths, downloadMutation]);

  const filteredCount = useGalleryStore((s) => {
    if (!s.searchQuery) return s.files.length;
    const q = s.searchQuery.toLowerCase();
    return s.files.filter((f) => f.relativePath.toLowerCase().includes(q)).length;
  });

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
      <GalleryToolbar
        totalCount={fileCount}
        filteredCount={filteredCount}
        onDeleteRequest={handleDeleteRequest}
        onMoveRequest={handleMoveRequest}
        onDownloadRequest={handleDownloadRequest}
      />
      <GalleryProgress />
      <ResizablePanelGroup orientation="horizontal" className="flex-1 min-h-0">
        <ResizablePanel defaultSize={70} minSize={40}>
          <GalleryGrid
            onDeleteRequest={handleDeleteRequest}
            onMoveRequest={handleMoveRequest}
            onDownloadRequest={handleDownloadRequest}
          />
        </ResizablePanel>
        {metadataPanelOpen && (
          <>
            <ResizableHandle />
            <ResizablePanel defaultSize={30} minSize={15} maxSize={50}>
              <GalleryMetadata />
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
      <GalleryLightbox />

      <DeleteConfirmDialog
        open={deleteDialogOpen}
        count={selectionCount}
        isPending={deleteMutation.isPending}
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteDialogOpen(false)}
      />
      <MoveToDialog
        open={moveDialogOpen}
        count={selectionCount}
        isPending={moveMutation.isPending}
        onConfirm={handleMoveConfirm}
        onCancel={() => setMoveDialogOpen(false)}
      />
    </div>
  );
}
