import { create } from "zustand";
import type { BrowserFolder, GalleryFile, GallerySort, CachedThumb } from "@/api/types/gallery";
import { getCachedFolder } from "@/lib/folderCache";

interface GalleryState {
  folders: BrowserFolder[];
  activeFolder: string | null;

  files: GalleryFile[];
  isLoadingFiles: boolean;
  loadProgress: { loaded: number; total: number | null };

  thumbs: Map<string, CachedThumb>;

  sortedFiles: GalleryFile[];

  selectedFile: GalleryFile | null;
  selectedThumb: CachedThumb | null;
  lightboxIndex: number | null;

  selectedIds: Set<string>;
  selectionAnchor: number | null;

  sort: GallerySort;
  searchQuery: string;
  thumbSize: number;
  layoutMode: "grid" | "masonry";
  metadataPanelOpen: boolean;

  setFolders: (f: BrowserFolder[]) => void;
  setActiveFolder: (path: string | null) => void;
  setFiles: (files: GalleryFile[]) => void;
  appendFile: (file: GalleryFile) => void;
  setLoadingFiles: (loading: boolean) => void;
  setLoadProgress: (loaded: number, total: number | null) => void;
  setThumb: (id: string, thumb: CachedThumb) => void;
  setThumbsBatch: (entries: [string, CachedThumb][]) => void;
  setSortedFiles: (files: GalleryFile[]) => void;
  selectFile: (file: GalleryFile | null, thumb?: CachedThumb | null) => void;
  openLightbox: (index: number) => void;
  closeLightbox: () => void;
  navigateLightbox: (delta: number, maxIndex: number) => void;
  setSort: (sort: GallerySort) => void;
  setSearchQuery: (q: string) => void;
  setThumbSize: (size: number) => void;
  setLayoutMode: (mode: "grid" | "masonry") => void;
  toggleMetadataPanel: () => void;
  toggleSelect: (fileId: string, index: number, shift: boolean, ctrl: boolean) => void;
  selectAll: () => void;
  deselectAll: () => void;
  removeFilesFromStore: (ids: string[]) => void;
  reset: () => void;
}

export const useGalleryStore = create<GalleryState>()((set) => ({
  folders: [],
  activeFolder: null,
  files: [],
  isLoadingFiles: false,
  loadProgress: { loaded: 0, total: null },
  thumbs: new Map(),
  sortedFiles: [],
  selectedFile: null,
  selectedThumb: null,
  lightboxIndex: null,
  selectedIds: new Set<string>(),
  selectionAnchor: null,
  sort: { field: "name", dir: "asc" },
  searchQuery: "",
  thumbSize: 180,
  layoutMode: "grid",
  metadataPanelOpen: true,

  setFolders: (folders) => set({ folders }),
  setActiveFolder: (path) => {
    if (path === null) {
      return set({ activeFolder: null, files: [], thumbs: new Map(), selectedFile: null, selectedThumb: null, lightboxIndex: null, selectedIds: new Set<string>(), selectionAnchor: null, isLoadingFiles: false, loadProgress: { loaded: 0, total: null } });
    }
    const cached = getCachedFolder(path);
    if (cached) {
      return set({ activeFolder: path, files: cached.files, thumbs: new Map(cached.thumbs), selectedFile: null, selectedThumb: null, lightboxIndex: null, selectedIds: new Set<string>(), selectionAnchor: null, isLoadingFiles: false, loadProgress: { loaded: cached.files.length, total: cached.files.length } });
    }
    return set({ activeFolder: path, files: [], thumbs: new Map(), selectedFile: null, selectedThumb: null, lightboxIndex: null, selectedIds: new Set<string>(), selectionAnchor: null, isLoadingFiles: false, loadProgress: { loaded: 0, total: null } });
  },
  setFiles: (files) => set({ files }),
  appendFile: (file) => set((s) => ({ files: [...s.files, file], loadProgress: { loaded: s.loadProgress.loaded + 1, total: s.loadProgress.total } })),
  setLoadingFiles: (loading) => set({ isLoadingFiles: loading }),
  setLoadProgress: (loaded, total) => set({ loadProgress: { loaded, total } }),
  setThumb: (id, thumb) => set((s) => { const m = new Map(s.thumbs); m.set(id, thumb); return { thumbs: m }; }),
  setThumbsBatch: (entries) => set((s) => {
    if (entries.length === 0) return {};
    const m = new Map(s.thumbs);
    for (const [id, thumb] of entries) m.set(id, thumb);
    return { thumbs: m };
  }),
  setSortedFiles: (sortedFiles) => set({ sortedFiles }),
  selectFile: (file, thumb) => set({ selectedFile: file, selectedThumb: thumb ?? null }),
  openLightbox: (index) => set({ lightboxIndex: index }),
  closeLightbox: () => set({ lightboxIndex: null }),
  navigateLightbox: (delta, maxIndex) => set((s) => {
    if (s.lightboxIndex === null) return {};
    const next = s.lightboxIndex + delta;
    if (next < 0 || next > maxIndex) return {};
    return { lightboxIndex: next };
  }),
  setSort: (sort) => set({ sort }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setThumbSize: (size) => set({ thumbSize: Math.max(120, Math.min(320, size)) }),
  setLayoutMode: (layoutMode) => set({ layoutMode }),
  toggleMetadataPanel: () => set((s) => ({ metadataPanelOpen: !s.metadataPanelOpen })),
  toggleSelect: (fileId, index, shift, ctrl) => set((s) => {
    if (shift && s.selectionAnchor !== null) {
      const start = Math.min(s.selectionAnchor, index);
      const end = Math.max(s.selectionAnchor, index);
      const next = new Set(s.selectedIds);
      for (let i = start; i <= end; i++) {
        if (i < s.sortedFiles.length) next.add(s.sortedFiles[i].id);
      }
      return { selectedIds: next };
    }
    if (ctrl) {
      const next = new Set(s.selectedIds);
      if (next.has(fileId)) next.delete(fileId);
      else next.add(fileId);
      return { selectedIds: next, selectionAnchor: index };
    }
    return { selectedIds: new Set([fileId]), selectionAnchor: index };
  }),
  selectAll: () => set((s) => ({ selectedIds: new Set(s.sortedFiles.map((f) => f.id)) })),
  deselectAll: () => set({ selectedIds: new Set<string>(), selectionAnchor: null }),
  removeFilesFromStore: (ids) => set((s) => {
    const idSet = new Set(ids);
    const files = s.files.filter((f) => !idSet.has(f.id));
    const sortedFiles = s.sortedFiles.filter((f) => !idSet.has(f.id));
    const thumbs = new Map(s.thumbs);
    const selectedIds = new Set(s.selectedIds);
    for (const id of ids) {
      thumbs.delete(id);
      selectedIds.delete(id);
    }
    const selectedFile = s.selectedFile && idSet.has(s.selectedFile.id) ? null : s.selectedFile;
    const selectedThumb = selectedFile ? s.selectedThumb : null;
    return { files, sortedFiles, thumbs, selectedIds, selectedFile, selectedThumb, loadProgress: { loaded: files.length, total: files.length } };
  }),
  reset: () => set({ folders: [], activeFolder: null, files: [], isLoadingFiles: false, loadProgress: { loaded: 0, total: null }, thumbs: new Map(), sortedFiles: [], selectedFile: null, selectedThumb: null, lightboxIndex: null, selectedIds: new Set<string>(), selectionAnchor: null, sort: { field: "name", dir: "asc" }, searchQuery: "", thumbSize: 180, layoutMode: "grid", metadataPanelOpen: true }),
}));
