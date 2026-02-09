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

  sort: GallerySort;
  searchQuery: string;
  thumbSize: number;

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
  sort: { field: "name", dir: "asc" },
  searchQuery: "",
  thumbSize: 180,

  setFolders: (folders) => set({ folders }),
  setActiveFolder: (path) => {
    if (path === null) {
      return set({ activeFolder: null, files: [], thumbs: new Map(), selectedFile: null, selectedThumb: null, lightboxIndex: null, isLoadingFiles: false, loadProgress: { loaded: 0, total: null } });
    }
    const cached = getCachedFolder(path);
    if (cached) {
      return set({ activeFolder: path, files: cached.files, thumbs: new Map(cached.thumbs), selectedFile: null, selectedThumb: null, lightboxIndex: null, isLoadingFiles: false, loadProgress: { loaded: cached.files.length, total: cached.files.length } });
    }
    return set({ activeFolder: path, files: [], thumbs: new Map(), selectedFile: null, selectedThumb: null, lightboxIndex: null, isLoadingFiles: false, loadProgress: { loaded: 0, total: null } });
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
  reset: () => set({ folders: [], activeFolder: null, files: [], isLoadingFiles: false, loadProgress: { loaded: 0, total: null }, thumbs: new Map(), sortedFiles: [], selectedFile: null, selectedThumb: null, lightboxIndex: null, sort: { field: "name", dir: "asc" }, searchQuery: "", thumbSize: 180 }),
}));
