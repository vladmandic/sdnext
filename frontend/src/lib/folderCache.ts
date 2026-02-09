import type { GalleryFile, CachedThumb } from "@/api/types/gallery";

export interface FolderSnapshot {
  files: GalleryFile[];
  thumbs: Map<string, CachedThumb>; // keyed by file.id
  serverMtime: number; // from backend folder-info endpoint
  cachedAt: number;
}

const MAX_CACHED_FOLDERS = 8;

/** Module-level LRU cache — survives Zustand resets and component unmounts. */
const cache = new Map<string, FolderSnapshot>();

/** Touch entry to mark it as most-recently-used (move to end of Map iteration order). */
function touch(key: string) {
  const snap = cache.get(key);
  if (snap) {
    cache.delete(key);
    cache.set(key, snap);
  }
}

/** Evict the least-recently-used entry if over capacity. */
function evictIfNeeded() {
  while (cache.size > MAX_CACHED_FOLDERS) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
    else break;
  }
}

export function getCachedFolder(path: string): FolderSnapshot | undefined {
  const snap = cache.get(path);
  if (snap) touch(path);
  return snap;
}

export function setCachedFolder(path: string, files: GalleryFile[], thumbs: Map<string, CachedThumb>, serverMtime: number): void {
  cache.set(path, { files, thumbs: new Map(thumbs), serverMtime, cachedAt: Date.now() });
  evictIfNeeded();
}

/** Merge new thumb entries into an existing folder snapshot (mutates in place). */
export function updateCachedThumbs(folder: string, entries: [string, CachedThumb][]): void {
  const snap = cache.get(folder);
  if (!snap || entries.length === 0) return;
  for (const [id, thumb] of entries) snap.thumbs.set(id, thumb);
}

export function invalidateFolder(path: string): void {
  cache.delete(path);
}
