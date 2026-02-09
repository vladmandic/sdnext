import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { BrowserFolder, BrowserSubdir, BrowserThumb, GalleryFile, CachedThumb } from "../types/gallery";
import { useGalleryStore } from "@/stores/galleryStore";
import { computeThumbHash, getThumb, putThumb, batchGetThumbs } from "@/lib/thumbnailCache";
import { getCachedFolder, setCachedFolder, updateCachedThumbs, mergeChildCaches } from "@/lib/folderCache";
import { Semaphore } from "@/lib/concurrency";

const thumbSemaphore = new Semaphore(16);

/** Module-level set shared between visible loader and background preloader. */
const inflightIds = new Set<string>();

export function useBrowserFolders() {
  return useQuery({
    queryKey: ["browser", "folders"],
    queryFn: () => api.get<BrowserFolder[]>("/sdapi/v1/browser/folders"),
    staleTime: 60_000,
  });
}

export function useSubdirs(folder: string | null) {
  return useQuery({
    queryKey: ["browser", "subdirs", folder],
    queryFn: () => api.get<BrowserSubdir[]>("/sdapi/v1/browser/subdirs", { folder: folder! }),
    enabled: folder !== null,
    staleTime: 30_000,
  });
}

function parseFileEntry(raw: string): GalleryFile {
  const sep = "##F##";
  const idx = raw.indexOf(sep);
  if (idx < 0) {
    const decoded = decodeURIComponent(raw);
    return { folder: "", relativePath: decoded, fullPath: decoded, id: raw };
  }
  let folderEnc = raw.slice(0, idx);
  let relEnc = raw.slice(idx + sep.length);
  // Restore Windows drive letter colons
  if (folderEnc.length > 3 && folderEnc[1] === "%" && folderEnc[2] === "3" && folderEnc[3] === "A") {
    folderEnc = folderEnc[0] + ":" + folderEnc.slice(4);
  }
  if (relEnc.length > 3 && relEnc[1] === "%" && relEnc[2] === "3" && relEnc[3] === "A") {
    relEnc = relEnc[0] + ":" + relEnc.slice(4);
  }
  const folder = decodeURIComponent(folderEnc);
  const relativePath = decodeURIComponent(relEnc);
  const fullPath = folder + "/" + relativePath;
  return { folder, relativePath, fullPath, id: raw };
}

const MEDIA_EXTENSIONS = new Set([
  // Images
  ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".tif", ".avif",
  // Video
  ".mp4", ".webm", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4v",
]);

function isMediaFile(path: string): boolean {
  const dot = path.lastIndexOf(".");
  if (dot < 0) return false;
  return MEDIA_EXTENSIONS.has(path.slice(dot).toLowerCase());
}

const FILE_BATCH_INTERVAL = 150; // ms between flushing accumulated files to the store

// ---------------------------------------------------------------------------
// Thumb preloading (batch + parallel)
// ---------------------------------------------------------------------------

/**
 * After all files are loaded, batch-check IndexedDB for cached thumbnails
 * and pre-populate the in-memory store in a single update.
 */
async function preloadCachedThumbs(files: GalleryFile[], folder: string) {
  if (files.length === 0) return;
  try {
    // Compute hashes in parallel (crypto.subtle.digest is async but very fast)
    const pairs = await Promise.all(
      files.map(async (f) => ({ id: f.id, hash: await computeThumbHash(f.fullPath) })),
    );

    const hashToId = new Map<string, string>();
    const hashes: string[] = [];
    for (const { id, hash } of pairs) {
      hashToId.set(hash, id);
      hashes.push(hash);
    }

    // Batch-read from IndexedDB in a single transaction
    const cached = await batchGetThumbs(hashes);

    if (cached.size > 0) {
      const batch: [string, CachedThumb][] = [];
      for (const [hash, thumb] of cached) {
        const fileId = hashToId.get(hash);
        if (fileId) batch.push([fileId, thumb]);
      }
      // Single Zustand update instead of N individual setThumb calls
      useGalleryStore.getState().setThumbsBatch(batch);
      // Keep folder cache in sync
      updateCachedThumbs(folder, batch);
    }
  } catch {
    // IndexedDB errors are non-fatal
  }
}

// ---------------------------------------------------------------------------
// WebSocket file streaming (shared between initial load and background refresh)
// ---------------------------------------------------------------------------

interface StreamResult {
  files: GalleryFile[];
}

/**
 * Open a WebSocket to stream file entries for a folder.
 * Returns a promise that resolves with the file list when #END# is received.
 * If `onFile` is provided, it is called for each file as they arrive (for progress).
 */
function streamFiles(
  folder: string,
  signal: AbortSignal,
  onFile?: (file: GalleryFile) => void,
): Promise<StreamResult> {
  return new Promise((resolve, reject) => {
    const wsUrl = api.getWebSocketUrl("/sdapi/v1/browser/files");
    const ws = new WebSocket(wsUrl);
    const allFiles: GalleryFile[] = [];

    const cleanup = () => {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
    };

    signal.addEventListener("abort", cleanup, { once: true });

    ws.onopen = () => ws.send(folder);

    ws.onmessage = (ev) => {
      const msg = ev.data as string;
      if (msg === "#END#") {
        ws.close();
        resolve({ files: allFiles });
        return;
      }
      const entry = parseFileEntry(msg);
      if (isMediaFile(entry.relativePath)) {
        allFiles.push(entry);
        onFile?.(entry);
      }
    };

    ws.onerror = () => reject(new Error("WebSocket error"));
    ws.onclose = (ev) => {
      // If we already resolved via #END#, this is a no-op (promise only resolves once)
      if (ev.code !== 1000) reject(new Error("WebSocket closed unexpectedly"));
      else resolve({ files: allFiles });
    };
  });
}

// ---------------------------------------------------------------------------
// Background refresh — check if folder changed, update if needed
// ---------------------------------------------------------------------------

async function backgroundRefresh(folder: string, signal: AbortSignal) {
  try {
    const info = await api.get<{ mtime: number }>("/sdapi/v1/browser/folder-info", { folder });
    const cached = getCachedFolder(folder);
    if (!cached || info.mtime === cached.serverMtime) return;

    // Mtime differs — re-stream file list silently
    const { files: newFiles } = await streamFiles(folder, signal);
    if (signal.aborted) return;

    // Diff: find files only in newFiles (added)
    const oldIds = new Set(cached.files.map((f) => f.id));
    const addedFiles = newFiles.filter((f) => !oldIds.has(f.id));

    // Build new thumb map: carry over existing thumbs, drop removed files
    const newIds = new Set(newFiles.map((f) => f.id));
    const newThumbs = new Map<string, CachedThumb>();
    for (const [id, thumb] of cached.thumbs) {
      if (newIds.has(id)) newThumbs.set(id, thumb);
    }

    // Update folder cache
    setCachedFolder(folder, newFiles, newThumbs, info.mtime);

    // Update store only if user is still on this folder
    const store = useGalleryStore.getState();
    if (store.activeFolder === folder) {
      store.setFiles(newFiles);
      store.setThumbsBatch(Array.from(newThumbs));
      store.setLoadProgress(newFiles.length, newFiles.length);
    }

    // Preload thumbs for newly added files only
    if (addedFiles.length > 0) {
      await preloadCachedThumbs(addedFiles, folder);
    }
  } catch {
    // Background refresh failures are silent
  }
}

// ---------------------------------------------------------------------------
// Main hook
// ---------------------------------------------------------------------------

export function useBrowserFiles(folder: string | null) {
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!folder) return;

    // Abort any previous load/refresh
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    const cached = getCachedFolder(folder);

    if (cached) {
      // ---- Cache hit: files already restored by setActiveFolder ----
      // Just kick off a background refresh to catch new/deleted files
      backgroundRefresh(folder, ac.signal);
    } else {
      // ---- Cache miss: stream from backend ----
      const store = useGalleryStore.getState();
      store.setLoadingFiles(true);
      store.setLoadProgress(0, null);

      // Pre-populate from cached child folders so user sees content instantly
      const childMerge = mergeChildCaches(folder);
      const hasChildSeed = !!childMerge;
      if (childMerge) {
        store.setFiles(childMerge.files);
        store.setThumbsBatch(Array.from(childMerge.thumbs));
        store.setLoadProgress(childMerge.files.length, null);
      }

      const buffer: GalleryFile[] = [];
      let flushTimer: ReturnType<typeof setTimeout> | null = null;
      const allFiles: GalleryFile[] = [];

      const flush = () => {
        if (buffer.length === 0) return;
        const batch = buffer.splice(0);
        allFiles.push(...batch);
        // Skip incremental store updates when seeded from child caches
        // to avoid duplicates — the #END# handler sets the authoritative list
        if (hasChildSeed) return;
        const s = useGalleryStore.getState();
        s.setFiles([...s.files, ...batch]);
        s.setLoadProgress(s.files.length + batch.length, null);
      };

      const scheduleFlush = () => {
        if (flushTimer) return;
        flushTimer = setTimeout(() => {
          flushTimer = null;
          flush();
        }, FILE_BATCH_INTERVAL);
      };

      const wsUrl = api.getWebSocketUrl("/sdapi/v1/browser/files");
      const ws = new WebSocket(wsUrl);

      const cleanupWs = () => {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) ws.close();
      };

      ac.signal.addEventListener("abort", cleanupWs, { once: true });

      ws.onopen = () => ws.send(folder);

      ws.onmessage = (ev) => {
        const msg = ev.data as string;
        if (msg === "#END#") {
          if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
          flush();
          const endStore = useGalleryStore.getState();

          // Re-key child-seeded thumbs to match parent-context file IDs.
          // Child cache files have IDs like "outputs%2Ftext##F##img.png" while
          // the parent stream produces "outputs##F##text%2Fimg.png" — different
          // IDs for the same fullPath. Build a fullPath→thumb map, then re-key.
          if (hasChildSeed && endStore.thumbs.size > 0) {
            const pathToThumb = new Map<string, CachedThumb>();
            const oldFileMap = new Map(endStore.files.map((f) => [f.id, f]));
            for (const [id, thumb] of endStore.thumbs) {
              const file = oldFileMap.get(id);
              if (file) pathToThumb.set(file.fullPath, thumb);
            }
            if (pathToThumb.size > 0) {
              const rekeyed: [string, CachedThumb][] = [];
              for (const f of allFiles) {
                const thumb = pathToThumb.get(f.fullPath);
                if (thumb) rekeyed.push([f.id, thumb]);
              }
              // Set files first, then apply re-keyed thumbs
              endStore.setFiles(allFiles);
              if (rekeyed.length > 0) endStore.setThumbsBatch(rekeyed);
            } else {
              endStore.setFiles(allFiles);
            }
          } else {
            endStore.setFiles(allFiles);
          }

          endStore.setLoadProgress(allFiles.length, allFiles.length);
          endStore.setLoadingFiles(false);
          ws.close();

          // Save to folder cache — fetch mtime for staleness tracking
          api.get<{ mtime: number }>("/sdapi/v1/browser/folder-info", { folder })
            .then((info) => {
              const thumbs = new Map(useGalleryStore.getState().thumbs);
              setCachedFolder(folder, allFiles, thumbs, info.mtime);
            })
            .catch(() => {
              // Still cache with mtime=0 so revisit is instant
              const thumbs = new Map(useGalleryStore.getState().thumbs);
              setCachedFolder(folder, allFiles, thumbs, 0);
            });

          // Pre-populate store from IndexedDB cache
          preloadCachedThumbs(allFiles, folder);
          return;
        }
        const entry = parseFileEntry(msg);
        if (isMediaFile(entry.relativePath)) {
          buffer.push(entry);
          scheduleFlush();
        }
      };

      ws.onerror = () => {
        useGalleryStore.getState().setLoadingFiles(false);
      };

      ws.onclose = () => {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        flush();
        useGalleryStore.getState().setLoadingFiles(false);
      };
    }

    return () => {
      ac.abort();
      abortRef.current = null;
    };
  }, [folder]);
}

// ---------------------------------------------------------------------------
// Thumbnail fetching
// ---------------------------------------------------------------------------

/**
 * Fetch a single thumbnail. Checks IndexedDB FIRST (no semaphore needed),
 * only falls back to the API on cache miss.
 */
export async function fetchThumb(file: GalleryFile): Promise<CachedThumb | null> {
  try {
    // Check IndexedDB cache first - no semaphore, no API call
    const hash = await computeThumbHash(file.fullPath);
    const cached = await getThumb(hash);
    if (cached) return cached;
  } catch {
    // IndexedDB errors are non-fatal, fall through to API
  }

  // Cache miss - fetch from backend (uses semaphore to limit concurrency)
  await thumbSemaphore.acquire();
  try {
    const resp = await api.get<BrowserThumb>("/sdapi/v1/browser/thumb", { file: file.fullPath });
    if (!resp || !resp.data) return null;

    const hash = await computeThumbHash(file.fullPath);
    const entry: CachedThumb = {
      hash,
      folder: file.folder,
      data: resp.data,
      width: resp.width,
      height: resp.height,
      size: resp.size,
      mtime: resp.mtime,
      exif: resp.exif ?? "",
    };

    // Store in IndexedDB (fire and forget)
    putThumb(entry).catch(() => {});

    return entry;
  } catch {
    return null;
  } finally {
    thumbSemaphore.release();
  }
}

/** Dispatch a single thumbnail fetch if not already cached or in-flight. */
function dispatchThumb(file: GalleryFile, setThumb: (id: string, thumb: CachedThumb) => void) {
  if (inflightIds.has(file.id) || useGalleryStore.getState().thumbs.has(file.id)) return;
  inflightIds.add(file.id);
  fetchThumb(file).then((thumb) => {
    inflightIds.delete(file.id);
    if (thumb) {
      setThumb(file.id, thumb);
      // Keep folder cache in sync
      const folder = useGalleryStore.getState().activeFolder;
      if (folder) updateCachedThumbs(folder, [[file.id, thumb]]);
    }
  });
}

/**
 * Load thumbnails for visible files (high priority — dispatched first).
 * Never aborts in-flight loads; uses a shared inflight set to avoid duplicates.
 */
export function useThumbnailLoader(visibleFileIds: string[], files: GalleryFile[]) {
  const setThumb = useGalleryStore((s) => s.setThumb);
  const filesRef = useRef(files);
  useEffect(() => { filesRef.current = files; }, [files]);

  useEffect(() => {
    const fileMap = new Map(filesRef.current.map((f) => [f.id, f]));
    for (const id of visibleFileIds) {
      const file = fileMap.get(id);
      if (file) dispatchThumb(file, setThumb);
    }
  }, [visibleFileIds, setThumb]);
}

/**
 * Background preloader: after the WS file stream finishes, continuously
 * preload ALL thumbnails so the progress bar reaches 100% and scrolling
 * through previously unseen rows is instant.
 */
export function useBackgroundPreloader(files: GalleryFile[]) {
  const isLoadingFiles = useGalleryStore((s) => s.isLoadingFiles);
  const setThumb = useGalleryStore((s) => s.setThumb);
  const cancelRef = useRef(false);

  useEffect(() => {
    // Only start after the file list is fully loaded
    if (isLoadingFiles || files.length === 0) return;
    cancelRef.current = false;

    // Small delay to let visible-area loader grab semaphore slots first
    const timer = setTimeout(() => {
      for (const file of files) {
        if (cancelRef.current) break;
        dispatchThumb(file, setThumb);
      }
    }, 500);

    return () => {
      cancelRef.current = true;
      clearTimeout(timer);
    };
  }, [isLoadingFiles, files, setThumb]);
}
