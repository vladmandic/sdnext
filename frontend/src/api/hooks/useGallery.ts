import { useEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../client";
import type { BrowserFolder, BrowserThumb, GalleryFile, CachedThumb } from "../types/gallery";
import { useGalleryStore } from "@/stores/galleryStore";
import { computeThumbHash, getThumb, putThumb, batchGetThumbs } from "@/lib/thumbnailCache";
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

/**
 * After all files are loaded, batch-check IndexedDB for cached thumbnails
 * and pre-populate the in-memory store. This makes folder revisits instant.
 */
async function preloadCachedThumbs(files: GalleryFile[]) {
  if (files.length === 0) return;
  try {
    // Compute hashes for all files (path-only, very fast)
    const hashToId = new Map<string, string>();
    const hashes: string[] = [];
    for (const f of files) {
      const hash = await computeThumbHash(f.fullPath);
      hashToId.set(hash, f.id);
      hashes.push(hash);
    }

    // Batch-read from IndexedDB in a single transaction
    const cached = await batchGetThumbs(hashes);

    // Populate store with all cache hits
    if (cached.size > 0) {
      const store = useGalleryStore.getState();
      for (const [hash, thumb] of cached) {
        const fileId = hashToId.get(hash);
        if (fileId) store.setThumb(fileId, thumb);
      }
    }
  } catch {
    // IndexedDB errors are non-fatal
  }
}

export function useBrowserFiles(folder: string | null) {
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!folder) return;

    const store = useGalleryStore.getState();
    store.setFiles([]);
    store.setLoadingFiles(true);
    store.setLoadProgress(0, null);

    const allFiles: GalleryFile[] = [];
    const buffer: GalleryFile[] = [];
    let flushTimer: ReturnType<typeof setTimeout> | null = null;

    const flush = () => {
      if (buffer.length === 0) return;
      const batch = buffer.splice(0);
      allFiles.push(...batch);
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
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(folder);
    };

    ws.onmessage = (ev) => {
      const msg = ev.data as string;
      if (msg === "#END#") {
        if (flushTimer) { clearTimeout(flushTimer); flushTimer = null; }
        flush();
        useGalleryStore.getState().setLoadingFiles(false);
        ws.close();
        // Pre-populate store from IndexedDB cache
        preloadCachedThumbs(allFiles);
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

    return () => {
      if (flushTimer) clearTimeout(flushTimer);
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
      wsRef.current = null;
    };
  }, [folder]);
}

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
    if (thumb) setThumb(file.id, thumb);
  });
}

/**
 * Load thumbnails for visible files (high priority — dispatched first).
 * Never aborts in-flight loads; uses a shared inflight set to avoid duplicates.
 */
export function useThumbnailLoader(visibleFileIds: string[], files: GalleryFile[]) {
  const setThumb = useGalleryStore((s) => s.setThumb);
  const filesRef = useRef(files);
  filesRef.current = files;

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
