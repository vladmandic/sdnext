import type { CachedThumb } from "@/api/types/gallery";

const DB_NAME = "SDNextReact";
const STORE_NAME = "thumbs";
const DB_VERSION = 2;

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      // V2: hash is now path-only, so clear stale entries from V1
      if (db.objectStoreNames.contains(STORE_NAME)) {
        db.deleteObjectStore(STORE_NAME);
      }
      const store = db.createObjectStore(STORE_NAME, { keyPath: "hash" });
      store.createIndex("folder", "folder", { unique: false });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

/** Hash based on file path only - enables cache-first lookup without an API call. */
export async function computeThumbHash(path: string): Promise<string> {
  const encoded = new TextEncoder().encode(path);
  const digest = await crypto.subtle.digest("SHA-256", encoded);
  const arr = Array.from(new Uint8Array(digest));
  return arr.map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function getThumb(hash: string): Promise<CachedThumb | undefined> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const req = tx.objectStore(STORE_NAME).get(hash);
    req.onsuccess = () => resolve(req.result as CachedThumb | undefined);
    req.onerror = () => reject(req.error);
  });
}

/** Batch-get multiple thumbs in a single transaction. */
export async function batchGetThumbs(hashes: string[]): Promise<Map<string, CachedThumb>> {
  if (hashes.length === 0) return new Map();
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const map = new Map<string, CachedThumb>();
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    let pending = hashes.length;
    for (const hash of hashes) {
      const req = store.get(hash);
      req.onsuccess = () => {
        if (req.result) map.set(hash, req.result as CachedThumb);
        if (--pending === 0) resolve(map);
      };
      req.onerror = () => {
        if (--pending === 0) resolve(map);
      };
    }
    tx.onerror = () => reject(tx.error);
  });
}

export async function putThumb(entry: CachedThumb): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function deleteFolder(folder: string): Promise<void> {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const index = store.index("folder");
    const req = index.openCursor(IDBKeyRange.only(folder));
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function cleanupFolder(folder: string, maxEntries: number): Promise<void> {
  const db = await openDb();
  const entries = await new Promise<CachedThumb[]>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const index = tx.objectStore(STORE_NAME).index("folder");
    const req = index.getAll(IDBKeyRange.only(folder));
    req.onsuccess = () => resolve(req.result as CachedThumb[]);
    req.onerror = () => reject(req.error);
  });
  if (entries.length <= maxEntries) return;
  entries.sort((a, b) => a.mtime - b.mtime);
  const toDelete = entries.slice(0, entries.length - maxEntries);
  const tx = db.transaction(STORE_NAME, "readwrite");
  const store = tx.objectStore(STORE_NAME);
  for (const entry of toDelete) {
    store.delete(entry.hash);
  }
  await new Promise<void>((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}
