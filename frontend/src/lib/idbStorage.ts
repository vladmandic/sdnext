import type { StateStorage } from "zustand/middleware";

/** Zustand StateStorage backed by IndexedDB with debounced writes. */
export function createIdbStorage(dbName: string, storeName: string, debounceMs = 2000): StateStorage {
  let dbPromise: Promise<IDBDatabase> | null = null;

  function openDb(): Promise<IDBDatabase> {
    if (!dbPromise) {
      dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
        const req = indexedDB.open(dbName, 1);
        req.onupgradeneeded = () => {
          if (!req.result.objectStoreNames.contains(storeName)) {
            req.result.createObjectStore(storeName);
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    return dbPromise;
  }

  function idbGet(key: string): Promise<string | null> {
    return openDb().then(
      (db) =>
        new Promise((resolve, reject) => {
          const tx = db.transaction(storeName, "readonly");
          const req = tx.objectStore(storeName).get(key);
          req.onsuccess = () => resolve((req.result as string) ?? null);
          req.onerror = () => reject(req.error);
        }),
    );
  }

  function idbSet(key: string, value: string): Promise<void> {
    return openDb().then(
      (db) =>
        new Promise((resolve, reject) => {
          const tx = db.transaction(storeName, "readwrite");
          tx.objectStore(storeName).put(value, key);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error);
        }),
    );
  }

  function idbDelete(key: string): Promise<void> {
    return openDb().then(
      (db) =>
        new Promise((resolve, reject) => {
          const tx = db.transaction(storeName, "readwrite");
          tx.objectStore(storeName).delete(key);
          tx.oncomplete = () => resolve();
          tx.onerror = () => reject(tx.error);
        }),
    );
  }

  // Debounce state for setItem
  let pendingKey: string | null = null;
  let pendingValue: string | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function flush() {
    if (pendingKey !== null && pendingValue !== null) {
      const k = pendingKey;
      const v = pendingValue;
      pendingKey = null;
      pendingValue = null;
      if (timer) { clearTimeout(timer); timer = null; }
      idbSet(k, v);
    }
  }

  if (typeof window !== "undefined") {
    window.addEventListener("beforeunload", flush);
  }

  return {
    getItem: (key) => idbGet(key),
    setItem: (key, value) => {
      pendingKey = key;
      pendingValue = value;
      if (timer) clearTimeout(timer);
      timer = setTimeout(flush, debounceMs);
    },
    removeItem: (key) => { idbDelete(key); },
  };
}
