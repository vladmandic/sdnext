/**
 * @type {?IDBDatabase}
 */
import { log } from './logger';

let db = null;

interface ThumbRecord {
  hash: string;
  folder?: string;
  [key: string]: unknown;
}

export async function initIndexDB(): Promise<void> {
  async function createDB(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('SDNext', 2);
      request.onerror = (evt: Event) => reject(evt);
      request.onsuccess = (evt: Event) => {
        db = (evt.target as IDBOpenDBRequest).result;
        const countAll = db
          .transaction(['thumbs'], 'readwrite')
          .objectStore('thumbs')
          .count();
        countAll.onsuccess = () => log('initIndexDB', countAll.result);
        resolve();
      };
      request.onupgradeneeded = (evt: IDBVersionChangeEvent) => {
        db = (evt.target as IDBOpenDBRequest).result;
        const oldver = evt.oldVersion;
        if (oldver < 1) {
          const store = db.createObjectStore('thumbs', { keyPath: 'hash' });
          store.createIndex('hash', 'hash', { unique: true });
        }
        if (oldver < 2) {
          const existingStore = request.transaction.objectStore('thumbs');
          existingStore.createIndex('folder', 'folder', { unique: false });
        }
        resolve();
      };
    });
  }

  if (!db) await createDB();
}

export function idbIsReady(): boolean {
  return db !== null;
}

/**
 * Reusable setup for handling IDB transactions.
 * @param {Object} resources - Required resources for implementation
 * @param {IDBTransaction} resources.transaction
 * @param {AbortSignal} resources.signal
 * @param {Function} resources.resolve
 * @param {Function} resources.reject
 * @param {*} resolveValue - Value to resolve the outer Promise with
 * @returns {() => void} - Function for manually aborting the transaction
 */
function configureTransactionAbort<T>({
  transaction,
  signal,
  resolve,
  reject,
}: {
  transaction: IDBTransaction;
  signal: AbortSignal;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
}, resolveValue: T): () => void {
  function abortTransaction() {
    signal.removeEventListener('abort', abortTransaction);
    transaction.abort();
  }
  signal.addEventListener('abort', abortTransaction);
  transaction.onabort = () => {
    signal.removeEventListener('abort', abortTransaction);
    reject(new DOMException(`Aborting database transaction. ${signal.reason}`, 'AbortError'));
  };
  transaction.onerror = (e) => {
    signal.removeEventListener('abort', abortTransaction);
    reject(new Error('Database transaction error.'));
  };
  transaction.oncomplete = () => {
    signal.removeEventListener('abort', abortTransaction);
    resolve(resolveValue);
  };
  return abortTransaction;
}

async function add(record: ThumbRecord): Promise<Event | null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const request = db
      .transaction(['thumbs'], 'readwrite')
      .objectStore('thumbs')
      .add(record);
    request.onsuccess = (evt) => resolve(evt);
    request.onerror = (evt) => reject(evt);
  });
}

async function del(hash: string): Promise<Event | null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const request = db
      .transaction(['thumbs'], 'readwrite')
      .objectStore('thumbs')
      .delete(hash);
    request.onsuccess = (evt) => resolve(evt);
    request.onerror = (evt) => reject(evt);
  });
}

async function get(hash: string): Promise<ThumbRecord | null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const request = db
      .transaction(['thumbs'], 'readwrite')
      .objectStore('thumbs')
      .get(hash);
    request.onsuccess = () => resolve(request.result);
    request.onerror = (evt) => reject(evt);
  });
}

async function put(record: ThumbRecord): Promise<Event | null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const request = db
      .transaction(['thumbs'], 'readwrite')
      .objectStore('thumbs')
      .put(record);
    request.onsuccess = (evt) => resolve(evt);
    request.onerror = (evt) => reject(evt);
  });
}

export async function idbGetAllKeys(index: string | null = null, query: IDBValidKey | IDBKeyRange | null = null): Promise<IDBValidKey[] | null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    try {
      let request;
      const transaction = db.transaction('thumbs', 'readonly');
      transaction.onabort = (e) => reject(e);

      const store = transaction.objectStore('thumbs');
      if (index) {
        request = store.index(index).getAllKeys(query);
      } else {
        request = store.getAllKeys(query);
      }
      request.onsuccess = () => resolve(request.result);
      request.onerror = (e) => reject(e);
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * Get the number of entries in the IndexedDB thumbnail cache.
 * @global
 * @param {IDBValidKey | IDBKeyRange | undefined} folder - If specified, get the count for this gallery folder. Otherwise get the total count.
 * @returns {Promise<number>}
 */
export async function idbCount(folder?: IDBValidKey | IDBKeyRange): Promise<number | null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    try {
      let request;
      const transaction = db.transaction('thumbs', 'readonly');
      transaction.onabort = (e) => reject(e);

      const store = transaction.objectStore('thumbs');
      if (folder) {
        request = store.index('folder').count(folder);
      } else {
        request = store.count();
      }
      request.onsuccess = () => resolve(request.result);
      request.onerror = (e) => reject(e);
    } catch (err) {
      reject(err);
    }
  });
}

/**
 * Cleanup function for IndexedDB thumbnail cache.
 * @global
 * @param {Set<string>} keepSet - Set containing the hashes of the current files in the folder
 * @param {IDBValidKey | IDBKeyRange} folder - Folder name/path or range
 * @param {AbortSignal} signal - Signal from the AbortController for thumbCacheCleanup()
 */
export async function idbFolderCleanup(keepSet: Set<string>, folder: IDBValidKey | IDBKeyRange, signal: AbortSignal): Promise<number | null> {
  if (!db) return null;
  const existing = await idbGetAllKeys('folder', folder);
  const removals = new Set((existing ?? []).map((entry) => String(entry)).filter((entry) => !keepSet.has(entry))); // Don't need to keep full set in memory
  const totalRemovals = removals.size;
  if (signal.aborted) {
    throw new Error(`Aborting. ${String(signal.reason)}`);
  }
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('thumbs', 'readwrite');
    const props = { transaction, signal, resolve, reject };
    configureTransactionAbort(props, totalRemovals);
    const store = transaction.objectStore('thumbs');
    removals.forEach((entry) => { store.delete(entry); });
  });
}

export async function idbClearAll(signal: AbortSignal): Promise<null> {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(['thumbs'], 'readwrite');
    const props = { transaction, signal, resolve, reject };
    configureTransactionAbort(props, null);
    transaction.objectStore('thumbs').clear();
  });
}

export const idbAdd = add;
export const idbDel = del;
export const idbGet = get;
export const idbPut = put;
