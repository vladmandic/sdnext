/**
 * @type {?IDBDatabase}
 */
let db = null;

async function initIndexDB() {
  async function createDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('SDNext', 2);
      request.onerror = (evt) => reject(evt);
      request.onsuccess = (evt) => {
        db = evt.target.result;
        const countAll = db
          .transaction(['thumbs'], 'readwrite')
          .objectStore('thumbs')
          .count();
        countAll.onsuccess = () => log('initIndexDB', countAll.result);
        resolve();
      };
      request.onupgradeneeded = (evt) => {
        db = evt.target.result;
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

function idbIsReady() {
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
function configureTransactionAbort({ transaction, signal, resolve, reject }, resolveValue) {
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
    reject(new Error('Database transaction error.', e));
  };
  transaction.oncomplete = () => {
    signal.removeEventListener('abort', abortTransaction);
    resolve(resolveValue);
  };
  return abortTransaction;
}

async function add(record) {
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

async function del(hash) {
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

async function get(hash) {
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

async function put(record) {
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

async function idbGetAllKeys(index = null, query = null) {
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
 * @param {?string} folder - If specified, get the count for this gallery folder. Otherwise get the total count.
 * @returns {Promise<number>}
 */
async function idbCount(folder = null) {
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
 * @param {string} folder - Folder name/path
 * @param {AbortSignal} signal - Signal from the AbortController for thumbCacheCleanup()
 */
async function idbFolderCleanup(keepSet, folder, signal) {
  if (!db) return null;
  if (!(keepSet instanceof Set)) {
    throw new TypeError('IndexedDB cleaning function must be given a Set() of the current gallery hashes');
  }
  if (typeof folder !== 'string') {
    throw new Error('IndexedDB cleaning function must be told the current active folder');
  }

  // Use range query to match folder and all its subdirectories
  const folderNormalized = folder.replace(/\/+/g, '/').replace(/\/$/, '');
  const range = IDBKeyRange.bound(folderNormalized, `${folderNormalized}\uffff`, false, true);
  let removals = new Set(await idbGetAllKeys('folder', range));
  removals = removals.difference(keepSet); // Don't need to keep full set in memory
  const totalRemovals = removals.size;
  if (signal.aborted) {
    throw `Aborting. ${signal.reason}`; // eslint-disable-line no-throw-literal
  }
  return new Promise((resolve, reject) => {
    const transaction = db.transaction('thumbs', 'readwrite');
    const props = { transaction, signal, resolve, reject };
    const abortTransaction = configureTransactionAbort(props, totalRemovals);
    try {
      const store = transaction.objectStore('thumbs');
      removals.forEach((entry) => { store.delete(entry); });
    } catch (err) {
      error(err);
      abortTransaction();
    }
  });
}

window.idbAdd = add;
window.idbDel = del;
window.idbGet = get;
window.idbPut = put;
