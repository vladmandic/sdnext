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
    let request;
    const store = db
      .transaction('thumbs', 'readonly')
      .objectStore('thumbs');
    if (index) {
      request = store.index(index).getAllKeys(query);
    } else {
      request = store.getAllKeys(query);
    }
    request.onsuccess = () => resolve(request.result);
    request.onerror = (evt) => reject(evt);
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
    let request;
    const store = db
      .transaction('thumbs', 'readonly')
      .objectStore('thumbs');
    if (folder) {
      request = store.index('folder').count(folder);
    } else {
      request = store.count();
    }
    request.onsuccess = () => resolve(request.result);
    request.onerror = (evt) => reject(evt);
  });
}

/**
 * Cleanup function for IndexedDB thumbnail cache.
 * @global
 * @param {Set<string>} keepSet - Set containing the hashes of the current files in the folder.
 * @param {string} folder - Folder name/path.
 * @param {UpdateMsgCallback} msgCallback - Callback for updating the overlay message progress.
 */
async function idbFolderCleanup(keepSet, folder, msgCallback) {
  if (!db) return null;
  if (!(keepSet instanceof Set)) {
    throw new TypeError('IndexedDB cleaning function must be given a Set() of the current gallery hashes');
  }
  if (!folder || typeof folder !== 'string') {
    throw new Error('IndexedDB cleaning function must be told the current active folder');
  }
  const removals = (new Set(await idbGetAllKeys('folder', folder))).difference(keepSet);
  const totalRemovals = removals.size;
  let counter = 0;
  return new Promise((resolve, reject) => {
    const folderIndex = db
      .transaction('thumbs', 'readwrite')
      .objectStore('thumbs')
      .index('folder');
    const request = folderIndex.openCursor(folder);

    request.onsuccess = (evt) => {
      const cursor = evt.target.result;
      if (cursor) {
        if (removals.has(cursor.primaryKey)) {
          counter++;
          cursor.delete();
        }
        if (counter === totalRemovals) {
          resolve(counter); // Got lucky with element order and can stop early
        } else {
          if (counter % 100 === 0 && counter !== 0) {
            msgCallback(Math.floor((counter / totalRemovals) * 100));
          }
          cursor.continue();
        }
      } else {
        resolve(counter);
      }
    };
    request.onerror = (evt) => reject(evt);
  });
}

window.idbAdd = add;
window.idbDel = del;
window.idbGet = get;
window.idbPut = put;
