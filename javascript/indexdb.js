let db;

async function initIndexDB() {
  async function createDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('SDNext');
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
        const store = db.createObjectStore('thumbs', { keyPath: 'hash' });
        store.createIndex('hash', 'hash', { unique: true });
        const index = store.index('hash');
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

async function idbGetAllKeys() {
  if (!db) return null;
  return new Promise((resolve, reject) => {
    const request = db
      .transaction("thumbs")
      .objectStore("thumbs")
      .getAllKeys();
    request.onsuccess = () => resolve(request.result);
    request.onerror = (evt) => reject(evt);
  });
}

async function idbClean(keepSet) {
  if (!db) return null;
  if (!keepSet instanceof Set) {
    console.error("IndexedDB cleaning function must be given a Set() of hashes to keep");
  };
  return new Promise((resolve, reject) => {
    let counter = 0;
    const request = db
      .transaction("thumbs", "readwrite")
      .objectStore("thumbs")
      .openCursor();
    request.onsuccess = (evt) => {
      const cursor = evt.target.result;
      if (cursor) {
        if (!keepSet.has(cursor.key)) {
          cursor.delete();
          counter++;
        }
        cursor.continue();
      }
      else {
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
