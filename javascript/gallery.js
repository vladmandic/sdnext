/* eslint-disable max-classes-per-file */
let ws;
let url;
let currentImage = null;
let currentGalleryFolder = null;
let pruneImagesTimer;
let outstanding = 0;
let lastSort = 0;
let lastSortName = 'None';
let gallerySelection = { files: [], index: -1 };
const galleryHashes = new Set();
let maintenanceController = new AbortController();
const folderStylesheet = new CSSStyleSheet();
const fileStylesheet = new CSSStyleSheet();
const iconStopwatch = String.fromCodePoint(9201);
// Store separator states for the session
const separatorStates = new Map();
const el = {
  folders: undefined,
  files: undefined,
  search: undefined,
  status: undefined,
  btnSend: undefined,
  clearCacheFolder: undefined,
};

const SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'tiff', 'jp2', 'jxl', 'gif', 'mp4', 'mkv', 'avi', 'mjpeg', 'mpg', 'avr'];

function getVisibleGalleryFiles() {
  if (!el.files) return [];
  return Array.from(el.files.children).filter((node) => node.name && node.offsetParent);
}

function updateGallerySelectionClasses(files = gallerySelection.files, index = gallerySelection.index) {
  files.forEach((file, i) => {
    file.classList.toggle('gallery-file-selected', i === index);
  });
}

function refreshGallerySelection() {
  updateGallerySelectionClasses(gallerySelection.files, -1);
  const files = getVisibleGalleryFiles();
  const index = files.findIndex((file) => file.src === currentImage);
  gallerySelection = { files, index };
  updateGallerySelectionClasses(files, index);
}

function resetGallerySelection() {
  updateGallerySelectionClasses(gallerySelection.files, -1);
  gallerySelection = { files: [], index: -1 };
  currentImage = null;
}

function applyGallerySelection(index, { send = true } = {}) {
  if (!gallerySelection.files.length) refreshGallerySelection();
  const files = gallerySelection.files;
  if (!files.length) return;
  if (!Number.isInteger(index) || index < 0 || index >= files.length) {
    log('gallery selection index out of range', index, files.length);
    resetGallerySelection();
    return;
  }
  gallerySelection.index = index;
  currentImage = files[index].src;
  updateGallerySelectionClasses(files, index);
  if (send && el.btnSend) el.btnSend.click();
}

function setGallerySelectionByElement(element, options) {
  if (!gallerySelection.files.length) refreshGallerySelection();
  let index = gallerySelection.files.findIndex((file) => file === element);
  if (index < 0) {
    refreshGallerySelection();
    index = gallerySelection.files.findIndex((file) => file === element);
  }
  if (index >= 0) applyGallerySelection(index, options);
}

function buildGalleryFileUrl(path) {
  return new URL(`/file=${encodeURI(path)}`, window.location.origin).toString();
}

window.getGallerySelection = () => ({ index: gallerySelection.index, files: gallerySelection.files });
window.setGallerySelection = (index, options) => applyGallerySelection(index, options);
window.getGallerySelectedUrl = () => (currentImage ? buildGalleryFileUrl(currentImage) : null);

/**
 * Wait for the `outstanding` variable to be below the specified value
 * @param {number} num - Threshold for `outstanding`
 * @param {AbortSignal} signal - AbortController signal
 */
async function awaitForOutstanding(num, signal) {
  while (outstanding > num && !signal.aborted) await new Promise((resolve) => { setTimeout(resolve, 50); });
  signal.throwIfAborted();
}

/**
 * Wait for gallery to finish populating
 * @param {number} expectedSize - Expected gallery size
 * @param {AbortSignal} signal - AbortController signal
 */
async function awaitForGallery(expectedSize, signal) {
  while (galleryHashes.size < expectedSize && !signal.aborted) await new Promise((resolve) => { setTimeout(resolve, 500); }); // longer interval because it's a low priority check
  signal.throwIfAborted();
}

function updateGalleryStyles() {
  if (opts.theme_type?.toLowerCase() === 'modern') {
    folderStylesheet.replaceSync(`
      .gallery-folder {
        cursor: pointer;
        padding: 8px 6px 8px 6px;
        background-color: var(--sd-button-normal-color);
        border-radius: var(--sd-border-radius);
        text-align: left;
        direction: rtl; /* Used to overflow the beginning instead of the end */
        min-width: 12em;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        transition-duration: 0.2s;
        transition-property: color, opacity, background-color, border-color;
        transition-timing-function: ease-out;
      }
      .gallery-folder:hover {
        background-color: var(--button-primary-background-fill-hover, var(--sd-button-hover-color));
      }
      .gallery-folder-selected {
        background-color: var(--sd-button-selected-color);
        color: var(--sd-button-selected-text-color);
      }
      .gallery-folder-icon {
        font-size: 1.2em;
        color: var(--sd-button-icon-color);
        margin-right: 1em;
        filter: drop-shadow(1px 1px 2px black);
        float: left;
      }
    `);
  } else {
    folderStylesheet.replaceSync(`
      .gallery-folder {
        cursor: pointer;
        padding: 8px 6px 8px 6px;
        max-width: 200px;
        overflow-x: hidden;
        text-wrap: nowrap;
        text-overflow: ellipsis;
      }
      .gallery-folder:hover {
        background-color: var(--button-primary-background-fill-hover);
      }
      .gallery-folder-selected {
        background-color: var(--button-primary-background-fill);
      }
    `);
  }
  fileStylesheet.replaceSync(`
    .gallery-file {
      object-fit: contain;
      cursor: pointer;
      height: ${opts.extra_networks_card_size}px;
      width: ${opts.browser_fixed_width ? `${opts.extra_networks_card_size}px` : 'unset'};
    }
    .gallery-file:hover {
      filter: grayscale(100%);
    }
    :host(.gallery-file-selected) .gallery-file {
      box-shadow: 0 0 0 2px var(--sd-button-selected-color);
    }
  `);
}

// Classes

class SimpleProgressBar {
  #container = document.createElement('div');
  #progress = document.createElement('div');
  #textDiv = document.createElement('div');
  #text = document.createElement('span');
  #visible = false;
  #hideTimeout = null;
  #interval = null;
  #max = 0;
  /** @type {Set} */
  #monitoredSet;

  constructor(monitoredSet) {
    this.#monitoredSet = monitoredSet; // This is required because incrementing a variable with a class method turned out to not be an atomic operation
    this.#container.style.cssText = 'position:relative;overflow:hidden;border-radius:var(--sd-border-radius);width:100%;background-color:hsla(0,0%,36%,0.3);height:1.2rem;margin:0;padding:0;display:none;';
    this.#progress.style.cssText = 'position:absolute;left:0;height:100%;width:0;transition:width 200ms;';
    this.#progress.style.backgroundColor = 'hsla(110, 32%, 35%, 0.80)'; // alt: '#27911d'
    this.#textDiv.style.cssText = 'position:relative;margin:auto;width:max-content;height:100%;';
    this.#text.style.cssText = 'user-select:none;color:white;';

    this.#textDiv.append(this.#text);
    this.#container.append(this.#progress, this.#textDiv);
  }

  start(total) {
    this.clear();
    this.#max = total;
    this.#interval = setInterval(() => {
      this.#update(this.#monitoredSet.size, this.#max);
    }, 250);
  }

  attachTo(element) {
    if (element.hasChildNodes) {
      element.innerHTML = '';
    }
    element.appendChild(this.#container);
  }

  clear() {
    this.#stop();
    clearTimeout(this.#hideTimeout);
    this.#hideTimeout = null;
    this.#container.style.display = 'none';
    this.#visible = false;
    this.#progress.style.width = '0';
    this.#text.textContent = '';
  }

  #update(loaded, max) {
    if (this.#hideTimeout) {
      this.#hideTimeout = null;
    }

    this.#progress.style.width = `${Math.floor((loaded / max) * 100)}%`;
    this.#text.textContent = `${loaded}/${max}`;

    if (!this.#visible) {
      this.#container.style.display = 'block';
      this.#visible = true;
    }
    if (loaded >= max) {
      this.#stop();
      this.#hideTimeout = setTimeout(() => {
        this.clear();
      }, 1000);
    }
  }

  #stop() {
    clearInterval(this.#interval);
    this.#interval = null;
  }
}

const galleryProgressBar = new SimpleProgressBar(galleryHashes);

/* This isn't as robust as the Web Locks API, but it will at least work if accessing a remote machine without HTTPS */
class SimpleFunctionQueue {
  #id;
  #running;
  #queue;

  constructor(id) {
    this.#id = id;
    this.#running = false;
    this.#queue = [];
  }

  static abortLogger(identifier, result) {
    if (typeof result === 'string' || (result instanceof DOMException && result.name === 'AbortError')) {
      log(identifier, result?.message || result);
    } else {
      error(identifier, result.message);
    }
  }

  /**
   * @param {{
   *  signal: AbortSignal,
   *  callback: Function
   * }} config
   */
  enqueue(config) {
    if (!(config.signal instanceof AbortSignal) || typeof config.callback !== 'function') {
      throw new Error('Invalid configuration. Object must contain an AbortSignal and a function');
    }
    if (config.signal.aborted) {
      debug(`${this.#id} Queue: Skipping addition to queue due to "${config.signal.reason}"`);
      return;
    }
    this.#queue.push(config);
    this.#tryRunNext();
  }

  async #tryRunNext() {
    if (this.#running || !this.#queue.length) return;
    try {
      const { signal, callback } = this.#queue.shift();
      if (signal.aborted) {
        return;
      }
      this.#running = true;
      if (callback.constructor.name.toLowerCase() === 'asyncfunction') {
        await callback();
      } else {
        callback();
      }
    } catch (err) {
      error(`${this.#id} Queue:`, err);
    } finally {
      this.#running = false;
      this.#tryRunNext();
    }
  }
}

// HTML Elements

class GalleryFolder extends HTMLElement {
  static folders = new Set();
  /** @type {GalleryFolder | null} */
  static #active = null;

  constructor(folder) {
    super();
    // Support both old format (string) and new format (object with path and label)
    if (typeof folder === 'object' && folder !== null) {
      this.name = decodeURI(folder.path || '');
      this.label = decodeURI(folder.label || folder.path || '');
    } else {
      this.name = decodeURI(folder);
      this.label = this.name;
    }
    this.style.overflowX = 'hidden';
    this.shadow = this.attachShadow({ mode: 'open' });
    this.shadow.adoptedStyleSheets = [folderStylesheet];

    this.div = document.createElement('div');
  }

  connectedCallback() {
    if (GalleryFolder.folders.has(this)) return; // Element is just being moved

    this.div.className = 'gallery-folder';
    this.div.innerHTML = `<span class="gallery-folder-icon">\uf03e</span> ${this.label}`;
    this.div.title = this.name; // Show full path on hover
    this.addEventListener('click', this.updateSelected);
    this.addEventListener('click', fetchFilesWS); // eslint-disable-line no-use-before-define
    this.shadow.appendChild(this.div);
    GalleryFolder.folders.add(this);
    if (this.name === currentGalleryFolder) {
      this.updateSelected();
    }
  }

  async disconnectedCallback() {
    await Promise.resolve(); // Wait for other microtasks (such as element moving)
    if (this.isConnected) return;
    GalleryFolder.folders.delete(this);
    if (GalleryFolder.#active === this) {
      GalleryFolder.#active = null;
    }
  }

  static getActive() {
    return GalleryFolder.#active;
  }

  updateSelected() {
    this.div.classList.add('gallery-folder-selected');
    GalleryFolder.#active = this;
    for (const folder of GalleryFolder.folders) {
      if (folder !== this) {
        folder.div.classList.remove('gallery-folder-selected');
      }
    }
  }
}

async function delayFetchThumb(fn, signal) {
  await awaitForOutstanding(16, signal);
  try {
    outstanding++;
    const ts = Date.now().toString();
    const res = await authFetch(`${window.api}/browser/thumb?file=${encodeURI(fn)}&ts=${ts}`, { priority: 'low' });
    if (!res.ok) {
      error(`fetchThumb: ${res.statusText}`);
      return undefined;
    }
    const json = await res.json();
    if (!res || !json || json.error || Object.keys(json).length === 0) {
      if (json.error) error(`fetchThumb: ${json.error}`);
      return undefined;
    }
    return json;
  } finally {
    outstanding--;
  }
}

class GalleryFile extends HTMLElement {
  /** @type {AbortSignal} */
  #signal;

  constructor(folder, file, signal) {
    super();
    this.folder = folder;
    this.name = file;
    this.#signal = signal;
    this.src = `${this.folder}/${this.name}`.replace(/\/+/g, '/'); // Ensure no //, ///, etc...
    this.fullFolder = this.src.replace(/\/[^/]+$/, '');
    this.size = 0;
    this.mtime = 0;
    this.hash = undefined;
    this.exif = '';
    this.width = 0;
    this.height = 0;
    this.shadow = this.attachShadow({ mode: 'open' });
    this.shadow.adoptedStyleSheets = [fileStylesheet];

    this.firstRun = true;
  }

  async connectedCallback() {
    if (!this.firstRun) return; // Element is just being moved
    this.firstRun = false;

    // Check separator state early to hide the element immediately
    const dir = this.name.match(/(.*)[/\\]/);
    if (dir && dir[1]) {
      const dirPath = dir[1];
      const isOpen = separatorStates.get(dirPath);
      if (isOpen === false) {
        this.style.display = 'none';
      }
    }

    this.hash = await getHash(`${this.src}/${this.size}/${this.mtime}`); // eslint-disable-line no-use-before-define
    const cachedData = (this.hash && opts.browser_cache) ? await idbGet(this.hash).catch(() => undefined) : undefined;
    const img = document.createElement('img');
    img.className = 'gallery-file';
    img.loading = 'lazy';
    img.onload = async () => {
      img.title += `\nResolution: ${this.width} x ${this.height}`;
      this.title = img.title;
      if (!cachedData && opts.browser_cache) {
        if ((this.width === 0) || (this.height === 0)) { // fetch thumb failed so we use actual image
          this.width = img.naturalWidth;
          this.height = img.naturalHeight;
        }
      }
    };
    let ok = true;
    if (cachedData?.img) {
      img.src = cachedData.img;
      this.exif = cachedData.exif;
      this.width = cachedData.width;
      this.height = cachedData.height;
      this.size = cachedData.size;
      this.mtime = new Date(cachedData.mtime);
    } else {
      try {
        const json = await delayFetchThumb(this.src, this.#signal);
        if (!json) {
          ok = false;
        } else {
          img.src = json.data;
          this.exif = json.exif;
          this.width = json.width;
          this.height = json.height;
          this.size = json.size;
          this.mtime = new Date(json.mtime);
          if (opts.browser_cache) {
            await idbAdd({
              hash: this.hash,
              folder: this.fullFolder,
              file: this.name,
              size: this.size,
              mtime: this.mtime,
              width: this.width,
              height: this.height,
              src: this.src,
              exif: this.exif,
              img: img.src,
              // exif: await getExif(img), // alternative client-side exif
              // img: await createThumb(img), // alternative client-side thumb
            });
          }
        }
      } catch (err) { // thumb fetch failed so assign actual image
        img.src = `file=${this.src}`;
      }
    }
    if (this.#signal.aborted) { // Do not change the operations order from here...
      return;
    }
    galleryHashes.add(this.hash);
    if (!ok) {
      return;
    } // ... to here unless modifications are also being made to maintenance functionality and the usage of AbortController/AbortSignal
    img.onclick = () => {
      setGallerySelectionByElement(this, { send: true });
    };
    img.title = `Folder: ${this.folder}\nFile: ${this.name}\nSize: ${this.size.toLocaleString()} bytes\nModified: ${this.mtime.toLocaleString()}`;
    this.title = img.title;

    // Final visibility check based on search term.
    const shouldDisplayBasedOnSearch = this.title.toLowerCase().includes(el.search.value.toLowerCase());
    if (this.style.display !== 'none') { // Only proceed if not already hidden by a closed separator
      this.style.display = shouldDisplayBasedOnSearch ? 'unset' : 'none';
    }

    this.shadow.appendChild(img);
  }
}

async function createThumb(img) {
  const height = opts.extra_networks_card_size;
  const width = opts.browser_fixed_width ? opts.extra_networks_card_size : 0;
  const canvas = document.createElement('canvas');
  const scaleY = height / img.height;
  const scaleX = width > 0 ? width / img.width : scaleY;
  const scale = Math.min(scaleX, scaleY);
  const scaledWidth = img.width * scale;
  const scaledHeight = img.height * scale;
  canvas.width = scaledWidth;
  canvas.height = scaledHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);
  const dataURL = canvas.toDataURL('image/jpeg', 0.5);
  return dataURL;
}

async function handleSeparator(separator) {
  separator.classList.toggle('gallery-separator-hidden');
  const nowHidden = separator.classList.contains('gallery-separator-hidden');

  // Store the state (true = open, false = closed)
  separatorStates.set(separator.title, !nowHidden);

  // Update arrow and count
  const arrow = separator.querySelector('.gallery-separator-arrow');
  arrow.style.transform = nowHidden ? 'rotate(0deg)' : 'rotate(90deg)';

  const all = Array.from(el.files.children);
  for (const f of all) {
    if (!f.name) continue; // Skip separators

    // Check if file belongs to this exact directory
    const fileDir = f.name.match(/(.*)[/\\]/);
    const fileDirPath = fileDir ? fileDir[1] : '';

    if (separator.title.length > 0 && fileDirPath === separator.title) {
      f.style.display = nowHidden ? 'none' : 'unset';
    }
  }
  // Note: Count is not updated here on manual toggle, as it reflects the total.
  // If I end up implementing it, the search function will handle dynamic count updates.
}

async function addSeparators() {
  document.querySelectorAll('.gallery-separator').forEach((node) => { el.files.removeChild(node); });
  const all = Array.from(el.files.children);
  let lastDir;

  // Count root files (files without a directory path)
  const hasRootFiles = all.some((f) => f.name && !f.name.match(/[/\\]/));
  // Only auto-open first separator if there are no root files to display
  let isFirstSeparator = !hasRootFiles;

  // First pass: create separators
  for (const f of all) {
    let dir = f.name?.match(/(.*)[/\\]/);
    if (!dir) dir = '';
    else dir = dir[1];
    if (dir !== lastDir) {
      lastDir = dir;
      if (dir.length > 0) {
        // Count files in this directory
        let fileCount = 0;
        for (const file of all) {
          if (!file.name) continue;
          const fileDir = file.name.match(/(.*)[/\\]/);
          const fileDirPath = fileDir ? fileDir[1] : '';
          if (fileDirPath === dir) fileCount++;
        }

        const sep = document.createElement('div');
        sep.className = 'gallery-separator';
        sep.title = dir;

        // Default to open for the first separator if no state is saved, otherwise closed.
        const isOpen = separatorStates.has(dir) ? separatorStates.get(dir) : isFirstSeparator;
        separatorStates.set(dir, isOpen); // Ensure it's in the map
        if (isFirstSeparator) isFirstSeparator = false; // Subsequent separators will default to closed

        if (!isOpen) {
          sep.classList.add('gallery-separator-hidden');
        }

        // Create arrow span
        const arrow = document.createElement('span');
        arrow.className = 'gallery-separator-arrow';
        arrow.textContent = '▶';
        arrow.style.transform = isOpen ? 'rotate(90deg)' : 'rotate(0deg)';

        // Create directory name span
        const dirName = document.createElement('span');
        dirName.className = 'gallery-separator-name';
        dirName.textContent = dir;
        dirName.title = dir; // Show full path on hover

        // Create count span
        const count = document.createElement('span');
        count.className = 'gallery-separator-count';
        count.textContent = `${fileCount} files`;
        sep.dataset.totalFiles = fileCount; // Store total count for search filtering

        sep.appendChild(arrow);
        sep.appendChild(dirName);
        sep.appendChild(count);

        sep.onclick = () => handleSeparator(sep);
        el.files.insertBefore(sep, f);
      }
    }
  }

  // Second pass: hide files in closed directories
  for (const f of all) {
    if (!f.name) continue; // Skip separators

    const dir = f.name.match(/(.*)[/\\]/);
    if (dir && dir[1]) {
      const dirPath = dir[1];
      const isOpen = separatorStates.get(dirPath);
      if (isOpen === false) {
        f.style.display = 'none';
      }
    }
  }
}

// methods

const gallerySendImage = (_images) => [currentImage]; // invoked by gradio button

async function getHash(str, algo = 'SHA-256') {
  try {
    let hex = '';
    const strBuf = new TextEncoder().encode(str);
    const hash = await crypto.subtle.digest(algo, strBuf);
    const view = new DataView(hash);
    for (let i = 0; i < hash.byteLength; i += 4) hex += (`00000000${view.getUint32(i).toString(16)}`).slice(-8);
    return hex;
  } catch {
    return undefined;
  }
}

/**
 * Helper function to update status with sort mode
 * @param  {...string|[string, string]} messages - Each can be either a string to use as-is, or an array of a string label and value
 * @returns {void}
 */
function updateStatusWithSort(...messages) {
  if (!el.status) return;
  messages.unshift(['Sort', lastSortName]);
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < messages.length; i++) {
    const div = document.createElement('div');
    if (Array.isArray(messages[i])) {
      const [text1, text2] = messages[i];
      const tDiv1 = document.createElement('div');
      tDiv1.innerText = `${text1}:`;
      const tDiv2 = document.createElement('div');
      tDiv2.innerText = text2;
      tDiv2.title = text2;
      div.append(tDiv1, tDiv2);
    } else {
      const tDiv1 = document.createElement('div');
      tDiv1.innerText = messages[i];
      div.append(tDiv1);
    }
    fragment.append(div);
  }
  if (el.status.hasChildNodes()) el.status.innerHTML = '';
  el.status.append(fragment);
}

async function injectGalleryStatusCSS() {
  const style = document.createElement('style');
  style.textContent = `
  #tab-gallery-status {
    display: inline-flex;
    flex-flow: row wrap;
    justify-content: ${opts.theme_type?.toLowerCase() === 'modern' ? 'flex-start' : 'flex-end'};
  }
  #tab-gallery-status > div {
    display: flex;
    max-width: 100%;
    white-space: nowrap;
    & div {
      &:first-child {
        flex-shrink: 0;
        margin-right: 4px;
      }
      &:last-child:not(:first-child) {
        flex-shrink: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        direction: rtl;
        text-align: left;
      }
    }
  }
  #tab-gallery-status > div:not(:last-child)::after {
    content: '|';
    margin-inline: 6px;
  }`;
  document.head.append(style);
}

async function wsConnect(socket, timeout = 5000) {
  const intrasleep = 100;
  const ttl = timeout / intrasleep;
  const isOpened = () => (socket.readyState === WebSocket.OPEN);
  if (socket.readyState !== WebSocket.CONNECTING) return isOpened();

  let loop = 0;
  while (socket.readyState === WebSocket.CONNECTING && loop < ttl) {
    await new Promise((resolve) => { setTimeout(resolve, intrasleep); });
    loop++;
  }
  return isOpened();
}

async function gallerySearch() {
  if (el.search.busy) clearTimeout(el.search.busy);
  el.search.busy = setTimeout(async () => {
    const t0 = performance.now();
    const str = el.search.value.toLowerCase();
    const allFiles = Array.from(el.files.children).filter((node) => node.name);
    const allSeparators = Array.from(el.files.children).filter((node) => node.classList.contains('gallery-separator'));

    // If search is cleared, restore original view
    if (str === '') {
      allSeparators.forEach((sep) => {
        sep.style.display = 'flex';
        const isOpen = separatorStates.has(sep.title) ? separatorStates.get(sep.title) : false;

        const countSpan = sep.querySelector('.gallery-separator-count');
        if (countSpan && sep.dataset.totalFiles) {
          countSpan.textContent = `${sep.dataset.totalFiles} files`;
        }

        const arrow = sep.querySelector('.gallery-separator-arrow');
        sep.classList.toggle('gallery-separator-hidden', !isOpen);
        if (arrow) arrow.style.transform = isOpen ? 'rotate(90deg)' : 'rotate(0deg)';
      });

      allFiles.forEach((f) => {
        const dir = f.name.match(/(.*)[/\\]/);
        const dirPath = (dir && dir[1]) ? dir[1] : '';
        const isOpen = separatorStates.get(dirPath);
        f.style.display = (!dirPath || isOpen) ? 'unset' : 'none';
      });

      updateStatusWithSort('Filter', 'Cleared', ['Images', allFiles.length.toLocaleString()]);
      return;
    }

    // --- Search logic ---
    let totalFound = 0;
    const directoryMatches = new Map();
    const fileMatches = new WeakSet();
    const r = /^(.+)([=<>])(.*)/;

    for (const f of allFiles) {
      let isMatch = false;
      if (r.test(str)) {
        const match = str.match(r);
        const key = match[1].trim();
        const op = match[2].trim();
        let val = match[3].trim();
        if (key === 'mtime') val = new Date(val);
        if (((op === '=') && (f[key] === val)) || ((op === '>') && (f[key] > val)) || ((op === '<') && (f[key] < val))) {
          isMatch = true;
        }
      } else if (f.title?.toLowerCase().includes(str) || f.exif?.toLowerCase().includes(str)) {
        isMatch = true;
      }

      if (isMatch) {
        fileMatches.add(f);
        totalFound++;
        const dir = f.name.match(/(.*)[/\\]/);
        const dirPath = (dir && dir[1]) ? dir[1] : '';
        directoryMatches.set(dirPath, (directoryMatches.get(dirPath) || 0) + 1);
      }
    }

    // Update separators based on search results
    for (const sep of allSeparators) {
      const dirPath = sep.title;
      const foundCount = directoryMatches.get(dirPath) || 0;

      if (foundCount > 0) {
        sep.style.display = 'flex'; // Show separator
        sep.classList.remove('gallery-separator-hidden'); // Force open

        const arrow = sep.querySelector('.gallery-separator-arrow');
        if (arrow) arrow.style.transform = 'rotate(90deg)';

        // Removed file count update during search as it was buggy.
      } else {
        sep.style.display = 'none'; // Hide separator
      }
    }

    // Update file visibility
    for (const f of allFiles) {
      f.style.display = fileMatches.has(f) ? 'unset' : 'none';
    }

    const t1 = performance.now();
    updateStatusWithSort('Filter', ['Images', `${totalFound.toLocaleString()} / ${allFiles.length.toLocaleString()}`], `${iconStopwatch} ${Math.floor(t1 - t0).toLocaleString()}ms`);
    refreshGallerySelection();
  }, 250);
}

const findDuplicates = (arr, key) => {
  const map = new Map();
  return arr.filter((item) => {
    const value = item[key];
    if (map.has(value)) return true;
    map.set(value, true);
    return false;
  });
};

async function gallerySort(btn) {
  const t0 = performance.now();
  const arr = Array.from(el.files.children).filter((node) => node.name); // filter out separators
  if (arr.length === 0) return; // no files to sort
  if (btn) lastSort = btn.charCodeAt(0);
  const fragment = document.createDocumentFragment();

  // Helper to get directory path from a file node
  const getDirPath = (node) => {
    const match = node.name.match(/(.*)[/\\]/);
    return match ? match[1] : '';
  };

  // Partition into root files and subfolder files - root files always stay at top
  const rootFiles = arr.filter((node) => !getDirPath(node));
  const subfolderFiles = arr.filter((node) => getDirPath(node));

  // Group subfolder files by directory
  const folderGroups = new Map();
  for (const file of subfolderFiles) {
    const dir = getDirPath(file);
    if (!folderGroups.has(dir)) {
      folderGroups.set(dir, []);
    }
    folderGroups.get(dir).push(file);
  }

  // Sort function based on current sort mode
  let sortFn;
  switch (lastSort) {
    case 61789: // name asc
      lastSortName = 'Name Ascending';
      sortFn = (a, b) => a.name.localeCompare(b.name);
      break;
    case 61790: // name dsc
      lastSortName = 'Name Descending';
      sortFn = (a, b) => b.name.localeCompare(a.name);
      break;
    case 61792: // size asc
      lastSortName = 'Size Ascending';
      sortFn = (a, b) => a.size - b.size;
      break;
    case 61793: // size dsc
      lastSortName = 'Size Descending';
      sortFn = (a, b) => b.size - a.size;
      break;
    case 61794: // resolution asc
      lastSortName = 'Resolution Ascending';
      sortFn = (a, b) => a.width * a.height - b.width * b.height;
      break;
    case 61795: // resolution dsc
      lastSortName = 'Resolution Descending';
      sortFn = (a, b) => b.width * b.height - a.width * a.height;
      break;
    case 61662:
      lastSortName = 'Modified Ascending';
      sortFn = (a, b) => a.mtime - b.mtime;
      break;
    case 61661:
      lastSortName = 'Modified Descending';
      sortFn = (a, b) => b.mtime - a.mtime;
      break;
    default:
      lastSortName = 'None';
      sortFn = null;
      break;
  }

  // Sort root files
  if (sortFn) {
    rootFiles.sort(sortFn);
  }
  rootFiles.forEach((node) => fragment.appendChild(node));

  // Sort folder names alphabetically, then sort files within each folder
  const sortedFolderNames = Array.from(folderGroups.keys()).sort((a, b) => a.localeCompare(b));
  for (const folderName of sortedFolderNames) {
    const files = folderGroups.get(folderName);
    if (sortFn) {
      files.sort(sortFn);
    }
    files.forEach((node) => fragment.appendChild(node));
  }

  if (fragment.children.length === 0) return;
  el.files.innerHTML = '';
  el.files.appendChild(fragment);
  addSeparators();

  // After sorting and adding separators, ensure files respect separator states
  const all = Array.from(el.files.children);
  for (const f of all) {
    if (!f.name) continue; // Skip separators

    const dir = f.name.match(/(.*)[/\\]/);
    if (dir && dir[1]) {
      const dirPath = dir[1];
      const isOpen = separatorStates.get(dirPath);
      if (isOpen === false) {
        f.style.display = 'none';
      }
    }
  }

  const t1 = performance.now();
  log(`gallerySort: char=${lastSort} len=${arr.length} time=${Math.floor(t1 - t0)} sort=${lastSortName}`);
  updateStatusWithSort(['Images', arr.length.toLocaleString()], `${iconStopwatch} ${Math.floor(t1 - t0).toLocaleString()}ms`);
  refreshGallerySelection();
}

/**
 * Function for removing the cleaning overlay
 * @callback ClearMsgCallback
 * @returns {void}
 */

/**
 * Generate and display the overlay to announce cleanup is in progress.
 * @param {number} count - Number of entries being cleaned up
 * @param {boolean} all - Indicate that all thumbnails are being cleared
 * @returns {ClearMsgCallback}
 */
function showCleaningMsg(count, all = false) {
  // Rendering performance isn't a priority since this doesn't run often
  const parent = el.folders.parentElement;
  const cleaningOverlay = document.createElement('div');
  const msgDiv = document.createElement('div');
  const msgText = document.createElement('div');
  const msgInfo = document.createElement('div');
  const anim = document.createElement('span');

  parent.style.position = 'relative';
  cleaningOverlay.style.cssText = 'position: absolute; height: 100%; width: 100%; background-color: hsl(210 50 20 / 0.8); display: flex; align-items: center; justify-content: center; align-content: center; flex-wrap: wrap;';
  msgDiv.style.cssText = 'display: block; background-color: hsl(0 0 10); color: white; padding: 12px; border-radius: 8px;';
  msgText.style.cssText = 'font-size: 1.2em';
  msgInfo.style.cssText = 'font-size: 0.9em; text-align: center;';
  msgText.innerText = 'Thumbnail cleanup...';
  msgInfo.innerText = all ? 'Clearing all entries' : `Found ${count} old entries`;
  anim.classList.add('idbBusyAnim');

  msgDiv.append(msgText, msgInfo);
  cleaningOverlay.append(msgDiv, anim);
  parent.append(cleaningOverlay);
  return () => { cleaningOverlay.remove(); };
}

const maintenanceQueue = new SimpleFunctionQueue('Gallery Maintenance');

/**
 * Handles calling the cleanup function for the thumbnail cache
 * @param {string} folder - Folder to clean
 * @param {number} imgCount - Expected number of images in gallery
 * @param {AbortController} controller - AbortController that's handling this task
 * @param {boolean} force - Force full cleanup of the folder
 */
async function thumbCacheCleanup(folder, imgCount, controller, force = false) {
  if (!opts.browser_cache && !force) return;
  try {
    if (typeof folder !== 'string' || typeof imgCount !== 'number') {
      throw new Error('Function called with invalid arguments');
    }
    debug('Thumbnail DB cleanup: Waiting for gallery data to settle');
    await awaitForGallery(imgCount, controller.signal);
  } catch (err) {
    debug(`Thumbnail DB cleanup: Skipping cleanup for "${folder}" due to "${err}"`);
    return;
  }

  maintenanceQueue.enqueue({
    signal: controller.signal,
    callback: async () => {
      log(`Thumbnail DB cleanup: Checking if "${folder}" needs cleaning`);
      const t0 = performance.now();
      const keptGalleryHashes = force ? new Set() : new Set(galleryHashes.values()); // External context should be safe since this function run is guarded by AbortController/AbortSignal in the SimpleFunctionQueue
      const folderNormalized = folder.replace(/\/+/g, '/').replace(/\/$/, '');
      const recursiveFolder = IDBKeyRange.bound(folderNormalized, `${folderNormalized}\uffff`, false, true);
      const cachedHashesCount = await idbCount(recursiveFolder)
        .catch((e) => {
          error(`Thumbnail DB cleanup: Error when getting entry count for "${folder}".`, e);
          return Infinity; // Forces next check to fail if something went wrong
        });
      const cleanupCount = cachedHashesCount - keptGalleryHashes.size;
      if (!force && (cleanupCount < 500 || !Number.isFinite(cleanupCount))) {
        // Don't run when there aren't many excess entries
        return;
      }

      if (controller.signal.aborted) {
        debug(`Thumbnail DB cleanup: Cancelling "${folder}" cleanup due to "${controller.signal.reason}"`);
        return;
      }
      const cb_clearMsg = showCleaningMsg(cleanupCount);
      await idbFolderCleanup(keptGalleryHashes, recursiveFolder, controller.signal)
        .then((delcount) => {
          const t1 = performance.now();
          log(`Thumbnail DB cleanup: folder=${folder} kept=${keptGalleryHashes.size} deleted=${delcount} time=${Math.floor(t1 - t0)}ms`);
          currentGalleryFolder = null;
          el.clearCacheFolder.innerText = '<select a folder first>';
          updateStatusWithSort('Thumbnail cache cleared');
        })
        .catch((reason) => {
          SimpleFunctionQueue.abortLogger('Thumbnail DB cleanup:', reason);
        })
        .finally(async () => {
          await new Promise((resolve) => { setTimeout(resolve, 1000); }); // Delay removal by 1 second to ensure at least minimum visibility
          cb_clearMsg();
        });
    },
  });
}

function resetGalleryState(reason) {
  maintenanceController.abort(reason);
  const controller = new AbortController();
  maintenanceController = controller;

  galleryHashes.clear(); // Must happen AFTER the AbortController steps
  galleryProgressBar.clear();
  resetGallerySelection();
  return controller;
}

function clearCacheIfDisabled(browser_cache) {
  if (browser_cache === false) {
    log('Thumbnail DB cleanup:', 'Image gallery cache setting disabled. Clearing cache.');
    const controller = resetGalleryState('Clearing all thumbnails from cache');
    maintenanceQueue.enqueue({
      signal: controller.signal,
      callback: async () => {
        const t0 = performance.now();
        const cb_clearMsg = showCleaningMsg(0, true);
        await idbClearAll(controller.signal)
          .then(() => {
            log(`Thumbnail DB cleanup: Cache cleared. time=${Math.floor(performance.now() - t0)}ms`);
            currentGalleryFolder = null;
            el.clearCacheFolder.innerText = '<select a folder first>';
            updateStatusWithSort('Thumbnail cache cleared');
          })
          .catch((e) => {
            SimpleFunctionQueue.abortLogger('Thumbnail DB cleanup:', e);
          })
          .finally(async () => {
            await new Promise((resolve) => { setTimeout(resolve, 1000); });
            cb_clearMsg();
          });
      },
    });
  }
}

function addCacheClearLabel() { // Don't use async
  const setting = document.querySelector('#setting_browser_cache');
  if (setting) {
    const div = document.createElement('div');
    div.style.marginBlock = '0.75rem';

    const span = document.createElement('span');
    span.style.cssText = 'font-weight: bold; text-decoration: underline; cursor: pointer; color: var(--color-blue); user-select: none;';
    span.innerText = '<select a folder first>';

    div.append('Clear the thumbnail cache for: ', span, ' (double-click)');
    setting.parentElement.insertAdjacentElement('afterend', div);
    el.clearCacheFolder = span;

    span.addEventListener('dblclick', (evt) => {
      evt.preventDefault();
      evt.stopPropagation();
      if (!currentGalleryFolder) return;
      el.clearCacheFolder.style.color = 'var(--color-green)';
      setTimeout(() => {
        el.clearCacheFolder.style.color = 'var(--color-blue)';
      }, 1000);
      const controller = resetGalleryState('Clearing folder thumbnails cache');
      el.files.innerHTML = '';
      thumbCacheCleanup(currentGalleryFolder, 0, controller, true);
    });
    return true;
  }
  return false;
}

async function fetchFilesHT(evt, controller) {
  const t0 = performance.now();
  const fragment = document.createDocumentFragment();
  updateStatusWithSort(['Folder', evt.target.name], 'in-progress');
  let numFiles = 0;

  const res = await authFetch(`${window.api}/browser/files?folder=${encodeURI(evt.target.name)}`);
  if (!res || res.status !== 200) {
    updateStatusWithSort(['Folder', evt.target.name], ['Failed', res?.statusText || 'No response']);
    return;
  }
  const jsonData = await res.json();
  for (const line of jsonData) {
    const data = decodeURI(line).split('##F##');
    const fileName = data[1];
    const ext = fileName.split('.').pop().toLowerCase();
    if (SUPPORTED_EXTENSIONS.includes(ext)) {
      numFiles++;
      const f = new GalleryFile(data[0], fileName, controller.signal);
      fragment.appendChild(f);
    }
  }

  if (controller.signal.aborted) return;
  el.files.appendChild(fragment);

  const t1 = performance.now();
  log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
  updateStatusWithSort(['Folder', evt.target.name], ['Images', numFiles.toLocaleString()], `${iconStopwatch} ${Math.floor(t1 - t0).toLocaleString()}ms`);
  galleryProgressBar.start(numFiles);
  addSeparators();
  refreshGallerySelection();
  thumbCacheCleanup(evt.target.name, numFiles, controller);
}

async function fetchFilesWS(evt) { // fetch file-by-file list over websockets
  if (!url) return;
  // Abort previous controller and point to new controller for next time
  const controller = resetGalleryState('Gallery update'); // Called here because fetchFilesHT isn't called directly

  el.files.innerHTML = '';
  updateGalleryStyles();
  if (ws && ws.readyState === WebSocket.OPEN) ws.close(); // abort previous request
  let wsConnected = false;
  try {
    ws = new WebSocket(`${url}/sdapi/v1/browser/files`);
    wsConnected = await wsConnect(ws);
  } catch (err) {
    log('gallery: ws connect error', err);
    return;
  }
  log(`gallery: connected=${wsConnected} state=${ws?.readyState} url=${ws?.url}`);
  currentGalleryFolder = evt.target.name;
  if (el.clearCacheFolder) {
    el.clearCacheFolder.innerText = currentGalleryFolder;
  }
  if (!wsConnected) {
    await fetchFilesHT(evt, controller); // fallback to http
    return;
  }
  updateStatusWithSort(['Folder', evt.target.name]);
  const t0 = performance.now();
  let numFiles = 0;
  let t1 = performance.now();
  let fragment = document.createDocumentFragment();

  ws.onmessage = (event) => {
    t1 = performance.now();
    const data = decodeURI(event.data).split('##F##');
    if (data[0] === '#END#') {
      ws.close();
    } else {
      const fileName = data[1];
      const ext = fileName.split('.').pop().toLowerCase();
      if (SUPPORTED_EXTENSIONS.includes(ext)) {
        const file = new GalleryFile(data[0], fileName, controller.signal);
        numFiles++;
        fragment.appendChild(file);
        if (numFiles % 100 === 0) {
          updateStatusWithSort(['Folder', evt.target.name], ['Images', numFiles.toLocaleString()], 'in-progress', `${iconStopwatch} ${Math.floor(t1 - t0).toLocaleString()}ms`);
          el.files.appendChild(fragment);
          fragment = document.createDocumentFragment();
        }
      }
    }
  };
  ws.onclose = (event) => {
    if (controller.signal.aborted) return;
    el.files.appendChild(fragment);
    // gallerySort();
    log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
    updateStatusWithSort(['Folder', evt.target.name], ['Images', numFiles.toLocaleString()], `${iconStopwatch} ${Math.floor(t1 - t0).toLocaleString()}ms`);
    galleryProgressBar.start(numFiles);
    addSeparators();
    refreshGallerySelection();
    thumbCacheCleanup(evt.target.name, numFiles, controller);
  };
  ws.onerror = (event) => {
    log('gallery ws error', event);
  };
  ws.send(encodeURI(evt.target.name));
}

async function updateFolders() {
  // if (el.folders.children.length > 0) return;
  const res = await authFetch(`${window.api}/browser/folders`);
  if (!res || res.status !== 200) return;
  url = res.url.split('/sdapi')[0].replace('http', 'ws'); // update global url as ws need fqdn
  const folders = await res.json();
  el.folders.innerHTML = '';
  for (const folder of folders) {
    const f = new GalleryFolder(folder);
    el.folders.appendChild(f);
  }
}

async function monitorGalleries() {
  async function galleryMutation(mutations) {
    const galleries = mutations.filter((m) => m.target?.classList?.contains('preview'));
    for (const gallery of galleries) {
      const links = gallery.target.querySelectorAll('a');
      for (const link of links) {
        const href = link.getAttribute('href');
        if (!href) continue;
        const fn = href.split('/').pop().split('\\').pop();
        link.setAttribute('download', fn);
      }
    }
  }

  const galleryElements = gradioApp().querySelectorAll('.gradio-gallery');
  for (const gallery of galleryElements) {
    const galleryObserver = new MutationObserver(galleryMutation);
    galleryObserver.observe(gallery, { childList: true, subtree: true, attributes: true });
  }
}

async function setOverlayAnimation() {
  const busyAnimation = document.createElement('style');
  // eslint-disable-next-line @stylistic/max-len
  busyAnimation.textContent = '.idbBusyAnim{width:16px;height:16px;border-radius:50%;display:block;margin:40px;position:relative;background:#ff3d00;color:#fff;box-shadow:-24px 0,24px 0;box-sizing:border-box;animation:2s ease-in-out infinite overlayRotation}@keyframes overlayRotation{0%{transform:rotate(0)}100%{transform:rotate(360deg)}}';
  document.head.append(busyAnimation);
}

async function galleryClearInit() {
  let galleryClearInitTimeout = 0;
  const tryCleanupInit = setInterval(() => {
    if (addCacheClearLabel() || galleryClearInitTimeout++ === 60) {
      clearInterval(tryCleanupInit);
      monitorOption('browser_cache', clearCacheIfDisabled);
    }
  }, 1000);
}

async function initGalleryAutoRefresh() {
  const isModern = opts.theme_type?.toLowerCase() === 'modern';
  let galleryTab = isModern ? document.getElementById('gallery_tabitem') : document.getElementById('tab_gallery');
  let timeout = 0;
  while (!galleryTab && timeout++ < 60) {
    await new Promise((resolve) => { setTimeout(resolve, 1000); });
    galleryTab = isModern ? document.getElementById('gallery_tabitem') : document.getElementById('tab_gallery');
  }
  if (!galleryTab) {
    throw new Error('Timed out waiting for gallery tab element');
  }
  const displayNoneRegEx = /display:\s*none/;
  async function galleryAutoRefresh(mutations) {
    if (!opts.browser_gallery_autoupdate) return;
    for (const mutation of mutations) {
      switch (mutation.attributeName) {
        case 'class':
          if (mutation.oldValue.includes('hidden') && !mutation.target.classList.contains('hidden')) {
            await updateFolders();
            GalleryFolder.getActive()?.click();
          }
          break;
        case 'style':
          if (displayNoneRegEx.test(mutation.oldValue) && !displayNoneRegEx.test(mutation.target.style.display)) {
            await updateFolders();
            GalleryFolder.getActive()?.click();
          }
          break;
        default:
          break;
      }
    }
  }
  const galleryVisObserver = new MutationObserver(galleryAutoRefresh);
  galleryVisObserver.observe(galleryTab, { attributeFilter: ['class', 'style'], attributeOldValue: true });
}

async function blockQueueUntilReady() {
  // Add block to maintenanceQueue until cache is ready
  maintenanceQueue.enqueue({
    signal: new AbortController().signal, // Use standalone AbortSignal that can't be aborted
    callback: async () => {
      let timeout = 0;
      while (!idbIsReady() && timeout++ < 60) {
        await new Promise((resolve) => { setTimeout(resolve, 1000); });
      }
      if (!idbIsReady()) {
        throw new Error('Timed out waiting for thumbnail cache');
      }
    },
  });
}

async function initGallery() { // triggered on gradio change to monitor when ui gets sufficiently constructed
  log('initGallery');
  el.folders = gradioApp().getElementById('tab-gallery-folders');
  el.files = gradioApp().getElementById('tab-gallery-files');
  el.status = gradioApp().getElementById('tab-gallery-status');
  el.search = gradioApp().querySelector('#tab-gallery-search textarea');
  if (!el.folders || !el.files || !el.status || !el.search) {
    error('initGallery', 'Missing gallery elements');
    return;
  }

  blockQueueUntilReady(); // Run first
  updateGalleryStyles();
  injectGalleryStatusCSS();
  setOverlayAnimation();
  galleryClearInit();
  const progress = gradioApp().getElementById('tab-gallery-progress');
  if (progress) {
    galleryProgressBar.attachTo(progress);
  } else {
    log('initGallery', 'Failed to attach loading progress bar');
  }
  el.search.addEventListener('input', gallerySearch);
  el.btnSend = gradioApp().getElementById('tab-gallery-send-image');
  document.getElementById('tab-gallery-files').style.height = opts.logmonitor_show ? '75vh' : '85vh';

  monitorGalleries();
  updateFolders();
  initGalleryAutoRefresh();
  [
    'browser_folders',
    'outdir_samples',
    'outdir_txt2img_samples',
    'outdir_img2img_samples',
    'outdir_control_samples',
    'outdir_extras_samples',
    'outdir_save',
    'outdir_video',
    'outdir_init_images',
    'outdir_grids',
    'outdir_txt2img_grids',
    'outdir_img2img_grids',
    'outdir_control_grids',
  ].forEach((op) => { monitorOption(op, updateFolders); });
}

// register on startup

customElements.define('gallery-folder', GalleryFolder);
customElements.define('gallery-file', GalleryFile);
