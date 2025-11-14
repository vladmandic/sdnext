/* eslint-disable max-classes-per-file */
let ws;
let url;
let currentImage;
let pruneImagesTimer;
let outstanding = 0;
let lastSort = 0;
let lastSortName = 'None';
// Store separator states for the session
const separatorStates = new Map();
const el = {
  folders: undefined,
  files: undefined,
  search: undefined,
  status: undefined,
  btnSend: undefined,
};

const SUPPORTED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp', 'tiff', 'jp2', 'jxl', 'gif', 'mp4', 'mkv', 'avi', 'mjpeg', 'mpg', 'avr'];

// HTML Elements

class GalleryFolder extends HTMLElement {
  constructor(name) {
    super();
    this.name = decodeURI(name);
    this.shadow = this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    const style = document.createElement('style'); // silly but necessasry since we're inside shadowdom
    if (window.opts.theme_type === 'Modern') {
      style.textContent = `
        .gallery-folder { cursor: pointer; padding: 8px 6px 8px 6px; background-color: var(--sd-button-normal-color); border-radius: var(--sd-border-radius); text-align: left; min-width: 12em;}
        .gallery-folder:hover { background-color: var(--button-primary-background-fill-hover); }
        .gallery-folder-selected { background-color: var(--sd-button-selected-color); color: var(--sd-button-selected-text-color); }
        .gallery-folder-icon { font-size: 1.2em; color: var(--sd-button-icon-color); margin-right: 1em; filter: drop-shadow(1px 1px 2px black); float: left; }
      `;
    } else {
      style.textContent = `
        .gallery-folder { cursor: pointer; padding: 8px 6px 8px 6px; }
        .gallery-folder:hover { background-color: var(--button-primary-background-fill-hover); }
        .gallery-folder-selected { background-color: var(--button-primary-background-fill); }
      `;
    }
    this.shadow.appendChild(style);
    const div = document.createElement('div');
    div.className = 'gallery-folder';
    div.innerHTML = `<span class="gallery-folder-icon">\uf03e</span> ${this.name}`;
    div.addEventListener('click', () => {
      for (const folder of el.folders.children) {
        if (folder.name === this.name) folder.shadow.children[1].classList.add('gallery-folder-selected');
        else folder.shadow.children[1].classList.remove('gallery-folder-selected');
      }
    });
    div.addEventListener('click', fetchFilesWS); // eslint-disable-line no-use-before-define
    this.shadow.appendChild(div);
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
    const fileDir = f.name.match(/(.*)[\/\\]/);
    const fileDirPath = fileDir ? fileDir[1] : '';

    if (separator.title.length > 0 && fileDirPath === separator.title) {
      f.style.display = nowHidden ? 'none' : 'unset';
    }
  }
  // Note: Count is not updated here on manual toggle, as it reflects the total.
  // If I end up implementing it, the search function will handle dynamic count updates.
}

async function addSeparators() {
  document.querySelectorAll('.gallery-separator').forEach((node) => el.files.removeChild(node));
  const all = Array.from(el.files.children);
  let lastDir;
  let isFirstSeparator = true; // Flag to open the first separator by default

  // First pass: create separators
  for (const f of all) {
    let dir = f.name?.match(/(.*)[\/\\]/);
    if (!dir) dir = '';
    else dir = dir[1];
    if (dir !== lastDir) {
      lastDir = dir;
      if (dir.length > 0) {
        // Count files in this directory
        let fileCount = 0;
        for (const file of all) {
          if (!file.name) continue;
          const fileDir = file.name.match(/(.*)[\/\\]/);
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

    const dir = f.name.match(/(.*)[\/\\]/);
    if (dir && dir[1]) {
      const dirPath = dir[1];
      const isOpen = separatorStates.get(dirPath);
      if (isOpen === false) {
        f.style.display = 'none';
      }
    }
  }
}

async function delayFetchThumb(fn) {
  while (outstanding > 16) await new Promise((resolve) => setTimeout(resolve, 50)); // eslint-disable-line no-promise-executor-return
  outstanding++;
  const ts = Date.now().toString();
  const res = await fetch(`${window.api}/browser/thumb?file=${encodeURI(fn)}&ts=${ts}`, { priority: 'low' });
  if (!res.ok) {
    error(`fetchThumb: ${res.statusText}`);
    outstanding--;
    return undefined;
  }
  const json = await res.json();
  outstanding--;
  if (!res || !json || json.error || Object.keys(json).length === 0) {
    if (json.error) error(`fetchThumb: ${json.error}`);
    return undefined;
  }
  return json;
}

class GalleryFile extends HTMLElement {
  constructor(folder, file) {
    super();
    this.folder = folder;
    this.name = file;
    this.size = 0;
    this.mtime = 0;
    this.hash = undefined;
    this.exif = '';
    this.width = 0;
    this.height = 0;
    this.src = `${this.folder}/${this.name}`;
    this.shadow = this.attachShadow({ mode: 'open' });
  }

  async connectedCallback() {
    if (this.shadow.children.length > 0) {
      return;
    }

    // Check separator state early to hide the element immediately
    const dir = this.name.match(/(.*)[\/\\]/);
    if (dir && dir[1]) {
      const dirPath = dir[1];
      const isOpen = separatorStates.get(dirPath);
      if (isOpen === false) {
        this.style.display = 'none';
      }
    }

    this.hash = await getHash(`${this.folder}/${this.name}/${this.size}/${this.mtime}`); // eslint-disable-line no-use-before-define
    const style = document.createElement('style');
    const width = opts.browser_fixed_width ? `${opts.extra_networks_card_size}px` : 'unset';
    style.textContent = `
      .gallery-file {
        object-fit: contain;
        cursor: pointer;
        height: ${opts.extra_networks_card_size}px;
        width: ${width};
      }
      .gallery-file:hover {
        filter: grayscale(100%);
      }
    `;

    const cache = (this.hash && opts.browser_cache) ? await idbGet(this.hash) : undefined;
    const img = document.createElement('img');
    img.className = 'gallery-file';
    img.loading = 'lazy';
    img.onload = async () => {
      img.title += `\nResolution: ${this.width} x ${this.height}`;
      this.title = img.title;
      if (!cache && opts.browser_cache) {
        if ((this.width === 0) || (this.height === 0)) { // fetch thumb failed so we use actual image
          this.width = img.naturalWidth;
          this.height = img.naturalHeight;
        }
      }
    };
    let ok = true;
    if (cache && cache.img) {
      img.src = cache.img;
      this.exif = cache.exif;
      this.width = cache.width;
      this.height = cache.height;
      this.size = cache.size;
      this.mtime = new Date(cache.mtime);
    } else {
      try {
        const json = await delayFetchThumb(this.src);
        if (!json) {
          ok = false;
        } else {
          img.src = json.data;
          this.exif = json.exif;
          this.width = json.width;
          this.height = json.height;
          this.size = json.size;
          this.mtime = new Date(json.mtime);
          await idbAdd({
            hash: this.hash,
            folder: this.folder,
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
      } catch (err) { // thumb fetch failed so assign actual image
        img.src = `file=${this.src}`;
      }
    }
    if (!ok) {
      return;
    }
    img.onclick = () => {
      currentImage = this.src;
      el.btnSend.click();
    };
    img.title = `Folder: ${this.folder}\nFile: ${this.name}\nSize: ${this.size.toLocaleString()} bytes\nModified: ${this.mtime.toLocaleString()}`;
    if (this.shadow.children.length > 0) {
      return; // avoid double-adding
    }
    this.title = img.title;

    // Final visibility check based on search term.
    const shouldDisplayBasedOnSearch = this.title.toLowerCase().includes(el.search.value.toLowerCase());
    if (this.style.display !== 'none') { // Only proceed if not already hidden by a closed separator
      this.style.display = shouldDisplayBasedOnSearch ? 'unset' : 'none';
    }

    this.shadow.appendChild(style);
    this.shadow.appendChild(img);
  }
}

// methods

const gallerySendImage = (_images) => [currentImage]; // invoked by gadio button

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

// Helper function to update status with sort mode
function updateStatusWithSort(message) {
  const sortIndicator = `Sort: ${lastSortName} | `;
  el.status.innerText = sortIndicator + message;
}

async function wsConnect(socket, timeout = 5000) {
  const intrasleep = 100;
  const ttl = timeout / intrasleep;
  const isOpened = () => (socket.readyState === WebSocket.OPEN);
  if (socket.readyState !== WebSocket.CONNECTING) return isOpened();

  let loop = 0;
  while (socket.readyState === WebSocket.CONNECTING && loop < ttl) {
    await new Promise((resolve) => setTimeout(resolve, intrasleep)); // eslint-disable-line no-promise-executor-return
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
        const dir = f.name.match(/(.*)[\/\\]/);
        const dirPath = (dir && dir[1]) ? dir[1] : '';
        const isOpen = separatorStates.get(dirPath);
        f.style.display = (!dirPath || isOpen) ? 'unset' : 'none';
      });

      updateStatusWithSort(`Filter | Cleared | ${allFiles.length.toLocaleString()} images`);
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
        const dir = f.name.match(/(.*)[\/\\]/);
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
    updateStatusWithSort(`Filter | ${totalFound.toLocaleString()} / ${allFiles.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`);
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
  lastSortName = 'None';
  const fragment = document.createDocumentFragment();
  switch (lastSort) {
    case 61789: // name asc
      lastSortName = 'Name Ascending';
      arr
        .sort((a, b) => a.name.localeCompare(b.name))
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61790: // name dsc
      lastSortName = 'Name Descending';
      arr
        .sort((b, a) => a.name.localeCompare(b.name))
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61792: // size asc
      lastSortName = 'Size Ascending';
      arr
        .sort((a, b) => a.size - b.size)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61793: // size dsc
      lastSortName = 'Size Descending';
      arr
        .sort((b, a) => a.size - b.size)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61794: // resolution asc
      lastSortName = 'Resolution Ascending';
      arr
        .sort((a, b) => a.width * a.height - b.width * b.height)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61795: // resolution dsc
      lastSortName = 'Resolution Descending';
      arr
        .sort((b, a) => a.width * a.height - b.width * b.height)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61662:
      lastSortName = 'Modified Ascending';
      arr
        .sort((a, b) => a.mtime - b.mtime)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61661:
      lastSortName = 'Modified Descending';
      arr
        .sort((b, a) => a.mtime - b.mtime)
        .forEach((node) => fragment.appendChild(node));
      break;
    default:
      break;
  }
  if (fragment.children.length === 0) return;
  el.files.innerHTML = '';
  el.files.appendChild(fragment);
  addSeparators();

  // After sorting and adding separators, ensure files respect separator states
  const all = Array.from(el.files.children);
  for (const f of all) {
    if (!f.name) continue; // Skip separators

    const dir = f.name.match(/(.*)[\/\\]/);
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
  updateStatusWithSort(`${arr.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`);
}

async function fetchFilesHT(evt) {
  const t0 = performance.now();
  const fragment = document.createDocumentFragment();
  updateStatusWithSort(`Folder: ${evt.target.name} | in-progress`);
  let numFiles = 0;

  const res = await fetch(`${window.api}/browser/files?folder=${encodeURI(evt.target.name)}`);
  if (!res || res.status !== 200) {
    updateStatusWithSort(`Folder: ${evt.target.name} | failed: ${res?.statusText}`);
    return;
  }
  const jsonData = await res.json();
  for (const line of jsonData) {
    const data = decodeURI(line).split('##F##');
    const fileName = data[1];
    const ext = fileName.split('.').pop().toLowerCase();
    if (SUPPORTED_EXTENSIONS.includes(ext)) {
      numFiles++;
      const f = new GalleryFile(data[0], fileName);
      fragment.appendChild(f);
    }
  }

  el.files.appendChild(fragment);

  const t1 = performance.now();
  log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
  updateStatusWithSort(`Folder: ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`);
  addSeparators();
}

async function fetchFilesWS(evt) { // fetch file-by-file list over websockets
  el.files.innerHTML = '';
  if (!url) return;
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
  if (!wsConnected) {
    await fetchFilesHT(evt); // fallback to http
    return;
  }
  updateStatusWithSort(`Folder: ${evt.target.name}`);
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
        const file = new GalleryFile(data[0], fileName);
        numFiles++;
        fragment.appendChild(file);
        if (numFiles % 100 === 0) {
          updateStatusWithSort(`Folder: ${evt.target.name} | ${numFiles.toLocaleString()} images | in-progress | ${Math.floor(t1 - t0).toLocaleString()}ms`);
          el.files.appendChild(fragment);
          fragment = document.createDocumentFragment();
        }
      }
    }
  };
  ws.onclose = (event) => {
    el.files.appendChild(fragment);
    // gallerySort();
    log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
    updateStatusWithSort(`Folder: ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`);
    addSeparators();
  };
  ws.onerror = (event) => {
    log('gallery ws error', event);
  };
  ws.send(encodeURI(evt.target.name));
}

async function pruneImages() {
  // TODO replace img.src with placeholder for images that are not visible
}

async function galleryVisible() {
  // if (el.folders.children.length > 0) return;
  const res = await fetch(`${window.api}/browser/folders`);
  if (!res || res.status !== 200) return;
  el.folders.innerHTML = '';
  url = res.url.split('/sdapi')[0].replace('http', 'ws'); // update global url as ws need fqdn
  const folders = await res.json();
  for (const folder of folders) {
    const f = new GalleryFolder(folder);
    el.folders.appendChild(f);
  }
  pruneImagesTimer = setInterval(pruneImages, 1000);
}

async function galleryHidden() {
  if (pruneImagesTimer) clearInterval(pruneImagesTimer);
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
  el.search.addEventListener('input', gallerySearch);
  el.btnSend = gradioApp().getElementById('tab-gallery-send-image');
  document.getElementById('tab-gallery-files').style.height = opts.logmonitor_show ? '75vh' : '85vh';

  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) galleryHidden();
    if (entries[0].intersectionRatio > 0) galleryVisible();
  });
  intersectionObserver.observe(el.folders);
  monitorGalleries();
}

// register on startup

customElements.define('gallery-folder', GalleryFolder);
customElements.define('gallery-file', GalleryFile);
