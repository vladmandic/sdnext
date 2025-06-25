/* eslint-disable max-classes-per-file */

let ws;
let url;
let currentImage;
let pruneImagesTimer;
let outstanding = 0;
let lastSort = 0;
let lastSortName = 'none';
const el = {
  folders: undefined,
  files: undefined,
  search: undefined,
  status: undefined,
  btnSend: undefined,
};

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
        .gallery-folder { cursor: pointer; padding: 8px 6px 8px 6px; background-color: var(--sd-secondary-color); }
        .gallery-folder:hover { background-color: var(--button-primary-background-fill-hover); }
        .gallery-folder-selected { background-color: var(--sd-button-selected-color); color: var(--sd-button-selected-text-color); }
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
    div.textContent = `\uf44a ${this.name}`;
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

async function addSeparators() {
  document.querySelectorAll('.gallery-separator').forEach((node) => el.files.removeChild(node));
  const all = Array.from(el.files.children);
  let lastDir;
  for (const f of all) {
    let dir = f.name.match(/(.*)[\/\\]/);
    if (!dir) dir = '';
    else dir = dir[1];
    if (dir !== lastDir) {
      lastDir = dir;
      if (dir.length > 0) {
        const sep = document.createElement('div');
        sep.className = 'gallery-separator';
        sep.innerText = dir;
        sep.title = dir;
        el.files.insertBefore(sep, f);
      }
    }
  }
}

async function delayFetchThumb(fn) {
  while (outstanding > 16) await new Promise((resolve) => setTimeout(resolve, 50)); // eslint-disable-line no-promise-executor-return
  outstanding++;
  const res = await fetch(`${window.api}/browser/thumb?file=${encodeURI(fn)}`, { priority: 'low' });
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
    const ext = this.name.split('.').pop().toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'gif', 'webp', 'jxl', 'svg', 'mp4'].includes(ext)) {
      console.error(`gallery: type=${ext} file=${this.name} unsupported`);
      return;
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
    this.style.display = this.title.toLowerCase().includes(el.search.value.toLowerCase()) ? 'unset' : 'none';
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

async function gallerySearch(evt) {
  if (el.search.busy) clearTimeout(el.search.busy);
  el.search.busy = setTimeout(async () => {
    let numFound = 0;
    const all = Array.from(el.files.children);
    const str = el.search.value.toLowerCase();
    const r = /^(.+)([=<>])(.*)/;
    const t0 = performance.now();
    for (const f of all) {
      if (r.test(str)) {
        const match = str.match(r);
        const key = match[1].trim();
        const op = match[2].trim();
        let val = match[3].trim();
        if (key === 'mtime') val = new Date(val);
        if (((op === '=') && (f[key] === val)) || ((op === '>') && (f[key] > val)) || ((op === '<') && (f[key] < val))) {
          f.style.display = 'unset';
          numFound++;
        } else {
          f.style.display = 'none';
        }
      } else if (f.title?.toLowerCase().includes(str) || f.exif?.toLowerCase().includes(str)) {
        f.style.display = 'unset';
        numFound++;
      } else {
        f.style.display = 'none';
      }
      const t1 = performance.now();
      el.status.innerText = `Filter | ${f.folder} | ${numFound.toLocaleString()}/${all.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
    }
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
  lastSortName = 'none';
  const fragment = document.createDocumentFragment();
  switch (lastSort) {
    case 61789: // name asc
      lastSortName = 'name asc';
      arr
        .sort((a, b) => a.name.localeCompare(b.name))
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61790: // name dsc
      lastSortName = 'name dsc';
      arr
        .sort((b, a) => a.name.localeCompare(b.name))
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61792: // size asc
      lastSortName = 'size asc';
      arr
        .sort((a, b) => a.size - b.size)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61793: // size dsc
      lastSortName = 'size dsc';
      arr
        .sort((b, a) => a.size - b.size)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61794: // resolution asc
      lastSortName = 'resolution asc';
      arr
        .sort((a, b) => a.width * a.height - b.width * b.height)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61795: // resolution dsc
      lastSortName = 'resolution dsc';
      arr
        .sort((b, a) => a.width * a.height - b.width * b.height)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61662:
      lastSortName = 'modified asc';
      arr
        .sort((a, b) => a.mtime - b.mtime)
        .forEach((node) => fragment.appendChild(node));
      break;
    case 61661:
      lastSortName = 'modified dsc';
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
  const t1 = performance.now();
  log(`gallerySort: char=${lastSort} len=${arr.length} time=${Math.floor(t1 - t0)} sort=${lastSortName}`);
  el.status.innerText = `Sort | ${lastSortName} | ${arr.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
}

async function fetchFilesHT(evt) {
  const t0 = performance.now();
  const fragment = document.createDocumentFragment();
  el.status.innerText = `Folder | ${evt.target.name} | in-progress`;
  let numFiles = 0;

  const res = await fetch(`${window.api}/browser/files?folder=${encodeURI(evt.target.name)}`);
  if (!res || res.status !== 200) {
    el.status.innerText = `Folder | ${evt.target.name} | failed: ${res?.statusText}`;
    return;
  }
  const jsonData = await res.json();
  for (const line of jsonData) {
    const data = decodeURI(line).split('##F##');
    numFiles++;
    const f = new GalleryFile(data[0], data[1]);
    fragment.appendChild(f);
  }

  el.files.appendChild(fragment);

  const t1 = performance.now();
  log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
  el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
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
  el.status.innerText = `Folder | ${evt.target.name}`;
  const t0 = performance.now();
  let numFiles = 0;
  let t1 = performance.now();
  let fragment = document.createDocumentFragment();

  ws.onmessage = (event) => {
    numFiles++;
    t1 = performance.now();
    const data = decodeURI(event.data).split('##F##');
    if (data[0] === '#END#') {
      ws.close();
    } else {
      const file = new GalleryFile(data[0], data[1]);
      fragment.appendChild(file);
      if (numFiles % 100 === 0) {
        el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | in-progress | ${Math.floor(t1 - t0).toLocaleString()}ms`;
        el.files.appendChild(fragment);
        fragment = document.createDocumentFragment();
      }
    }
  };
  ws.onclose = (event) => {
    el.files.appendChild(fragment);
    // gallerySort();
    log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
    el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
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

async function initGallery() { // triggered on gradio change to monitor when ui gets sufficiently constructed
  log('initGallery');
  el.folders = gradioApp().getElementById('tab-gallery-folders');
  el.files = gradioApp().getElementById('tab-gallery-files');
  el.status = gradioApp().getElementById('tab-gallery-status');
  el.search = gradioApp().querySelector('#tab-gallery-search textarea');
  el.search.addEventListener('input', gallerySearch);
  el.btnSend = gradioApp().getElementById('tab-gallery-send-image');
  document.getElementById('tab-gallery-files').style.height = opts.logmonitor_show ? '75vh' : '85vh';

  const intersectionObserver = new IntersectionObserver((entries) => {
    if (entries[0].intersectionRatio <= 0) galleryHidden();
    if (entries[0].intersectionRatio > 0) galleryVisible();
  });
  intersectionObserver.observe(el.folders);
}

// register on startup

customElements.define('gallery-folder', GalleryFolder);
customElements.define('gallery-file', GalleryFile);
