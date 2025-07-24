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

// Add CSS for collapsible separators
function addSeparatorStyles() {
  if (document.getElementById('gallery-separator-styles')) return;
  const style = document.createElement('style');
  style.id = 'gallery-separator-styles';
  // FIX: Use existing theme variables and adjust flex properties for spacing.
  style.textContent = `
    .gallery-separator {
      cursor: pointer;
      padding: 8px 12px;
      margin: 4px 0;
      background-color: var(--sd-group-background-color, #2c2c2c);
      color: var(--sd-input-text-color, #e0e0e0);
      border-radius: 4px;
      font-weight: bold;
      display: flex;
      align-items: center;
      gap: 8px;
      user-select: none;
      transition: background-color 0.2s;
    }
    .gallery-separator:hover {
      background-color: var(--sd-panel-background-color, #444);
    }
    .gallery-separator-title {
      flex-grow: 1; /* Changed from flex: 1 */
      margin-right: 8px; /* Added margin for spacing */
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .gallery-separator-icon {
      font-family: monospace;
      font-size: 14px;
      width: 16px;
      text-align: center;
      transition: transform 0.2s ease;
      display: inline-block;
      transform-origin: center;
      transform: rotate(90deg);
    }
    .gallery-separator.collapsed .gallery-separator-icon {
      transform: rotate(0deg);
    }
    .gallery-separator-count {
      font-size: 12px;
      color: var(--sd-label-color, #b0b0b0);
      font-weight: normal;
      margin-left: auto; /* Keeps it to the right */
      flex-shrink: 0; /* Prevents count from shrinking */
    }
    .gallery-section {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      overflow: hidden;
    }
    .gallery-section.collapsed {
      display: none !important;
    }
  `;
  document.head.appendChild(style);
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

// Load collapsed state from localStorage
function loadCollapsedState() {
  try {
    const state = localStorage.getItem('gallery-collapsed-sections');
    return state ? JSON.parse(state) : {};
  } catch {
    return {};
  }
}

// Check if a section has been explicitly set to expanded
function isSectionExpanded(sectionName) {
  const state = loadCollapsedState();
  // If no state is saved, default to collapsed (true)
  // If state exists, use it (false = expanded, true = collapsed)
  // TODO: I didn't end up liking this function, but I already wrote it, need to make it a setting
  return state.hasOwnProperty(sectionName) ? !state[sectionName] : false;
}

// Save collapsed state to localStorage
function saveCollapsedState(sectionName, isCollapsed) {
  try {
    const state = loadCollapsedState();
    state[sectionName] = isCollapsed;
    localStorage.setItem('gallery-collapsed-sections', JSON.stringify(state));
  } catch {
    // Ignore localStorage errors
  }
}

// This function is no longer used after the refactor, kept for compatibility
async function addSeparators() {
  // Function stub - separators are now created during file loading
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
    // Skip if already loaded
    if (this.shadow.children.length > 0) {
      return;
    }
    
    // Check if this element has a data-skip-load attribute
    if (this.dataset.skipLoad === 'true') {
      return;
    }
    
    const ext = this.name.split('.').pop().toLowerCase();
    if (!['jpg', 'jpeg', 'png', 'gif', 'webp', 'jxl', 'svg', 'mp4'].includes(ext)) {
      // console.error(`gallery: type=${ext} file=${this.name} unsupported`);
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
    const str = el.search.value.toLowerCase();
    const r = /^(.+)([=<>])(.*)/;
    const t0 = performance.now();
    
    // If searching, ensure all sections are loaded
    if (str.length > 0) {
      const sections = document.querySelectorAll('.gallery-section');
      for (const section of sections) {
        if (section.dataset.loaded === 'false') {
          section.dataset.loaded = 'true';
          const images = section.querySelectorAll('gallery-file');
          for (const img of images) {
            delete img.dataset.skipLoad;
            await img.connectedCallback();
          }
        }
      }
    }
    
    // Search through all gallery-file elements
    const allFiles = el.files.querySelectorAll('gallery-file');
    for (const f of allFiles) {
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
    }
    
    const t1 = performance.now();
    el.status.innerText = `Filter | ${numFound.toLocaleString()}/${allFiles.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
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
  
  // Collect all files from all sections and root
  const allFiles = [];
  
  // Get root files (not in any section)
  el.files.querySelectorAll(':scope > gallery-file').forEach(file => {
    file.sectionName = '';
    file.wasLoaded = true; // Root files are always loaded
    allFiles.push(file);
  });
  
  // Get files from sections
  const sections = document.querySelectorAll('.gallery-section');
  sections.forEach(section => {
    const files = Array.from(section.querySelectorAll('gallery-file'));
    files.forEach(file => {
      file.sectionName = section.dataset.sectionName;
      file.wasLoaded = !file.dataset.skipLoad;
      allFiles.push(file);
    });
  });
  
  if (allFiles.length === 0) return;
  if (btn) lastSort = btn.charCodeAt(0);
  lastSortName = 'none';
  
  // Sort the files
  switch (lastSort) {
    case 61789: // name asc
      lastSortName = 'name asc';
      allFiles.sort((a, b) => a.name.localeCompare(b.name));
      break;
    case 61790: // name dsc
      lastSortName = 'name dsc';
      allFiles.sort((b, a) => a.name.localeCompare(b.name));
      break;
    case 61792: // size asc
      lastSortName = 'size asc';
      allFiles.sort((a, b) => a.size - b.size);
      break;
    case 61793: // size dsc
      lastSortName = 'size dsc';
      allFiles.sort((b, a) => a.size - b.size);
      break;
    case 61794: // resolution asc
      lastSortName = 'resolution asc';
      allFiles.sort((a, b) => a.width * a.height - b.width * b.height);
      break;
    case 61795: // resolution dsc
      lastSortName = 'resolution dsc';
      allFiles.sort((b, a) => a.width * a.height - b.width * b.height);
      break;
    case 61662:
      lastSortName = 'modified asc';
      allFiles.sort((a, b) => a.mtime - b.mtime);
      break;
    case 61661:
      lastSortName = 'modified dsc';
      allFiles.sort((b, a) => a.mtime - b.mtime);
      break;
    default:
      return;
  }
  
  // Clear sections and remove root files
  el.files.querySelectorAll(':scope > gallery-file').forEach(file => file.remove());
  sections.forEach(section => {
    section.innerHTML = '';
  });
  
  // Redistribute files
  allFiles.forEach(file => {
    if (file.sectionName === '') {
      // Root file - find the right position among separators
      const separators = Array.from(el.files.querySelectorAll('.gallery-separator'));
      if (separators.length > 0) {
        el.files.insertBefore(file, separators[0]);
      } else {
        el.files.appendChild(file);
      }
    } else {
      // Section file
      const section = Array.from(sections).find(s => s.dataset.sectionName === file.sectionName);
      if (section) {
        // Restore skipLoad state if file wasn't loaded
        if (!file.wasLoaded) {
          file.dataset.skipLoad = 'true';
        }
        section.appendChild(file);
      }
    }
  });
  
  // Update counts
  sections.forEach(section => {
    const count = section.children.length;
    const sectionName = section.dataset.sectionName;
    const separator = Array.from(document.querySelectorAll('.gallery-separator')).find(
      sep => sep.querySelector('span:nth-child(2)').textContent === sectionName
    );
    if (separator) {
      separator.querySelector('.gallery-separator-count').textContent = `${count} files`;
    }
  });
  
  const t1 = performance.now();
  log(`gallerySort: char=${lastSort} len=${allFiles.length} time=${Math.floor(t1 - t0)} sort=${lastSortName}`);
  el.status.innerText = `Sort | ${lastSortName} | ${allFiles.length.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
}

async function fetchFilesHT(evt) {
  const t0 = performance.now();
  el.status.innerText = `Folder | ${evt.target.name} | in-progress`;
  let numFiles = 0;

  const res = await fetch(`${window.api}/browser/files?folder=${encodeURI(evt.target.name)}`);
  if (!res || res.status !== 200) {
    el.status.innerText = `Folder | ${evt.target.name} | failed: ${res?.statusText}`;
    return;
  }
  const jsonData = await res.json();
  
  // Collect all files first
  const allFiles = [];
  for (const line of jsonData) {
    const data = decodeURI(line).split('##F##');
    numFiles++;
    const f = new GalleryFile(data[0], data[1]);
    allFiles.push(f);
  }
  
  // Sort files by name to ensure proper grouping
  allFiles.sort((a, b) => a.name.localeCompare(b.name));
  
  // Organize files by directory
  const filesByDir = new Map();
  for (const f of allFiles) {
    let dir = f.name.match(/(.*)[\/\\]/);
    if (!dir) dir = '';
    else dir = dir[1];
    
    if (!filesByDir.has(dir)) {
      filesByDir.set(dir, []);
    }
    filesByDir.get(dir).push(f);
  }

  // Now add organized files with separators
  addSeparatorStyles();
  el.files.innerHTML = '';
  
  // Handle files without directories first
  const rootFiles = filesByDir.get('');
  if (rootFiles && rootFiles.length > 0) {
    rootFiles.forEach(file => {
      el.files.appendChild(file);
      // Root files should always load immediately
      file.connectedCallback();
    });
  }
  
  // Sort directory names for consistent display
  const sortedDirs = Array.from(filesByDir.keys())
    .filter(dir => dir.length > 0)
    .sort((a, b) => a.localeCompare(b));
  
  for (const dir of sortedDirs) {
    const files = filesByDir.get(dir);
    
    // Create separator
    const sep = document.createElement('div');
    sep.className = 'gallery-separator collapsed'; // Start collapsed
    
    const icon = document.createElement('span');
    icon.className = 'gallery-separator-icon';
    icon.textContent = '▶';
    
    const text = document.createElement('span');
    text.className = 'gallery-separator-title';
    text.textContent = dir;
    
    const count = document.createElement('span');
    count.className = 'gallery-separator-count';
    count.textContent = `${files.length} files`;
    
    sep.appendChild(icon);
    sep.appendChild(text);
    sep.appendChild(count);
    sep.title = `Click to toggle ${dir}`;
    
    // Create section container
    const section = document.createElement('div');
    section.className = 'gallery-section collapsed'; // Start collapsed
    section.dataset.sectionName = dir;
    section.dataset.loaded = 'false';
    
    // Add files to section with skipLoad
    files.forEach(file => {
      file.dataset.skipLoad = 'true';
      section.appendChild(file);
    });
    
    // Check if section should be expanded
    const isExpanded = isSectionExpanded(dir);
    if (isExpanded) {
      sep.classList.remove('collapsed');
      section.classList.remove('collapsed');
      section.dataset.loaded = 'true';
      // Load images for expanded sections
      files.forEach(file => {
        delete file.dataset.skipLoad;
        file.connectedCallback();
      });
    }
    
    // Add click handler
    sep.addEventListener('click', async () => {
      const isCollapsed = sep.classList.toggle('collapsed');
      section.classList.toggle('collapsed');
      saveCollapsedState(dir, isCollapsed);
      
      // Load images when expanding for the first time
      if (!isCollapsed && section.dataset.loaded === 'false') {
        section.dataset.loaded = 'true';
        const countSpan = sep.querySelector('.gallery-separator-count');
        const originalText = countSpan.textContent;
        countSpan.textContent = 'Loading...';
        
        const images = section.querySelectorAll('gallery-file');
        let loaded = 0;
        for (const img of images) {
          delete img.dataset.skipLoad;
          await img.connectedCallback();
          loaded++;
          if (loaded % 5 === 0) {
            countSpan.textContent = `Loading... ${loaded}/${images.length}`;
          }
        }
        countSpan.textContent = originalText;
      }
    });
    
    el.files.appendChild(sep);
    el.files.appendChild(section);
  }

  const t1 = performance.now();
  log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
  el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
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
  
  // Collect files by directory
  const filesByDir = new Map();
  const pendingFiles = [];

  ws.onmessage = (event) => {
    numFiles++;
    t1 = performance.now();
    const data = decodeURI(event.data).split('##F##');
    if (data[0] === '#END#') {
      ws.close();
    } else {
      const file = new GalleryFile(data[0], data[1]);
      pendingFiles.push(file);
      
      if (numFiles % 100 === 0) {
        el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | in-progress | ${Math.floor(t1 - t0).toLocaleString()}ms`;
      }
    }
  };
  
  ws.onclose = (event) => {
    // Sort files by name to ensure proper grouping
    pendingFiles.sort((a, b) => a.name.localeCompare(b.name));
    
    // Organize all files by directory
    for (const file of pendingFiles) {
      let dir = file.name.match(/(.*)[\/\\]/);
      if (!dir) dir = '';
      else dir = dir[1];
      
      if (!filesByDir.has(dir)) {
        filesByDir.set(dir, []);
      }
      filesByDir.get(dir).push(file);
    }
    
    // Now add organized files with separators
    addSeparatorStyles();
    
    // Handle files without directories first
    const rootFiles = filesByDir.get('');
    if (rootFiles && rootFiles.length > 0) {
      rootFiles.forEach(file => {
        el.files.appendChild(file);
        // Root files should always load immediately
        file.connectedCallback();
      });
    }
    
    // Sort directory names for consistent display
    const sortedDirs = Array.from(filesByDir.keys())
      .filter(dir => dir.length > 0)
      .sort((a, b) => a.localeCompare(b));
    
    for (const dir of sortedDirs) {
      const files = filesByDir.get(dir);
      
      // Create separator
      const sep = document.createElement('div');
      sep.className = 'gallery-separator collapsed';
      
      const icon = document.createElement('span');
      icon.className = 'gallery-separator-icon';
      icon.textContent = '▶';
      
      const text = document.createElement('span');
      text.className = 'gallery-separator-title';
      text.textContent = dir;
      
      const count = document.createElement('span');
      count.className = 'gallery-separator-count';
      count.textContent = `${files.length} files`;
      
      sep.appendChild(icon);
      sep.appendChild(text);
      sep.appendChild(count);
      sep.title = `Click to toggle ${dir}`;
      
      // Create section container
      const section = document.createElement('div');
      section.className = 'gallery-section collapsed';
      section.dataset.sectionName = dir;
      section.dataset.loaded = 'false';
      
      // Add files to section with skipLoad
      files.forEach(file => {
        file.dataset.skipLoad = 'true';
        section.appendChild(file);
      });
      
      // Check if section should be expanded
      const isExpanded = isSectionExpanded(dir);
      if (isExpanded) {
        sep.classList.remove('collapsed');
        section.classList.remove('collapsed');
        section.dataset.loaded = 'true';
        // Load images for expanded sections
        files.forEach(file => {
          delete file.dataset.skipLoad;
          file.connectedCallback();
        });
      }
      
      // Add click handler
      sep.addEventListener('click', async () => {
        const isCollapsed = sep.classList.toggle('collapsed');
        section.classList.toggle('collapsed');
        saveCollapsedState(dir, isCollapsed);
        
        // Load images when expanding for the first time
        if (!isCollapsed && section.dataset.loaded === 'false') {
          section.dataset.loaded = 'true';
          const countSpan = sep.querySelector('.gallery-separator-count');
          const originalText = countSpan.textContent;
          countSpan.textContent = 'Loading...';
          
          const images = section.querySelectorAll('gallery-file');
          let loaded = 0;
          for (const img of images) {
            delete img.dataset.skipLoad;
            await img.connectedCallback();
            loaded++;
            if (loaded % 5 === 0) {
              countSpan.textContent = `Loading... ${loaded}/${images.length}`;
            }
          }
          countSpan.textContent = originalText;
        }
      });
      
      el.files.appendChild(sep);
      el.files.appendChild(section);
    }
    
    log(`gallery: folder=${evt.target.name} num=${numFiles} time=${Math.floor(t1 - t0)}ms`);
    el.status.innerText = `Folder | ${evt.target.name} | ${numFiles.toLocaleString()} images | ${Math.floor(t1 - t0).toLocaleString()}ms`;
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
}

// register on startup

customElements.define('gallery-folder', GalleryFolder);
customElements.define('gallery-file', GalleryFile);
