/*
 * Tag autocomplete for SD.Next prompt textareas.
 *
 * Ported from Enso's CodeMirror-based autocomplete (autocomplete.ts).
 * Uses binary search on sorted tag arrays for O(log n) prefix lookup,
 * with substring fallback for 4+ char queries.
 */

// -- Category colors (unified 14-category scheme) --

const CATEGORY_COLORS = {
  0: '#0075f8', // general
  1: '#cc0000', // artist
  2: '#ff4500', // studio
  3: '#9900ff', // copyright
  4: '#00ab2c', // character
  5: '#ed5d1f', // species
  6: '#8a66ff', // genre
  7: '#00cccc', // medium
  8: '#6b7280', // meta
  9: '#228b22', // lore
  10: '#e67e22', // lens
  11: '#f1c40f', // lighting
  12: '#1abc9c', // composition
  13: '#e84393', // color
};

const CATEGORY_NAMES = {
  0: 'general',
  1: 'artist',
  2: 'studio',
  3: 'copyright',
  4: 'character',
  5: 'species',
  6: 'genre',
  7: 'medium',
  8: 'meta',
  9: 'lore',
  10: 'lens',
  11: 'lighting',
  12: 'composition',
  13: 'color',
};

let active = false;

// -- Utilities (ported from Enso) --

/** Binary search for the first tag where tag.name >= prefix. */
function lowerBound(tags, prefix) {
  let lo = 0;
  let hi = tags.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (tags[mid].name < prefix) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Format post count as abbreviated string. */
function formatCount(count) {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${Math.round(count / 1_000)}k`;
  return String(count);
}

/**
 * Estimate viewport Y of the bottom of the caret line using a persistent
 * offscreen mirror div. Styles and width are re-read from the textarea on
 * every call so resized textareas are handled correctly.
 */
let caretMirror = null;
let caretMarker = null;
const MIRROR_PROPS = ['fontFamily', 'fontSize', 'fontWeight', 'fontStyle',
  'lineHeight', 'letterSpacing', 'wordSpacing', 'textTransform',
  'padding', 'border', 'boxSizing'];

function caretViewportY(textarea) {
  if (!caretMirror) {
    caretMirror = document.createElement('div');
    caretMirror.className = 'autocomplete-mirror';
    caretMirror.style.whiteSpace = 'pre-wrap';
    caretMirror.style.wordWrap = 'break-word';
    caretMirror.style.position = 'absolute';
    caretMirror.style.left = '-9999px';
    caretMirror.style.overflow = 'hidden';
    caretMarker = document.createElement('span');
    caretMarker.textContent = '\u200b';
    document.body.appendChild(caretMirror);
  }
  const cs = getComputedStyle(textarea);
  for (const p of MIRROR_PROPS) caretMirror.style[p] = cs[p];
  caretMirror.style.width = `${textarea.offsetWidth}px`;
  caretMirror.textContent = textarea.value.substring(0, textarea.selectionStart);
  caretMirror.appendChild(caretMarker);
  const offset = caretMarker.offsetTop + caretMarker.offsetHeight;
  return textarea.getBoundingClientRect().top + offset - textarea.scrollTop;
}

// -- TagIndex --

class TagIndex {
  constructor(data) {
    this.categories = data.categories || {};
    // Build sorted array of {name, category, count} from raw [name, catId, count] tuples
    this.tags = data.tags.map(([name, category, count]) => ({
      name: name.toLowerCase(),
      display: name,
      category,
      count,
    }));
    this.tags.sort((a, b) => a.name.localeCompare(b.name));
  }

  /** Prefix search with binary search. Returns matches sorted by count descending. */
  search(prefix, limit = 20) {
    const query = prefix.toLowerCase().replace(/ /g, '_');
    if (!query) return [];
    const start = lowerBound(this.tags, query);
    const matches = [];
    for (let i = start; i < this.tags.length && matches.length < limit * 5; i++) {
      if (!this.tags[i].name.startsWith(query)) break;
      matches.push(this.tags[i]);
    }
    // Substring fallback for 4+ chars if prefix found nothing
    if (matches.length === 0 && query.length >= 4) {
      for (let i = 0; i < this.tags.length && matches.length < limit * 5; i++) {
        if (this.tags[i].name.includes(query)) matches.push(this.tags[i]);
      }
    }
    matches.sort((a, b) => b.count - a.count);
    return matches.slice(0, limit);
  }
}

// -- Engine --

const engine = {
  indices: new Map(), // name -> TagIndex
  categoryColors: { ...CATEGORY_COLORS },
  categoryNames: { ...CATEGORY_NAMES },

  async loadEnabled() {
    const enabled = window.opts?.autocomplete_enabled || [];
    active = window.opts?.autocomplete_active || false;
    if (!active) {
      this.indices.clear();
      return;
    }
    const t0 = performance.now();
    const toLoad = enabled.filter((n) => !this.indices.has(n));
    const toRemove = [...this.indices.keys()].filter((n) => !enabled.includes(n));
    toRemove.forEach((n) => this.indices.delete(n));
    await Promise.all(toLoad.map(async (name) => {
      try {
        const resp = await fetch(`${window.api}/autocomplete/${name}`, { credentials: 'include' });
        if (!resp.ok) throw new Error(`${resp.status}`);
        const data = await resp.json();
        this.indices.set(name, new TagIndex(data));
        // Extract category colors from first loaded file
        if (data.categories) {
          Object.entries(data.categories).forEach(([id, cat]) => {
            if (cat.color) this.categoryColors[id] = cat.color;
            if (cat.name) this.categoryNames[id] = cat.name;
          });
        }
        const t1 = performance.now();
        log('autoComplete', { loaded: name, tags: data.tags?.length || 0, time: Math.round(t1 - t0) });
        timer(`autocompleteLoad:${name}`, t1 - t0);
      } catch (e) {
        log('autoComplete', { failed: name, error: e });
      }
    }));
  },

  searchAll(prefix, limit = 20) {
    if (this.indices.size === 0) return [];
    const all = [];
    this.indices.forEach((index) => {
      all.push(...index.search(prefix, limit));
    });
    // Deduplicate by name, keeping highest count
    const seen = new Map();
    all.forEach((tag) => {
      const existing = seen.get(tag.name);
      if (!existing || tag.count > existing.count) seen.set(tag.name, tag);
    });
    const results = [...seen.values()];
    results.sort((a, b) => b.count - a.count);
    return results.slice(0, limit);
  },
};

// -- Textarea integration --

/** Extract the current word being typed at the cursor position. */
function getCurrentWord(textarea) {
  const { value, selectionStart } = textarea;
  if (selectionStart !== textarea.selectionEnd) return null; // has selection
  // Scan backward from cursor to find word start
  let start = selectionStart;
  while (start > 0) {
    const ch = value[start - 1];
    if (ch === ',' || ch === '\n') break;
    start--;
  }
  // Skip leading whitespace
  while (start < selectionStart && value[start] === ' ') start++;
  const word = value.slice(start, selectionStart);
  if (!word) return null;
  // Skip if inside angle brackets (LoRA/embedding syntax)
  const before = value.slice(0, selectionStart);
  const lastOpen = before.lastIndexOf('<');
  const lastClose = before.lastIndexOf('>');
  if (lastOpen > lastClose) return null;
  // Skip if inside wildcard syntax
  const wcBefore = before.slice(start);
  if (wcBefore.startsWith('__') && !wcBefore.endsWith('__')) return null;
  return { word, start, end: selectionStart };
}

/** Insert a tag at the current word position, replacing the typed prefix. */
function insertTag(textarea, tagName) {
  const info = getCurrentWord(textarea);
  if (!info) return;
  const { value } = textarea;
  const before = value.slice(0, info.start);
  const after = value.slice(info.end);
  // Build insertion: tag + separator
  const useComma = window.opts?.autocomplete_append_comma ?? true;
  const sep = useComma ? ',' : '';
  const needsSepBefore = before.length > 0 && before.trimEnd().length > 0 && !before.trimEnd().endsWith(',');
  const prefix = needsSepBefore ? `${sep} ` : '';
  let suffix = `${sep} `;
  if (after.length > 0 && after.trimStart().startsWith(',')) suffix = ' ';
  const insertion = `${prefix}${tagName}${suffix}`;
  textarea.value = before.trimEnd() + (before.trimEnd().length > 0 ? ' ' : '') + insertion + after.trimStart();
  // Position cursor after the inserted tag + separator
  const cursorPos = before.trimEnd().length + (before.trimEnd().length > 0 ? 1 : 0) + insertion.length;
  textarea.selectionStart = cursorPos;
  textarea.selectionEnd = cursorPos;
  // Sync with Gradio
  if (typeof updateInput === 'function') updateInput(textarea);
}

// -- Dropdown --

const dropdown = {
  el: null,
  listEl: null,
  selectedIndex: -1,
  results: [],
  textarea: null,
  query: '',
  visible: false,

  init() {
    this.el = document.createElement('div');
    this.el.className = 'autocompleteResults';
    this.el.style.display = 'none';
    this.listEl = document.createElement('ul');
    this.listEl.className = 'autocompleteResultsList';
    this.el.appendChild(this.listEl);
    document.body.appendChild(this.el);
    this.el.addEventListener('mousedown', (e) => e.preventDefault()); // prevent blur on click
    this.el.addEventListener('click', (e) => {
      const li = e.target.closest('li');
      if (!li) return;
      const idx = [...this.listEl.children].indexOf(li);
      if (idx >= 0 && idx < this.results.length) {
        this.selectedIndex = idx;
        this.accept();
      }
    });
    this.resizeObserver = new ResizeObserver(() => {
      if (this.visible) this.position();
    });
  },

  show(results, textarea, query) {
    if (results.length === 0) { this.hide(); return; }
    if (this.textarea !== textarea) {
      if (this.textarea) this.resizeObserver.unobserve(this.textarea);
      this.resizeObserver.observe(textarea);
    }
    this.results = results;
    this.textarea = textarea;
    this.query = query || '';
    this.selectedIndex = -1;
    this.render();
    this.position();
    this.el.style.display = '';
    this.visible = true;
  },

  hide() {
    if (this.textarea) this.resizeObserver.unobserve(this.textarea);
    this.textarea = null;
    this.el.style.display = 'none';
    this.visible = false;
    this.results = [];
    this.selectedIndex = -1;
  },

  render() {
    const replaceUnderscores = window.opts?.autocomplete_replace_underscores ?? true;
    const queryNorm = this.query.toLowerCase().replace(/ /g, '_');
    this.listEl.replaceChildren();
    this.results.forEach((tag, i) => {
      const li = document.createElement('li');
      if (i === this.selectedIndex) li.classList.add('selected');
      const dot = document.createElement('span');
      dot.className = 'autocomplete-category';
      dot.style.color = engine.categoryColors[tag.category] || '#888';
      dot.textContent = '\u25CF';
      dot.title = engine.categoryNames[tag.category] || '';
      const name = document.createElement('span');
      name.className = 'autocomplete-tag';
      const tagText = replaceUnderscores ? tag.display.replace(/_/g, ' ') : tag.display;
      const matchPos = tag.name.indexOf(queryNorm);
      if (matchPos >= 0 && queryNorm.length > 0) {
        const mark = document.createElement('mark');
        mark.textContent = tagText.slice(matchPos, matchPos + queryNorm.length);
        name.append(
          document.createTextNode(tagText.slice(0, matchPos)),
          mark,
          document.createTextNode(tagText.slice(matchPos + queryNorm.length)),
        );
      } else {
        name.textContent = tagText;
      }
      const count = document.createElement('span');
      count.className = 'autocomplete-count';
      count.textContent = tag.count > 0 ? formatCount(tag.count) : '';
      li.append(dot, name, count);
      li.addEventListener('mouseenter', () => {
        this.selectedIndex = i;
        this.updateSelection();
      });
      this.listEl.appendChild(li);
    });
  },

  position() {
    if (!this.textarea) return;
    const rect = this.textarea.getBoundingClientRect();
    // Position near the caret line instead of the textarea bottom
    const cursorBottom = caretViewportY(this.textarea);
    const anchorY = Math.max(rect.top, Math.min(cursorBottom, rect.bottom));
    const spaceBelow = window.innerHeight - anchorY;
    const dropHeight = Math.min(this.el.scrollHeight, 300);
    if (spaceBelow >= dropHeight || spaceBelow >= anchorY - rect.top) {
      this.el.style.top = `${anchorY + 2}px`;
    } else {
      this.el.style.top = `${anchorY - dropHeight - 2}px`;
    }
    this.el.style.left = `${rect.left}px`;
    this.el.style.width = `${rect.width}px`;
  },

  updateSelection() {
    [...this.listEl.children].forEach((li, i) => {
      li.classList.toggle('selected', i === this.selectedIndex);
    });
    const selected = this.listEl.children[this.selectedIndex];
    if (selected) selected.scrollIntoView({ block: 'nearest' });
  },

  navigate(dir) {
    if (this.results.length === 0) return;
    if (this.selectedIndex === -1) {
      this.selectedIndex = dir > 0 ? 0 : this.results.length - 1;
    } else {
      this.selectedIndex = (this.selectedIndex + dir + this.results.length) % this.results.length;
    }
    this.updateSelection();
  },

  accept() {
    if (this.selectedIndex < 0 || this.selectedIndex >= this.results.length) {
      // Tab with no selection: select first
      if (this.results.length > 0) {
        this.selectedIndex = 0;
        this.updateSelection();
      }
      return;
    }
    const tag = this.results[this.selectedIndex];
    if (this.textarea) insertTag(this.textarea, tag.display);
    this.hide();
  },
};

// -- Event handlers --

let debounceTimer = null;

function onInput(textarea) {
  if (!active) return;
  const minChars = window.opts?.autocomplete_min_chars ?? 3;
  const info = getCurrentWord(textarea);
  if (!info || info.word.length < minChars) {
    dropdown.hide();
    return;
  }
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    const results = engine.searchAll(info.word);
    dropdown.show(results, textarea, info.word);
  }, 150);
}

function onKeyDown(e) {
  if (!dropdown.visible) return;
  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault();
      e.stopPropagation();
      dropdown.navigate(1);
      break;
    case 'ArrowUp':
      e.preventDefault();
      e.stopPropagation();
      dropdown.navigate(-1);
      break;
    case 'Enter':
      if (dropdown.selectedIndex >= 0) {
        e.preventDefault();
        e.stopPropagation();
        dropdown.accept();
      }
      break;
    case 'Tab':
      e.preventDefault();
      e.stopPropagation();
      dropdown.accept();
      break;
    case 'Escape':
      e.preventDefault();
      e.stopPropagation();
      dropdown.hide();
      break;
    default:
      break;
  }
}

/** Attach autocomplete to a single textarea. */
function attachAutocomplete(textarea) {
  textarea.addEventListener('input', () => onInput(textarea));
  textarea.addEventListener('keydown', onKeyDown);
  textarea.addEventListener('focusout', () => {
    setTimeout(() => dropdown.hide(), 200);
  });
}

// -- Prompt textarea IDs --

const PROMPT_IDS = [
  'txt2img_prompt', 'txt2img_neg_prompt',
  'img2img_prompt', 'img2img_neg_prompt',
  'control_prompt', 'control_neg_prompt',
  'video_prompt', 'video_neg_prompt',
];

// -- Active button --

function patchActiveButton() {
  const buttons = [...gradioApp().querySelectorAll('.autocomplete-active')];
  active = window.opts?.autocomplete_active || false;
  buttons.forEach((btn) => {
    btn.classList.toggle('autocomplete-active', active);
    btn.classList.toggle('autocomplete-inactive', !active);
    btn.parentElement.onclick = () => {
      active = !active;
      window.opts.autocomplete_active = !active;
      btn.classList.toggle('autocomplete-active', active);
      btn.classList.toggle('autocomplete-inactive', !active);
    };
  });
}

// -- Config bridge --

/** Monkey-patch script config bridge textboxes to push autocomplete config changes to window.opts immediately. */
function patchConfigBridge() {
  const elements = gradioApp().querySelectorAll('[id$="_tag_autocomplete_config_json"]');
  for (const el of elements) {
    const textarea = el.querySelector('textarea');
    if (!textarea || textarea.acBridgePatched) continue;
    textarea.acBridgePatched = true;
    const proto = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
    Object.defineProperty(textarea, 'value', {
      set(newValue) {
        const oldValue = proto.get.call(textarea);
        proto.set.call(textarea, newValue);
        if (oldValue !== newValue && newValue) {
          try {
            const cfg = JSON.parse(newValue);
            for (const [key, val] of Object.entries(cfg)) window.opts[key] = val;
            executeCallbacks(optionsChangedCallbacks);
          } catch { /* ignore parse errors */ }
        }
      },
      get() { return proto.get.call(textarea); },
    });
  }
}

// -- Initialization --

async function initAutocomplete() {
  const t0 = performance.now();
  const enabled = window.opts?.autocomplete_enabled || [];
  active = window.opts?.autocomplete_active || false;
  log('autoComplete', { active, enabled });
  // Inject styles (CSS files in javascript/ are not auto-loaded)
  const style = document.createElement('style');
  style.textContent = [
    '.autocompleteResults { position: fixed; z-index: 9999; max-height: 300px; overflow-y: auto;',
    '  background: var(--sd-main-background-color, var(--background-fill-primary, #1f2937));',
    '  border: 1px solid var(--sd-input-border-color, var(--border-color-primary, #374151));',
    '  border-radius: var(--sd-border-radius, 6px); box-shadow: 0 4px 16px rgba(0,0,0,0.4);',
    '  font-size: 13px; scrollbar-width: thin; }',
    '.autocompleteResultsList { list-style: none; margin: 0; padding: 4px 0; }',
    '.autocompleteResultsList > li { display: flex; align-items: center; padding: 6px 12px; cursor: pointer;',
    '  gap: 8px; line-height: 1.4; transition: background 0.1s ease; border-bottom: 1px solid rgba(255,255,255,0.03); }',
    '.autocompleteResultsList > li:last-child { border-bottom: none; }',
    '.autocompleteResultsList > li:hover { background: var(--sd-panel-background-color, var(--input-background-fill-focus, #374151)); }',
    '.autocompleteResultsList > li.selected { background: var(--sd-main-accent-color, var(--button-primary-background-fill, #4b5563)); }',
    '.autocomplete-category { font-size: 10px; flex-shrink: 0; width: 10px; text-align: center; cursor: help; }',
    '.autocomplete-tag { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }',
    '.autocomplete-tag mark { background: transparent; color: inherit; font-weight: 700; }',
    '.autocomplete-count { font-size: 0.75em; opacity: 0.45; flex-shrink: 0; font-variant-numeric: tabular-nums;',
    '  background: rgba(255,255,255,0.06); padding: 1px 6px; border-radius: 8px; min-width: 28px; text-align: right; }',
  ].join('\n');
  document.head.appendChild(style);
  dropdown.init();
  await engine.loadEnabled();
  // Attach to all prompt textareas; even if no dictionaries loaded yet, they may be enabled later via script UI
  let attached = 0;
  PROMPT_IDS.forEach((id) => {
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (textarea) {
      attachAutocomplete(textarea);
      attached++;
    }
  });
  // Reload when settings change

  async function optionsChangedCallback() {
    const newActive = window.opts?.autocomplete_active || false;
    const newEnabled = window.opts?.autocomplete_enabled || [];
    const currentKeys = [...engine.indices.keys()].sort().join(',');
    const newKeys = [...newEnabled].sort().join(',');
    if ((currentKeys !== newKeys) || (active !== newActive)) {
      log('autoComplete', { reload: newEnabled });
      await engine.loadEnabled();
      active = newActive;
      patchActiveButton();
    }
  }
  onOptionsChanged(optionsChangedCallback);
  // Watch for config updates from the script UI bridge
  patchConfigBridge();
  patchActiveButton();
  onAfterUiUpdate(patchConfigBridge);
  const t1 = performance.now();
  log('autoComplete', { attached, dicts: engine.indices.size, time: Math.round(t1 - t0) });
  timer('autocompleteInit', t1 - t0);
}
