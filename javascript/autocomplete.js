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

// Glyph + color per result kind. Renders in place of the category dot for non-tag results.
const KIND_GLYPHS = {
  tag: { glyph: '●', color: null }, // color pulled from tag category
  lora: { glyph: '◆', color: '#8a66ff' },
  embed: { glyph: '▲', color: '#1abc9c' },
  wildcard: { glyph: '★', color: '#f1c40f' },
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
    // Tuples are [name, catId, count] or [name, catId, count, aliases]. Default `aliases = []`
    // keeps legacy 3-tuple dictionaries working unchanged.
    this.tags = data.tags.map(([name, category, count, aliases = []]) => ({
      name: name.toLowerCase(),
      display: name,
      category,
      count,
      aliases,
    }));
    this.tags.sort((a, b) => a.name.localeCompare(b.name));
    // Alias index parallel to this.tags. Each entry has .name so lowerBound works on both.
    this.aliasEntries = [];
    for (const tag of this.tags) {
      if (!tag.aliases || tag.aliases.length === 0) continue;
      for (const alias of tag.aliases) {
        this.aliasEntries.push({ name: alias.toLowerCase(), display: alias, tag });
      }
    }
    this.aliasEntries.sort((a, b) => a.name.localeCompare(b.name));
    // Optional translations companion: foreign_term -> canonical_tag_name.
    // tagByName is keyed on canonical lowercased name for O(1) resolution from a translation hit.
    this.translations = new Map();
    this.tagByName = new Map(this.tags.map((t) => [t.name, t]));
    if (data.translations && typeof data.translations === 'object') {
      for (const [foreign, canonical] of Object.entries(data.translations)) {
        if (typeof foreign !== 'string' || typeof canonical !== 'string') continue;
        this.translations.set(foreign.toLowerCase(), { canonical: canonical.toLowerCase(), foreign });
      }
    }
    // Sorted translation keys for prefix+substring scan via lowerBound.
    this.translationEntries = [...this.translations.entries()]
      .map(([foreignLower, { canonical, foreign }]) => ({ name: foreignLower, foreign, canonical }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }

  /** Prefix search with binary search across canonical names and aliases. Returns matches sorted by count descending. */
  search(prefix, limit = 20) {
    const query = prefix.toLowerCase().replace(/ /g, '_');
    if (!query) return [];
    // Canonical prefix matches
    const matches = [];
    const start = lowerBound(this.tags, query);
    for (let i = start; i < this.tags.length && matches.length < limit * 5; i++) {
      if (!this.tags[i].name.startsWith(query)) break;
      matches.push(this.tags[i]);
    }
    // Alias prefix matches. Annotate so render can show "canonical (alias)".
    const aliasStart = lowerBound(this.aliasEntries, query);
    for (let i = aliasStart; i < this.aliasEntries.length && matches.length < limit * 10; i++) {
      const entry = this.aliasEntries[i];
      if (!entry.name.startsWith(query)) break;
      matches.push({ ...entry.tag, matchedVia: 'alias', matchedAlias: entry.display });
    }
    // Substring fallback (canonical + aliases) for 4+ char queries when prefix matching returned nothing.
    if (matches.length === 0 && query.length >= 4) {
      for (let i = 0; i < this.tags.length && matches.length < limit * 5; i++) {
        if (this.tags[i].name.includes(query)) matches.push(this.tags[i]);
      }
      for (let i = 0; i < this.aliasEntries.length && matches.length < limit * 10; i++) {
        const entry = this.aliasEntries[i];
        if (entry.name.includes(query)) matches.push({ ...entry.tag, matchedVia: 'alias', matchedAlias: entry.display });
      }
    }
    // Translation lookup. Prefix scan over foreign terms, resolving to canonical tags when present.
    if (this.translationEntries.length > 0) {
      const tStart = lowerBound(this.translationEntries, query);
      for (let i = tStart; i < this.translationEntries.length && matches.length < limit * 10; i++) {
        const entry = this.translationEntries[i];
        if (!entry.name.startsWith(query)) break;
        const canonicalTag = this.tagByName.get(entry.canonical);
        if (canonicalTag) matches.push({ ...canonicalTag, matchedVia: 'translation', matchedTerm: entry.foreign });
      }
      // Substring fallback over translation keys (CJK/short foreign terms benefit from 2-char threshold)
      if (query.length >= 2) {
        for (let i = 0; i < this.translationEntries.length && matches.length < limit * 10; i++) {
          const entry = this.translationEntries[i];
          if (entry.name.includes(query) && !entry.name.startsWith(query)) {
            const canonicalTag = this.tagByName.get(entry.canonical);
            if (canonicalTag) matches.push({ ...canonicalTag, matchedVia: 'translation', matchedTerm: entry.foreign });
          }
        }
      }
    }
    // Dedupe by canonical name; prefer canonical (no matchedVia) over alias/translation matches.
    const seen = new Map();
    for (const tag of matches) {
      const existing = seen.get(tag.name);
      if (!existing || (existing.matchedVia && !tag.matchedVia)) seen.set(tag.name, tag);
    }
    const result = [...seen.values()];
    result.sort((a, b) => b.count - a.count);
    return result.slice(0, limit);
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

/**
 * Extract the current completion context at the cursor position.
 *
 * Returns { word, start, end, mode } where:
 *   mode === 'tag':      ordinary tag completion
 *   mode === 'lora':     inside an unclosed `<lora:...` span; `start` points at the `<`
 *   mode === 'wildcard': inside an unclosed `__...` span; `start` points at the first `_`
 *
 * `start..end` is the replacement range the appropriate insert function should overwrite.
 * Embeddings are served under `mode === 'tag'` and merged into tag-mode results by the engine.
 */
function getCurrentWord(textarea) {
  const { value, selectionStart } = textarea;
  if (selectionStart !== textarea.selectionEnd) return null; // has selection
  // Scan backward from cursor to the nearest hard separator
  let wordStart = selectionStart;
  while (wordStart > 0) {
    const ch = value[wordStart - 1];
    if (ch === ',' || ch === '\n') break;
    wordStart--;
  }
  // Skip leading whitespace between the separator and the typed word
  while (wordStart < selectionStart && value[wordStart] === ' ') wordStart++;
  const segment = value.slice(wordStart, selectionStart);
  // LoRA / extra-network trigger: unclosed `<` with `kind:` prefix
  const before = value.slice(0, selectionStart);
  const lastOpen = before.lastIndexOf('<');
  const lastClose = before.lastIndexOf('>');
  if (lastOpen > lastClose && lastOpen >= wordStart) {
    const inside = before.slice(lastOpen + 1); // e.g. "lora:foo" or "lora:" or "lor"
    const colon = inside.indexOf(':');
    // Require `<lora:`; before the colon the kind is ambiguous (could be lora/embed/hypernet).
    if (colon >= 0 && inside.slice(0, colon).toLowerCase() === 'lora') {
      return { word: inside.slice(colon + 1), start: lastOpen, end: selectionStart, mode: 'lora' };
    }
    // Inside `<...` but not yet a recognized kind, suppress completion.
    return null;
  }
  // Wildcard trigger: unclosed `__` that doesn't close within the current word
  if (segment.startsWith('__') && !segment.slice(2).includes('__')) {
    return { word: segment.slice(2), start: wordStart, end: selectionStart, mode: 'wildcard' };
  }
  // Ordinary tag
  if (!segment) return null;
  return { word: segment, start: wordStart, end: selectionStart, mode: 'tag' };
}

/** Escape bare parens so tag names like `fate_(series)` aren't parsed as attention syntax. */
function escapeParensForPrompt(name) {
  return name.replace(/([()])/g, '\\$1');
}

/**
 * Insert an extra-network reference at the current trigger position.
 *   kind === 'lora':     inserts `<lora:name:1.0>` over the range including the leading `<`
 *   kind === 'wildcard': inserts `__name__` over the range including the leading `__`
 * Embeddings use insertTag directly so they go through comma-separator and paren-escape logic.
 */
function insertExtraNetwork(textarea, item, kind) {
  const info = getCurrentWord(textarea);
  if (!info || info.mode !== kind) return;
  const { value } = textarea;
  const before = value.slice(0, info.start);
  const after = value.slice(info.end);
  let insertion;
  if (kind === 'lora') {
    insertion = `<lora:${item.display ?? item.name}:1.0>`;
  } else if (kind === 'wildcard') {
    insertion = `__${item.display ?? item.name}__`;
  } else {
    return;
  }
  textarea.value = before + insertion + after;
  const cursorPos = before.length + insertion.length;
  textarea.selectionStart = cursorPos;
  textarea.selectionEnd = cursorPos;
  if (typeof updateInput === 'function') updateInput(textarea);
}

/** Insert a tag at the current word position, replacing the typed prefix. */
function insertTag(textarea, tagName) {
  const info = getCurrentWord(textarea);
  if (!info || info.mode !== 'tag') return;
  const { value } = textarea;
  const before = value.slice(0, info.start);
  const after = value.slice(info.end);
  // Build insertion: tag + separator. Parens in tag names are escaped so the prompt parser doesn't read them as attention syntax.
  const useComma = window.opts?.autocomplete_append_comma ?? true;
  const sep = useComma ? ',' : '';
  const needsSepBefore = before.length > 0 && before.trimEnd().length > 0 && !before.trimEnd().endsWith(',');
  const prefix = needsSepBefore ? `${sep} ` : '';
  let suffix = `${sep} `;
  if (after.length > 0 && after.trimStart().startsWith(',')) suffix = ' ';
  const insertion = `${prefix}${escapeParensForPrompt(tagName)}${suffix}`;
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
    // Switching textareas: clear prior state so a stale render can't leak across.
    if (this.textarea && this.textarea !== textarea) this.hide();
    if (this.textarea !== textarea) this.resizeObserver.observe(textarea);
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
      const kind = tag.kind || 'tag';
      const kindStyle = KIND_GLYPHS[kind] || KIND_GLYPHS.tag;
      dot.style.color = kindStyle.color || engine.categoryColors[tag.category] || '#888';
      dot.textContent = kindStyle.glyph;
      dot.title = kind === 'tag' ? (engine.categoryNames[tag.category] || '') : kind;
      const name = document.createElement('span');
      name.className = 'autocomplete-tag';
      const tagText = replaceUnderscores ? tag.display.replace(/_/g, ' ') : tag.display;
      const canonicalMatch = tag.name.indexOf(queryNorm);
      if (canonicalMatch >= 0 && queryNorm.length > 0) {
        const mark = document.createElement('mark');
        mark.textContent = tagText.slice(canonicalMatch, canonicalMatch + queryNorm.length);
        name.append(
          document.createTextNode(tagText.slice(0, canonicalMatch)),
          mark,
          document.createTextNode(tagText.slice(canonicalMatch + queryNorm.length)),
        );
      } else {
        name.textContent = tagText;
      }
      // Alias/translation-matched rows append " (foreign)" with the query fragment highlighted.
      let annotationTerm = null;
      if (tag.matchedVia === 'alias') annotationTerm = tag.matchedAlias;
      else if (tag.matchedVia === 'translation') annotationTerm = tag.matchedTerm;
      if (annotationTerm) {
        const annotationDisplay = replaceUnderscores ? annotationTerm.replace(/_/g, ' ') : annotationTerm;
        const annotationLower = annotationTerm.toLowerCase();
        const annotationMatch = annotationLower.indexOf(queryNorm);
        const prefix = tag.matchedVia === 'translation' ? ' \u{1F310} ' : ' (';
        const suffix = tag.matchedVia === 'translation' ? '' : ')';
        name.appendChild(document.createTextNode(prefix));
        if (annotationMatch >= 0 && queryNorm.length > 0) {
          const mark = document.createElement('mark');
          mark.textContent = annotationDisplay.slice(annotationMatch, annotationMatch + queryNorm.length);
          name.append(
            document.createTextNode(annotationDisplay.slice(0, annotationMatch)),
            mark,
            document.createTextNode(annotationDisplay.slice(annotationMatch + queryNorm.length)),
          );
        } else {
          name.appendChild(document.createTextNode(annotationDisplay));
        }
        if (suffix) name.appendChild(document.createTextNode(suffix));
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
    const result = this.results[this.selectedIndex];
    if (this.textarea) {
      if (result.kind === 'lora' || result.kind === 'wildcard') {
        insertExtraNetwork(this.textarea, result, result.kind);
      } else {
        // 'embed' kind and untagged tag results both go through insertTag (comma-aware, paren-escaped).
        insertTag(this.textarea, result.display ?? result.name);
      }
    }
    this.hide();
  },
};

// -- Event handlers --

let debounceTimer = null;

function onInput(textarea) {
  if (!active) return;
  // IME candidate window open: value isn't committed, and Enter would race with tag accept.
  if (textarea.dataset.imeActive === '1') return;
  const minChars = window.opts?.autocomplete_min_chars ?? 3;
  const info = getCurrentWord(textarea);
  if (!info) {
    dropdown.hide();
    return;
  }
  // Extra-network triggers have a zero threshold so `<lora:` alone surfaces results.
  const threshold = info.mode === 'tag' ? minChars : 0;
  if (info.word.length < threshold) {
    dropdown.hide();
    return;
  }
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    let results;
    if (info.mode === 'lora') {
      results = window.autocompleteXn ? window.autocompleteXn.searchLoras(info.word) : [];
    } else if (info.mode === 'wildcard') {
      results = window.autocompleteXn ? window.autocompleteXn.searchWildcards(info.word) : [];
    } else {
      const tagResults = engine.searchAll(info.word);
      const embedResults = window.autocompleteXn ? window.autocompleteXn.searchEmbeddings(info.word) : [];
      // Embeddings fold into tag-mode results (a1111 tagcomplete parity).
      results = [...embedResults, ...tagResults];
    }
    dropdown.show(results, textarea, info.word);
  }, 150);
}

function onKeyDown(e) {
  if (!dropdown.visible) return;
  if (e.isComposing) return; // IME candidate selection, let the browser commit the candidate
  // Modifier + nav/accept keys belong to other handlers (editAttention.js on Ctrl+Arrow,
  // generate hotkey on Ctrl+Enter). Let them through even with the dropdown open.
  const hasModifier = e.ctrlKey || e.metaKey || e.altKey;
  switch (e.key) {
    case 'ArrowDown':
      if (hasModifier) return;
      e.preventDefault();
      e.stopPropagation();
      dropdown.navigate(1);
      break;
    case 'ArrowUp':
      if (hasModifier) return;
      e.preventDefault();
      e.stopPropagation();
      dropdown.navigate(-1);
      break;
    case 'Enter':
      if (hasModifier) return;
      if (dropdown.selectedIndex >= 0) {
        e.preventDefault();
        e.stopPropagation();
        dropdown.accept();
      }
      break;
    case 'Tab':
      if (hasModifier) return;
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
  textarea.addEventListener('compositionstart', () => { textarea.dataset.imeActive = '1'; });
  textarea.addEventListener('compositionend', () => { delete textarea.dataset.imeActive; });
  textarea.addEventListener('focusin', () => {
    if (dropdown.visible && dropdown.textarea && dropdown.textarea !== textarea) dropdown.hide();
  });
  textarea.addEventListener('focusout', () => {
    // Cancel any in-flight debounced dropdown.show; otherwise it fires against a stale textarea.
    clearTimeout(debounceTimer);
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
let bridgeWarnedMissingDescriptor = false;
function patchConfigBridge() {
  const proto = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
  if (!proto?.get || !proto?.set) {
    if (!bridgeWarnedMissingDescriptor) {
      log('autoComplete', { bridge: 'skipped', reason: 'HTMLTextAreaElement.prototype.value descriptor missing' });
      bridgeWarnedMissingDescriptor = true;
    }
    return;
  }
  const elements = gradioApp().querySelectorAll('[id$="_tag_autocomplete_config_json"]');
  for (const el of elements) {
    const textarea = el.querySelector('textarea');
    if (!textarea || textarea.acBridgePatched) continue;
    textarea.acBridgePatched = true;
    Object.defineProperty(textarea, 'value', {
      set(newValue) {
        const oldValue = proto.get.call(textarea);
        proto.set.call(textarea, newValue);
        if (oldValue !== newValue && newValue) {
          try {
            const cfg = JSON.parse(newValue);
            for (const [key, val] of Object.entries(cfg)) window.opts[key] = val;
            executeCallbacks(optionsChangedCallbacks);
          } catch { /* ignore parse errors; the bridge is best-effort */ }
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
  if (window.autocompleteXn) window.autocompleteXn.loadAll();
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
    if (window.autocompleteXn) window.autocompleteXn.loadAll();
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
