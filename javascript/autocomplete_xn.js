/*
 * Extra-networks completion for SD.Next prompt textareas.
 *
 * Companion to autocomplete.js: exposes sorted indices for LoRAs, embeddings, and wildcards,
 * each backed by an existing enumeration endpoint. Dispatch and insertion are driven from
 * autocomplete.js via the mode returned by getCurrentWord().
 *
 * This file relies on globals declared in autocomplete.js (lowerBound, log, engine).
 */

/* global lowerBound */

// -- Indices --

class XnIndex {
  constructor(items) {
    // items: [{ name, display }]. Sorted in-place by lowercase name.
    this.items = items.map(({ name, display }) => ({
      name: String(name).toLowerCase(),
      display: display ?? name,
    }));
    this.items.sort((a, b) => a.name.localeCompare(b.name));
  }

  search(prefix, limit = 20) {
    const query = String(prefix).toLowerCase();
    // Empty query returns the first `limit` items so `<lora:` or `__` alone shows a browsable list.
    if (!query) return this.items.slice(0, limit);
    const start = lowerBound(this.items, query);
    const matches = [];
    for (let i = start; i < this.items.length && matches.length < limit; i++) {
      if (!this.items[i].name.startsWith(query)) break;
      matches.push(this.items[i]);
    }
    // Substring fallback for 3+ char queries (extra-network names are usually short)
    if (matches.length === 0 && query.length >= 3) {
      for (let i = 0; i < this.items.length && matches.length < limit; i++) {
        if (this.items[i].name.includes(query)) matches.push(this.items[i]);
      }
    }
    return matches.slice(0, limit);
  }
}

// -- Engine --

const xnEngine = {
  lora: new XnIndex([]),
  embed: new XnIndex([]),
  wildcard: new XnIndex([]),

  async fetchJson(path) {
    try {
      const resp = await fetch(`${window.api}${path}`, { credentials: 'include' });
      if (!resp.ok) throw new Error(`${resp.status}`);
      return await resp.json();
    } catch (e) {
      log('autoComplete', { xnFetchFailed: path, error: String(e) });
      return null;
    }
  },

  async loadAll() {
    // LoRAs: [{name, alias, path, metadata}, ...]
    const loraData = await this.fetchJson('/loras');
    if (Array.isArray(loraData)) {
      const items = [];
      for (const lo of loraData) {
        if (lo?.name) items.push({ name: lo.name });
        if (lo?.alias && lo.alias !== lo.name) items.push({ name: lo.alias });
      }
      this.lora = new XnIndex(items);
    }
    // Embeddings: {loaded: [...], skipped: [...]}
    const embData = await this.fetchJson('/embeddings');
    if (embData && typeof embData === 'object') {
      const loaded = Array.isArray(embData.loaded) ? embData.loaded : [];
      this.embed = new XnIndex(loaded.map((name) => ({ name })));
    }
    // Wildcards: [{name}, ...]
    const wcData = await this.fetchJson('/wildcards');
    if (Array.isArray(wcData)) {
      this.wildcard = new XnIndex(wcData.filter((w) => w?.name).map((w) => ({ name: w.name })));
    }
    log('autoComplete', {
      xnLoaded: true,
      lora: this.lora.items.length,
      embed: this.embed.items.length,
      wildcard: this.wildcard.items.length,
    });
  },

  searchLoras(prefix, limit = 20) {
    return this.lora.search(prefix, limit).map((item) => ({ ...item, kind: 'lora' }));
  },

  searchEmbeddings(prefix, limit = 20) {
    return this.embed.search(prefix, limit).map((item) => ({ ...item, kind: 'embed' }));
  },

  searchWildcards(prefix, limit = 20) {
    return this.wildcard.search(prefix, limit).map((item) => ({ ...item, kind: 'wildcard' }));
  },
};

// Expose globally so autocomplete.js can dispatch to it.
window.autocompleteXn = xnEngine;
