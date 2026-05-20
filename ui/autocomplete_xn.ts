import { log } from './logger';

/*
 * Extra-networks completion for SD.Next prompt textareas.
 *
 * Companion to autocomplete.js: exposes sorted indices for LoRAs, embeddings, and wildcards,
 * each backed by an existing enumeration endpoint. Dispatch and insertion are driven from
 * autocomplete.js via the mode returned by getCurrentWord().
 */

interface XnItem {
  name: string;
  display?: string;
}

interface SearchItem {
  name: string;
  display: string;
}

type SearchResult = SearchItem & { kind: 'lora' | 'embed' | 'wildcard' };

/** Binary search for the first item where item.name >= query. */
export function lowerBound(items: { name: string }[], query: string): number {
  let lo = 0;
  let hi = items.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (items[mid].name < query) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

interface XnEngine {
  lora: XnIndex;
  embed: XnIndex;
  wildcard: XnIndex;
  fetchJson(path: string): Promise<unknown>;
  loadAll(): Promise<void>;
  searchLoras(prefix: string, limit?: number): SearchResult[];
  searchEmbeddings(prefix: string, limit?: number): SearchResult[];
  searchWildcards(prefix: string, limit?: number): SearchResult[];
}

// -- Indices --

class XnIndex {
  items: SearchItem[];

  constructor(items: XnItem[]) {
    // items: [{ name, display }]. Sorted in-place by lowercase name.
    this.items = items.map(({ name, display }) => ({
      name: String(name).toLowerCase(),
      display: display ?? name,
    }));
    this.items.sort((a, b) => a.name.localeCompare(b.name));
  }

  search(prefix: string, limit = 20): SearchItem[] {
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

export const xnEngine: XnEngine = {
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
      const items: XnItem[] = [];
      for (const lo of loraData) {
        if (typeof lo === 'object' && lo && 'name' in lo && typeof lo.name === 'string') items.push({ name: lo.name });
        if (typeof lo === 'object' && lo && 'alias' in lo && typeof lo.alias === 'string' && lo.alias !== (lo as { name?: string }).name) items.push({ name: lo.alias });
      }
      this.lora = new XnIndex(items);
    }
    // Embeddings: {loaded: [...], skipped: [...]}
    const embData = await this.fetchJson('/embeddings') as Record<string, unknown> | null;
    if (embData && typeof embData === 'object') {
      const loaded = Array.isArray(embData.loaded) ? embData.loaded : [];
      this.embed = new XnIndex(loaded.map((name) => ({ name: String(name) })));
    }
    // Wildcards: [{name}, ...]
    const wcData = await this.fetchJson('/wildcards');
    if (Array.isArray(wcData)) {
      this.wildcard = new XnIndex(
        wcData
          .filter((w) => typeof w === 'object' && w && 'name' in w && typeof w.name === 'string')
          .map((w) => ({ name: w.name })),
      );
    }
    log('autoComplete', {
      xnLoaded: true,
      lora: this.lora.items.length,
      embed: this.embed.items.length,
      wildcard: this.wildcard.items.length,
    });
  },

  searchLoras(prefix, limit = 20) {
    return this.lora.search(prefix, limit).map((item) => ({ ...item, kind: 'lora' as const }));
  },

  searchEmbeddings(prefix, limit = 20) {
    return this.embed.search(prefix, limit).map((item) => ({ ...item, kind: 'embed' as const }));
  },

  searchWildcards(prefix, limit = 20) {
    return this.wildcard.search(prefix, limit).map((item) => ({ ...item, kind: 'wildcard' as const }));
  },
};
