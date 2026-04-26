#!/usr/bin/env python3
"""Fetch and convert booru tag databases to SD.Next autocomplete format.

Usage:
    python cli/tags-fetch.py danbooru [--output PATH] [--min-count N]
    python cli/tags-fetch.py e621 [--output PATH] [--min-count N]
    python cli/tags-fetch.py rule34 --key USER_ID:API_KEY [--output PATH] [--min-count N]
    python cli/tags-fetch.py sankaku [--output PATH] [--min-count N]
    python cli/tags-fetch.py idol [--output PATH] [--min-count N]
    python cli/tags-fetch.py all --key USER_ID:API_KEY [--output-dir DIR] [--min-count N]

Progress is saved every 50 pages to a .partial file so interrupted
runs can be resumed. Transient HTTP errors are retried with backoff.

Output format:
    JSON with { name, version, categories, tags: [[name, category_id, post_count], ...] }
    Tags sorted by post_count descending.
"""

import argparse
import json
import os
import sys
import time
from datetime import date

import requests

# Unified category scheme - all sources map their native type IDs to these.
# Every dict file uses these same IDs and colors.
UNIFIED_CATEGORIES = {
    "0": {"name": "general", "color": "#0075f8"},
    "1": {"name": "artist", "color": "#cc0000"},
    "2": {"name": "studio", "color": "#ff4500"},
    "3": {"name": "copyright", "color": "#9900ff"},
    "4": {"name": "character", "color": "#00ab2c"},
    "5": {"name": "species", "color": "#ed5d1f"},
    "6": {"name": "genre", "color": "#8a66ff"},
    "7": {"name": "medium", "color": "#00cccc"},
    "8": {"name": "meta", "color": "#6b7280"},
    "9": {"name": "lore", "color": "#228b22"},
    "10": {"name": "lens", "color": "#e67e22"},
    "11": {"name": "lighting", "color": "#f1c40f"},
    "12": {"name": "composition", "color": "#1abc9c"},
    "13": {"name": "color", "color": "#e84393"},
}

# Source → unified type maps. Each maps the source's native category IDs
# to the unified IDs above. Unmapped IDs default to 0 (general).
DANBOORU_TYPE_MAP = {0: 0, 1: 1, 3: 3, 4: 4, 5: 8}
E621_TYPE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0, 7: 8, 8: 9}
RULE34_TYPE_MAP = {0: 0, 1: 1, 3: 3, 4: 4, 5: 8}
SANKAKU_TYPE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 8: 7, 9: 8}

# Idol Complex uses Sankaku's granular tag type system (21 subcategories).
# API type → unified category:
#  0=studio, 1=artist, 2=studio, 3=franchise, 4=character,
#  5=photoset, 6=genre, 8=medium, 9=meta, 10=fashion,
#  11=anatomy, 12=pose, 13=activity, 14=role, 15=flora,
#  17=fauna/entity, 18=object/setting, 19=substance, 20=general,
#  21=language, 22=automatic
IDOL_TYPE_MAP = {
    0: 2, 1: 1, 2: 2, 3: 3, 4: 4, 5: 8, 6: 6, 8: 7, 9: 8,
    10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 17: 0, 18: 0,
    19: 0, 20: 0, 21: 8, 22: 8,
}

USER_AGENT = "SDNext-DictFetcher/1.0 (tag autocomplete)"
SAVE_INTERVAL = 50  # save progress every N pages
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, multiplied by attempt number


# -- Partial save/resume --

def save_partial(path: str, page: int, tags: list):
    """Save fetch progress to a .partial file."""
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"page": page, "tags": tags}, f, separators=(",", ":"))
    os.replace(tmp, path)


def load_partial(path: str) -> tuple[int, list] | tuple[None, list]:
    """Load saved progress from a .partial file. Returns (page, tags) or (None, [])."""
    if not path or not os.path.isfile(path):
        return None, []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        page = data["page"]
        tags = data["tags"]
        print(f"  Resuming from partial: page {page}, {len(tags)} tags", file=sys.stderr)
        return page, tags
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Warning: corrupt .partial file, starting fresh ({e})", file=sys.stderr)
        return None, []


def clear_partial(path: str):
    """Remove .partial file after successful completion."""
    if path and os.path.isfile(path):
        os.remove(path)


# -- HTTP retry helper --

def fetch_with_retry(session: requests.Session, url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """GET with retry and exponential backoff for transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF * (attempt + 1)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} in {wait}s: {e}", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


# -- Auth --

def parse_login_key(key: str | None) -> tuple[str, str] | None:
    """Parse a `login:api_key` string. Returns (login, api_key) or None if unset/invalid."""
    if not key or ":" not in key:
        return None
    login, api_key = key.split(":", 1)
    login = login.strip()
    api_key = api_key.strip()
    if not login or not api_key:
        return None
    return login, api_key


# -- Fetchers --

def fetch_danbooru(min_count: int = 10, partial_path: str = "", api_key: str | None = None, **_kwargs) -> list:
    """Fetch tags from Danbooru API, paginated.
    With `api_key` set (login:api_key) the rate limit rises from anon (~1 rps) to authenticated (~10 rps).
    """
    auth = parse_login_key(api_key)
    sleep_sec = 0.1 if auth else 1.0
    start_page, tags = load_partial(partial_path)
    page = (start_page + 1) if start_page is not None else 1
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        params: dict[str, str] = {"limit": "1000", "page": str(page), "search[order]": "count"}
        if auth:
            params["login"], params["api_key"] = auth
        try:
            resp = fetch_with_retry(session, "https://danbooru.donmai.us/tags.json", params=params)
        except requests.RequestException as e:
            print(f"  Error on page {page}: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        for tag in data:
            count = tag.get("post_count", 0)
            if count < min_count:
                continue
            tags.append([tag["name"], tag["category"], count])
        below_threshold = all(t.get("post_count", 0) < min_count for t in data)
        print(f"  Page {page}: {len(data)} tags (total: {len(tags)})", file=sys.stderr)
        if below_threshold:
            break
        if page % SAVE_INTERVAL == 0:
            save_partial(partial_path, page, tags)
        page += 1
        time.sleep(sleep_sec)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_e621(min_count: int = 10, partial_path: str = "", api_key: str | None = None, **_kwargs) -> list:
    """Fetch tags from e621 API, paginated.
    With `api_key` set (login:api_key) the rate limit rises from anon (~1 rps) to authenticated.
    """
    auth = parse_login_key(api_key)
    sleep_sec = 0.25 if auth else 1.0
    start_page, tags = load_partial(partial_path)
    page = (start_page + 1) if start_page is not None else 1
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        params: dict[str, str] = {"limit": "320", "page": str(page), "search[order]": "count"}
        if auth:
            params["login"], params["api_key"] = auth
        try:
            resp = fetch_with_retry(session, "https://e621.net/tags.json", params=params)
        except requests.RequestException as e:
            print(f"  Error on page {page}: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        for tag in data:
            count = tag.get("post_count", 0)
            if count < min_count:
                continue
            tags.append([tag["name"], tag["category"], count])
        below_threshold = all(t.get("post_count", 0) < min_count for t in data)
        print(f"  Page {page}: {len(data)} tags (total: {len(tags)})", file=sys.stderr)
        if below_threshold:
            break
        if page % SAVE_INTERVAL == 0:
            save_partial(partial_path, page, tags)
        page += 1
        time.sleep(sleep_sec)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_danbooru_style_aliases(base_url: str, api_key: str | None, partial_path: str, sleep_sec: float) -> dict[str, list[str]]:
    """Fetch active tag aliases from a Danbooru-style /tag_aliases.json endpoint.
    Works for both danbooru.donmai.us and e621.net, which share the same schema.
    Returns {consequent_name: [antecedent_names]}.
    """
    partial = {}
    start_page, collected = load_partial(partial_path)
    if collected and isinstance(collected, dict):
        partial = collected
    page = (start_page + 1) if start_page is not None else 1
    auth = parse_login_key(api_key)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        params: dict[str, str] = {"limit": "1000", "page": str(page), "search[status]": "active"}
        if auth:
            params["login"], params["api_key"] = auth
        try:
            resp = fetch_with_retry(session, base_url, params=params)
        except requests.RequestException as e:
            print(f"  Error on alias page {page}: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        for row in data:
            ant = row.get("antecedent_name")
            con = row.get("consequent_name")
            if not ant or not con:
                continue
            partial.setdefault(con, []).append(ant)
        print(f"  Alias page {page}: {len(data)} rows (total consequents: {len(partial)})", file=sys.stderr)
        if page % SAVE_INTERVAL == 0:
            # .partial for alias harvests stores the dict directly under `tags` for reuse of the loader.
            save_partial(partial_path, page, partial)  # type: ignore[arg-type]
        page += 1
        time.sleep(sleep_sec)
    return partial


def fetch_danbooru_aliases(api_key: str | None = None, partial_path: str = "", **_kwargs) -> dict[str, list[str]]:
    """Harvest active tag aliases from Danbooru. Authenticated runs go ~10x faster."""
    sleep_sec = 0.1 if parse_login_key(api_key) else 1.0
    return fetch_danbooru_style_aliases("https://danbooru.donmai.us/tag_aliases.json", api_key, partial_path, sleep_sec)


def fetch_e621_aliases(api_key: str | None = None, partial_path: str = "", **_kwargs) -> dict[str, list[str]]:
    """Harvest active tag aliases from e621 (same schema as danbooru)."""
    sleep_sec = 0.25 if parse_login_key(api_key) else 1.0
    return fetch_danbooru_style_aliases("https://e621.net/tag_aliases.json", api_key, partial_path, sleep_sec)


def fetch_gelbooru(base_url: str, min_count: int = 10, api_key: str | None = None, rate_limit: float = 0.5, partial_path: str = "") -> list:
    """Fetch tags from a Gelbooru-compatible API (rule34, gelbooru, etc.).

    Unlike Danbooru/e621, the Gelbooru tag endpoint doesn't support ordering
    by count, so we paginate through all tags and filter client-side.
    The tag endpoint returns XML (json=1 is not supported for tags).
    """
    import xml.etree.ElementTree as ET

    start_page, tags = load_partial(partial_path)
    page = (start_page + 1) if start_page is not None else 0
    page_size = 1000
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    params: dict[str, str] = {
        "page": "dapi", "s": "tag", "q": "index",
        "limit": str(page_size),
    }
    if api_key:
        if ":" not in api_key:
            print("  Error: --key must be USER_ID:API_KEY format", file=sys.stderr)
            return []
        uid, key = api_key.split(":", 1)
        params["user_id"] = uid
        params["api_key"] = key

    while True:
        params["pid"] = str(page)
        try:
            resp = fetch_with_retry(session, base_url, params=params)
        except requests.RequestException as e:
            print(f"  Error on page {page} after {MAX_RETRIES} retries: {e}", file=sys.stderr)
            break
        text = resp.text.strip()
        if not text or text.startswith('"') or not text.startswith('<?xml'):
            if page == 0:
                print(f"  Error: {text[:200]}", file=sys.stderr)
            break
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            print(f"  Skipping page {page}: malformed XML ({e})", file=sys.stderr)
            page += 1
            time.sleep(rate_limit)
            continue
        elements = root.findall('tag')
        for el in elements:
            count = int(el.get("count", "0"))
            if count >= min_count:
                tags.append([el.get("name", ""), int(el.get("type", "0")), count])
        print(f"  Page {page}: {len(elements)} tags (total: {len(tags)})", file=sys.stderr)
        if len(elements) < page_size:
            break
        if page % SAVE_INTERVAL == 0:
            save_partial(partial_path, page, tags)
        page += 1
        time.sleep(rate_limit)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_rule34(min_count: int = 10, api_key: str | None = None, partial_path: str = "", **_kwargs) -> list:
    """Fetch tags from rule34.xxx."""
    if not api_key:
        print("  Warning: no --key provided, rule34 may rate-limit aggressively", file=sys.stderr)
    return fetch_gelbooru("https://api.rule34.xxx/index.php", min_count=min_count, api_key=api_key, partial_path=partial_path)


def fetch_danbooru_translations(api_key: str | None = None, partial_path: str = "", **_kwargs) -> dict[str, str]:
    """Harvest foreign-name to canonical mappings from Danbooru's wiki.
    Each wiki page has `title` (canonical tag) and `other_names[]` (alternate names, often JA/KR/ZH).
    Returns {other_name_lower: title_lower}. Full harvest; authenticated runs go ~10x faster.
    """
    auth = parse_login_key(api_key)
    sleep_sec = 0.1 if auth else 1.0
    partial: dict[str, str] = {}
    start_page, collected = load_partial(partial_path)
    if collected and isinstance(collected, dict):
        partial = collected
    page = (start_page + 1) if start_page is not None else 1
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        params: dict[str, str] = {"limit": "1000", "page": str(page), "only": "title,other_names"}
        if auth:
            params["login"], params["api_key"] = auth
        try:
            resp = fetch_with_retry(session, "https://danbooru.donmai.us/wiki_pages.json", params=params)
        except requests.RequestException as e:
            print(f"  Error on wiki page {page}: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        added = 0
        for wiki in data:
            title = (wiki.get("title") or "").strip().lower()
            if not title:
                continue
            for other in wiki.get("other_names") or []:
                other_norm = (other or "").strip().lower()
                if not other_norm or other_norm == title:
                    continue
                # First-wins: if two wiki pages claim the same foreign term, keep the first seen.
                partial.setdefault(other_norm, title)
                added += 1
        print(f"  Wiki page {page}: {len(data)} entries ({added} names, total: {len(partial)})", file=sys.stderr)
        if page % SAVE_INTERVAL == 0:
            save_partial(partial_path, page, partial)  # type: ignore[arg-type]
        page += 1
        time.sleep(sleep_sec)
    return partial


def fetch_rule34_aliases(partial_path: str = "", **_kwargs) -> dict[str, list[str]]:
    """Harvest rule34.xxx aliases by scraping the public /index.php?page=alias&s=list listing.
    No API key required (alias list is public). 50 rows per page; `pid` advances by 50; empty page = done.
    """
    import re
    from html import unescape
    partial: dict[str, list[str]] = {}
    start_pid, collected = load_partial(partial_path)
    if collected and isinstance(collected, dict):
        partial = collected
    pid = (start_pid + 50) if start_pid is not None else 0
    session = requests.Session()
    # A browser-like UA is needed; the default python-requests UA gets 403.
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36"
    # Matches the 4 <td> cells of an alias row: checkbox, alias-link, canonical-link, reason.
    # Each tag link's text is `tagname` (possibly with ` (count)` suffix), so we extract from the tags= URL param instead.
    row_re = re.compile(r'<tr class="(?:even|odd)"[^>]*>.*?</tr>', re.DOTALL)
    tag_re = re.compile(r'tags=([^"&]+)')
    while True:
        try:
            resp = fetch_with_retry(session, "https://rule34.xxx/index.php",
                                    params={"page": "alias", "s": "list", "pid": str(pid)})
        except requests.RequestException as e:
            print(f"  Error on alias pid={pid}: {e}", file=sys.stderr)
            break
        rows = row_re.findall(resp.text)
        if not rows:
            break
        added = 0
        for row in rows:
            matches = tag_re.findall(row)
            if len(matches) < 2:
                continue
            ant = unescape(matches[0]).strip()
            con = unescape(matches[1]).strip()
            if not ant or not con:
                continue
            partial.setdefault(con, []).append(ant)
            added += 1
        print(f"  Alias pid={pid}: {len(rows)} rows ({added} parsed, total consequents: {len(partial)})", file=sys.stderr)
        if len(rows) < 50:
            break  # last page
        if (pid // 50) % SAVE_INTERVAL == 0:
            save_partial(partial_path, pid, partial)  # type: ignore[arg-type]
        pid += 50
        time.sleep(1.0)
    return partial


def fetch_sankaku(min_count: int = 10, partial_path: str = "", translations_out: dict | None = None, **_kwargs) -> list:
    """Fetch tags from Sankaku Complex (chan.sankakucomplex.com).

    Uses the public JSON API at sankakuapi.com which supports order=count,
    so we can stop early when counts drop below min_count.
    Tag names come as English with spaces - converted to lowercase.

    If `translations_out` is provided, it is populated with {foreign_term_lower: canonical_lower}
    harvested from each tag's name_ja + translations[] fields in the same pass, avoiding a second
    full-API walk.
    """
    start_page, tags = load_partial(partial_path)
    page = (start_page + 1) if start_page is not None else 1
    page_size = 200
    # Languages to include in translations. EN is already the canonical tag, so it's excluded.
    translation_langs = {"ja", "ko", "zh", "de", "it", "pt", "ru", "fr", "es"}
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        try:
            resp = fetch_with_retry(session, "https://sankakuapi.com/tags",
                                    params={"limit": page_size, "page": page, "order": "count"})
        except requests.RequestException as e:
            print(f"  Error on page {page} after {MAX_RETRIES} retries: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        for tag in data:
            count = tag.get("post_count", 0)
            if count < min_count:
                continue
            name = tag.get("name_en") or tag.get("name_ja", "")
            if not name:
                continue
            name = name.strip().lower()
            tags.append([name, tag.get("type", 0), count])
            if translations_out is None:
                continue
            # name_ja is present on most tags; translations[] covers other languages when available.
            name_ja = (tag.get("name_ja") or "").strip().lower()
            if name_ja and name_ja != name:
                translations_out.setdefault(name_ja, name)
            for entry in tag.get("translations") or []:
                lang = (entry.get("lang") or "").strip().lower()
                term = (entry.get("translation") or "").strip().lower()
                if not term or lang not in translation_langs or term == name:
                    continue
                translations_out.setdefault(term, name)
        below_threshold = all(tag.get("post_count", 0) < min_count for tag in data)
        print(f"  Page {page}: {len(data)} tags (total: {len(tags)})", file=sys.stderr)
        if below_threshold:
            break
        if page % SAVE_INTERVAL == 0:
            save_partial(partial_path, page, tags)
        page += 1
        time.sleep(0.5)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_idol(min_count: int = 10, partial_path: str = "", **_kwargs) -> list:
    """Fetch tags from Idol Complex (idol.sankakucomplex.com).

    Uses the legacy JSON API at iapi.sankakucomplex.com. Supports order=count.
    Max 50 tags per page. Type IDs are non-standard - remapped in write_dict.
    """
    start_page, tags = load_partial(partial_path)
    page = (start_page + 1) if start_page is not None else 1
    page_size = 50
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/138.0.0.0 Safari/537.36"
    while True:
        try:
            resp = fetch_with_retry(session, "https://iapi.sankakucomplex.com/tags.json",
                                    params={"limit": page_size, "page": page, "order": "count"})
        except requests.RequestException as e:
            print(f"  Error on page {page} after {MAX_RETRIES} retries: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        for tag in data:
            count = tag.get("count", 0)
            if count < min_count:
                continue
            name = tag.get("name", "")
            if not name:
                continue
            tags.append([name, tag.get("type", 0), count])
        below_threshold = all(tag.get("count", 0) < min_count for tag in data)
        print(f"  Page {page}: {len(data)} tags (total: {len(tags)})", file=sys.stderr)
        if below_threshold:
            break
        if page % SAVE_INTERVAL == 0:
            save_partial(partial_path, page, tags)
        page += 1
        time.sleep(1.0)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


SOURCES = {
    "danbooru": {
        "fetch": fetch_danbooru, "type_map": DANBOORU_TYPE_MAP,
        "fetch_aliases": fetch_danbooru_aliases,
        "fetch_translations": fetch_danbooru_translations,
    },
    "e621": {
        "fetch": fetch_e621, "type_map": E621_TYPE_MAP,
        "fetch_aliases": fetch_e621_aliases,
    },
    "rule34": {
        "fetch": fetch_rule34, "type_map": RULE34_TYPE_MAP,
        "fetch_aliases": fetch_rule34_aliases,
    },
    "sankaku": {"fetch": fetch_sankaku, "type_map": SANKAKU_TYPE_MAP},  # translations harvested inline
    "idol": {"fetch": fetch_idol, "type_map": IDOL_TYPE_MAP},
}


def write_dict(name: str, tags: list, type_map: dict, output_path: str,
               separator: str = "_", aliases: dict[str, list[str]] | None = None):
    """Write dict JSON file atomically.

    type_map remaps source-native category IDs to unified IDs.
    separator controls the word separator in tag names:
      "_" (default) → "high_resolution" (booru convention, anime/illustration models)
      " " → "high resolution" (natural language, SDXL/Flux-style models)
    aliases, if provided, is {canonical_name: [alternative_names]}. A 4-tuple is emitted
      only for tags that have non-empty aliases, keeping the on-disk size of alias-free
      tags unchanged for backward compatibility with clients reading 3-tuples.
    """
    aliases = aliases or {}
    normalized = []
    matched_aliases = 0
    for t in tags:
        source_name = t[0]
        canonical_key = source_name.strip().lower()
        tag_name = source_name.replace(" ", "_") if separator == "_" else source_name.replace("_", " ")
        category = type_map.get(t[1], 0)
        tag_aliases = aliases.get(canonical_key) or aliases.get(tag_name.lower())
        if tag_aliases:
            # Normalize alias word-separator to match the tag's separator choice.
            if separator == "_":
                tag_aliases = [a.replace(" ", "_") for a in tag_aliases]
            else:
                tag_aliases = [a.replace("_", " ") for a in tag_aliases]
            normalized.append([tag_name, category, t[2], tag_aliases])
            matched_aliases += 1
        else:
            normalized.append([tag_name, category, t[2]])
    data = {
        "name": name,
        "version": date.today().isoformat(),
        "categories": UNIFIED_CATEGORIES,
        "tags": normalized,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    alias_note = f", {matched_aliases} with aliases" if matched_aliases else ""
    print(f"  Written: {output_path} ({len(tags)} tags{alias_note}, {size_mb:.1f} MB)", file=sys.stderr)


def write_translations(mapping: dict[str, str], output_path: str):
    """Write the translations companion file atomically.
    `mapping` is {foreign_term: canonical_tag_name}, both lowercased and underscore-normalized.
    """
    if not mapping:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    os.replace(tmp_path, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Translations: {output_path} ({len(mapping)} terms, {size_kb:.1f} KB)", file=sys.stderr)


def fetch_source(name: str, output: str, min_count: int, keys: dict[str, str] | None = None, separator: str = "_"):
    """Fetch and write a single source. `keys` maps source name to `login:token` (or rule34's `uid:token`)."""
    if name not in SOURCES:
        print(f"Unknown source: {name}. Available: {', '.join(SOURCES.keys())}", file=sys.stderr)
        sys.exit(1)
    source = SOURCES[name]
    keys = keys or {}
    api_key = keys.get(name)
    partial_path = output + ".partial"
    print(f"Fetching {name} (min_count={min_count})...", file=sys.stderr)

    # Sankaku collects translations inline during the tag walk to avoid a second full API pass.
    translations_inline: dict[str, str] | None = {} if name == "sankaku" else None
    tags = source["fetch"](
        min_count=min_count, api_key=api_key, partial_path=partial_path,
        translations_out=translations_inline,
    ) if translations_inline is not None else source["fetch"](
        min_count=min_count, api_key=api_key, partial_path=partial_path,
    )
    if not tags:
        print(f"  No tags fetched for {name}", file=sys.stderr)
        return

    # Optional alias harvest via a separate .partial file (independent resume).
    aliases: dict[str, list[str]] = {}
    if source.get("fetch_aliases"):
        alias_partial = output + ".aliases.partial"
        print(f"Fetching {name} aliases...", file=sys.stderr)
        try:
            aliases = source["fetch_aliases"](api_key=api_key, partial_path=alias_partial)
            clear_partial(alias_partial)
        except Exception as e:
            print(f"  Alias fetch failed for {name}: {e}", file=sys.stderr)
            aliases = {}

    write_dict(name, tags, source["type_map"], output, separator=separator, aliases=aliases)
    clear_partial(partial_path)

    # Translations: either inline (sankaku) or via a dedicated fetcher (danbooru wiki).
    translations: dict[str, str] = translations_inline or {}
    if source.get("fetch_translations"):
        tr_partial = output + ".translations.partial"
        print(f"Fetching {name} translations...", file=sys.stderr)
        try:
            translations = source["fetch_translations"](api_key=api_key, partial_path=tr_partial) or {}
            clear_partial(tr_partial)
        except Exception as e:
            print(f"  Translation fetch failed for {name}: {e}", file=sys.stderr)
    if translations:
        tr_output = output.replace(".json", ".translations.json") if output.endswith(".json") else f"{output}.translations.json"
        write_translations(translations, tr_output)

    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("tags_manifest", os.path.join(os.path.dirname(__file__), "tags-manifest.py"))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    update_manifest = mod.update_manifest
    update_manifest(os.path.dirname(output) or ".")


def main():
    parser = argparse.ArgumentParser(description="Fetch booru tag databases for SD.Next autocomplete")
    parser.add_argument("source", choices=list(SOURCES.keys()) + ["all"], help="Tag source to fetch")
    parser.add_argument("--output", "-o", help="Output file path (for single source)")
    parser.add_argument("--output-dir", "-d", help="Output directory (for 'all')")
    parser.add_argument("--min-count", "-m", type=int, default=10, help="Minimum post count to include (default: 10)")
    parser.add_argument("--key", "-k", help="Rule34 API key as USER_ID:API_KEY")
    parser.add_argument("--danbooru-key", help="Danbooru login:api_key (raises rate limit from 1 to 10 rps)")
    parser.add_argument("--e621-key", help="e621 login:api_key (raises rate limit)")
    parser.add_argument("--spaces", action="store_true", help="Use spaces instead of underscores in tag names (for natural language models like SDXL/Flux)")
    args = parser.parse_args()
    separator = " " if args.spaces else "_"
    # Per-source key map. Rule34 keeps the original `--key` (uid:token); danbooru/e621 get dedicated flags.
    keys: dict[str, str] = {}
    if args.key:
        keys["rule34"] = args.key
    if args.danbooru_key:
        keys["danbooru"] = args.danbooru_key
    if args.e621_key:
        keys["e621"] = args.e621_key

    if args.source == "all":
        output_dir = args.output_dir or "."
        for name in SOURCES:
            output = os.path.join(output_dir, f"{name}.json")
            fetch_source(name, output, args.min_count, keys=keys, separator=separator)
    else:
        output = args.output or f"{args.source}.json"
        fetch_source(args.source, output, args.min_count, keys=keys, separator=separator)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
