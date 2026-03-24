#!/usr/bin/env python3
"""Fetch and convert booru tag databases to SD.Next dict format.

Usage:
    python cli/fetch_dicts.py danbooru [--output PATH] [--min-count N]
    python cli/fetch_dicts.py e621 [--output PATH] [--min-count N]
    python cli/fetch_dicts.py rule34 --key USER_ID:API_KEY [--output PATH] [--min-count N]
    python cli/fetch_dicts.py sankaku [--output PATH] [--min-count N]
    python cli/fetch_dicts.py all --key USER_ID:API_KEY [--output-dir DIR] [--min-count N]

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

DANBOORU_CATEGORIES = {
    "0": {"name": "general", "color": "#0075f8"},
    "1": {"name": "artist", "color": "#a800aa"},
    "3": {"name": "copyright", "color": "#dd00dd"},
    "4": {"name": "character", "color": "#00ab2c"},
    "5": {"name": "meta", "color": "#ee8800"},
}

E621_CATEGORIES = {
    "0": {"name": "general", "color": "#0075f8"},
    "1": {"name": "artist", "color": "#a800aa"},
    "3": {"name": "copyright", "color": "#dd00dd"},
    "4": {"name": "character", "color": "#00ab2c"},
    "5": {"name": "species", "color": "#ed5d1f"},
    "6": {"name": "invalid", "color": "#ff3d3d"},
    "7": {"name": "meta", "color": "#ee8800"},
    "8": {"name": "lore", "color": "#228b22"},
}

# Rule34 uses the same category IDs as Danbooru (Gelbooru-compatible)
RULE34_CATEGORIES = {
    "0": {"name": "general", "color": "#0075f8"},
    "1": {"name": "artist", "color": "#a800aa"},
    "3": {"name": "copyright", "color": "#dd00dd"},
    "4": {"name": "character", "color": "#00ab2c"},
    "5": {"name": "meta", "color": "#ee8800"},
}

SANKAKU_CATEGORIES = {
    "0": {"name": "general", "color": "#0075f8"},
    "1": {"name": "artist", "color": "#a800aa"},
    "2": {"name": "studio", "color": "#a800aa"},
    "3": {"name": "copyright", "color": "#dd00dd"},
    "4": {"name": "character", "color": "#00ab2c"},
    "5": {"name": "species", "color": "#ed5d1f"},
    "8": {"name": "medium", "color": "#ee8800"},
    "9": {"name": "meta", "color": "#ee8800"},
}

USER_AGENT = "SDNext-DictFetcher/1.0 (tag autocomplete)"


def fetch_danbooru(min_count: int = 10, **_kwargs) -> list:
    """Fetch tags from Danbooru API, paginated."""
    tags = []
    page = 1
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        url = f"https://danbooru.donmai.us/tags.json?limit=1000&page={page}&search[order]=count"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
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
        page += 1
        time.sleep(0.5)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_e621(min_count: int = 10, **_kwargs) -> list:
    """Fetch tags from e621 API, paginated."""
    tags = []
    page = 1
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        url = f"https://e621.net/tags.json?limit=320&page={page}&search[order]=count"
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
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
        page += 1
        time.sleep(1.0)  # e621 rate limit is stricter
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_gelbooru(base_url: str, min_count: int = 10, api_key: str | None = None, rate_limit: float = 0.5) -> list:
    """Fetch tags from a Gelbooru-compatible API (rule34, gelbooru, etc.).

    Unlike Danbooru/e621, the Gelbooru tag endpoint doesn't support ordering
    by count, so we paginate through all tags and filter client-side.
    The tag endpoint returns XML (json=1 is not supported for tags).
    """
    import xml.etree.ElementTree as ET

    tags = []
    page = 0
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
            resp = session.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error on page {page}: {e}", file=sys.stderr)
            break
        text = resp.text.strip()
        if not text or text.startswith('"') or not text.startswith('<?xml'):
            # Non-XML response (auth error, empty, etc.)
            if page == 0:
                print(f"  Error: {text[:200]}", file=sys.stderr)
            break
        root = ET.fromstring(text)
        elements = root.findall('tag')
        for el in elements:
            count = int(el.get("count", "0"))
            if count >= min_count:
                tags.append([el.get("name", ""), int(el.get("type", "0")), count])
        print(f"  Page {page}: {len(elements)} tags (total: {len(tags)})", file=sys.stderr)
        if len(elements) < page_size:
            break
        page += 1
        time.sleep(rate_limit)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


def fetch_rule34(min_count: int = 10, api_key: str | None = None, **_kwargs) -> list:
    """Fetch tags from rule34.xxx."""
    if not api_key:
        print("  Warning: no --key provided, rule34 may rate-limit aggressively", file=sys.stderr)
    return fetch_gelbooru("https://api.rule34.xxx/index.php", min_count=min_count, api_key=api_key)


def fetch_sankaku(min_count: int = 10, **_kwargs) -> list:
    """Fetch tags from Sankaku Complex (chan.sankakucomplex.com).

    Uses the public JSON API at sankakuapi.com which supports order=count,
    so we can stop early when counts drop below min_count.
    Tag names come as English with spaces — converted to lowercase underscores.
    """
    tags = []
    page = 1
    page_size = 200  # API max
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    while True:
        try:
            resp = session.get(
                "https://sankakuapi.com/tags",
                params={"limit": page_size, "page": page, "order": "count"},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  Error on page {page}: {e}", file=sys.stderr)
            break
        data = resp.json()
        if not data:
            break
        below_threshold = True
        for tag in data:
            count = tag.get("post_count", 0)
            if count >= min_count:
                below_threshold = False
                name = tag.get("name_en") or tag.get("name_ja", "")
                if not name:
                    continue
                name = name.strip().lower()
                tags.append([name, tag.get("type", 0), count])
            else:
                below_threshold = True
        print(f"  Page {page}: {len(data)} tags (total: {len(tags)})", file=sys.stderr)
        if below_threshold:
            break
        page += 1
        time.sleep(0.5)
    tags.sort(key=lambda t: t[2], reverse=True)
    return tags


SOURCES = {
    "danbooru": {
        "fetch": fetch_danbooru,
        "categories": DANBOORU_CATEGORIES,
    },
    "e621": {
        "fetch": fetch_e621,
        "categories": E621_CATEGORIES,
    },
    "rule34": {
        "fetch": fetch_rule34,
        "categories": RULE34_CATEGORIES,
    },
    "sankaku": {
        "fetch": fetch_sankaku,
        "categories": SANKAKU_CATEGORIES,
    },
}


def write_dict(name: str, tags: list, categories: dict, output_path: str, separator: str = "_"):
    """Write dict JSON file atomically.

    separator controls the word separator in tag names:
      "_" (default) → "high_resolution" (booru convention, anime/illustration models)
      " " → "high resolution" (natural language, SDXL/Flux-style models)
    """
    if separator == "_":
        tags = [[t[0].replace(" ", "_"), t[1], t[2]] for t in tags]
    else:
        tags = [[t[0].replace("_", " "), t[1], t[2]] for t in tags]
    data = {
        "name": name,
        "version": date.today().isoformat(),
        "categories": categories,
        "tags": tags,
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp_path, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Written: {output_path} ({len(tags)} tags, {size_mb:.1f} MB)", file=sys.stderr)


def fetch_source(name: str, output: str, min_count: int, api_key: str | None = None, separator: str = "_"):
    """Fetch and write a single source."""
    if name not in SOURCES:
        print(f"Unknown source: {name}. Available: {', '.join(SOURCES.keys())}", file=sys.stderr)
        sys.exit(1)
    source = SOURCES[name]
    print(f"Fetching {name} (min_count={min_count})...", file=sys.stderr)
    tags = source["fetch"](min_count=min_count, api_key=api_key)
    if not tags:
        print(f"  No tags fetched for {name}", file=sys.stderr)
        return
    write_dict(name, tags, source["categories"], output, separator=separator)


def main():
    parser = argparse.ArgumentParser(description="Fetch booru tag databases for SD.Next dict autocomplete")
    parser.add_argument("source", choices=list(SOURCES.keys()) + ["all"], help="Tag source to fetch")
    parser.add_argument("--output", "-o", help="Output file path (for single source)")
    parser.add_argument("--output-dir", "-d", help="Output directory (for 'all')")
    parser.add_argument("--min-count", "-m", type=int, default=10, help="Minimum post count to include (default: 10)")
    parser.add_argument("--key", "-k", help="API key as USER_ID:API_KEY (required for rule34)")
    parser.add_argument("--spaces", action="store_true", help="Use spaces instead of underscores in tag names (for natural language models like SDXL/Flux)")
    args = parser.parse_args()
    separator = " " if args.spaces else "_"

    if args.source == "all":
        output_dir = args.output_dir or "."
        for name in SOURCES:
            output = os.path.join(output_dir, f"{name}.json")
            fetch_source(name, output, args.min_count, api_key=args.key, separator=separator)
    else:
        output = args.output or f"{args.source}.json"
        fetch_source(args.source, output, args.min_count, api_key=args.key, separator=separator)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
