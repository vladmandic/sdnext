#!/usr/bin/env python3
"""Fetch and convert booru tag databases to SD.Next dict format.

Usage:
    python cli/fetch_dicts.py danbooru [--output PATH] [--min-count N]
    python cli/fetch_dicts.py e621 [--output PATH] [--min-count N]
    python cli/fetch_dicts.py all [--output-dir DIR] [--min-count N]

Output format:
    JSON with { name, version, categories, tags: [[name, category_id, post_count], ...] }
    Tags sorted by post_count descending.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, timezone

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

USER_AGENT = "SDNext-DictFetcher/1.0 (tag autocomplete)"


def fetch_danbooru(min_count: int = 10) -> list:
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


def fetch_e621(min_count: int = 10) -> list:
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


SOURCES = {
    "danbooru": {
        "fetch": fetch_danbooru,
        "categories": DANBOORU_CATEGORIES,
    },
    "e621": {
        "fetch": fetch_e621,
        "categories": E621_CATEGORIES,
    },
}


def write_dict(name: str, tags: list, categories: dict, output_path: str):
    """Write dict JSON file atomically."""
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


def fetch_source(name: str, output: str, min_count: int):
    """Fetch and write a single source."""
    if name not in SOURCES:
        print(f"Unknown source: {name}. Available: {', '.join(SOURCES.keys())}", file=sys.stderr)
        sys.exit(1)
    source = SOURCES[name]
    print(f"Fetching {name} (min_count={min_count})...", file=sys.stderr)
    tags = source["fetch"](min_count=min_count)
    if not tags:
        print(f"  No tags fetched for {name}", file=sys.stderr)
        return
    write_dict(name, tags, source["categories"], output)


def main():
    parser = argparse.ArgumentParser(description="Fetch booru tag databases for SD.Next dict autocomplete")
    parser.add_argument("source", choices=list(SOURCES.keys()) + ["all"], help="Tag source to fetch")
    parser.add_argument("--output", "-o", help="Output file path (for single source)")
    parser.add_argument("--output-dir", "-d", help="Output directory (for 'all')")
    parser.add_argument("--min-count", "-m", type=int, default=10, help="Minimum post count to include (default: 10)")
    args = parser.parse_args()

    if args.source == "all":
        output_dir = args.output_dir or "."
        for name in SOURCES:
            output = os.path.join(output_dir, f"{name}.json")
            fetch_source(name, output, args.min_count)
    else:
        output = args.output or f"{args.source}.json"
        fetch_source(args.source, output, args.min_count)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
