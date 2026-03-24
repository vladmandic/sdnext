#!/usr/bin/env python3
"""Generate manifest.json for the HuggingFace dict repository.

Reads dict JSON files and writes a manifest with accurate tag counts,
file sizes, and versions. Only listed files are included - curated dicts
that ship bundled with the repo should not be passed as arguments.

Usage:
    python cli/gen_manifest.py data/dicts/danbooru.json data/dicts/e621.json ...
    python cli/gen_manifest.py data/dicts/danbooru.json -o data/dicts/manifest.json
    python cli/gen_manifest.py data/dicts/*.json --exclude art photography quality negative
"""

import argparse
import json
import os
import sys

# Human-readable descriptions keyed by dict name.
# Add entries here when new sources are added to fetch_dicts.py.
DESCRIPTIONS = {
    "art": "Art movements, styles, and techniques",
    "danbooru": "Danbooru image board tags - anime/illustration focused",
    "e621": "e621 tags - furry/animal art focused",
    "idol": "Idol Complex tags - Japanese idol photography",
    "negative": "Common negative prompt terms",
    "photography": "Photography terms - lens, lighting, composition, color",
    "quality": "Quality and aesthetic meta tags",
    "rule34": "Rule34.xxx tags - multi-fandom",
    "sankaku": "Sankaku Complex tags - anime/illustration with granular categories",
}


def build_entry(filepath: str) -> dict:
    """Build a manifest entry from a dict JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("name") or os.path.splitext(os.path.basename(filepath))[0]
    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
    return {
        "name": name,
        "description": DESCRIPTIONS.get(name, ""),
        "version": data.get("version", ""),
        "tag_count": len(data.get("tags", [])),
        "size_mb": size_mb,
    }


def update_manifest(dicts_dir: str) -> bool:
    """Regenerate manifest.json in dicts_dir if one already exists.

    Only updates entries for dicts already listed in the manifest - does not
    add new dicts. Returns True if the manifest was updated, False if no
    manifest exists to update.
    """
    manifest_path = os.path.join(dicts_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    existing_names = {e["name"] for e in manifest.get("dicts", [])}
    entries = []
    for name in sorted(existing_names):
        filepath = os.path.join(dicts_dir, f"{name}.json")
        if not os.path.isfile(filepath):
            continue
        try:
            entries.append(build_entry(filepath))
        except (json.JSONDecodeError, KeyError):
            continue
    manifest["dicts"] = entries
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"  Manifest updated: {manifest_path} ({len(entries)} dicts)", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate manifest.json for HF dict repo")
    parser.add_argument("files", nargs="+", help="Dict JSON files to include")
    parser.add_argument("-o", "--output", default="manifest.json", help="Output path (default: manifest.json)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Dict names to exclude (e.g. art quality)")
    args = parser.parse_args()

    exclude = set(args.exclude)
    entries = []
    for filepath in args.files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if basename in exclude or basename == "manifest":
            continue
        if not os.path.isfile(filepath):
            print(f"  Skipping {filepath}: not found", file=sys.stderr)
            continue
        try:
            entry = build_entry(filepath)
            entries.append(entry)
            print(f"  {entry['name']:>12}: {entry['tag_count']:>10,} tags, {entry['size_mb']:>6.1f} MB, v={entry['version']}", file=sys.stderr)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Skipping {filepath}: {e}", file=sys.stderr)

    entries.sort(key=lambda e: e["name"])
    manifest = {"dicts": entries}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\nManifest written: {args.output} ({len(entries)} dicts)", file=sys.stderr)


if __name__ == "__main__":
    main()
