#!/usr/bin/env python3
"""Generate manifest.json for the HuggingFace autocomplete repository.

Reads tag JSON files and writes a manifest with accurate tag counts,
file sizes, and versions.

Usage:
    python cli/tags-manifest.py models/autocomplete/danbooru.json models/autocomplete/e621.json ...
    python cli/tags-manifest.py models/autocomplete/*.json -o models/autocomplete/manifest.json
"""

import argparse
import json
import os
import sys

# Human-readable descriptions keyed by file name.
# Add entries here when new sources are added to tags-fetch.py.
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
    """Build a manifest entry from a tag JSON file.
    If a `<name>.translations.json` companion sits next to the file, the entry gets
    `translations: true` so the client-side downloader knows to pull the companion too.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    name = data.get("name") or os.path.splitext(os.path.basename(filepath))[0]
    size_mb = round(os.path.getsize(filepath) / (1024 * 1024), 1)
    entry = {
        "name": name,
        "description": DESCRIPTIONS.get(name, ""),
        "version": data.get("version", ""),
        "tag_count": len(data.get("tags", [])),
        "size_mb": size_mb,
    }
    translations_path = os.path.join(os.path.dirname(filepath), f"{name}.translations.json")
    if os.path.isfile(translations_path):
        entry["translations"] = True
    return entry


def update_manifest(directory: str) -> bool:
    """Regenerate manifest.json in directory if one already exists.

    Only updates entries already listed in the manifest - does not
    add new entries. Returns True if the manifest was updated, False
    if no manifest exists to update.
    """
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    existing_names = {e["name"] for e in manifest.get("entries", [])}
    entries = []
    for name in sorted(existing_names):
        filepath = os.path.join(directory, f"{name}.json")
        if not os.path.isfile(filepath):
            continue
        try:
            entries.append(build_entry(filepath))
        except (json.JSONDecodeError, KeyError):
            continue
    manifest["entries"] = entries
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"  Manifest updated: {manifest_path} ({len(entries)} entries)", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate manifest.json for HF autocomplete repo")
    parser.add_argument("files", nargs="+", help="Tag JSON files to include")
    parser.add_argument("-o", "--output", default="manifest.json", help="Output path (default: manifest.json)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Names to exclude (e.g. art quality)")
    args = parser.parse_args()

    exclude = set(args.exclude)
    entries = []
    for filepath in args.files:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if basename in exclude or basename == "manifest":
            continue
        # Skip translation companion files. They're pulled in automatically via `translations: true`
        # flags on their parent dict entries; standalone entries would be malformed.
        if basename.endswith(".translations"):
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
    manifest = {"entries": entries}

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")

    print(f"\nManifest written: {args.output} ({len(entries)} entries)", file=sys.stderr)


if __name__ == "__main__":
    main()
