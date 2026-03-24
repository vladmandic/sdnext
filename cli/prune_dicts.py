#!/usr/bin/env python3
"""Prune tag dictionaries with per-category minimum post counts.

Usage:
    python cli/prune_dicts.py data/dicts/danbooru.json
    python cli/prune_dicts.py data/dicts/*.json --general 500 --artist 20
    python cli/prune_dicts.py data/dicts/sankaku.json -o data/dicts/sankaku-pruned.json
    python cli/prune_dicts.py data/dicts/*.json --dry-run

Category-aware pruning: artists and characters are kept at lower thresholds
(proper nouns that need autocomplete), while general tags are pruned more
aggressively (common vocabulary you'd type naturally).
"""

import argparse
import json
import os
import sys

# Category ID → name (must match UNIFIED_CATEGORIES in fetch_dicts.py)
CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    2: "studio",
    3: "copyright",
    4: "character",
    5: "species",
    6: "genre",
    7: "medium",
    8: "meta",
    9: "lore",
}

# Default minimum post counts per category.
# Low for proper nouns (hard to guess), high for common vocabulary (easy to type).
DEFAULTS = {
    "general": 200,
    "artist": 10,
    "studio": 10,
    "copyright": 20,
    "character": 10,
    "species": 50,
    "genre": 100,
    "medium": 100,
    "meta": 1000,
    "lore": 50,
}


def prune(tags: list, thresholds: dict[str, int]) -> tuple[list, dict[str, tuple[int, int]]]:
    """Prune tags by per-category thresholds.

    Returns (pruned_tags, stats) where stats maps category name
    to (before_count, after_count).
    """
    stats: dict[str, tuple[int, int]] = {}
    kept = []
    for tag in tags:
        cat_id = tag[1]
        cat_name = CATEGORY_NAMES.get(cat_id, "general")
        threshold = thresholds.get(cat_name, thresholds.get("general", 10))
        before, after = stats.get(cat_name, (0, 0))
        before += 1
        if tag[2] >= threshold:
            kept.append(tag)
            after += 1
        stats[cat_name] = (before, after)
    return kept, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prune tag dictionaries with per-category minimum post counts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Defaults: " + ", ".join(f"{k}={v}" for k, v in DEFAULTS.items()),
    )
    parser.add_argument("files", nargs="+", help="Dict JSON files to prune")
    parser.add_argument("-o", "--output", help="Output file (single file mode only)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pruned without writing")
    parser.add_argument("--in-place", "-i", action="store_true", help="Overwrite input files")

    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=int, default=default, help=f"Min posts for {name} (default: {default})")

    args = parser.parse_args()

    if args.output and len(args.files) > 1:
        print("Error: --output can only be used with a single input file", file=sys.stderr)
        sys.exit(1)

    if not args.output and not args.in_place and not args.dry_run:
        print("Error: specify --in-place, --output, or --dry-run", file=sys.stderr)
        sys.exit(1)

    thresholds = {name: getattr(args, name) for name in DEFAULTS}

    for filepath in args.files:
        if not os.path.isfile(filepath):
            print(f"  Skipping {filepath}: not found", file=sys.stderr)
            continue

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        tags = data.get("tags", [])
        pruned, stats = prune(tags, thresholds)
        total_before = sum(s[0] for s in stats.values())
        total_after = sum(s[1] for s in stats.values())

        name = data.get("name", os.path.basename(filepath))
        print(f"\n{name}: {total_before:,} → {total_after:,} tags ({total_before - total_after:,} removed)", file=sys.stderr)
        for cat_name in sorted(stats, key=lambda c: list(CATEGORY_NAMES.values()).index(c) if c in CATEGORY_NAMES.values() else 99):
            before, after = stats[cat_name]
            threshold = thresholds.get(cat_name, 10)
            removed = before - after
            if removed > 0:
                print(f"  {cat_name:>12}: {before:>8,} → {after:>8,}  (min {threshold:>5}, -{removed:,})", file=sys.stderr)
            else:
                print(f"  {cat_name:>12}: {before:>8,}  (min {threshold:>5}, all kept)", file=sys.stderr)

        if args.dry_run:
            continue

        data["tags"] = pruned
        output_path = args.output or filepath
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_path, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Written: {output_path} ({size_mb:.1f} MB)", file=sys.stderr)
        from gen_manifest import update_manifest
        update_manifest(os.path.dirname(output_path) or ".")


if __name__ == "__main__":
    main()
