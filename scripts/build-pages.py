#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
README_URL = "https://raw.githubusercontent.com/vladmandic/sdnext/refs/heads/dev/README.md"
GITHUB_MARKDOWN_API = "https://api.github.com/markdown"
OUT_DIR = ROOT / "_includes" / "readme"
DATA_FILE = ROOT / "_data" / "readme_sections.json"

pattern = re.compile(r"^(##)\s+(.*)$")


def slugify(text: str) -> str:
    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def github_render(markdown_text: str) -> str:
    headers = {
        "User-Agent": "sdnext-pages-script/1.0",
        "Content-Type": "application/json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    payload = json.dumps({"text": markdown_text, "mode": "gfm"}).encode("utf-8")
    request = Request(GITHUB_MARKDOWN_API, data=payload, headers=headers)
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    request = Request(README_URL, headers={"User-Agent": "sdnext-pages-script/1.0"})
    with urlopen(request, timeout=20) as response:
        data = response.read().decode("utf-8")

    lines = data.splitlines()
    sections = []
    current = {"slug": "hero", "title": "hero", "lines": []}

    for line in lines:
        match = pattern.match(line)
        if match:
            if current["lines"]:
                sections.append(current)
            current = {
                "slug": slugify(match.group(2)),
                "title": match.group(2).strip(),
                "lines": [line],
            }
        else:
            current["lines"].append(line)

    if current["lines"]:
        sections.append(current)

    for section in sections:
        source = "\n".join(section["lines"]).rstrip() + "\n"
        md_path = OUT_DIR / f"{section['slug']}.md"
        html_path = OUT_DIR / f"{section['slug']}.html"

        md_path.write_text(source, encoding="utf-8")

        try:
            rendered = github_render(source)
        except Exception as exc:
            print(f"Warning: GitHub markdown render failed for {section['slug']}: {exc}")
            rendered = source

        html_path.write_text(rendered, encoding="utf-8")

    DATA_FILE.write_text(
        json.dumps(
            [{"slug": section["slug"], "title": section["title"]} for section in sections],
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(sections)} README section includes in {OUT_DIR}")
    print(f"Saved section metadata to {DATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
