"""Downloads exactly the works listed in config.yaml from Project Gutenberg.

Nothing is scraped or auto-discovered — the curated list *is* the approved
corpus boundary from the PRD. Each file is saved to:

    data/raw/<author_key>/<mode>/<slug>.txt

and we print the "Title:" line parsed out of the Gutenberg header so you can
eyeball-confirm the ID actually points at the work you expect.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import requests

from src.config import CONFIG, get_env, project_path, iter_works  # noqa: F401 (iter_works re-exported for scripts)

GUTENBERG_URL_TEMPLATE = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
TITLE_RE = re.compile(r"^\s*Title:\s*(.+)$", re.MULTILINE)


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "untitled"


def download_one(gutenberg_id: int, dest: Path) -> str:
    """Download a single Gutenberg text. Returns the parsed Title: line."""
    url = GUTENBERG_URL_TEMPLATE.format(id=gutenberg_id)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "LiteraViewAI/0.1 (educational RAG prototype)"})
    resp.raise_for_status()
    text = resp.text

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")

    match = TITLE_RE.search(text)
    return match.group(1).strip() if match else "(no Title: line found)"


def download_all() -> None:
    raw_root = project_path(CONFIG["paths"]["raw_dir"])

    for mode in ("literature", "life_process"):
        for author_key, display_name, work_title, gutenberg_id in iter_works(mode):
            dest = raw_root / author_key / mode / f"{slugify(work_title)}.txt"
            if dest.exists():
                print(f"[skip] already have: {dest}")
                continue

            print(f"[download] {display_name} / {mode} / {work_title} (id={gutenberg_id}) ...")
            found_title = download_one(gutenberg_id, dest)
            print(f"           -> saved {dest}")
            print(f"           -> Gutenberg header says: \"{found_title}\"")
            if slugify(found_title) != slugify(work_title) and found_title not in work_title and work_title not in found_title:
                print(
                    "           !! WARNING: header title doesn't obviously match "
                    "config.yaml title — double check gutenberg_id before trusting this file."
                )
            time.sleep(1)  # be polite to Gutenberg's mirrors


if __name__ == "__main__":
    download_all()
