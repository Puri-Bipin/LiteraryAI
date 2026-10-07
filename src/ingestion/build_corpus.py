"""Turns cleaned raw text files into section-level corpus records.

Each record already carries the metadata fields the PRD's architecture
section calls for (author, source type, work/document title, mode,
chapter/act/section, source reference) *before* chunking happens, so every
downstream chunk inherits correct provenance no matter how the text gets
split later.

Output: data/processed/corpus.jsonl, one JSON object per line:
{
  "author_key": "mark_twain",
  "author_display": "Mark Twain",
  "mode": "literature" | "life_process",
  "work_title": "Adventures of Huckleberry Finn",
  "section_title": "Chapter 3",
  "section_index": 3,
  "text": "...",
  "source_ref": "https://www.gutenberg.org/ebooks/76"
}
"""
from __future__ import annotations

import json
import re

from src.config import CONFIG, project_path, iter_works
from src.ingestion.clean_text import clean
from src.ingestion.download_gutenberg import slugify

# Matches four known Gutenberg heading styles used across Twain's works:
#   1. "CHAPTER I.", "Chapter 3", "ACT III", "SCENE 2"
#   2. "CHAPTERS FROM MY AUTOBIOGRAPHY.--III."  (title + dash + roman numeral)
#   3. A bare, short, ALL-CAPS title standing alone on its own line, blank
#      lines both before and after -- the format most short-story
#      collections use for each story's heading, e.g. "THE JUMPING FROG",
#      "MY WATCH" (no "CHAPTER" word, no numeral). Requires blank-line
#      isolation on both sides specifically to avoid matching an all-caps
#      word or short shout embedded inside a paragraph.
#   4. "XLV. LETTERS, 1906, TO VARIOUS PERSONS..." -- the format Twain's
#      published Letters volumes use: a roman numeral, a period, then a
#      summary of that letter/chapter's contents, all on one line. Note:
#      in the raw plain-text files, a long heading like this can sometimes
#      wrap across two physical lines, in which case this pattern will
#      miss it and the surrounding text gets grouped into the previous
#      section instead -- check section counts after a first run on a new
#      Letters volume rather than assuming perfect per-letter granularity.
SECTION_HEADING_RE = re.compile(
    r"(?:"
    r"^\s*(?:CHAPTER|Chapter|ACT|Act|SCENE|Scene)\s+[IVXLC\d]+\.?.*$"
    r"|^\s*[A-Z][A-Z '\.]{4,80}[.\-\u2013\u2014]{1,3}\s*[IVXLC]+\.?\s*$"
    r"|(?<=\n\n)[A-Z][A-Z0-9 ,'\.\-]{3,69}(?=\n\n)"
    r"|^\s*[IVXLC]+\.\s+[A-Z].{3,120}$"
    r")",
    re.MULTILINE,
)


def split_into_sections(body_text: str) -> list[tuple[str, str]]:
    """Return [(section_title, section_text), ...].

    Falls back to a single section covering the whole work if no chapter
    headings are found (e.g. short essays).
    """
    headings = list(SECTION_HEADING_RE.finditer(body_text))
    if not headings:
        return [("Full text", body_text)]

    sections = []
    for i, match in enumerate(headings):
        title = match.group(0).strip()
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body_text)
        section_text = body_text[start:end].strip()
        if section_text:
            sections.append((title, section_text))
    return sections


def build_corpus() -> None:
    raw_root = project_path(CONFIG["paths"]["raw_dir"])
    out_path = project_path(CONFIG["paths"]["corpus_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record_count = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for mode in ("literature", "life_process"):
            for author_key, display_name, work_title, gutenberg_id in iter_works(mode):
                raw_path = raw_root / author_key / mode / f"{slugify(work_title)}.txt"
                if not raw_path.exists():
                    print(f"[skip] not downloaded yet: {raw_path}")
                    continue

                raw_text = raw_path.read_text(encoding="utf-8")
                body = clean(raw_text)
                sections = split_into_sections(body)

                for idx, (section_title, section_text) in enumerate(sections, start=1):
                    record = {
                        "author_key": author_key,
                        "author_display": display_name,
                        "mode": mode,
                        "work_title": work_title,
                        "section_title": section_title,
                        "section_index": idx,
                        "text": section_text,
                        "source_ref": f"https://www.gutenberg.org/ebooks/{gutenberg_id}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    record_count += 1

                print(f"[ok] {display_name} / {work_title}: {len(sections)} sections")

    print(f"\nWrote {record_count} section records to {out_path}")


if __name__ == "__main__":
    build_corpus()