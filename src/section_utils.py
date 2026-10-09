"""Removes table-of-contents entries that were mistaken for real sections.

Found Oct 2026 while debugging why "What books appear on the table in
Chapter XVII of Huckleberry Finn?" could not be answered. Gutenberg texts
open with a table of contents, and every contents line (for example
"CHAPTER XVII. | An Evening Call.--The Farm in Arkansaw...", about 100
characters) was being split off as if it were a chapter. Huck Finn had 92
"sections" for 43 real chapters: each chapter existed twice, once as a
stub and once as the real text. The stubs compete with real passages in
search and show up as bogus citations (an earlier answer quoted a
contents line as though it were story text).
"""
from __future__ import annotations

import re

_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_SECTION_TITLE_RE = re.compile(r"^\s*(CHAPTER|ACT|SCENE)\s+([IVXLC]+|\d+)\b", re.IGNORECASE)


def _roman_to_int(token: str) -> int | None:
    token = token.upper()
    if not token or any(ch not in _ROMAN_VALUES for ch in token):
        return None
    total = 0
    for i, ch in enumerate(token):
        value = _ROMAN_VALUES[ch]
        if i + 1 < len(token) and _ROMAN_VALUES[token[i + 1]] > value:
            total -= value
        else:
            total += value
    return total or None


def _identity(title: str):
    """What a heading refers to. "CHAPTER XVII." and "Chapter 17" are the same
    chapter; unnumbered headings ("CHAPTER THE LAST", "PREFACE") are
    identified by their upper-cased title, ignoring punctuation."""
    match = _SECTION_TITLE_RE.match(title or "")
    if match:
        token = match.group(2)
        number = int(token) if token.isdigit() else _roman_to_int(token)
        if number:
            return (match.group(1).lower(), number)
    # Punctuation-insensitive: the contents stub is "CHAPTER THE LAST." but the
    # real heading is "CHAPTER THE LAST" (no period) in Huck Finn.
    return re.sub(r"[^A-Z0-9 ]", "", (title or "").upper()).strip()


def drop_toc_stubs(
    sections: list[tuple[str, str]],
    max_stub_chars: int = 400,
    ratio: int = 5,
) -> tuple[list[tuple[str, str]], int]:
    """Returns (kept_sections, number_dropped).

    A section is dropped only when ALL of these hold: another section in the
    same work has the same identity, this one is shorter than
    `max_stub_chars`, and the longest section with that identity is at least
    `ratio` times longer. A short section that appears only once (such as
    Huck Finn's famous one-paragraph "NOTICE.") is never touched.
    """
    groups: dict = {}
    for index, (title, _text) in enumerate(sections):
        groups.setdefault(_identity(title), []).append(index)

    drop = set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        longest = max(len(sections[i][1]) for i in indices)
        for i in indices:
            length = len(sections[i][1])
            if length < max_stub_chars and length * ratio < longest:
                drop.add(i)

    kept = [section for i, section in enumerate(sections) if i not in drop]
    return kept, len(drop)