"""Strips Project Gutenberg's license header/footer and normalizes text.

Gutenberg files wrap the actual work between two marker lines:
    *** START OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
    *** END OF THE PROJECT GUTENBERG EBOOK <TITLE> ***
Everything outside that band is boilerplate we never want in the corpus.
"""
from __future__ import annotations

import re

START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    start_match = START_RE.search(raw_text)
    end_match = END_RE.search(raw_text)

    body = raw_text[start_match.end():end_match.start()] if (start_match and end_match) else raw_text
    return body.strip()


def normalize_whitespace(text: str) -> str:
    # Collapse runs of 3+ blank lines to a double newline (paragraph break),
    # and strip trailing whitespace on each line, without touching single
    # blank lines that separate paragraphs.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean(raw_text: str) -> str:
    return normalize_whitespace(strip_gutenberg_boilerplate(raw_text))
