"""Implements PRD Section 3 Step 3-4: "System identifies whether the
question concerns a specific work, the oeuvre, or life/process" and then
"Retrieval is restricted to the correct dataset boundary."

The author and mode are already chosen by the user via the UI (Step 1-2).
This router has three jobs:

1. Detect "meta" / navigational questions -- "what works are available?" --
   answered directly from config.yaml, no retrieval/LLM involved.
2. Inside "Literature" mode, detect whether the question names one specific
   work (-> restrict retrieval to just that work) or reads as a cross-work
   question (-> retrieve across the author's whole literary corpus).
3. When a specific work AND a chapter are named ("Chapter XVII of
   Huckleberry Finn"), record the chapter so retrieval can restrict the
   search to just that chapter's chunks. (Added Oct 2026: the chapter
   heading lives in metadata, not in the embedded text, so a plain
   similarity search can never use "Chapter XVII" -- it has to be a filter.)

All plain string matching / keyword heuristics, deliberately NOT an LLM
call: free, instant, deterministic, and testable.

Title matching uses a WINDOWED comparison (title vs. same-length sliding
windows of the question) rather than a whole-question comparison, because
a long question dilutes an otherwise clear title mention: "What books
appear on the table in Chapter XVII of Huckleberry Finn?" scored only
0.479 against the title as a whole sentence, but 0.778 as a window.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass



@dataclass
class RouteDecision:
    scope: str                  # "meta" | "specific_work" | "oeuvre" | "life_process"
    work_title: str | None      # set only when scope == "specific_work"
    chapter: str | None = None  # e.g. "XVII" or "THE LAST"; only with specific_work


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# --- Meta / navigational question detection ---------------------------------
_AVAILABILITY_WORDS = {"available", "have", "offer", "cover", "discuss", "access", "include", "got"}
_CORPUS_NOUNS = {
    "works", "work", "texts", "text", "books", "book", "corpus",
    "material", "materials", "sources", "source", "titles", "title", "stories",
}
_META_STARTERS = ("what", "which", "list")


def is_meta_question(question: str) -> bool:
    norm = _normalize(question)
    if not norm:
        return False
    tokens = set(norm.split())
    starts_meta = norm.startswith(_META_STARTERS)
    return starts_meta and bool(tokens & _AVAILABILITY_WORDS) and bool(tokens & _CORPUS_NOUNS)


# --- Specific-work vs. oeuvre detection --------------------------------------
def _windowed_best_ratio(norm_question: str, norm_title: str) -> float:
    q_words = norm_question.split()
    t_words = norm_title.split()
    t_len = len(t_words)
    if t_len == 0:
        return 0.0
    if len(q_words) <= t_len:
        return difflib.SequenceMatcher(None, norm_question, norm_title).ratio()

    best = 0.0
    for start in range(0, len(q_words) - t_len + 1):
        window = " ".join(q_words[start:start + t_len])
        score = difflib.SequenceMatcher(None, window, norm_title).ratio()
        best = max(best, score)
    return best


def _best_title_match(question: str, known_titles: list[str], threshold: float = 0.5) -> str | None:
    norm_question = _normalize(question)
    norm_titles = {title: _normalize(title) for title in known_titles}

    for title, norm_title in norm_titles.items():
        if norm_title in norm_question:
            return title

    best_title, best_score = None, 0.0
    for title, norm_title in norm_titles.items():
        score = _windowed_best_ratio(norm_question, norm_title)
        if score > best_score:
            best_title, best_score = title, score

    return best_title if best_score >= threshold else None


# --- Chapter detection --------------------------------------------------------
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_ROMAN_STEPS = [(100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
                (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _int_to_roman(n: int) -> str:
    out = []
    for value, symbol in _ROMAN_STEPS:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def _roman_to_int(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        value = _ROMAN_VALUES[ch]
        if i + 1 < len(s) and _ROMAN_VALUES[s[i + 1]] > value:
            total -= value
        else:
            total += value
    return total


def parse_chapter(question: str) -> str | None:
    """Returns a normalized chapter label ("XVII", "THE LAST") or None.

    Accepts "Chapter XVII", "chapter 17", and "the last/final chapter".
    A roman-numeral-looking word is only accepted if it round-trips
    (e.g. "civil" is made of roman letters but is not a valid numeral).
    """
    q = question.lower()
    if re.search(r"\bchapter the last\b|\b(last|final) chapter\b", q):
        return "THE LAST"

    m = re.search(r"\bchapter\s+(\d{1,3}|[ivxlc]{1,8})\b", q)
    if not m:
        return None

    token = m.group(1)
    if token.isdigit():
        n = int(token)
    else:
        n = _roman_to_int(token.upper())
        if _int_to_roman(n) != token.upper():
            return None
    if not 1 <= n <= 199:
        return None
    return _int_to_roman(n)


def chapter_variants(label: str) -> list[str]:
    """Every section_title spelling we expect Gutenberg texts to use for a
    chapter, e.g. "CHAPTER XVII.", "CHAPTER XVII", "Chapter 17"."""
    if label == "THE LAST":
        bases = ["THE LAST", "the Last", "the last"]
    else:
        bases = [label, str(_roman_to_int(label))]
    return [
        f"{prefix} {base}{suffix}"
        for prefix in ("CHAPTER", "Chapter")
        for base in bases
        for suffix in ("", ".")
    ]


def route_question(question: str, mode: str, known_literature_titles: list[str]) -> RouteDecision:
    """
    mode: "literature" or "life_process" (selected by the user in the UI).
    known_literature_titles: the selected author's literature work titles,
        used only when mode == "literature".
    """
    if is_meta_question(question):
        return RouteDecision(scope="meta", work_title=None)

    if mode == "life_process":
        return RouteDecision(scope="life_process", work_title=None)

    matched_title = _best_title_match(question, known_literature_titles)
    if matched_title:
        # A chapter number only means something once we know which book.
        return RouteDecision(
            scope="specific_work",
            work_title=matched_title,
            chapter=parse_chapter(question),
        )

    return RouteDecision(scope="oeuvre", work_title=None)