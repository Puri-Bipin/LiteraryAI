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

Title matching (rewritten Oct 2026): a work is matched when its full title
or a curator-listed alias (config.yaml `aliases`, e.g. "Huck Finn") appears
as a phrase in the question, or when at least two of its DISTINCTIVE title
words appear (typo-tolerant, ignoring filler such as "the", "adventures",
"new", "old"). History worth knowing: the first version compared the whole
question to the title letter-by-letter, which missed "What books appear on
the table in Chapter XVII of Huckleberry Finn?" (0.479). A second version
compared sliding windows and fixed that, but scored "tell me about the
jumping" at 0.520 against "the prince and the pauper" -- shared filler
words, not title content -- so "Tell me about the jumping frog story" was
wrongly scoped to The Prince and the Pauper and answered "not covered".
Letter-similarity cannot tell filler from content, so it is gone.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from src.config import CONFIG


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
# Words that appear in many titles and say nothing about WHICH work is meant.
_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "and", "in", "on", "to", "for", "with", "by",
    "adventures", "new", "old", "complete", "part",
}


def _aliases_for(title: str) -> list[str]:
    """Curator-listed alternative names for a work (config.yaml `aliases`)."""
    for author_cfg in CONFIG.get("authors", {}).values():
        for key in ("literature", "life_process"):
            for work in author_cfg.get(key) or []:
                if work.get("title") == title:
                    return list(work.get("aliases") or [])
    return []


def _contains_phrase(norm_question: str, norm_phrase: str) -> bool:
    return bool(norm_phrase) and re.search(r"\b" + re.escape(norm_phrase) + r"\b", norm_question) is not None


def _content_words(norm_title: str) -> list[str]:
    return [w for w in norm_title.split() if w not in _TITLE_STOPWORDS and not w.isdigit()]


def _word_present(word: str, question_words: list[str]) -> bool:
    """Exact match, or a close typo (e.g. "huckelberry", "fin")."""
    for q in question_words:
        if q == word:
            return True
        if len(q) >= 3 and len(word) >= 3 and difflib.SequenceMatcher(None, q, word).ratio() >= 0.85:
            return True
    return False


def _best_title_match(question: str, known_titles: list[str], min_fraction: float = 0.66) -> str | None:
    """Find the work a question names, or None if it names none.

    1. Full title or a curated alias appears as a whole phrase.
    2. Otherwise, at least two distinctive title words appear and they make
       up at least `min_fraction` of the title's distinctive words.
    A question that merely shares filler words with a title never matches.
    """
    norm_question = _normalize(question)
    question_words = norm_question.split()

    for title in known_titles:
        for phrase in [title] + _aliases_for(title):
            if _contains_phrase(norm_question, _normalize(phrase)):
                return title

    best_title, best_fraction = None, 0.0
    for title in known_titles:
        words = _content_words(_normalize(title))
        if len(words) < 2:
            continue
        hits = sum(1 for w in words if _word_present(w, question_words))
        fraction = hits / len(words)
        if hits >= 2 and fraction >= min_fraction and fraction > best_fraction:
            best_title, best_fraction = title, fraction
    return best_title


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