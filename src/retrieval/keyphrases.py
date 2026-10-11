"""Finds the named things in a student's question ("Model Boy", "Aunt Polly",
"Smiley") so retrieval can look for them word for word.

Why this exists (Oct 2026): "Who is the Model Boy in The Adventures of Tom
Sawyer?" could not be answered. Chapter V says it outright -- "last of all
came the Model Boy, Willie Mufferson" -- but that chunk is mostly about a
church service, so by MEANING it sits far from a short question about a
"Model Boy", while an illustration list that merely repeats the caption
matched every slot. A meaning-based search cannot tell that a rare phrase in
the question is exactly the thing to look for. Searching for the literal
phrase can.

Deliberately simple and deterministic (no LLM): capitalised runs of words and
quoted phrases, minus question words, chapter references, the author's name,
and the names of the works themselves.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019\-]*")
_QUOTED_RE = re.compile(r"[\"\u201c]([^\"\u201d]{3,60})[\"\u201d]")
_POSSESSIVE_RE = re.compile(r"['\u2019]s?$", re.IGNORECASE)

_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    "tell", "give", "describe", "explain", "summarize", "summarise", "list",
    "name", "show", "is", "are", "was", "were", "do", "does", "did", "can",
    "could", "would", "should", "will", "has", "have", "had",
}
_NOT_A_NAME = {"chapter", "act", "scene"}
MAX_PHRASES = 3


def _is_capitalised(token: str) -> bool:
    # "Model", "Jim" yes; "I", "XVII", "USA" no (single letters, numerals, shouting).
    return len(token) >= 2 and token[0].isupper() and not token.isupper()


def _base(word: str) -> str:
    return _POSSESSIVE_RE.sub("", word.lower())


def extract_key_phrases(question: str, ignore_words: set[str]) -> list[str]:
    """Up to MAX_PHRASES literal phrases worth searching for, longest first.

    `ignore_words` are lower-case words that carry no information here (the
    author's name, words from work titles). Ignored words are trimmed off the
    ends of a run, so "Tom Sawyer's Aunt Polly" becomes "Aunt Polly".
    """
    tokens = list(_TOKEN_RE.finditer(question))
    runs: list[list[str]] = []
    current: list = []

    def flush():
        if current:
            runs.append([t.group(0) for t in current])
            current.clear()

    for index, match in enumerate(tokens):
        word = match.group(0)
        if not _is_capitalised(word):
            flush()
            continue
        if current and question[current[-1].end():match.start()] != " ":
            flush()  # punctuation or extra space between: a new run
        current.append(match)
    flush()

    candidates: list[str] = []

    for run in runs:
        words = list(run)
        # The first word of a question is capitalised by grammar, not because it is a name.
        if question.lstrip().startswith(words[0]) and _base(words[0]) in _QUESTION_WORDS:
            words = words[1:]
        while words and _base(words[0]) in ignore_words:
            words = words[1:]
        while words and _base(words[-1]) in ignore_words:
            words = words[:-1]
        if not words or _base(words[0]) in _NOT_A_NAME:
            continue
        phrase = " ".join(words)
        phrase = _POSSESSIVE_RE.sub("", phrase)
        if len(phrase) >= 3:
            candidates.append(phrase)

    for match in _QUOTED_RE.finditer(question):
        phrase = match.group(1).strip()
        if len(phrase) >= 3 and not all(_base(w) in ignore_words for w in phrase.split()):
            candidates.append(phrase)

    seen, unique = set(), []
    for phrase in candidates:
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            unique.append(phrase)

    # Multi-word phrases are more specific than single words; keep the best few.
    unique.sort(key=lambda p: -len(p.split()))
    return unique[:MAX_PHRASES]


def phrase_variants(phrases: list[str]) -> list[str]:
    """Spellings to look for. Chroma's text match is case-sensitive, and
    Gutenberg texts print captions in capitals ("THE MODEL BOY.") and
    apostrophes as curly quotes."""
    variants: list[str] = []
    for phrase in phrases:
        for form in (phrase, phrase.lower(), phrase.title(), phrase.upper()):
            for spelling in (form, form.replace("'", "\u2019")):
                if spelling not in variants:
                    variants.append(spelling)
    return variants