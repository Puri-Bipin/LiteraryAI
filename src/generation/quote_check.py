"""Checks that quotations in an answer really appear in the sources.

Why this exists (Oct 2026): an answer about Twain's wife's death quoted the
letter as "...we were holding the oxygen pipe to her mouth..." The letter
says "they were holding". A language model will quietly tidy a quotation
(change a pronoun, drop a word) while still wrapping it in quotation marks,
and for a tool whose first requirement is that the texts be entirely
accurate, that is the kind of slip a close reader catches. Asking the model
to be careful helps but cannot guarantee it, so this check is deterministic:
every quoted passage is looked up, word for word, in the text the model was
shown. Anything that cannot be found is flagged for the student to verify.

Deliberately forgiving about things that are not wording: capitalisation,
punctuation, curly vs straight quotes, dashes, line breaks, and the editorial
marks "..." and [brackets]. Each stretch of quoted words between such marks
is checked on its own. Quotations shorter than MIN_WORDS words (a name, a
title, a single word) are not checked.
"""
from __future__ import annotations

import re

MIN_WORDS = 3

_QUOTED_RE = re.compile(r'"([^"]{6,}?)"|\u201c([^\u201d]{6,}?)\u201d', re.DOTALL)
# Ellipses and [bracketed editorial insertions] split a quotation into stretches.
_BREAK_RE = re.compile(r"\u2026|\.(?:\s?\.){2,}|\[[^\]]*\]")
_DASH_RE = re.compile(r"[\u2013\u2014]|-{2,}")
_LONE_APOSTROPHE_RE = re.compile(r"(?<![a-z])'|'(?![a-z])")


def normalize(text: str) -> str:
    """Lower-case, straighten quotes, and strip punctuation, so only the words are compared."""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = _DASH_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = _LONE_APOSTROPHE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_unmatched_quotes(answer: str, seen_text: str, min_words: int = MIN_WORDS) -> list[str]:
    """Quoted passages in `answer` that do not appear word for word in `seen_text`.

    `seen_text` should be everything the model was shown (the source passages,
    plus the student's question and the background notes, which it may also quote).
    """
    haystack = f" {normalize(seen_text)} "
    flagged: list[str] = []
    for match in _QUOTED_RE.finditer(answer):
        quoted = (match.group(1) or match.group(2) or "").strip()
        for stretch in _BREAK_RE.split(quoted):
            words = normalize(stretch)
            if len(words.split()) >= min_words and f" {words} " not in haystack:
                if quoted not in flagged:
                    flagged.append(quoted)
                break
    return flagged


def quote_check_note(flagged: list[str], limit: int = 3, width: int = 140) -> str:
    """A short note to append to an answer; empty when nothing was flagged."""
    if not flagged:
        return ""
    shown = []
    for quoted in flagged[:limit]:
        text = " ".join(quoted.split())
        shown.append("\u201c" + (text if len(text) <= width else text[: width - 1].rstrip() + "\u2026") + "\u201d")
    more = f" (and {len(flagged) - limit} more)" if len(flagged) > limit else ""
    return (
        "\n\n*Quotation check: this wording could not be matched word for word to the "
        "sources shown, so please verify it against the original text before relying on it: "
        + "; ".join(shown) + more + ".*"
    )