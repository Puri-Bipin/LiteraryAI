"""Implements PRD Section 3 Step 3-4: "System identifies whether the
question concerns a specific work, the oeuvre, or life/process" and then
"Retrieval is restricted to the correct dataset boundary."

The author and mode are already chosen by the user via the UI (Step 1-2).
This router has two jobs:

1. Detect "meta" / navigational questions -- "what works are available?",
   "what can I ask about?" -- which should be answered directly from
   config.yaml's own list of titles, not routed through retrieval or the
   LLM at all. (Added Oct 2026 after testing found the chatbot answering
   "what literary works are available?" as if it were a content question,
   listing books mentioned INSIDE Twain's stories rather than the actual
   corpus.)
2. Inside "Literature" mode, detect whether the question names one
   specific work (-> restrict retrieval to just that work) or reads as a
   cross-work question (-> retrieve across the author's whole approved
   literary corpus).

Both are done with plain string matching / keyword heuristics, deliberately
NOT an LLM call: free, instant, deterministic, and testable -- and it keeps
limited LLM free-tier credits for generation, where they're actually needed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass
class RouteDecision:
    scope: str              # "meta" | "specific_work" | "oeuvre" | "life_process"
    work_title: str | None  # set only when scope == "specific_work"


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# --- Meta / navigational question detection ---------------------------------
# A question is treated as "meta" if it combines an availability word
# ("available", "have", "offer", "cover", "discuss", "include") with a
# corpus noun ("works", "texts", "books", "corpus", "sources", "titles").
# This is a heuristic, not exhaustive -- tune these sets if real usage
# surfaces phrasings that should (or shouldn't) be caught.
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
def _best_title_match(question: str, known_titles: list[str], threshold: float = 0.5) -> str | None:
    """Find a known work title referenced in the question.

    Checks for substantial substring containment first (handles near-exact
    title mentions reliably), then falls back to difflib similarity ratio
    against each title.
    """
    norm_question = _normalize(question)
    norm_titles = {title: _normalize(title) for title in known_titles}

    for title, norm_title in norm_titles.items():
        if norm_title in norm_question:
            return title

    best_title, best_score = None, 0.0
    for title, norm_title in norm_titles.items():
        score = difflib.SequenceMatcher(None, norm_question, norm_title).ratio()
        if score > best_score:
            best_title, best_score = title, score

    return best_title if best_score >= threshold else None


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
        return RouteDecision(scope="specific_work", work_title=matched_title)

    return RouteDecision(scope="oeuvre", work_title=None)