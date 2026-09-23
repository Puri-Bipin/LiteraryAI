"""Implements PRD Section 3 Step 3-4: "System identifies whether the
question concerns a specific work, the oeuvre, or life/process" and then
"Retrieval is restricted to the correct dataset boundary."

The author and mode are already chosen by the user via the UI (Step 1-2),
so this router's only real job — inside "Literature" mode — is to detect
whether the question names one specific work (-> restrict retrieval to
just that work) or reads as a cross-work question (-> retrieve across the
author's whole approved literary corpus).

This is done with plain fuzzy string matching against the known work
titles, deliberately *not* an LLM call: it's fast, free, deterministic,
and testable — and it keeps your limited LLM free-tier credits for the
generation step where they're actually needed.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass


@dataclass
class RouteDecision:
    scope: str            # "specific_work" | "oeuvre" | "life_process"
    work_title: str | None  # set only when scope == "specific_work"


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def _best_title_match(question: str, known_titles: list[str], threshold: float = 0.5) -> str | None:
    """Find a known work title referenced in the question.

    Checks for substantial substring containment first (handles "Huck
    Finn" style shorthand poorly, but catches exact/near-exact title
    mentions reliably), then falls back to difflib similarity ratio on a
    sliding window of the question against each title.
    """
    norm_question = _normalize(question)
    norm_titles = {title: _normalize(title) for title in known_titles}

    # 1) direct substring match (title appears verbatim, or vice versa for short titles)
    for title, norm_title in norm_titles.items():
        if norm_title in norm_question:
            return title

    # 2) fuzzy match against the whole question string
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
    if mode == "life_process":
        return RouteDecision(scope="life_process", work_title=None)

    matched_title = _best_title_match(question, known_literature_titles)
    if matched_title:
        return RouteDecision(scope="specific_work", work_title=matched_title)

    return RouteDecision(scope="oeuvre", work_title=None)
