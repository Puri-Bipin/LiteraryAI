"""Builds a metadata-filtered similarity search over the shared Chroma
collection, scoped by the router's decision. This is what enforces
"session boundaries ... without cross-author contamination" and the three
routing rows in PRD Section 4.

Chapter-aware retrieval (Oct 2026): when the router found a chapter
("Chapter XVII of Huckleberry Finn"), the search is further restricted to
chunks whose section_title is one of that chapter's known spellings. This
needs no re-indexing -- section_title is already stored as metadata. If a
chapter filter matches nothing (a book whose headings use an unexpected
format), we quietly fall back to a whole-book search rather than failing.

Alias retrieval (Oct 2026): a pure meaning-based search cannot connect the
word in a student's question to the name the source actually uses ("wife"
vs "Livy" / "Mrs. Clemens"), nor find a story whose passages never repeat
its title ("the jumping frog story" -> Smiley, Simon Wheeler). So when a
question contains a trigger word from the author's `retrieval_aliases`
(config.yaml), a SECOND search runs over only the chunks that literally
contain one of the alias terms, ranked by meaning within that small set.
Those results are merged in ahead of the normal ones. If the alias search
fails for any reason, the normal results are returned unchanged.

A question can also carry a TOPIC word ("death", "died"). Topic words are
used only alongside a name match, and then the alias search demands a chunk
containing BOTH ("Mrs. Clemens" AND "dead"). Without that, "How did Twain
feel about the death of his wife?" ranked ~170 chunks that mention Livy by
meaning alone and missed the one that says "she was dead".

Relevance gate: scoped searches (a named work/chapter, or an alias match)
use the looser `max_distance_scoped`; see config.yaml for why.
"""
from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CONFIG, project_path
from src.indexing.embeddings import get_embeddings
from src.retrieval.router import RouteDecision, chapter_variants

_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(project_path(CONFIG["vectorstore"]["persist_dir"])),
            embedding_function=get_embeddings(),
            collection_name=CONFIG["vectorstore"]["collection_name"],
        )
    return _vectorstore


def _build_filter(author_key: str, decision: RouteDecision) -> dict:
    clauses = [
        {"author_key": author_key},
        {"mode": "life_process" if decision.scope == "life_process" else "literature"},
    ]
    if decision.scope == "specific_work":
        clauses.append({"work_title": decision.work_title})
        if decision.chapter:
            clauses.append({"section_title": {"$in": chapter_variants(decision.chapter)}})
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _top_k_for(decision: RouteDecision) -> int:
    retrieval_cfg = CONFIG["retrieval"]
    # A whole chapter is only ~8-14 chunks, so when a chapter is named we
    # fetch enough to cover most of it.
    if decision.scope == "specific_work" and decision.chapter:
        return retrieval_cfg.get("top_k_chapter", 12)
    # A single-work search is already narrowed to one book, so it's cheap
    # and safe to search deeper than a cross-corpus search.
    if decision.scope == "specific_work":
        return retrieval_cfg.get("top_k_specific_work", retrieval_cfg["top_k"])
    return retrieval_cfg["top_k"]


def _alias_terms(question: str, author_key: str, decision: RouteDecision) -> tuple[list[str], list[str]]:
    """(entity_terms, topic_terms) whose trigger words appear in the question
    (whole-word, case-insensitive), limited to groups for the current mode.

    Topic terms are only returned when at least one entity group also fired:
    "death" on its own must not turn into a text filter."""
    mode = "life_process" if decision.scope == "life_process" else "literature"
    groups = CONFIG["authors"].get(author_key, {}).get("retrieval_aliases") or []
    lowered = question.lower()

    entity_terms: list[str] = []
    topic_terms: list[str] = []
    for group in groups:
        if group.get("mode") not in (None, mode):
            continue
        triggered = any(
            re.search(r"\b" + re.escape(str(trigger).lower()) + r"\b", lowered)
            for trigger in group.get("triggers", [])
        )
        if not triggered:
            continue
        bucket = topic_terms if group.get("role") == "topic" else entity_terms
        for term in group.get("terms", []):
            if term not in bucket:
                bucket.append(term)

    if not entity_terms:
        return [], []
    return entity_terms, topic_terms


def _any_of(terms: list[str]) -> dict:
    # Chroma's $or needs at least two clauses.
    if len(terms) == 1:
        return {"$contains": terms[0]}
    return {"$or": [{"$contains": term} for term in terms]}


def _where_document(entity_terms: list[str], topic_terms: list[str] | None = None) -> dict:
    """A chunk must contain one entity term and, if given, one topic term."""
    if topic_terms:
        return {"$and": [_any_of(entity_terms), _any_of(topic_terms)]}
    return _any_of(entity_terms)


def _chunk_key(doc: Document) -> tuple:
    meta = doc.metadata
    return (meta.get("work_title"), meta.get("section_title"), meta.get("chunk_index"), doc.page_content[:60])


def _merge(alias_results: list, normal_results: list, limit: int) -> list:
    """Alias matches first, then the normal results, without repeats."""
    merged, seen = [], set()
    for doc, score in list(alias_results) + list(normal_results):
        key = _chunk_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append((doc, score))
    return merged[:limit]


def _alias_search(vectorstore, question: str, author_key: str, decision: RouteDecision,
                  entity_terms: list[str], topic_terms: list[str]) -> list:
    """Alias-matched chunks, tagged so the relevance gate can treat them as scoped.

    With a topic word, try "name AND topic" first and top up with name-only
    matches if that finds fewer than asked for."""
    k = CONFIG["retrieval"].get("alias_top_k", 8)
    where = _build_filter(author_key, decision)

    found: list = []
    attempts = []
    if topic_terms:
        attempts.append(_where_document(entity_terms, topic_terms))
    attempts.append(_where_document(entity_terms))

    for where_document in attempts:
        try:
            batch = vectorstore.similarity_search_with_score(
                query=question, k=k, filter=where, where_document=where_document
            )
        except Exception as exc:  # never let the extra search break an answer
            print(f"[retriever] alias search skipped: {exc}")
            continue
        found = _merge(found, batch, k)
        if len(found) >= k:
            break

    for doc, _score in found:
        doc.metadata["_via_alias"] = True
    return found


def retrieve(question: str, author_key: str, decision: RouteDecision) -> list[tuple[Document, float]]:
    """Returns [(document, distance), ...] -- lower distance = more similar.

    May clear decision.chapter if the chapter filter matched nothing, so
    callers can tell whether chapter scoping actually took effect.
    """
    vectorstore = get_vectorstore()
    top_k = _top_k_for(decision)

    results = vectorstore.similarity_search_with_score(
        query=question, k=top_k, filter=_build_filter(author_key, decision)
    )

    if not results and decision.chapter:
        decision.chapter = None
        top_k = _top_k_for(decision)
        results = vectorstore.similarity_search_with_score(
            query=question, k=top_k, filter=_build_filter(author_key, decision)
        )

    entity_terms, topic_terms = _alias_terms(question, author_key, decision)
    if entity_terms:
        alias_results = _alias_search(vectorstore, question, author_key, decision, entity_terms, topic_terms)
        limit = max(top_k, CONFIG["retrieval"].get("max_results_with_aliases", 10))
        results = _merge(alias_results, results, limit)

    return results


def filter_by_relevance(
    scored_docs: list[tuple[Document, float]], decision: RouteDecision | None = None
) -> list[tuple[Document, float]]:
    """Drops results too far from the question to be useful.

    Scoped results (the question named a work/chapter, or the chunk was found
    by a keyword alias) get the looser `max_distance_scoped`; everything else
    uses `max_distance`."""
    cfg = CONFIG["retrieval"]
    strict = cfg["max_distance"]
    scoped = cfg.get("max_distance_scoped", strict)
    named_work = decision is not None and decision.scope == "specific_work"

    kept = []
    for doc, score in scored_docs:
        limit = scoped if (named_work or doc.metadata.get("_via_alias")) else strict
        if score <= limit:
            kept.append((doc, score))
    return kept