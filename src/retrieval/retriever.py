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


def _alias_terms(question: str, author_key: str, decision: RouteDecision) -> list[str]:
    """Alias terms whose trigger words appear in the question (whole-word,
    case-insensitive), limited to groups that apply to the current mode."""
    mode = "life_process" if decision.scope == "life_process" else "literature"
    groups = CONFIG["authors"].get(author_key, {}).get("retrieval_aliases") or []
    lowered = question.lower()

    terms: list[str] = []
    for group in groups:
        if group.get("mode") not in (None, mode):
            continue
        triggered = any(
            re.search(r"\b" + re.escape(str(trigger).lower()) + r"\b", lowered)
            for trigger in group.get("triggers", [])
        )
        if triggered:
            for term in group.get("terms", []):
                if term not in terms:
                    terms.append(term)
    return terms


def _where_document(terms: list[str]) -> dict:
    # Chroma's $or needs at least two clauses.
    if len(terms) == 1:
        return {"$contains": terms[0]}
    return {"$or": [{"$contains": term} for term in terms]}


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

    terms = _alias_terms(question, author_key, decision)
    if terms:
        retrieval_cfg = CONFIG["retrieval"]
        try:
            alias_results = vectorstore.similarity_search_with_score(
                query=question,
                k=retrieval_cfg.get("alias_top_k", 5),
                filter=_build_filter(author_key, decision),
                where_document=_where_document(terms),
            )
        except Exception as exc:  # never let the extra search break an answer
            print(f"[retriever] alias search skipped: {exc}")
            alias_results = []
        limit = max(top_k, retrieval_cfg.get("max_results_with_aliases", 8))
        results = _merge(alias_results, results, limit)

    return results


def filter_by_relevance(scored_docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    max_distance = CONFIG["retrieval"]["max_distance"]
    return [(doc, score) for doc, score in scored_docs if score <= max_distance]