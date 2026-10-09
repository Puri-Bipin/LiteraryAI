"""Builds a metadata-filtered similarity search over the shared Chroma
collection, scoped by the router's decision. This is what enforces
"session boundaries ... without cross-author contamination" and the three
routing rows in PRD Section 4.

Chapter-aware retrieval (added Oct 2026): when the router found a chapter
("Chapter XVII of Huckleberry Finn"), the search is further restricted to
chunks whose section_title is one of that chapter's known spellings. This
needs no re-indexing -- section_title is already stored as metadata. If a
chapter filter matches nothing (a book whose headings use an unexpected
format), we quietly fall back to a whole-book search rather than failing.
"""
from __future__ import annotations

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


def retrieve(question: str, author_key: str, decision: RouteDecision) -> list[tuple[Document, float]]:
    """Returns [(document, distance), ...] -- lower distance = more similar.

    May clear decision.chapter if the chapter filter matched nothing, so
    callers can tell whether chapter scoping actually took effect.
    """
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query=question, k=_top_k_for(decision), filter=_build_filter(author_key, decision)
    )

    if not results and decision.chapter:
        decision.chapter = None
        results = vectorstore.similarity_search_with_score(
            query=question, k=_top_k_for(decision), filter=_build_filter(author_key, decision)
        )

    return results


def filter_by_relevance(scored_docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    max_distance = CONFIG["retrieval"]["max_distance"]
    return [(doc, score) for doc, score in scored_docs if score <= max_distance]