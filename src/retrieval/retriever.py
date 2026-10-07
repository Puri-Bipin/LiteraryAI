"""Builds a metadata-filtered similarity search over the shared Chroma
collection, scoped by the router's decision. This is what enforces
"session boundaries ... without cross-author contamination" and the three
routing rows in PRD Section 4.
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CONFIG, project_path
from src.indexing.embeddings import get_embeddings
from src.retrieval.router import RouteDecision

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
    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _top_k_for(decision: RouteDecision) -> int:
    # A specific-work search is already narrowed to one book, so it's cheap
    # and safe to search deeper -- helps surface buried, very specific
    # details (e.g. "what books sit on the table in Chapter XVII") that a
    # shallower top_k can miss in a broader pool.
    if decision.scope == "specific_work":
        return CONFIG["retrieval"].get("top_k_specific_work", CONFIG["retrieval"]["top_k"])
    return CONFIG["retrieval"]["top_k"]


def retrieve(question: str, author_key: str, decision: RouteDecision) -> list[tuple[Document, float]]:
    """Returns [(document, distance), ...] -- lower distance = more similar."""
    vectorstore = get_vectorstore()
    where = _build_filter(author_key, decision)
    top_k = _top_k_for(decision)

    return vectorstore.similarity_search_with_score(query=question, k=top_k, filter=where)


def filter_by_relevance(scored_docs: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    max_distance = CONFIG["retrieval"]["max_distance"]
    return [(doc, score) for doc, score in scored_docs if score <= max_distance]