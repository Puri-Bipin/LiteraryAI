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

Literal-phrase search (Oct 2026): capitalised terms in the question ("Model
Boy", "Aunt Polly", "Smiley") are also searched for word for word. The
Model Boy passage in Chapter V is mostly about a church service, so by
meaning alone it lost every slot to an illustration list that repeats the
caption; a literal search finds it. See src/retrieval/keyphrases.py.

A question can carry SEVERAL topic words ("feel" + "death"); each triggered
topic group must match, so "How did Twain feel about the death of his wife?"
asks for a chunk with a name for her AND a death word AND a reaction word
("our hearts broke", "I wish I were with Livy"). That singled out the Howells
letter, which a name+death search alone ranked below Paine's narration.
Strictest combination first, then progressively looser ones fill any gaps.

Relevance gate: scoped searches (a named work/chapter, or an alias match)
use the looser `max_distance_scoped`; see config.yaml for why.
"""
from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CONFIG, project_path
from src.indexing.embeddings import get_embeddings
from src.retrieval.keyphrases import extract_key_phrases, phrase_variants
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
        titles = decision.work_titles or [decision.work_title]
        clauses.append(
            {"work_title": titles[0]} if len(titles) == 1 else {"work_title": {"$in": titles}}
        )
        if decision.chapter and len(titles) == 1:
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


def _alias_terms(question: str, author_key: str, decision: RouteDecision) -> tuple[list[str], list[list[str]]]:
    """(entity_terms, topic_groups) whose trigger words appear in the question
    (whole-word, case-insensitive), limited to groups for the current mode.

    topic_groups is one term list per triggered topic group, in config order.
    Topic terms are only returned when at least one entity group also fired:
    "death" on its own must not turn into a text filter."""
    mode = "life_process" if decision.scope == "life_process" else "literature"
    groups = CONFIG["authors"].get(author_key, {}).get("retrieval_aliases") or []
    lowered = question.lower()

    entity_terms: list[str] = []
    topic_groups: list[list[str]] = []
    for group in groups:
        if group.get("mode") not in (None, mode):
            continue
        triggered = any(
            re.search(r"\b" + re.escape(str(trigger).lower()) + r"\b", lowered)
            for trigger in group.get("triggers", [])
        )
        if not triggered:
            continue
        if group.get("role") == "topic":
            terms = list(group.get("terms", []))
            if terms:
                topic_groups.append(terms)
        else:
            for term in group.get("terms", []):
                if term not in entity_terms:
                    entity_terms.append(term)

    if not entity_terms:
        return [], []
    return entity_terms, topic_groups


def _any_of(terms: list[str]) -> dict:
    # Chroma's $or needs at least two clauses.
    if len(terms) == 1:
        return {"$contains": terms[0]}
    return {"$or": [{"$contains": term} for term in terms]}


def _where_document(entity_terms: list[str], topic_groups: list[list[str]] | None = None) -> dict:
    """A chunk must contain one entity term and one term from EACH topic group given."""
    parts = [_any_of(entity_terms)] + [_any_of(group) for group in (topic_groups or []) if group]
    return parts[0] if len(parts) == 1 else {"$and": parts}


def _ignore_words(author_key: str) -> set[str]:
    """Words that never count as a 'named thing': the author's name and every
    word in a work's title or alias ("Tom", "Sawyer", "Huckleberry", "Finn")."""
    author_cfg = CONFIG["authors"].get(author_key, {})
    words = set(re.findall(r"[a-z']+", str(author_cfg.get("display_name", "")).lower()))
    for key in ("literature", "life_process"):
        for work in author_cfg.get(key) or []:
            for phrase in [work.get("title", "")] + list(work.get("aliases") or []):
                for word in re.findall(r"[a-z']+", phrase.lower()):
                    words.add(re.sub(r"['\u2019]s?$", "", word))
    return words


def _keyword_search(vectorstore, question: str, author_key: str, decision: RouteDecision) -> list:
    """Chunks that contain a capitalised term from the question word for word,
    ranked by meaning within that set. Tagged so the relevance gate treats
    them as scoped. Returns [] on any problem."""
    phrases = extract_key_phrases(question, _ignore_words(author_key))
    if not phrases:
        return []
    try:
        found = vectorstore.similarity_search_with_score(
            query=question,
            k=CONFIG["retrieval"].get("keyword_top_k", 6),
            filter=_build_filter(author_key, decision),
            where_document=_any_of(phrase_variants(phrases)),
        )
    except Exception as exc:  # never let the extra search break an answer
        print(f"[retriever] keyword search skipped: {exc}")
        return []
    for doc, _score in found:
        doc.metadata["_via_alias"] = True
    return found


NEIGHBOUR_HITS = 2  # expand only the best few hits, to keep the context short


def _neighbours(vectorstore, hits: list) -> list:
    """The chunks immediately before and after the best alias/keyword hits.

    A hit often lands mid-passage: "she was dead" is in one chunk and Twain's
    own reaction ("How sweet she was in death...") is in the next, where he
    says "she" rather than naming her, so a name-based search cannot match it.
    Same section only; each neighbour inherits its hit's distance. Returns []
    on any problem."""
    out: list = []
    for doc, score in hits[:NEIGHBOUR_HITS]:
        meta = doc.metadata
        index = meta.get("chunk_index")
        if not isinstance(index, int) or "work_title" not in meta or "section_title" not in meta:
            continue
        try:
            got = vectorstore.get(
                where={"$and": [
                    {"work_title": meta["work_title"]},
                    {"section_title": meta["section_title"]},
                    {"chunk_index": {"$in": [index - 1, index + 1]}},
                ]},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            print(f"[retriever] neighbour lookup skipped: {exc}")
            continue
        for text, neighbour_meta in zip(got.get("documents") or [], got.get("metadatas") or []):
            neighbour = Document(page_content=text, metadata={**neighbour_meta, "_via_alias": True})
            out.append((neighbour, score))
    return out


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
                  entity_terms: list[str], topic_groups: list[list[str]]) -> list:
    """Alias-matched chunks, tagged so the relevance gate can treat them as scoped.

    Tries the strictest combination first (name AND every topic), then drops
    topics one at a time, and finally name-only, topping up until alias_top_k
    chunks are found."""
    k = CONFIG["retrieval"].get("alias_top_k", 8)
    where = _build_filter(author_key, decision)

    attempts = [_where_document(entity_terms, topic_groups[:n]) for n in range(len(topic_groups), 0, -1)]
    attempts.append(_where_document(entity_terms))

    found: list = []
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

    extra: list = []
    entity_terms, topic_groups = _alias_terms(question, author_key, decision)
    if entity_terms:
        extra += _alias_search(vectorstore, question, author_key, decision, entity_terms, topic_groups)
    extra += _keyword_search(vectorstore, question, author_key, decision)

    if extra:
        extra = extra + _neighbours(vectorstore, extra)
        limit = max(top_k, CONFIG["retrieval"].get("max_results_with_aliases", 12))
        results = _merge(extra, results, limit)

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