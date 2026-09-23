"""Ties router + retriever + LLM together into one grounded answer call.

This is the piece that enforces PRD Section 2's "Primary-source grounded"
and "Transparent" principles and Section 6's "Grounded generation",
"Source visibility", and "Safe fallback" requirements all in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.generation.llm import get_llm
from src.retrieval.retriever import filter_by_relevance, retrieve
from src.retrieval.router import RouteDecision, route_question

SYSTEM_PROMPT = """You are LiteraView AI, a research assistant that helps students engage \
more deeply with an author's primary-source texts. You are having a conversation about \
{author_display}.

Hard rules, no exceptions:
1. Answer ONLY using the numbered SOURCE PASSAGES below. Do not use outside knowledge about \
the author, the work, or literary history, even if you believe it to be true.
2. Every substantive claim must be traceable to at least one numbered source passage. Refer \
to passages by their number, e.g. "(Source 2)".
3. If the source passages do not contain enough information to answer the question, say so \
plainly — something like "The retrieved passages don't cover that" — instead of filling the \
gap from general knowledge. Never invent quotes, plot details, or biographical facts.
4. You are extending close reading of the primary text, not replacing it — where useful, \
point the student back to the specific chapter/section for further reading, rather than \
just summarizing it away.
"""

NO_RELEVANT_CONTEXT_MESSAGE = (
    "I don't have any passages in the approved corpus for this author/mode that are "
    "relevant enough to answer that reliably. Try rephrasing, or ask about a specific "
    "work or theme covered in the corpus."
)


@dataclass
class RagAnswer:
    answer: str
    scope: str
    work_title: str | None
    grounded: bool
    sources: list[dict] = field(default_factory=list)


def _format_context(scored_docs) -> str:
    blocks = []
    for i, (doc, _distance) in enumerate(scored_docs, start=1):
        meta = doc.metadata
        header = f"[Source {i}] {meta['work_title']} — {meta['section_title']}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _sources_payload(scored_docs) -> list[dict]:
    return [
        {
            "index": i,
            "work_title": doc.metadata["work_title"],
            "section_title": doc.metadata["section_title"],
            "source_ref": doc.metadata["source_ref"],
            "snippet": doc.page_content[:300].strip() + ("…" if len(doc.page_content) > 300 else ""),
            "distance": round(distance, 4),
        }
        for i, (doc, distance) in enumerate(scored_docs, start=1)
    ]


def answer_question(
    question: str,
    author_key: str,
    author_display: str,
    mode: str,
    known_literature_titles: list[str],
) -> RagAnswer:
    decision: RouteDecision = route_question(question, mode, known_literature_titles)

    scored_docs = retrieve(question, author_key, decision)
    relevant_docs = filter_by_relevance(scored_docs)

    if not relevant_docs:
        return RagAnswer(
            answer=NO_RELEVANT_CONTEXT_MESSAGE,
            scope=decision.scope,
            work_title=decision.work_title,
            grounded=False,
            sources=[],
        )

    context = _format_context(relevant_docs)
    system = SYSTEM_PROMPT.format(author_display=author_display)
    human = f"SOURCE PASSAGES:\n\n{context}\n\nSTUDENT QUESTION: {question}"

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])

    return RagAnswer(
        answer=response.content,
        scope=decision.scope,
        work_title=decision.work_title,
        grounded=True,
        sources=_sources_payload(relevant_docs),
    )
