"""Ties router + retriever + LLM together into one grounded answer call.

This is the piece that enforces PRD Section 2's "Primary-source grounded"
and "Transparent" principles and Section 6's "Grounded generation",
"Source visibility", and "Safe fallback" requirements all in one place.

Updated Oct 2026 after the first round of English-faculty testing surfaced
four concrete issues, each addressed below:

1. "What literary works are available?" was answered as a content
   question (listing books mentioned INSIDE Twain's stories) instead of a
   navigational one. -> scope == "meta" now answers directly from
   config.yaml, no retrieval/LLM involved at all.
2. Interpretive questions (theme, symbolism, "the major conflict") were
   sometimes presented as one single authoritative answer, which runs
   against the PRD's own "Pedagogy first" principle. -> Literature-mode
   questions now get an explicit "present evidence, offer one reading,
   invite the student's own interpretation" framing.
3. Family-relationship confusion in Life & Process answers ("mama" read as
   Twain's mother instead of his wife Olivia; a daughter called his
   "sister"). -> a short, curator-maintained reference glossary
   (config.yaml's life_process_context) is now included as labeled
   background context, separate from citable sources.
4. A tester found one earlier response used period-typical offensive
   language in the chatbot's OWN voice, versus a later response that
   correctly kept such language only inside a directly quoted excerpt.
   -> explicit prompt rule plus a post-generation safety-net filter
   (_apply_restricted_term_filter) that only touches the model's own
   narration, never the quoted source text shown in the Sources panel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import CONFIG
from src.generation.llm import get_llm
from src.retrieval.retriever import filter_by_relevance, retrieve
from src.retrieval.router import RouteDecision, route_question

BASE_RULES_TEMPLATE = """You are LiteraView AI, a research assistant that helps students engage \
more deeply with an author's primary-source texts. You are having a conversation about \
{author_display}.

Hard rules, no exceptions:
1. Answer ONLY using the numbered SOURCE PASSAGES below. Do not use outside knowledge about \
the author, the work, or literary history, even if you believe it to be true.
2. Every substantive claim must be directly and specifically supported by at least one numbered \
source passage -- refer to passages by number, e.g. "(Source 2)". If the best available evidence \
only loosely or indirectly supports a claim, say so explicitly (e.g. "the evidence here is \
indirect, but suggests...") rather than stating it as settled fact.
3. If the source passages do not contain enough information to answer the question, say so \
plainly instead of filling the gap from general knowledge. Never invent quotes, plot details, \
or biographical facts.
   If the passages stop partway through an episode, say that they stop there. Do NOT describe, \
summarize, or hint at what happens next, how it ends, or any twist -- not even a famous one \
you remember -- and do not mention events the passages do not contain, even to say they are \
"not quoted here".
4. You are extending close reading of the primary text, not replacing it -- where useful, point \
the student back to the specific chapter/section for further reading.
5. Never use offensive, slurring, or otherwise restricted language in your own voice or \
explanation. Such language may only appear when it occurs INSIDE a direct quotation copied from \
the source text (wrapped in quotation marks), and only when the surrounding discussion is \
clearly about the text's historical, racial, or social context -- never gratuitously, and never \
as part of your own narration.
"""

LITERATURE_FRAMING = """
Additional guidance for this Literature-mode question:
- For questions about theme, symbolism, character, or conflict, do NOT present a single \
interpretation as the definitive or only correct answer.
- Frame your reading as one possible interpretation supported by the evidence, e.g. "Looking at \
the following passages, one could argue that..."
- Present the supporting passage(s), then a brief synthesis.
- End by inviting the student's own reading, e.g. "What's your own take on this?" or "Does the \
evidence suggest a different reading to you?"
- This framing does NOT apply to purely factual questions (e.g. "what happens in Chapter 3", \
"how did Tom convince the boys to paint the fence") -- answer those directly and factually, \
without the interpretive hedge.
"""

LIFE_PROCESS_FRAMING = """
Additional guidance for this Life & Process question:
- You are discussing {author_display}'s own letters, autobiography, and essays -- be thorough. \
For a broad question (e.g. "tell me about his life"), cover multiple phases or aspects if the \
sources allow it, rather than resting on a single anecdote.
- Pay close attention to family relationships and nicknames; do not assume a term means what it \
would in general usage if the reference context below says otherwise.
{context_block}
"""

NO_RELEVANT_CONTEXT_MESSAGE = (
    "I don't have any passages in the approved corpus for this author/mode that are "
    "relevant enough to answer that reliably. Try rephrasing, or ask about a specific "
    "work or theme covered in the corpus."
)

# Matches a straight or curly double-quoted span so the restricted-term
# filter can skip anything the model is directly quoting from the source.
_QUOTE_SPAN_RE = re.compile(r'(".*?"|\u201c.*?\u201d)', re.DOTALL)


@dataclass
class RagAnswer:
    answer: str
    scope: str
    work_title: str | None
    grounded: bool
    sources: list[dict] = field(default_factory=list)
    chapter: str | None = None  # set when retrieval was restricted to one chapter


def _build_system_prompt(author_display: str, author_key: str, mode: str) -> str:
    parts = [BASE_RULES_TEMPLATE.format(author_display=author_display)]
    if mode == "literature":
        parts.append(LITERATURE_FRAMING)
    else:
        raw_context = CONFIG["authors"].get(author_key, {}).get("life_process_context", "")
        raw_context = (raw_context or "").strip()
        context_block = (
            f"Reference context about {author_display}'s family (curator-provided -- use ONLY "
            f"to correctly identify who people and nicknames refer to; this is NOT a source "
            f"citation and should never be the sole basis for a claim):\n{raw_context}"
            if raw_context else ""
        )
        parts.append(LIFE_PROCESS_FRAMING.format(author_display=author_display, context_block=context_block))
    return "\n".join(parts)


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


def _titles_for_mode(author_key: str, mode: str) -> list[str]:
    key = "literature" if mode == "literature" else "life_process"
    return [w["title"] for w in CONFIG["authors"].get(author_key, {}).get(key, [])]


def _meta_answer(mode: str, author_display: str, titles: list[str]) -> str:
    label = "literary works" if mode == "literature" else "letters/autobiography/essay sources"
    if not titles:
        return f"There are currently no {label} loaded for {author_display}."
    bullet_list = "\n".join(f"- {t}" for t in titles)
    return (
        f"Right now, you can ask about these {label} for {author_display}:\n\n"
        f"{bullet_list}\n\n"
        f"Feel free to ask about one of them by name, or ask a question that spans all of them."
    )


def _apply_restricted_term_filter(text: str, restricted_terms: list[str]) -> str:
    """Redacts restricted terms from the model's OWN narration only.

    Anything inside a quoted span (the model directly quoting the source
    text) is left untouched -- this is what lets a quoted historical slur
    stay accurate while stopping the model from using the same word in its
    own voice. The actual source excerpts shown in the UI's Sources panel
    are a completely separate code path and are never touched by this
    function at all.
    """
    cleaned_terms = [t.strip() for t in restricted_terms if t and t.strip() and t.strip() != "REPLACE_WITH_YOUR_TERM"]
    if not cleaned_terms:
        return text

    patterns = [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in cleaned_terms]

    parts = _QUOTE_SPAN_RE.split(text)
    for i in range(0, len(parts), 2):  # even indices = text outside quotes
        for pattern in patterns:
            parts[i] = pattern.sub("[restricted term omitted]", parts[i])
    return "".join(parts)


def answer_question(
    question: str,
    author_key: str,
    author_display: str,
    mode: str,
    known_literature_titles: list[str],
) -> RagAnswer:
    decision: RouteDecision = route_question(question, mode, known_literature_titles)

    if decision.scope == "meta":
        titles = _titles_for_mode(author_key, mode)
        return RagAnswer(
            answer=_meta_answer(mode, author_display, titles),
            scope="meta",
            work_title=None,
            grounded=True,
            sources=[],
        )

    scored_docs = retrieve(question, author_key, decision)

    # NOTE: retrieve() clears decision.chapter if the chapter filter matched
    # nothing, so check it AFTER the call.
    if decision.chapter:
        # The student named a chapter. The metadata filter already guarantees
        # these passages are the right ones, so don't let the similarity
        # threshold throw away a chapter the student explicitly asked for --
        # and present them in reading order, not similarity order.
        relevant_docs = sorted(scored_docs, key=lambda pair: pair[0].metadata.get("chunk_index", 0))
    else:
        relevant_docs = filter_by_relevance(scored_docs, decision)

    if not relevant_docs:
        return RagAnswer(
            answer=NO_RELEVANT_CONTEXT_MESSAGE,
            scope=decision.scope,
            work_title=decision.work_title,
            grounded=False,
            sources=[],
            chapter=decision.chapter,
        )

    context = _format_context(relevant_docs)
    system = _build_system_prompt(author_display, author_key, mode)
    chapter_note = (
        f"NOTE: The student asked specifically about Chapter {decision.chapter} of "
        f"{decision.work_title}. All passages below come from that chapter, in reading order.\n\n"
        if decision.chapter else ""
    )
    human = f"{chapter_note}SOURCE PASSAGES:\n\n{context}\n\nSTUDENT QUESTION: {question}"

    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])

    restricted_terms = CONFIG.get("generation", {}).get("restricted_terms", [])
    filtered_answer = _apply_restricted_term_filter(response.content, restricted_terms)

    return RagAnswer(
        answer=filtered_answer,
        scope=decision.scope,
        work_title=decision.work_title,
        grounded=True,
        sources=_sources_payload(relevant_docs),
        chapter=decision.chapter,
    )