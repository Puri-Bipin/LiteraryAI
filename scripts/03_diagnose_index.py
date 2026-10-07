"""Diagnostic: inspect what's actually in the persisted corpus + vector store.

Run this any time retrieval results look wrong or empty for content you
know should be there. It answers the most basic question first -- is the
data actually indexed -- before anyone starts tuning retrieval parameters
based on guesses.

Usage (from repo root):
    python scripts/03_diagnose_index.py
    python scripts/03_diagnose_index.py "your test question here"
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIG, project_path


def check_deduped_corpus():
    path = project_path(CONFIG["paths"].get("deduped_corpus_file", CONFIG["paths"]["corpus_file"]))
    print(f"=== corpus file on disk: {path} ===")
    if not path.exists():
        print("  FILE DOES NOT EXIST. Run scripts/02_build_index.py first.\n")
        return

    counts = Counter()
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts[(rec["author_key"], rec["mode"], rec["work_title"])] += 1
            total += 1

    print(f"  {total} total section records\n")
    for (author_key, mode, work_title), count in sorted(counts.items()):
        print(f"  {author_key:15s} {mode:14s} {work_title:48s} {count:5d} sections")
    print()


def check_vectorstore():
    print("=== Chroma vector store (what's actually embedded + searchable) ===")
    from src.retrieval.retriever import get_vectorstore

    vs = get_vectorstore()
    collection = vs._collection
    total = collection.count()
    print(f"  Total vectors in collection: {total}")
    if total == 0:
        print("  COLLECTION IS EMPTY. The index was never built, or was built to a different path than the app is reading from.\n")
        return

    sample = collection.get(limit=min(total, 20000), include=["metadatas"])
    counts = Counter()
    for meta in sample["metadatas"]:
        counts[(meta.get("author_key"), meta.get("mode"), meta.get("work_title"))] += 1

    print(f"  Chunk counts by work (from {len(sample['metadatas'])} vectors sampled):")
    for key, count in sorted(counts.items()):
        print(f"    {key[0]:15s} {key[1]:14s} {str(key[2]):48s} {count:5d} chunks")
    print()


def test_raw_query(question: str):
    print(f"=== Raw similarity search, NO metadata filter, for: {question!r} ===")
    from src.retrieval.retriever import get_vectorstore

    vs = get_vectorstore()
    results = vs.similarity_search_with_score(question, k=8)
    if not results:
        print("  No results at all -- the collection itself may be empty (see above).")
        return
    for doc, score in results:
        meta = doc.metadata
        print(f"  distance={score:.4f}  work_title={meta.get('work_title')!r}  section={meta.get('section_title')!r}")
    print()


def test_filtered_query(question: str, author_key: str, mode: str):
    """Runs the EXACT same path the live app runs: route_question -> retrieve
    -> filter_by_relevance. This is the real test -- the raw query above can
    be misleading on its own, since a title-as-proper-noun question often
    matches passages that MENTION a book's title more strongly than the
    book's own prose (which rarely repeats its own title), even when the
    metadata filter would correctly restrict the search to just that book.
    """
    print(f"=== Filtered search (mirrors the live app exactly) for: {question!r} ===")
    from src.retrieval.router import route_question
    from src.retrieval.retriever import retrieve, filter_by_relevance, _build_filter, _top_k_for
    from src.config import CONFIG

    known_titles = [w["title"] for w in CONFIG["authors"][author_key].get("literature", [])]
    decision = route_question(question, mode, known_titles)
    print(f"  router decision: scope={decision.scope!r}  work_title={decision.work_title!r}")
    print(f"  metadata filter used: {_build_filter(author_key, decision)}")
    print(f"  top_k used: {_top_k_for(decision)}")

    scored = retrieve(question, author_key, decision)
    print(f"  {len(scored)} results from the FILTERED search:")
    for doc, score in scored:
        meta = doc.metadata
        print(f"    distance={score:.4f}  work_title={meta.get('work_title')!r}  section={meta.get('section_title')!r}")

    survivors = filter_by_relevance(scored)
    max_distance = CONFIG["retrieval"]["max_distance"]
    print(f"  after filter_by_relevance (max_distance={max_distance}): {len(survivors)} survive")
    if not survivors and scored:
        best = min(d for _, d in scored)
        print(f"  >>> Best available distance was {best:.4f} -- threshold rejected it. "
              f"This means real content WAS found but max_distance is cutting it off.")
    print()


if __name__ == "__main__":
    check_deduped_corpus()
    check_vectorstore()
    question = sys.argv[1] if len(sys.argv) > 1 else "Tell me more about the story of Huckleberry Finn"
    test_raw_query(question)
    test_filtered_query(question, author_key="mark_twain", mode="literature")