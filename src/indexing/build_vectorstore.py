"""Embeds all chunks and persists them into a single Chroma collection.

We use ONE collection for every author/mode, and rely on metadata filters
(author_key, mode, work_title) at query time to enforce the retrieval
boundaries from the PRD. This scales better than one collection per
author: adding Chekhov or Eliot later means re-running this script, not
restructuring storage.

Inserts are done in BATCHES, not all at once. Chroma's embedded
(sqlite-backed) client enforces a hard per-call maximum batch size
(computed from SQLite's own variable limit -- it showed up as 5,461 in
testing, but treat that as implementation-dependent, not a fixed
constant). A corpus large enough -- which happened the moment multiple
story collections and six Letters volumes were added -- can produce more
chunks than that in one go, and a single `Chroma.from_documents()` call
fails outright once the corpus crosses that line. This will only get more
likely as more authors and works are added, so it's handled generally
here rather than patched around once.
"""
from __future__ import annotations

from langchain_chroma import Chroma

from src.config import CONFIG, project_path
from src.indexing.chunker import build_documents, load_corpus_records
from src.indexing.embeddings import get_embeddings

# Comfortably under the ~5,461 ceiling observed in practice, with headroom
# for that internal limit to vary slightly across Chroma/SQLite versions.
INSERT_BATCH_SIZE = 4000


def build_vectorstore() -> None:
    corpus_path = project_path(
        CONFIG["paths"].get("deduped_corpus_file", CONFIG["paths"]["corpus_file"])
    )
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found. Run build_corpus.py and dedupe_corpus.py "
            f"(via scripts/02_build_index.py) first."
        )

    records = load_corpus_records(corpus_path)
    print(f"Loaded {len(records)} section records.")

    documents = build_documents(records)
    print(f"Split into {len(documents)} chunks "
          f"(chunk_size={CONFIG['chunking']['chunk_size']}, "
          f"overlap={CONFIG['chunking']['chunk_overlap']}).")

    embeddings = get_embeddings()
    persist_dir = str(project_path(CONFIG["vectorstore"]["persist_dir"]))
    collection_name = CONFIG["vectorstore"]["collection_name"]

    print(f"Embedding + persisting to {persist_dir} in batches of {INSERT_BATCH_SIZE} "
          f"... (this can take a few minutes on CPU)")

    # Create (or connect to) the collection empty first, then add in batches,
    # rather than Chroma.from_documents() which inserts everything in one
    # call and is exactly what hits the batch-size ceiling on a large corpus.
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    total = len(documents)
    for start in range(0, total, INSERT_BATCH_SIZE):
        batch = documents[start:start + INSERT_BATCH_SIZE]
        vectorstore.add_documents(batch)
        end = min(start + INSERT_BATCH_SIZE, total)
        print(f"  inserted {end}/{total} chunks")

    print("Done. Vector store is ready.")


if __name__ == "__main__":
    build_vectorstore()