"""Embeds all chunks and persists them into a single Chroma collection.

We use ONE collection for every author/mode, and rely on metadata filters
(author_key, mode, work_title) at query time to enforce the retrieval
boundaries from the PRD. This scales better than one collection per
author: adding Chekhov or Eliot later means re-running this script, not
restructuring storage.
"""
from __future__ import annotations

from langchain_chroma import Chroma

from src.config import CONFIG, project_path
from src.indexing.chunker import build_documents, load_corpus_records
from src.indexing.embeddings import get_embeddings


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

    print(f"Embedding + persisting to {persist_dir} ... (this can take a few minutes on CPU)")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=CONFIG["vectorstore"]["collection_name"],
        collection_metadata={"hnsw:space": "cosine"},
    )
    print("Done. Vector store is ready.")


if __name__ == "__main__":
    build_vectorstore()