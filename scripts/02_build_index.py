"""Step 2 of the pipeline: clean raw text -> corpus.jsonl -> chunk -> embed
-> persist Chroma vector store.

Usage (from repo root):
    python scripts/02_build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.build_corpus import build_corpus
from src.ingestion.dedupe_corpus import dedupe
from src.indexing.build_vectorstore import build_vectorstore

if __name__ == "__main__":
    print("=== Building corpus.jsonl from raw downloads ===")
    build_corpus()

    print("\n=== Detecting and removing duplicate/near-duplicate sections ===")
    dedupe()

    print("\n=== Chunking + embedding + persisting vector store ===")
    build_vectorstore()

    print("\nAll done. Run: streamlit run src/app/streamlit_app.py")