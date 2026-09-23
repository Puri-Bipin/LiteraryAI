"""Chunks corpus.jsonl records into LangChain Documents.

We chunk *within* each section (chapter/act/essay) rather than across
section boundaries, so a chunk never silently blends text from two
chapters — that would break the "cite supporting passages" requirement,
since a citation needs to point at one coherent place in the source.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import CONFIG


def load_corpus_records(corpus_path: Path) -> list[dict]:
    records = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_documents(records: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunking"]["chunk_size"],
        chunk_overlap=CONFIG["chunking"]["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for record in records:
        chunks = splitter.split_text(record["text"])
        for i, chunk_text in enumerate(chunks):
            metadata = {
                "author_key": record["author_key"],
                "author_display": record["author_display"],
                "mode": record["mode"],
                "work_title": record["work_title"],
                "section_title": record["section_title"],
                "section_index": record["section_index"],
                "chunk_index": i,
                "source_ref": record["source_ref"],
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))

    return documents
