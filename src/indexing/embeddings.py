"""Returns the embedding model configured in config.yaml.

Both build_vectorstore.py (write path) and retriever.py (read path) must
use the *same* embedding function, or similarity search will silently
return garbage — this module is the single place that decision is made.
"""
from __future__ import annotations

from src.config import CONFIG, get_env


def get_embeddings():
    provider = CONFIG["embedding"]["provider"]

    if provider == "local":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=CONFIG["embedding"]["local_model"])

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=CONFIG["embedding"]["gemini_model"],
            google_api_key=get_env("GOOGLE_API_KEY"),
        )

    raise ValueError(f"Unknown embedding provider in config.yaml: {provider!r}")
