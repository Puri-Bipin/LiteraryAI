"""Returns the chat model configured in config.yaml.

Switching providers (e.g. Gemini free tier is rate-limited today, Groq is
faster for short prompts) is a one-line config.yaml change — nothing else
in the app needs to know which provider is behind `get_llm()`.
"""
from __future__ import annotations

from src.config import CONFIG, get_env


def get_llm():
    provider = CONFIG["llm"]["provider"]

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=CONFIG["llm"]["gemini_model"],
            google_api_key=get_env("GOOGLE_API_KEY"),
            temperature=CONFIG["llm"]["temperature"],
            max_output_tokens=CONFIG["llm"]["max_output_tokens"],
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=CONFIG["llm"]["groq_model"],
            api_key=get_env("GROQ_API_KEY"),
            temperature=CONFIG["llm"]["temperature"],
            max_tokens=CONFIG["llm"]["max_output_tokens"],
        )

    raise ValueError(f"Unknown llm provider in config.yaml: {provider!r}")
