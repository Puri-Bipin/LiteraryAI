"""Loads config.yaml + .env once and exposes typed accessors.

Every other module imports `CONFIG` and `PROJECT_ROOT` from here instead of
re-parsing the YAML, so there is exactly one source of truth for paths and
settings.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG: dict = yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    return PROJECT_ROOT / relative


def get_env(name: str, required: bool = True) -> str:
    value = os.getenv(name, "")
    if required and not value:
        raise RuntimeError(
            f"Missing environment variable '{name}'. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def iter_works(mode: str):
    """Yield (author_key, display_name, work_title, gutenberg_id) for a mode.

    mode is "literature" or "life_process".
    """
    for author_key, author_cfg in CONFIG["authors"].items():
        for work in author_cfg.get(mode, []):
            yield author_key, author_cfg["display_name"], work["title"], work["gutenberg_id"]
