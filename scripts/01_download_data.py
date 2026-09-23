"""Step 1 of the pipeline: download the curated corpus from Project Gutenberg.

Usage (from repo root):
    python scripts/01_download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.download_gutenberg import download_all

if __name__ == "__main__":
    download_all()
