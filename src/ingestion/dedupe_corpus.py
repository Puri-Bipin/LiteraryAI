"""Detects and removes duplicate/near-duplicate sections in corpus.jsonl.

Necessary because Gutenberg publishes many short-story collections that
overlap heavily: the same story can appear standalone AND inside two or
three different anthology volumes, sometimes under a different title.
Without this step, the vector store ends up with several redundant
copies of the same story, which skews retrieval (it looks like broader
"coverage" of a theme than actually exists) and pads the corpus with
noise.

Runs on the SECTION-level corpus.jsonl, before chunking -- comparing
whole sections (chapters/stories) rather than small chunks keeps the
comparison count manageable and matches how a human curator actually
thinks about "is this the same piece."

Two-pass detection, both restricted to comparisons within the same
author_key + mode (never cross-author, never literature vs life_process):

  1. EXACT duplicates -- a hash of the normalized text. Catches an
     identical reprint (the common case: two anthologies built from the
     same underlying Gutenberg source text).
  2. NEAR duplicates -- word-shingle Jaccard similarity. Catches the
     same story with minor formatting/editorial differences between
     editions, which an exact hash would miss.

Output:
  - data/processed/corpus.deduped.jsonl  (one row kept per duplicate group)
  - data/processed/dedup_report.jsonl    (every dropped section, which
    kept section it matched, and the similarity score -- for human review)

This does not silently decide and move on: it always writes the report,
so borderline near-duplicate calls stay visible and reviewable, the same
way manual curation was, just narrowed to the pairs actually flagged.
"""
from __future__ import annotations

import hashlib
import json
import re

from src.config import CONFIG, project_path, iter_works

DEFAULTS = {
    "shingle_size": 10,
    "jaccard_threshold": 0.6,
    "length_ratio_tolerance": 0.3,
}


def _cfg(key: str):
    return CONFIG.get("dedup", {}).get(key, DEFAULTS[key])


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _exact_hash(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def _shingles(normalized_text: str, k: int) -> set:
    words = normalized_text.split()
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a) + len(b) - inter
    return inter / union if union else 0.0


class _UnionFind:
    """Groups indices into duplicate clusters (a story can appear 3+ times)."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _canonical_priority() -> dict:
    """Maps (author_key, work_title) -> its position in config.yaml's listed
    order. Lower = earlier in the config = kept when a duplicate is found.
    Reorder config.yaml's work lists to change which edition wins."""
    priority = {}
    idx = 0
    for mode in ("literature", "life_process"):
        for author_key, _display, work_title, _gid in iter_works(mode):
            priority[(author_key, work_title)] = idx
            idx += 1
    return priority


def dedupe() -> None:
    shingle_size = _cfg("shingle_size")
    jaccard_threshold = _cfg("jaccard_threshold")
    length_ratio_tolerance = _cfg("length_ratio_tolerance")

    corpus_path = project_path(CONFIG["paths"]["corpus_file"])
    out_path = project_path(CONFIG["paths"].get("deduped_corpus_file", "data/processed/corpus.deduped.jsonl"))
    report_path = project_path(CONFIG["paths"].get("dedup_report_file", "data/processed/dedup_report.jsonl"))

    records = [json.loads(line) for line in open(corpus_path, "r", encoding="utf-8") if line.strip()]
    print(f"Loaded {len(records)} sections from {corpus_path}")

    normalized = [_normalize(r["text"]) for r in records]
    exact_hashes = [_exact_hash(n) for n in normalized]
    lengths = [len(n) for n in normalized]
    n = len(records)
    uf = _UnionFind(n)

    # --- Pass 1: exact duplicates -------------------------------------
    hash_groups: dict = {}
    for i, r in enumerate(records):
        key = (r["author_key"], r["mode"], exact_hashes[i])
        hash_groups.setdefault(key, []).append(i)
    for group in hash_groups.values():
        for j in group[1:]:
            uf.union(group[0], j)

    # --- Pass 2: near duplicates via shingle overlap -------------------
    by_author_mode: dict = {}
    for i, r in enumerate(records):
        by_author_mode.setdefault((r["author_key"], r["mode"]), []).append(i)

    shingle_cache: dict = {}

    def get_shingles(i: int) -> set:
        if i not in shingle_cache:
            shingle_cache[i] = _shingles(normalized[i], shingle_size)
        return shingle_cache[i]

    comparisons = 0
    for _, indices in by_author_mode.items():
        indices_sorted = sorted(indices, key=lambda i: lengths[i])
        for a_pos in range(len(indices_sorted)):
            i = indices_sorted[a_pos]
            if lengths[i] < shingle_size * 3:
                continue  # too short to fingerprint meaningfully
            for b_pos in range(a_pos + 1, len(indices_sorted)):
                j = indices_sorted[b_pos]
                if uf.find(i) == uf.find(j):
                    continue
                if lengths[j] == 0:
                    continue
                # lengths are sorted ascending, so once the ratio drops
                # below tolerance it only gets worse going forward -- stop
                if lengths[i] / lengths[j] < (1 - length_ratio_tolerance):
                    break
                comparisons += 1
                if _jaccard(get_shingles(i), get_shingles(j)) >= jaccard_threshold:
                    uf.union(i, j)

    print(f"Ran {comparisons} near-duplicate comparisons.")

    clusters: dict = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    priority = _canonical_priority()

    def rank(i: int) -> int:
        r = records[i]
        return priority.get((r["author_key"], r["work_title"]), 10_000)

    kept_indices = set()
    report_rows = []

    for members in clusters.values():
        if len(members) == 1:
            kept_indices.add(members[0])
            continue
        members_sorted = sorted(members, key=rank)
        keeper = members_sorted[0]
        kept_indices.add(keeper)
        for dup in members_sorted[1:]:
            is_exact = exact_hashes[dup] == exact_hashes[keeper]
            sim = 1.0 if is_exact else _jaccard(get_shingles(dup), get_shingles(keeper))
            report_rows.append({
                "dropped_work_title": records[dup]["work_title"],
                "dropped_section_title": records[dup]["section_title"],
                "kept_work_title": records[keeper]["work_title"],
                "kept_section_title": records[keeper]["section_title"],
                "match_type": "exact" if is_exact else "near",
                "similarity": round(sim, 3),
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for i in sorted(kept_indices):
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    with open(report_path, "w", encoding="utf-8") as f:
        for row in report_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nKept {len(kept_indices)} sections -> {out_path}")
    print(f"Flagged {len(report_rows)} duplicate sections -> {report_path}")
    if report_rows:
        print("\nReview the report before trusting it blindly:")
        print("  - 'exact' matches are safe to accept as-is")
        print("  - 'near' matches worth a skim -- two genuinely different")
        print("    pieces sharing a lot of stock phrasing could in theory")
        print("    trip this. If you see a false match, raise dedup.jaccard_threshold")
        print("    in config.yaml and re-run.")


if __name__ == "__main__":
    dedupe()