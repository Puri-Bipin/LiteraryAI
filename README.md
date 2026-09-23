# LiteraView AI — Fall 2026 MVP (Mark Twain prototype)

Retrieval-grounded conversational system for exploring an author's literary
works and life/process writing, scoped so the model can only answer from an
approved public-domain corpus (Project Gutenberg).

This repo currently ships **one author (Mark Twain)** end to end. The
folder/config layout is built so Chekhov and George Eliot can be added later
by extending `config.yaml` — no code changes required.

## Architecture (matches PRD Section 7)

```
Project Gutenberg (curated list in config.yaml)
        │  download_gutenberg.py
        ▼
data/raw/<author>/<mode>/<work>.txt
        │  clean_text.py  (strip PG header/footer, normalize whitespace)
        ▼
build_corpus.py  (split into chapters/acts, attach metadata)
        ▼
data/processed/corpus.jsonl
        │  chunker.py (RecursiveCharacterTextSplitter)
        ▼
build_vectorstore.py (embed + persist)
        ▼
Chroma vector store (data/processed/chroma/)
        │
        ▼
router.py  → decides: specific work | across works (oeuvre) | life & process
        │
        ▼
retriever.py  → metadata-filtered similarity search (author + mode + work)
        │
        ▼
rag_chain.py  → LLM (Gemini/Groq) generates answer strictly from retrieved
                 chunks, with citations, safe "I don't know" fallback
        │
        ▼
streamlit_app.py  → chat UI (author + mode selection, source display)
```

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste in your GOOGLE_API_KEY and/or GROQ_API_KEY
```

Get a free Gemini key (no card required) at https://aistudio.google.com →
"Get API key". Get a free Groq key at https://console.groq.com/keys.

## 2. Download & build the corpus

```bash
python scripts/01_download_data.py     # pulls .txt files from Gutenberg
python scripts/02_build_index.py       # cleans, chunks, embeds, persists Chroma
```

`01_download_data.py` prints the `Title:` line it finds inside each
downloaded file so you can visually confirm the Gutenberg ID in
`config.yaml` actually points at the work you think it does before you
build the index on top of it — Gutenberg IDs occasionally get reassigned
editions, so this is a cheap sanity check, not paranoia.

## 3. Run the app

```bash
streamlit run src/app/streamlit_app.py
```

## Repo layout

```
literaview-ai/
├── config.yaml                 # curated corpus list, chunking & model config
├── .env.example
├── requirements.txt
├── data/
│   ├── raw/                    # downloaded Gutenberg .txt (gitignored)
│   └── processed/              # corpus.jsonl + chroma/ (gitignored)
├── src/
│   ├── config.py                # loads .env + config.yaml
│   ├── ingestion/
│   │   ├── download_gutenberg.py
│   │   ├── clean_text.py
│   │   └── build_corpus.py
│   ├── indexing/
│   │   ├── chunker.py
│   │   └── build_vectorstore.py
│   ├── retrieval/
│   │   ├── router.py
│   │   └── retriever.py
│   ├── generation/
│   │   ├── llm.py
│   │   └── rag_chain.py
│   └── app/
│       └── streamlit_app.py
├── scripts/
│   ├── 01_download_data.py
│   └── 02_build_index.py
└── tests/
    └── test_router.py
```

## Adding a second author later

1. Add an entry under `authors:` in `config.yaml` with its `literature` and
   `life_process` work lists (title + Gutenberg ID).
2. Re-run `01_download_data.py` and `02_build_index.py` — they read the
   whole config, so new authors are picked up automatically and added to
   the same Chroma store as new metadata-filtered slices.
3. Add the author to the `AUTHORS` list at the top of `streamlit_app.py`.

## Notes on the "Safe fallback" requirement

`rag_chain.py` enforces this two ways: (1) if the retriever returns no
chunks above a similarity threshold, the app returns a fixed "not
supported by the approved corpus" message *without* calling the LLM at
all (saves API credits); (2) the prompt itself instructs the model to say
so explicitly if the retrieved context doesn't answer the question, so a
partially-relevant-but-insufficient retrieval doesn't get hallucinated
over.
