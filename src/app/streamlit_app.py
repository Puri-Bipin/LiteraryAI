"""LiteraView AI — Streamlit MVP UI with persistent conversation history.

Implements PRD Section 3 step by step:
  1. Choose an author       -> sidebar selectbox
  2. Choose a mode          -> sidebar radio (Literature / Life & Process)
  3. Ask a question         -> chat input
  4/5. Route + retrieve + generate -> src.generation.rag_chain.answer_question

Plus: new / resume / delete conversations, backed by SQLite
(src/app/conversation_store.py). A conversation is anchored to one
author+mode pair, chosen when its first message is sent.

Run with:  streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from src.app import conversation_store as store
from src.config import CONFIG
from src.generation.rag_chain import answer_question

st.set_page_config(page_title="LiteraView AI", page_icon="📚", layout="centered")

store.init_db()

AUTHOR_KEYS = list(CONFIG["authors"].keys())


def literature_titles(author_key: str) -> list[str]:
    return [w["title"] for w in CONFIG["authors"][author_key].get("literature", [])]


def has_life_process(author_key: str) -> bool:
    return bool(CONFIG["authors"][author_key].get("life_process"))


def mode_label_to_key(label: str) -> str:
    return "literature" if label == "Literature" else "life_process"


def mode_key_to_label(key: str) -> str:
    return "Literature" if key == "literature" else "Life & Process"


# --- Session state defaults --------------------------------------------------
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "author_key" not in st.session_state:
    st.session_state.author_key = AUTHOR_KEYS[0]
if "mode_label" not in st.session_state:
    st.session_state.mode_label = "Literature"


def start_new_conversation():
    st.session_state.current_conversation_id = None
    st.session_state.chat_history = []


def load_conversation(conv: dict):
    st.session_state.current_conversation_id = conv["id"]
    st.session_state.author_key = conv["author_key"]
    st.session_state.mode_label = mode_key_to_label(conv["mode"])
    st.session_state.chat_history = store.get_messages(conv["id"])


def delete_conversation(conv_id: int):
    store.delete_conversation(conv_id)
    if st.session_state.current_conversation_id == conv_id:
        start_new_conversation()


# --- Sidebar: conversation list ---------------------------------------------
st.sidebar.title("📚 LiteraView AI")
st.sidebar.caption("Fall 2026 MVP — Mark Twain prototype")

st.sidebar.button("+ New conversation", use_container_width=True, on_click=start_new_conversation)

st.sidebar.divider()
st.sidebar.caption("Conversations")

conversations = store.list_conversations()
if not conversations:
    st.sidebar.caption("_No conversations yet — ask something to start one._")

for conv in conversations:
    is_active = conv["id"] == st.session_state.current_conversation_id
    col_title, col_delete = st.sidebar.columns([5, 1])
    label = f"{'▶ ' if is_active else ''}{conv['title']}"
    if col_title.button(label, key=f"open_{conv['id']}", use_container_width=True):
        load_conversation(conv)
        st.rerun()
    if col_delete.button("🗑", key=f"delete_{conv['id']}"):
        delete_conversation(conv["id"])
        st.rerun()

st.sidebar.divider()

# --- Sidebar: Steps 1 & 2 of the PRD flow -----------------------------------
# Disabled once a conversation has a first message, since a conversation is
# anchored to the author/mode it was started with -- switching mid-thread
# would silently change what corpus the rest of the conversation is grounded in.
locked = st.session_state.current_conversation_id is not None

selected_author_key = st.sidebar.selectbox(
    "Choose an author",
    options=AUTHOR_KEYS,
    format_func=lambda k: CONFIG["authors"][k]["display_name"],
    key="author_key",
    disabled=locked,
)

mode_options = ["Literature"]
if has_life_process(selected_author_key):
    mode_options.append("Life & Process")

selected_mode_label = st.sidebar.radio(
    "Choose a mode", options=mode_options, key="mode_label", disabled=locked
)
selected_mode = mode_label_to_key(selected_mode_label)

if locked:
    st.sidebar.caption("Author/mode locked for this conversation. Start a new one to change them.")

st.sidebar.divider()
st.sidebar.caption(
    "Answers are generated only from the approved public-domain corpus "
    "(Project Gutenberg) for the selected author and mode. This is a "
    "reading companion, not a substitute for the primary text."
)

# --- Main pane ---------------------------------------------------------------
author_display = CONFIG["authors"][selected_author_key]["display_name"]
st.title(f"Ask {author_display}")
st.caption(f"Mode: {selected_mode_label}")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("sources"):
            with st.expander("Sources"):
                for src in turn["sources"]:
                    st.markdown(
                        f"**[{src['index']}] {src['work_title']} — {src['section_title']}**  \n"
                        f"{src['snippet']}  \n"
                        f"[View on Project Gutenberg]({src['source_ref']})"
                    )

placeholder = (
    f"Ask about a specific {author_display} work, common themes across his work, "
    "or his life & writing process..."
    if selected_mode == "literature"
    else f"Ask about {author_display}'s life, writing process, or worldview..."
)

question = st.chat_input(placeholder)

if question:
    # Lazily create the conversation on the first message, anchored to
    # whatever author/mode is currently selected.
    if st.session_state.current_conversation_id is None:
        st.session_state.current_conversation_id = store.create_conversation(
            author_key=selected_author_key, mode=selected_mode, title=question
        )

    conv_id = st.session_state.current_conversation_id

    st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
    store.add_message(conv_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving grounded passages and generating a response..."):
            result = answer_question(
                question=question,
                author_key=selected_author_key,
                author_display=author_display,
                mode=selected_mode,
                known_literature_titles=literature_titles(selected_author_key),
            )

        scope_label = {
            "specific_work": f"Scoped to: {result.work_title}",
            "oeuvre": "Scoped to: across all approved literary works",
            "life_process": "Scoped to: letters / autobiography / essays",
        }[result.scope]
        st.caption(scope_label)
        st.markdown(result.answer)

        if result.sources:
            with st.expander("Sources"):
                for src in result.sources:
                    st.markdown(
                        f"**[{src['index']}] {src['work_title']} — {src['section_title']}**  \n"
                        f"{src['snippet']}  \n"
                        f"[View on Project Gutenberg]({src['source_ref']})"
                    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
    store.add_message(conv_id, "assistant", result.answer, result.sources)
    st.rerun()  # refresh sidebar so the conversation title/order updates