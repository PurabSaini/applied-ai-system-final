"""Streamlit UI for the RAG music recommender.

Run from the project root:
    streamlit run src/app.py

Type a free-text request; the app parses it, retrieves candidate tracks from
Last.fm, ranks them, and has Gemini write grounded recommendations that are
guardrail-checked against what was actually retrieved.
"""
import os
import sys

# Make sibling modules importable whether run via `streamlit run src/app.py`
# or from within src/.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import config
import rag

st.set_page_config(page_title="Music RAG Recommender", page_icon="🎵")

EXAMPLES = [
    "something like Bon Iver, chill and acoustic",
    "upbeat pop for a workout, mainstream is fine",
    "moody indie rock for a rainy night",
    "deep-cut jazz, nothing mainstream",
]


def _tag_lookup(result):
    """Map a recommendation title -> its retrieved tags (for transparency)."""
    return {
        song.title.lower(): [name for name, _ in song.tags[:6]]
        for song in result.candidates
    }


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.markdown(
        "Natural-language music recommendations grounded in **Last.fm** data, "
        "written by **Gemini**, and checked by a **guardrail** that rejects any "
        "song the model invents."
    )
    st.divider()
    k = st.slider("How many recommendations?", 1, 8, 4)
    keys_ok = bool(config.LASTFM_API_KEY) and bool(config.GEMINI_API_KEY)
    if keys_ok:
        st.success("API keys loaded")
    else:
        st.error("Missing API key(s). Copy .env.example to .env and fill it in.")
    st.caption(f"Model: {config.GEMINI_MODEL}")


# --- Main ------------------------------------------------------------------
st.title("🎵 Music RAG Recommender")
st.write("Describe what you want to hear, in your own words.")

# Example chips populate the input box.
st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["query"] = ex

query = st.text_input(
    "Your request",
    key="query",
    placeholder="e.g. mellow acoustic songs for studying, like Bon Iver",
)

go = st.button("Recommend", type="primary", disabled=not keys_ok)

if go and query.strip():
    with st.spinner("Retrieving from Last.fm and generating..."):
        try:
            result = rag.recommend(query.strip(), k=k)
        except Exception as exc:  # last-resort guard so the UI never crashes
            st.error(f"Something went wrong: {exc}")
            st.stop()

    # Parsed intent (transparency into the retrieval step).
    intent = result.intent
    st.caption(
        f"Understood as → tags: {intent.desired_tags or '—'} · "
        f"seed: {intent.seed_artist or '—'} · popularity: {intent.popularity_pref}"
    )

    if not result.recommendations:
        st.warning(result.intro or "No recommendations found. Try rephrasing.")
    else:
        if result.intro:
            st.subheader(result.intro)
        tags_by_title = _tag_lookup(result)
        for i, rec in enumerate(result.recommendations, 1):
            with st.container(border=True):
                title_md = f"**{i}. {rec.title}** — {rec.artist}"
                if rec.url:
                    title_md += f"  ·  [Last.fm]({rec.url})"
                st.markdown(title_md)
                if rec.reason:
                    st.write(rec.reason)
                tags = tags_by_title.get(rec.title.lower())
                if tags:
                    st.caption("Grounded in tags: " + ", ".join(tags))

    # Guardrail / status messages.
    if result.warnings:
        with st.expander("⚠️ Guardrail & status notes"):
            for w in result.warnings:
                st.write("• " + w)

    # Full retrieved set — the "documents" the answer is grounded in.
    with st.expander(f"🔎 Retrieved candidates ({len(result.candidates)})"):
        for song in result.candidates:
            tags = ", ".join(name for name, _ in song.tags[:6]) or "no tags"
            st.write(f"- **{song.title}** — {song.artist}  ·  _{tags}_")

elif go:
    st.info("Type a request first.")
