# 🎵 Music RAG Recommender

A natural-language music recommender built as a **Retrieval-Augmented Generation (RAG)** system. You describe what you want to hear in plain English; the app parses your intent, **retrieves real candidate songs from Last.fm**, ranks them, and has an LLM (Google Gemini) write grounded recommendations — with a **guardrail that rejects any song the model invents**.

This extends an earlier feature-scored music recommender. The original numeric scoring engine survives as a ranking layer; Last.fm replaces the static CSV as a live retrieval source.

**Rubric mapping:** Option 1 — *RAG + testing + guardrails*, plus graceful-degradation reliability.

---

## How it works

```
User query
  → Parse intent      (Gemini → {desired_tags, seed_artist, popularity_pref})
  → Retrieve          (Last.fm: get_similar / tag.getTopTracks / search, then enrich)
  → Rank              (score_song: tag overlap + similarity match + popularity)
  → Generate          (Gemini writes picks, choosing ONLY from retrieved songs)
  → GUARDRAIL         (every cited song must be in the retrieved set, else drop + regenerate)
  → Show              (Streamlit UI / CLI, with the grounding sources visible)
```

See [diagrams/architecture.mmd](diagrams/architecture.mmd) for the diagram. Where AI output is checked:
- **Guardrail (runtime):** picks not in the retrieved set are dropped and the model is asked to regenerate — the anti-hallucination gate.
- **Human (runtime):** the user sees the parsed intent, the reasons, and the full retrieved candidate set, then accepts or refines.
- **Automated tests (offline):** a pytest suite asserts the guardrail holds and the pipeline degrades gracefully (see below).

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (both free)
cp .env.example .env
#    then edit .env:
#      LAST_FM_API_KEY  -> https://www.last.fm/api/account/create
#      GEMINI_API_KEY   -> https://aistudio.google.com/apikey
```

> **Model note:** the default `gemini-flash-latest` is used because pinned `gemini-2.0-flash` / `gemini-2.5-flash` are blocked on the free tier for new keys.

---

## Sample command executions

```bash
# Interactive web UI
streamlit run src/app.py

# CLI demo — runs the built-in example queries + guardrail demonstration
python src/demo.py

# CLI with your own query
python src/demo.py "moody indie rock for a rainy night"

# Run the offline test suite (no API keys / no network required)
pytest
```

Full captured runs live in [logs/example_run.txt](logs/example_run.txt) and [logs/test_results.txt](logs/test_results.txt).

---

## Example inputs & outputs

Real output captured from `python src/demo.py` (see [logs/example_run.txt](logs/example_run.txt) for the complete run).

### Example 1 — similarity-seeded query

**Input:** `"something like Bon Iver, chill and acoustic"`

```text
Parsed intent -> tags=['chill', 'acoustic'] seed='Bon Iver' popularity=any

Here are some chill, acoustic tracks that capture the atmospheric folk vibe of Bon Iver.
  1. Flume - Bon Iver
     why: Straight from Bon Iver's debut, this track showcases their signature chill, stripped-down acoustic sound.
  2. The Wolves (Act I and II) - Bon Iver
     why: Another quintessential Bon Iver piece delivering an atmospheric and intimate acoustic experience.
  3. Big Black Car - Gregory Alan Isakov
     why: Tagged with acoustic, folk, and indie folk, this song delivers a gentle, introspective tone perfect for fans of Bon Iver.
  4. Amsterdam - Gregory Alan Isakov
     why: Tagged with acoustic, folk, and beautiful, it offers a soothing and mellow arrangement.

  Retrieved candidates (8) [grounding source]:
    - Flume - Bon Iver  [no tags]
    - The Wolves (Act I and II) - Bon Iver  [no tags]
    - Big Black Car - Gregory Alan Isakov  [folk, acoustic, indie, singer-songwriter, indie folk]
    - Stubborn Love - The Lumineers  [folk, indie, indie folk, acoustic, folk rock]
    - Amsterdam - Gregory Alan Isakov  [indie, acoustic, waltz, folk, beautiful]
    - Heartbeats - José González  [no tags]
    - Old Pine - Ben Howard  [no tags]
    - First Day of My Life - Bright Eyes  [no tags]
```

### Example 2 — tag/mood query with popularity preference

**Input:** `"upbeat pop for a workout, mainstream is fine"`

```text
Parsed intent -> tags=['upbeat', 'pop', 'workout'] seed=None popularity=mainstream

Here are some high-energy, mainstream pop tracks from your options to keep your workout moving.
  1. Hey, Soul Sister - Train
     why: With its upbeat, pop, and happy tags alongside massive popularity, this track brings cheerful, high-tempo energy to any workout.
  2. Pocketful of Sunshine - Natasha Bedingfield
     why: Tagged as both pop and upbeat, this mainstream hit provides an optimistic, driving rhythm perfect for staying active.
  3. Locked Out of Heaven - Bruno Mars
     why: A hugely popular pop and R&B anthem with a propulsive tempo that fits right into a workout playlist.
  4. I Bet My Life - Imagine Dragons
     why: Tagged as upbeat, this mainstream pop-rock track offers an anthemic, energetic feel ideal for keeping your momentum up.
```

Every recommended song appears in the retrieved candidate set, and explanations reference the songs' **actual Last.fm tags** — the grounding the RAG design guarantees.

---

## Reliability / guardrail results

### 1. Guardrail rejects hallucinated songs

The demo injects a fake pick ("Imaginary Anthem") that was never retrieved, to prove the guardrail catches invented songs. Captured from `python src/demo.py`:

```text
======================================================================
GUARDRAIL DEMONSTRATION (injected hallucination)
======================================================================
Retrieved set: 'The Scientist', 'Fix You' (both Coldplay)

Model proposed: 'The Scientist' (real) + 'Imaginary Anthem' (invented)

After guardrail:
  KEPT: The Scientist - Coldplay
  Guardrail (attempt 1): dropped un-retrieved pick(s): Imaginary Anthem
  Guardrail (attempt 2): dropped un-retrieved pick(s): Imaginary Anthem

Result: the invented song was rejected; only grounded picks remain.
```

### 2. Graceful degradation

If Gemini is unreachable (quota/network), the pipeline **falls back to the ranked list with score-based reasons** instead of crashing; if Last.fm fails mid-retrieval, those calls are skipped and retrieval continues. Both paths are covered by tests.

### 3. Automated test suite (offline, no keys/network)

Captured from `pytest -v` (full log: [logs/test_results.txt](logs/test_results.txt)):

```text
collected 26 items

test\test_gemini_client.py .....                                         [ 19%]
test\test_lastfm_client.py .......                                       [ 46%]
test\test_rag.py ........                                                [ 76%]
test\test_recommender.py ......                                          [100%]

============================= 26 passed in 0.11s ==============================
```

What the suite locks down:

| Area | Checks |
|---|---|
| Ranking (`test_recommender.py`) | tag overlap, seed-gated similarity, popularity direction, top-k sort |
| Retrieval (`test_lastfm_client.py`) | response normalization, single/list coercion, canonical-name enrich, HTTP/API/network errors |
| LLM client (`test_gemini_client.py`) | JSON extraction from fenced/prose output, HTTP 429 → clear error |
| **Guardrail (`test_rag.py`)** | **hallucinated pick dropped + warned** (case-insensitive), dedup/cap, LLM/Last.fm fallbacks |

---

## Project layout

```
src/
  app.py            Streamlit UI
  demo.py           CLI demo (generates logs/example_run.txt)
  rag.py            RAG orchestration + guardrail
  recommender.py    Song/QueryIntent + score_song ranking
  lastfm_client.py  Last.fm retrieval + normalization
  gemini_client.py  Gemini REST wrapper
  config.py         loads .env
test/               offline pytest suite (26 tests)
logs/               captured example run + test results
diagrams/           architecture.mmd
```

## Notes on data honesty

Recommendations are grounded in **Last.fm-native fields only** — tags, listener/play counts, and similarity match scores. The project does **not** fabricate audio features (energy, danceability, etc.); those were dropped precisely because Last.fm doesn't measure them.
