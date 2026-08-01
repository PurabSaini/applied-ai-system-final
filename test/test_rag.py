"""Offline tests for the RAG orchestration, focused on the grounding guardrail.

Both external services (Last.fm, Gemini) are monkeypatched, so these run with
no keys and no network.
"""
import gemini_client
import lastfm_client
import rag
from recommender import QueryIntent, Song


# --- retrieve() ------------------------------------------------------------
def test_retrieve_dedups_and_caps(monkeypatch):
    dupes = [
        {"title": "Song A", "artist": "X", "tags": [], "listeners": 1},
        {"title": "song a", "artist": "x", "tags": [], "listeners": 1},  # dup
        {"title": "Song B", "artist": "Y", "tags": [], "listeners": 1},
        {"title": "Song C", "artist": "Z", "tags": [], "listeners": 1},
    ]
    monkeypatch.setattr(lastfm_client, "get_tag_top_tracks",
                        lambda tag, limit=8: list(dupes))
    monkeypatch.setattr(lastfm_client, "enrich", lambda c: c)  # identity

    intent = QueryIntent(desired_tags=["pop"])
    songs = rag.retrieve(intent, max_candidates=2)
    assert all(isinstance(s, Song) for s in songs)
    titles = [s.title for s in songs]
    assert titles == ["Song A", "Song B"]  # deduped + capped at 2


# --- guardrail (the core anti-hallucination check) -------------------------
def _ranked(*titles):
    return [
        (Song(t, "Artist", tags=[("pop", 100)], listeners=100), 3.0, ["tagged pop"])
        for t in titles
    ]


def test_guardrail_drops_hallucinated_pick(monkeypatch):
    ranked = _ranked("Real One", "Real Two")
    monkeypatch.setattr(rag, "_generate_once", lambda q, c, k: {
        "intro": "x",
        "picks": [
            {"title": "Real One", "artist": "Artist", "reason": "ok"},
            {"title": "Invented Song", "artist": "Ghost", "reason": "nope"},
        ],
    })
    res = rag.generate("q", ranked, k=5)
    kept = [r.title for r in res.recommendations]
    assert kept == ["Real One"]
    assert any("Invented Song" in w for w in res.warnings)


def test_guardrail_accepts_all_grounded(monkeypatch):
    ranked = _ranked("Real One", "Real Two")
    monkeypatch.setattr(rag, "_generate_once", lambda q, c, k: {
        "intro": "x",
        "picks": [
            {"title": "Real One", "artist": "Artist", "reason": "a"},
            {"title": "Real Two", "artist": "Artist", "reason": "b"},
        ],
    })
    res = rag.generate("q", ranked, k=5)
    assert [r.title for r in res.recommendations] == ["Real One", "Real Two"]
    assert not res.warnings


def test_guardrail_matches_ignore_case(monkeypatch):
    ranked = _ranked("Karma Police")
    monkeypatch.setattr(rag, "_generate_once", lambda q, c, k: {
        "intro": "x",
        "picks": [{"title": "karma police", "artist": "Radiohead", "reason": "r"}],
    })
    res = rag.generate("q", ranked, k=5)
    assert [r.title for r in res.recommendations] == ["Karma Police"]  # canonical case


# --- graceful degradation --------------------------------------------------
def test_fallback_when_gemini_unavailable(monkeypatch):
    ranked = _ranked("Real One", "Real Two")

    def boom(q, c, k):
        raise gemini_client.GeminiError("quota")

    monkeypatch.setattr(rag, "_generate_once", boom)
    res = rag.generate("q", ranked, k=5)
    # Falls back to the ranked list with score reasons instead of crashing.
    assert [r.title for r in res.recommendations] == ["Real One", "Real Two"]
    assert any("LLM unavailable" in w for w in res.warnings)


def test_parse_intent_fallback_on_llm_error(monkeypatch):
    def boom(*a, **k):
        raise gemini_client.GeminiError("down")

    monkeypatch.setattr(gemini_client, "generate_json", boom)
    intent = rag.parse_intent("amazing chill acoustic vibes please")
    assert intent.seed_artist is None
    assert "chill" in intent.desired_tags  # significant words become tags


# --- full pipeline wiring --------------------------------------------------
def test_recommend_attaches_intent_and_grounds(monkeypatch):
    monkeypatch.setattr(rag, "parse_intent",
                        lambda q: QueryIntent(desired_tags=["pop"]))
    monkeypatch.setattr(rag, "retrieve",
                        lambda intent, **k: [Song("Real One", "A",
                                                  tags=[("pop", 100)], listeners=1)])
    monkeypatch.setattr(rag, "_generate_once", lambda q, c, k: {
        "intro": "here you go",
        "picks": [{"title": "Real One", "artist": "A", "reason": "pop"}],
    })
    res = rag.recommend("some pop please", k=3)
    assert res.intent.desired_tags == ["pop"]
    assert [r.title for r in res.recommendations] == ["Real One"]
    assert res.intro == "here you go"


def test_recommend_handles_empty_retrieval(monkeypatch):
    monkeypatch.setattr(rag, "parse_intent",
                        lambda q: QueryIntent(desired_tags=["xyzzy"]))
    monkeypatch.setattr(rag, "retrieve", lambda intent, **k: [])
    res = rag.recommend("nonsense", k=3)
    assert res.recommendations == []
    assert res.warnings
