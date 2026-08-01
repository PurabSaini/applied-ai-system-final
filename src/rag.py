"""RAG orchestration: free-text query -> grounded music recommendations.

Pipeline (see diagrams/architecture.mmd):
    parse_intent  -> Gemini turns the query into a QueryIntent
    retrieve      -> Last.fm fetches + enriches candidate tracks (the "documents")
    rank          -> recommend_songs() orders them
    generate      -> Gemini writes picks, choosing ONLY from the candidates
    GUARDRAIL     -> drop/regenerate any pick not in the retrieved set

The guardrail is the anti-hallucination core: the model may only recommend
tracks that Last.fm actually returned, and every pick is verified against that
set before the user sees it.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import gemini_client
import lastfm_client
from recommender import QueryIntent, Song, recommend_songs

MAX_CANDIDATES = 10        # cap enrich calls per query (latency vs. coverage)
MAX_REGEN_TRIES = 2        # regeneration attempts when the model hallucinates


@dataclass
class Recommendation:
    title: str
    artist: str
    reason: str
    url: str = ""
    confidence: Optional[float] = None  # set by optional self-critique pass


@dataclass
class RagResult:
    query: str
    intent: QueryIntent
    intro: str
    recommendations: List[Recommendation] = field(default_factory=list)
    candidates: List[Song] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _norm(s: str) -> str:
    """Normalize a title/artist for set membership checks."""
    return " ".join((s or "").lower().split())


# --------------------------------------------------------------------------
# 1. Parse intent
# --------------------------------------------------------------------------
_INTENT_SYSTEM = (
    "You extract music search intent from a user's request. "
    "Respond ONLY with JSON, no prose."
)


def parse_intent(query: str) -> QueryIntent:
    """Use Gemini to turn a free-text query into a structured QueryIntent.

    Falls back to a naive keyword split if the model output is unusable, so the
    pipeline still runs when the LLM misbehaves.
    """
    prompt = (
        'Extract music search intent from the request below. Return JSON with '
        'exactly these keys:\n'
        '  "desired_tags": list of lowercase genre/mood/descriptor strings\n'
        '  "seed_artist": an artist name to find similar songs to, or null\n'
        '  "popularity_pref": one of "mainstream", "niche", "any"\n\n'
        f'Request: "{query}"'
    )
    try:
        data = gemini_client.generate_json(prompt, system=_INTENT_SYSTEM)
        tags = data.get("desired_tags") or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        pref = str(data.get("popularity_pref") or "any").lower()
        if pref not in {"mainstream", "niche", "any"}:
            pref = "any"
        return QueryIntent(
            desired_tags=[str(t).lower() for t in tags if str(t).strip()],
            seed_artist=(data.get("seed_artist") or None),
            popularity_pref=pref,
        )
    except gemini_client.GeminiError:
        # Degrade gracefully: treat significant words as tags.
        words = [w.lower() for w in query.split() if len(w) > 3]
        return QueryIntent(desired_tags=words[:4])


# --------------------------------------------------------------------------
# 2. Retrieve
# --------------------------------------------------------------------------
def retrieve(intent: QueryIntent, max_candidates: int = MAX_CANDIDATES) -> List[Song]:
    """Fetch candidate tracks from Last.fm based on the intent.

    Strategy: if a seed artist is named, pull tracks similar to that artist's
    top track; otherwise pull top tracks for the desired tags; fall back to a
    plain search. Candidates are deduped, capped, then enriched so each has
    tags + popularity for ranking and grounding.
    """
    raw: List[dict] = []

    if intent.seed_artist:
        try:
            seed_track = lastfm_client.get_artist_top_track(intent.seed_artist)
            if seed_track:
                raw += lastfm_client.get_similar(
                    intent.seed_artist, seed_track, limit=max_candidates
                )
        except lastfm_client.LastFmError:
            pass

    for tag in intent.desired_tags[:2]:
        try:
            raw += lastfm_client.get_tag_top_tracks(tag, limit=8)
        except lastfm_client.LastFmError:
            pass

    if not raw:  # last-resort fallback
        query = " ".join(intent.desired_tags) or (intent.seed_artist or "")
        if query:
            try:
                raw += lastfm_client.search_track(query, limit=max_candidates)
            except lastfm_client.LastFmError:
                pass

    # Dedupe by (title, artist), preserving the first (highest-priority) hit.
    seen = set()
    deduped: List[dict] = []
    for c in raw:
        key = (_norm(c.get("title", "")), _norm(c.get("artist", "")))
        if key == ("", "") or key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    deduped = deduped[:max_candidates]

    # Enrich each candidate so it has tags + popularity (match score preserved).
    return [Song.from_dict(lastfm_client.enrich(c)) for c in deduped]


# --------------------------------------------------------------------------
# 3. Rank + build grounding context
# --------------------------------------------------------------------------
def build_context(ranked: List) -> str:
    """Build the grounding block from retrieved fields ONLY."""
    lines = []
    for i, (song, score, reasons) in enumerate(ranked, 1):
        tags = ", ".join(name for name, _ in song.tags[:6]) or "no tags"
        why = "; ".join(reasons) if reasons else "no strong signal"
        lines.append(
            f'{i}. "{song.title}" by {song.artist} '
            f'[tags: {tags}] [listeners: {song.listeners}] '
            f'(match signals: {why})'
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4 + 5. Generate with guardrail
# --------------------------------------------------------------------------
_GEN_SYSTEM = (
    "You are a music recommender. You may ONLY recommend songs from the "
    "provided candidate list. Never invent songs or artists. Cite each pick "
    "using its EXACT title and artist from the list. Respond ONLY with JSON."
)


def _generate_once(query: str, context: str, k: int) -> dict:
    prompt = (
        f'User request: "{query}"\n\n'
        f'Candidate songs (choose ONLY from these):\n{context}\n\n'
        f'Pick up to {k} songs that best fit the request, best first. '
        'Return JSON:\n'
        '{ "intro": "one short framing sentence",\n'
        '  "picks": [ { "title": "...", "artist": "...", '
        '"reason": "why it fits, referencing its tags" } ] }'
    )
    return gemini_client.generate_json(prompt, system=_GEN_SYSTEM)


def generate(query: str, ranked: List, k: int = 5) -> RagResult:
    """Generate recommendations and enforce the grounding guardrail.

    Retries generation when the model cites a track outside the retrieved set;
    after retries, any remaining ungrounded picks are dropped (with a warning).
    """
    context = build_context(ranked)
    # Retrieved set: normalized title -> Song (for guardrail + url lookup).
    index = {_norm(song.title): song for song, _, _ in ranked}

    warnings: List[str] = []
    grounded: List[Recommendation] = []
    intro = ""

    for attempt in range(1, MAX_REGEN_TRIES + 1):
        try:
            data = _generate_once(query, context, k)
        except gemini_client.GeminiError as exc:
            # LLM unavailable: fall back to the ranked list + score reasons.
            warnings.append(f"LLM unavailable ({exc}); showing ranked matches.")
            return _fallback_result(query, ranked, k, warnings)

        intro = str(data.get("intro", "")).strip()
        picks = data.get("picks") or []
        grounded, hallucinated = [], []
        for p in picks:
            title = str(p.get("title", "")).strip()
            song = index.get(_norm(title))
            if song is None:
                hallucinated.append(title or "(unnamed)")
                continue
            grounded.append(Recommendation(
                title=song.title,
                artist=song.artist,
                reason=str(p.get("reason", "")).strip(),
                url=song.url,
            ))

        if not hallucinated:
            break  # all picks grounded -> accept
        warnings.append(
            f"Guardrail (attempt {attempt}): dropped un-retrieved pick(s): "
            + ", ".join(hallucinated)
        )
        # Retry with a stricter reminder appended to the context.
        context += "\n(REMINDER: recommend ONLY songs from the list above.)"

    return RagResult(
        query=query, intent=QueryIntent(), intro=intro,
        recommendations=grounded[:k], candidates=[s for s, _, _ in ranked],
        warnings=warnings,
    )


def _fallback_result(query, ranked, k, warnings) -> RagResult:
    """When the LLM can't be reached, present the ranked list with its reasons."""
    recs = [
        Recommendation(
            title=song.title, artist=song.artist,
            reason="; ".join(reasons) if reasons else "matched your request",
            url=song.url,
        )
        for song, _, reasons in ranked[:k]
    ]
    return RagResult(
        query=query, intent=QueryIntent(), intro="Here are your top matches:",
        recommendations=recs, candidates=[s for s, _, _ in ranked],
        warnings=warnings,
    )


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------
def recommend(query: str, k: int = 5) -> RagResult:
    """Full RAG pipeline: query -> grounded recommendations."""
    intent = parse_intent(query)
    candidates = retrieve(intent)
    if not candidates:
        return RagResult(
            query=query, intent=intent,
            intro="No matching songs found on Last.fm for that request.",
            warnings=["Retrieval returned no candidates."],
        )
    ranked = recommend_songs(intent, candidates, k=max(k, 8))
    result = generate(query, ranked, k=k)
    result.intent = intent  # attach the real intent for transparency/UI
    return result
