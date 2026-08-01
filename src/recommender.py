"""Ranking logic for the music recommender.

Reworked for the Last.fm RAG pipeline. The old version scored songs on numeric
audio features (energy, valence, acousticness...) from a local CSV. Last.fm
doesn't provide those, so scoring now uses Last.fm-native signals:

    score = W_TAGS * tag_overlap      # how well the song's tags match the ask
          + W_MATCH * similarity      # Last.fm's own similar-track match score
          + W_POP  * popularity_fit   # mainstream vs niche preference

The score->sort->top-k shape of recommend_songs() is unchanged from the
original; only the per-song signals differ. score_song() also returns a list
of human-readable reasons, which the RAG layer feeds to the LLM as grounding.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Signal weights (relative importance). Tags matter most, then similarity to a
# named seed, then the popularity preference.
W_TAGS = 3.0
W_MATCH = 2.0
W_POP = 1.0

# Listener count (log10) treated as "fully popular". ~10M listeners -> 1.0.
_POP_LOG_CEIL = 7.0


@dataclass
class QueryIntent:
    """Structured form of a user's free-text request (produced by the LLM)."""
    desired_tags: List[str] = field(default_factory=list)
    seed_artist: Optional[str] = None
    popularity_pref: str = "any"  # "mainstream" | "niche" | "any"


@dataclass
class Song:
    """A candidate track, normalized from a Last.fm response."""
    title: str
    artist: str
    tags: List[Tuple[str, int]] = field(default_factory=list)  # (name, weight)
    listeners: int = 0
    playcount: int = 0
    match: Optional[float] = None  # similarity score, only from get_similar
    url: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Song":
        """Build a Song from the dict shape produced by lastfm_client."""
        return cls(
            title=d.get("title", ""),
            artist=d.get("artist", ""),
            tags=list(d.get("tags", []) or []),
            listeners=int(d.get("listeners", 0) or 0),
            playcount=int(d.get("playcount", 0) or 0),
            match=d.get("match"),
            url=d.get("url", ""),
        )

    @property
    def tag_map(self) -> dict:
        """Lowercased tag name -> weight, for matching."""
        return {name.lower(): weight for name, weight in self.tags}


def _popularity_norm(listeners: int) -> float:
    """Map a listener count onto [0, 1] on a log scale (big ranges compress)."""
    if listeners <= 0:
        return 0.0
    return min(1.0, math.log10(listeners + 1) / _POP_LOG_CEIL)


def score_song(intent: QueryIntent, song: Song) -> Tuple[float, List[str]]:
    """Score one song against the intent. Returns (score, reasons).

    Each signal contributes a sub-score in [0, 1] scaled by its weight, mirroring
    the weighted-judge design of the original CSV recommender.
    """
    score = 0.0
    reasons: List[str] = []

    # --- Tag overlap (weight W_TAGS) ---
    # Fraction of desired tags the song carries, each weighted by how strongly
    # Last.fm associates that tag with the track (count is 0-100).
    desired = [t.lower() for t in intent.desired_tags if t.strip()]
    if desired:
        tag_map = song.tag_map
        matched = [t for t in desired if t in tag_map]
        if matched:
            confidence = sum(min(1.0, (tag_map[t] or 0) / 100.0) or 0.2 for t in matched)
            tag_sub = min(1.0, confidence / len(desired))
            score += W_TAGS * tag_sub
            reasons.append("tagged " + ", ".join(matched))

    # --- Similarity match (weight W_MATCH) ---
    # Only counts when the user actually named a seed to be "similar to".
    # (In the pipeline, match scores only exist for get_similar results, but
    # guarding on seed_artist keeps scoring correct for mixed candidate lists.)
    if song.match is not None and intent.seed_artist:
        match_sub = max(0.0, min(1.0, float(song.match)))
        score += W_MATCH * match_sub
        if match_sub >= 0.3:
            seed = f" to {intent.seed_artist}" if intent.seed_artist else ""
            reasons.append(f"similar{seed} (match {match_sub:.2f})")

    # --- Popularity fit (weight W_POP) ---
    pop_norm = _popularity_norm(song.listeners)
    pref = (intent.popularity_pref or "any").lower()
    if pref == "mainstream":
        score += W_POP * pop_norm
        if pop_norm >= 0.7:
            reasons.append("a popular, widely-loved pick")
    elif pref == "niche":
        niche = 1.0 - pop_norm
        score += W_POP * niche
        if niche >= 0.7 and song.listeners > 0:
            reasons.append("more of a deep cut")
    # "any" -> popularity does not affect the score

    return score, reasons


def recommend_songs(
    intent: QueryIntent, songs: List[Song], k: int = 5
) -> List[Tuple[Song, float, List[str]]]:
    """Score every song, then return the k highest as (song, score, reasons).

    Same score->sort->top-k structure as the original recommend_songs().
    """
    scored = [
        (song, score, reasons)
        for song in songs
        for score, reasons in [score_song(intent, song)]
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
