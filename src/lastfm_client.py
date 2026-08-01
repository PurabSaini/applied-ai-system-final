"""Client for the Last.fm API.

This is the retrieval layer of the RAG pipeline: it fetches candidate tracks
and their tags / popularity from Last.fm. Everything is normalized into plain
dicts with a stable shape so the recommender can rank them without caring
where they came from.

Normalized song dict shape:
    {
        "title":     str,
        "artist":    str,
        "tags":      [(name: str, weight: int), ...],   # may be empty
        "listeners": int,
        "playcount": int,
        "match":     float | None,   # similarity score, only from get_similar
        "url":       str,
    }
"""
from typing import List, Optional

import requests

import config

_BASE = "http://ws.audioscrobbler.com/2.0/"
_TIMEOUT = 20


class LastFmError(RuntimeError):
    """Raised when a Last.fm request fails (network or API-level error)."""


def _call(method: str, **params) -> dict:
    """Make one Last.fm API call and return the parsed JSON.

    Last.fm returns HTTP 200 even for API errors, signalling them with an
    "error" field in the body, so we check for that explicitly.
    """
    query = {
        "method": method,
        "api_key": config.require("LAST_FM_API_KEY"),
        "format": "json",
        **params,
    }
    try:
        resp = requests.get(_BASE, params=query, timeout=_TIMEOUT)
    except requests.RequestException as exc:
        raise LastFmError(f"Could not reach Last.fm: {exc}") from exc

    if resp.status_code != 200:
        raise LastFmError(f"Last.fm HTTP {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    if isinstance(data, dict) and "error" in data:
        raise LastFmError(f"Last.fm error {data['error']}: {data.get('message', '')}")
    return data


def _as_list(value) -> list:
    """Last.fm returns a dict for a single result and a list for many."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _to_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_tags(raw_tags) -> List[tuple]:
    """Turn a Last.fm toptags block into [(name, weight), ...]."""
    tags = []
    for t in _as_list(raw_tags):
        name = (t.get("name") or "").strip()
        if name:
            tags.append((name, _to_int(t.get("count"))))
    return tags


def search_track(query: str, limit: int = 10) -> List[dict]:
    """Search tracks by free text. Returns normalized song dicts (no tags yet)."""
    data = _call("track.search", track=query, limit=limit)
    matches = data.get("results", {}).get("trackmatches", {}).get("track")
    songs = []
    for t in _as_list(matches):
        songs.append({
            "title": t.get("name", ""),
            "artist": t.get("artist", ""),
            "tags": [],
            "listeners": _to_int(t.get("listeners")),
            "playcount": 0,
            "match": None,
            "url": t.get("url", ""),
        })
    return songs


def get_similar(artist: str, track: str, limit: int = 20) -> List[dict]:
    """Tracks similar to a seed track, carrying Last.fm's match score."""
    data = _call("track.getSimilar", artist=artist, track=track,
                 limit=limit, autocorrect=1)
    similar = data.get("similartracks", {}).get("track")
    songs = []
    for t in _as_list(similar):
        try:
            match = float(t.get("match"))
        except (TypeError, ValueError):
            match = None
        songs.append({
            "title": t.get("name", ""),
            "artist": (t.get("artist") or {}).get("name", ""),
            "tags": [],
            "listeners": 0,
            "playcount": _to_int(t.get("playcount")),
            "match": match,
            "url": t.get("url", ""),
        })
    return songs


def get_tag_top_tracks(tag: str, limit: int = 20) -> List[dict]:
    """Top tracks for a tag (genre/mood). Good when there's no seed artist."""
    data = _call("tag.getTopTracks", tag=tag, limit=limit)
    tracks = data.get("tracks", {}).get("track")
    songs = []
    for t in _as_list(tracks):
        songs.append({
            "title": t.get("name", ""),
            "artist": (t.get("artist") or {}).get("name", ""),
            "tags": [(tag, 100)],  # it matched this tag by construction
            "listeners": 0,
            "playcount": 0,
            "match": None,
            "url": t.get("url", ""),
        })
    return songs


def get_top_tags(artist: str, track: str) -> List[tuple]:
    """Top tags for one track, as [(name, weight), ...]."""
    data = _call("track.getTopTags", artist=artist, track=track, autocorrect=1)
    return _normalize_tags(data.get("toptags", {}).get("tag"))


def get_track_info(artist: str, track: str) -> dict:
    """Full info for one track: listeners, playcount, tags, url, bio summary."""
    data = _call("track.getInfo", artist=artist, track=track, autocorrect=1)
    t = data.get("track", {})
    wiki = (t.get("wiki") or {}).get("summary", "")
    return {
        "title": t.get("name", track),
        "artist": (t.get("artist") or {}).get("name", artist),
        "tags": _normalize_tags((t.get("toptags") or {}).get("tag")),
        "listeners": _to_int(t.get("listeners")),
        "playcount": _to_int(t.get("playcount")),
        "match": None,
        "url": t.get("url", ""),
        "bio": wiki.strip(),
    }


def enrich(song: dict) -> dict:
    """Fill in tags + popularity for a song that only has title/artist.

    Best-effort: if the info call fails, the original song is returned
    unchanged so retrieval degrades gracefully instead of crashing.
    """
    if not song.get("artist") or not song.get("title"):
        return song
    try:
        info = get_track_info(song["artist"], song["title"])
    except LastFmError:
        return song
    merged = dict(song)
    # Adopt Last.fm's canonical title/artist so messy search hits display cleanly.
    merged["title"] = info["title"] or song.get("title", "")
    merged["artist"] = info["artist"] or song.get("artist", "")
    merged["tags"] = info["tags"] or song.get("tags", [])
    merged["listeners"] = info["listeners"] or song.get("listeners", 0)
    merged["playcount"] = info["playcount"] or song.get("playcount", 0)
    merged["url"] = song.get("url") or info["url"]
    return merged
