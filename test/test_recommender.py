"""Offline tests for the ranking logic (pure functions, no network)."""
from recommender import QueryIntent, Song, recommend_songs, score_song


def _song(title, tags, listeners=1000, match=None, artist="A"):
    return Song(title=title, artist=artist, tags=tags,
                listeners=listeners, match=match)


def test_tag_overlap_scores_higher():
    intent = QueryIntent(desired_tags=["chill", "acoustic"])
    strong = _song("Strong", [("chill", 100), ("acoustic", 90)])
    weak = _song("Weak", [("metal", 100)])
    s_strong, reasons = score_song(intent, strong)
    s_weak, _ = score_song(intent, weak)
    assert s_strong > s_weak
    assert any("tagged" in r for r in reasons)


def test_similarity_ignored_without_seed():
    """A match score must not count when the user named no seed artist."""
    intent = QueryIntent(desired_tags=["acoustic"])  # no seed
    with_match = _song("HasMatch", [("acoustic", 50)], match=0.99)
    score, reasons = score_song(intent, with_match)
    assert not any("similar" in r for r in reasons)


def test_similarity_counts_with_seed():
    intent = QueryIntent(desired_tags=["acoustic"], seed_artist="Bon Iver")
    song = _song("HasMatch", [("acoustic", 50)], match=0.99)
    _, reasons = score_song(intent, song)
    assert any("similar" in r for r in reasons)


def test_popularity_preference_direction():
    mainstream = QueryIntent(desired_tags=[], popularity_pref="mainstream")
    niche = QueryIntent(desired_tags=[], popularity_pref="niche")
    popular = _song("Popular", [], listeners=8_000_000)
    obscure = _song("Obscure", [], listeners=50)
    assert score_song(mainstream, popular)[0] > score_song(mainstream, obscure)[0]
    assert score_song(niche, obscure)[0] > score_song(niche, popular)[0]


def test_recommend_songs_sorted_topk():
    intent = QueryIntent(desired_tags=["pop"])
    songs = [
        _song("no", [("jazz", 100)]),
        _song("yes", [("pop", 100)]),
        _song("meh", [("pop", 20)]),
    ]
    ranked = recommend_songs(intent, songs, k=2)
    assert len(ranked) == 2
    assert ranked[0][0].title == "yes"
    # scores are descending
    assert ranked[0][1] >= ranked[1][1]


def test_song_from_dict_roundtrip():
    d = {"title": "T", "artist": "Ar", "tags": [("indie", 55)],
         "listeners": 10, "match": None, "url": "u"}
    song = Song.from_dict(d)
    assert song.tag_map == {"indie": 55}
    assert song.title == "T" and song.url == "u"
