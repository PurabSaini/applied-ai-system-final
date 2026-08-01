"""Offline tests for the Last.fm client: response normalization + errors.

All HTTP is mocked, so these run with no API key and no network.
"""
import pytest

import config
import lastfm_client


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    """Provide a dummy key so config.require() passes without a real .env."""
    monkeypatch.setattr(config, "LASTFM_API_KEY", "test-key")


def _patch_get(monkeypatch, payload, status=200):
    monkeypatch.setattr(lastfm_client.requests, "get",
                        lambda *a, **k: FakeResp(payload, status))


def test_search_track_normalizes(monkeypatch):
    _patch_get(monkeypatch, {"results": {"trackmatches": {"track": [
        {"name": "Yellow", "artist": "Coldplay", "listeners": "4719", "url": "u"},
    ]}}})
    songs = lastfm_client.search_track("yellow")
    assert songs[0]["title"] == "Yellow"
    assert songs[0]["artist"] == "Coldplay"
    assert songs[0]["listeners"] == 4719


def test_get_similar_parses_match(monkeypatch):
    _patch_get(monkeypatch, {"similartracks": {"track": [
        {"name": "The Scientist", "artist": {"name": "Coldplay"},
         "match": "0.95", "playcount": "100"},
    ]}})
    songs = lastfm_client.get_similar("Coldplay", "Yellow")
    assert songs[0]["match"] == pytest.approx(0.95)
    assert songs[0]["artist"] == "Coldplay"


def test_single_result_coerced_to_list(monkeypatch):
    """Last.fm returns a dict (not list) for a single result."""
    _patch_get(monkeypatch, {"results": {"trackmatches": {"track":
        {"name": "Solo", "artist": "X", "listeners": "1"}}}})
    songs = lastfm_client.search_track("solo")
    assert len(songs) == 1 and songs[0]["title"] == "Solo"


def test_api_error_field_raises(monkeypatch):
    _patch_get(monkeypatch, {"error": 6, "message": "Track not found"})
    with pytest.raises(lastfm_client.LastFmError):
        lastfm_client.get_track_info("Nobody", "Nothing")


def test_http_error_raises(monkeypatch):
    _patch_get(monkeypatch, "server error", status=500)
    with pytest.raises(lastfm_client.LastFmError):
        lastfm_client.search_track("x")


def test_network_error_raises(monkeypatch):
    def boom(*a, **k):
        raise lastfm_client.requests.RequestException("no network")
    monkeypatch.setattr(lastfm_client.requests, "get", boom)
    with pytest.raises(lastfm_client.LastFmError):
        lastfm_client.search_track("x")


def test_enrich_adopts_canonical_name(monkeypatch):
    """enrich() should replace a messy title with Last.fm's canonical one."""
    _patch_get(monkeypatch, {"track": {
        "name": "Karma Police", "artist": {"name": "Radiohead"},
        "listeners": "3352165", "playcount": "10",
        "toptags": {"tag": [{"name": "alternative", "count": 100}]},
        "url": "u",
    }})
    messy = {"title": "Radiohead - Karma Police", "artist": "Radiohead", "tags": []}
    out = lastfm_client.enrich(messy)
    assert out["title"] == "Karma Police"
    assert out["tags"] == [("alternative", 100)]
    assert out["listeners"] == 3352165
