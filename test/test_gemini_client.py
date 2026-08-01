"""Offline tests for the Gemini client: JSON extraction + error mapping."""
import pytest

import config
import gemini_client


def test_generate_json_strips_code_fence(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate",
                        lambda *a, **k: '```json\n{"desired_tags": ["chill"]}\n```')
    out = gemini_client.generate_json("prompt")
    assert out == {"desired_tags": ["chill"]}


def test_generate_json_extracts_from_prose(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate",
                        lambda *a, **k: 'Sure! {"a": 1, "b": 2} hope that helps')
    assert gemini_client.generate_json("p") == {"a": 1, "b": 2}


def test_generate_json_raises_on_garbage(monkeypatch):
    monkeypatch.setattr(gemini_client, "generate",
                        lambda *a, **k: "no json here at all")
    with pytest.raises(gemini_client.GeminiError):
        gemini_client.generate_json("p")


class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_quota_429_maps_to_gemini_error(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini_client.requests, "post",
                        lambda *a, **k: FakeResp(429, text="quota"))
    with pytest.raises(gemini_client.GeminiError):
        gemini_client.generate("hi")


def test_generate_parses_text(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    payload = {"candidates": [{"content": {"parts": [{"text": "  hello  "}]}}]}
    monkeypatch.setattr(gemini_client.requests, "post",
                        lambda *a, **k: FakeResp(200, payload))
    assert gemini_client.generate("hi") == "hello"
