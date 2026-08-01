"""Thin wrapper around the Google Gemini REST API.

Just enough to send a prompt and get text back. Used for two things in the
RAG pipeline: parsing a free-text query into structured intent, and writing
the grounded recommendation. No SDK -- plain HTTP via requests.
"""
import json
import re
from typing import Optional

import requests

import config

_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    """Raised when the Gemini API call fails (network, quota, bad response)."""


def generate(prompt: str, system: Optional[str] = None, temperature: float = 0.4) -> str:
    """Send a prompt to Gemini and return the model's text reply.

    Raises GeminiError with a clear message on network/quota/parse failure so
    callers can degrade gracefully instead of crashing.
    """
    api_key = config.require("GEMINI_API_KEY")
    model = config.GEMINI_MODEL

    payload: dict = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    url = f"{_BASE}/models/{model}:generateContent"
    try:
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise GeminiError(f"Could not reach Gemini API: {exc}") from exc

    if resp.status_code == 429:
        raise GeminiError(
            f"Gemini quota/rate limit hit for model '{model}' (HTTP 429). "
            "Try again shortly or check your free-tier limits."
        )
    if resp.status_code != 200:
        raise GeminiError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise GeminiError(f"Unexpected Gemini response shape: {resp.text[:300]}") from exc


def generate_json(prompt: str, system: Optional[str] = None) -> dict:
    """Ask Gemini for JSON and parse it, tolerating ```json fences and stray text.

    Returns the parsed dict. Raises GeminiError if no valid JSON is found.
    """
    raw = generate(prompt, system=system, temperature=0.1)

    # Strip a ```json ... ``` fence if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else raw

    # Fall back to the first {...} block if the model added prose around it.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass
    raise GeminiError(f"Could not parse JSON from model output: {raw[:300]}")
