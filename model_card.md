# Model Card — Music RAG Recommender

See [README.md](README.md) for setup/examples and [diagrams/architecture.mmd](diagrams/architecture.mmd) for the architecture.

---

## Limitations and Biases

- **Popularity bias:** Last.fm retrieval favors well-known, mostly Western/English artists; niche and non-English music surfaces less often.
- **Uneven tags:** tags are community-generated, so some tracks return few or none, weakening ranking and grounding.
- **No audio understanding:** the system reasons over tags, popularity, and similarity — not how a song actually sounds.
- **Narrow window:** retrieval is capped (~10 candidates) to limit API latency.

---

## Potential for Misuse and Mitigations

- **Hallucinated songs:** the guardrail checks every recommendation against the songs Last.fm actually returned and drops anything invented — the user also sees the retrieved set.
- **Prompt abuse:** the system prompt locks the model to the task and JSON output; even a jailbreak can't surface a song outside the retrieved set.
- **Key/cost abuse:** keys live in a git-ignored `.env` and both clients fail gracefully on rate limits.

---

## What Surprised Me About Reliability

How reliably the LLM turned a vague query into a structured Last.fm request. I expected intent parsing to be brittle, but *"something like Bon Iver, chill and acoustic"* consistently became clean JSON — `{desired_tags: ["chill","acoustic"], seed_artist: "Bon Iver", popularity_pref: "any"}` — mapping straight onto the API's parameters.

---

## Collaboration With AI

- **Helpful:** the AI produced a clear, phased implementation outline (`implementation.txt`) that sequenced the build and kept dependencies between components explicit.
- **Flawed:** it first suggested running the model locally via Ollama, which required a multi-gigabyte download and install. Switching to the Gemini API — just a free key and a REST call — was far less work.
