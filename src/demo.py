"""Command-line demo of the RAG music recommender.

Usage:
    python src/demo.py                # runs the built-in example queries
    python src/demo.py "your query"   # runs a single custom query

Prints, for each query: the parsed intent, the grounded recommendations, any
guardrail/status warnings, and the retrieved candidate set. The final section
demonstrates the guardrail rejecting a deliberately hallucinated pick.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag
from recommender import Song

EXAMPLE_QUERIES = [
    "something like Bon Iver, chill and acoustic",
    "upbeat pop for a workout, mainstream is fine",
    "moody indie rock for a rainy night",
]

RULE = "=" * 70


def run_query(query: str, k: int = 4) -> None:
    print(RULE)
    print(f"INPUT:  {query!r}")
    print(RULE)
    result = rag.recommend(query, k=k)

    intent = result.intent
    print(f"Parsed intent -> tags={intent.desired_tags} "
          f"seed={intent.seed_artist!r} popularity={intent.popularity_pref}")
    print()
    if result.intro:
        print(result.intro)
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec.title} - {rec.artist}")
        print(f"     why: {rec.reason}")

    if result.warnings:
        print("\n  [guardrail/status]")
        for w in result.warnings:
            print(f"    - {w}")

    print(f"\n  Retrieved candidates ({len(result.candidates)}) [grounding source]:")
    for s in result.candidates:
        tags = ", ".join(name for name, _ in s.tags[:5]) or "no tags"
        print(f"    - {s.title} - {s.artist}  [{tags}]")
    print()


def guardrail_demo() -> None:
    """Show the guardrail dropping a song the model invented."""
    print(RULE)
    print("GUARDRAIL DEMONSTRATION (injected hallucination)")
    print(RULE)

    # A fixed, retrieved candidate set (what Last.fm 'returned').
    ranked = [
        (Song("The Scientist", "Coldplay", tags=[("rock", 100)], listeners=5000),
         3.0, ["tagged rock"]),
        (Song("Fix You", "Coldplay", tags=[("rock", 90)], listeners=4000),
         2.8, ["tagged rock"]),
    ]
    print("Retrieved set: 'The Scientist', 'Fix You' (both Coldplay)\n")

    # Force the model to return one real pick and one invented pick.
    original = rag._generate_once
    rag._generate_once = lambda q, c, k: {
        "intro": "Here are some picks:",
        "picks": [
            {"title": "The Scientist", "artist": "Coldplay", "reason": "real, retrieved"},
            {"title": "Imaginary Anthem", "artist": "Nobody", "reason": "INVENTED - not retrieved"},
        ],
    }
    try:
        result = rag.generate("coldplay-ish rock", ranked, k=5)
    finally:
        rag._generate_once = original

    print("Model proposed: 'The Scientist' (real) + 'Imaginary Anthem' (invented)\n")
    print("After guardrail:")
    for rec in result.recommendations:
        print(f"  KEPT: {rec.title} - {rec.artist}")
    for w in result.warnings:
        print(f"  {w}")
    print("\nResult: the invented song was rejected; only grounded picks remain.\n")


def main() -> None:
    args = sys.argv[1:]
    queries = [" ".join(args)] if args else EXAMPLE_QUERIES
    for q in queries:
        run_query(q)
    if not args:
        guardrail_demo()


if __name__ == "__main__":
    main()
