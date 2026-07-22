import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of dicts."""
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            })
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences, returning (score, reasons)."""
    # Weighted "judge" algorithm from the README. Each feature returns a
    # sub-score in [0, 1] that is multiplied by its weight; the weights are the
    # only thing that decides how important each feature is. Max score = 7.5.
    #
    #   score = 3.0*genre + 2.0*mood + 1.5*energy + 1.0*acoustic
    #
    # user_prefs keys: genre, mood, energy, likes_acoustic (acoustic optional).
    score = 0.0
    reasons: List[str] = []

    # genre: exact match, weight 3.0
    fav_genre = user_prefs.get("genre")
    if fav_genre is not None:
        genre_sub = 1.0 if song["genre"] == fav_genre else 0.0
        score += 3.0 * genre_sub
        if genre_sub:
            reasons.append(f"matches your favorite genre ({fav_genre})")

    # mood: exact match, weight 2.0
    fav_mood = user_prefs.get("mood")
    if fav_mood is not None:
        mood_sub = 1.0 if song["mood"] == fav_mood else 0.0
        score += 2.0 * mood_sub
        if mood_sub:
            reasons.append(f"matches your favorite mood ({fav_mood})")

    # energy: closeness, weight 1.5
    target_energy = user_prefs.get("energy")
    if target_energy is not None:
        energy_sub = max(0.0, 1.0 - abs(song["energy"] - target_energy))
        score += 1.5 * energy_sub
        if energy_sub >= 0.8:
            reasons.append(f"energy ({song['energy']:.2f}) is close to your target")

    # acousticness: directional, weight 1.0
    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        acoustic_sub = song["acousticness"] if likes_acoustic else 1.0 - song["acousticness"]
        score += 1.0 * acoustic_sub
        if acoustic_sub >= 0.7:
            reasons.append("acoustic" if likes_acoustic else "not too acoustic")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the k highest-scoring songs as (song, score, explanation) tuples."""
    # Score every song, then keep the k highest. Each item is
    # (song_dict, score, explanation).
    scored = [
        (song, score, ", ".join(reasons) if reasons else "no strong matches")
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]