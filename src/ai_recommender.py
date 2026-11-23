from google import genai
import os
from src.recommender import Song, UserProfile, recommend_songs
from typing import List, Dict, Tuple, Optional
import csv
import numpy as np

EMBED_MODEL = "gemini-embeding-001"
GEMINI_MODEL_NAME = "gemini-flash-lite-latest"

api_key = os.getenv("GEMINI_API_KEY")

def client() -> genai.Client:

    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY environment variable. "
        )

    client = genai.Client(api_key=api_key)
    return client

def song_text(song: Song) -> str:
    return f"{song.title} by {song.artist}. Genre: {song.genre}. Mood: {song.mood}."


def embed_songs(songs: List[Song]) -> Dict[int, np.ndarray]:
    """Embed each song once. Returns {song.id: vector}."""

    vectors = {}
    for song in songs:
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=song_text(song),
        )
        vectors[song.id] = np.array(result.embeddings[0].values)
    return vectors

def rag_recommend(
    user_prefs: UserProfile,
    songs: List[Song],
    song_vectors: Dict[int, np.ndarray],
    client: genai.Client,
    k: int = 5,
    candidate_pool: int = 15,
) -> str:
    """
    Retrieve the top candidates with the scoring logic, then let Gemini
    choose the best k and explain why.
    """
    # 1. Retrieve: existing scoring gives the candidate pool
    candidates = recommend_songs(user_prefs, songs, k=candidate_pool)

    listing = "\n".join(
        f"- id={song.id} | {song.title} by {song.artist} | genre: {song.genre} | "
        f"mood: {song.mood} | energy: {song.energy} | valence: {song.valence} | "
        f"acousticness: {song.acousticness} | score: {score:.2f}"
        for song, score, _ in candidates
    )

    # 2. Generate: Gemini picks from the candidates only
    prompt = (
        "You are a music recommender. Pick the best songs for this user.\n\n"
        f"User profile: favorite genre={user_prefs.favorite_genre}, "
        f"target energy={user_prefs.target_energy}, "
        f"valence={user_prefs.valence_lvl}, "
        f"acousticness={user_prefs.acoustic_lvl}.\n\n"
        f"Candidate songs (pre-scored by the retriever):\n{listing}\n\n"
        f"Choose the {k} best songs, in order. Only choose from the candidate list. "
        "For each, give the song title, and one sentence on why it fits this user."
    )
    response = client.models.generate_content(model=GEMINI_MODEL_NAME, contents=prompt)
    return response.text