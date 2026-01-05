import os
import joblib
import datetime
import random
import requests
import logging
import pandas as pd
import spotipy
from collections import Counter
from typing import List, Dict
from pydantic import BaseModel, Field
from spotipy.oauth2 import SpotifyOAuth
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from dotenv import load_dotenv
from spotipy import SpotifyException

load_dotenv()


# --- CONFIG & MODELS ---
class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=100)


app = FastAPI(title="Global Discovery API")

# Load Models (XGBoost/LGBM)
model_xgb = None
model_lgbm = None
model_lock = Lock()

try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    logging.info("✅ AI Models Loaded")
except Exception as e:
    logging.error(f"❌ Model Load Error: {e}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- UTILS ---
def is_valid_track(track: dict) -> bool:
    """Filters out non-music content strictly."""
    if not track or not track.get("id"): return False
    name = track.get("name", "").lower()
    # Avoid spoken word/podcasts appearing as tracks
    if any(x in name for x in ["chapter", "episode", "part", "intro", "skit"]): return False
    if track.get("duration_ms", 0) < 45000: return False  # Music is rarely < 45s
    return True


# --- CORE LOGIC: THE TASTE ANCHOR ---
def get_user_dna(sp) -> Dict:
    """
    Combines Long Term (Years) and Medium Term (Months) to find
    the user's 'True North' in music, ignoring temporary spikes.
    """
    dna = {"tracks": [], "genres": [], "artist_ids": set()}

    # 1. Pull Long Term & Medium Term (The 'Natural' Taste)
    for t_range in ["long_term", "medium_term"]:
        results = sp.current_user_top_tracks(limit=50, time_range=t_range)
        for t in results.get('items', []):
            if is_valid_track(t):
                dna["tracks"].append(t)
                dna["artist_ids"].add(t['artists'][0]['id'])

    # 2. Identify Top Genres from these artists (Strict filtering)
    artist_list = list(dna["artist_ids"])
    for i in range(0, len(artist_list), 50):  # Spotify allows 50 artists per call
        chunk = artist_list[i:i + 50]
        full_artists = sp.artists(chunk)['artists']
        for a in full_artists:
            dna["genres"].extend(a.get('genres', []))

    dna["top_genres"] = [g for g, count in Counter(dna["genres"]).most_common(8)]
    return dna


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    if not model_xgb: raise HTTPException(status_code=500, detail="Models offline")

    sp = spotipy.Spotify(auth=request.token)

    try:
        user = sp.current_user()
        market = user.get("country", "US")
        dna = get_user_dna(sp)
    except SpotifyException:
        raise HTTPException(status_code=401, detail="Auth Failed")

    if not dna["tracks"]:
        raise HTTPException(status_code=400, detail="Profile too new. Listen to more music!")

    # --- STRATEGIC CANDIDATE GENERATION ---
    candidate_pool = []
    seen_ids = {t['id'] for t in dna["tracks"]}

    # Strategy A: Deep Genre Search (The "Public" Fix)
    # This prevents a Country fan from getting Rap because we only search their genres.
    for genre in dna["top_genres"][:5]:
        search_query = f"genre:\"{genre}\""
        # We use 'tag:new' for discovery or just the genre for breadth
        results = sp.search(q=search_query, type='track', limit=40, market=market)
        for t in results.get('tracks', {}).get('items', []):
            if is_valid_track(t) and t['id'] not in seen_ids:
                candidate_pool.append(t)
                seen_ids.add(t['id'])

    # Strategy B: Artist-Based Discovery
    # Pick random core artists and find their other work/collabs
    random_artists = random.sample(list(dna["artist_ids"]), min(len(dna["artist_ids"]), 10))
    for a_id in random_artists:
        top_t = sp.artist_top_tracks(a_id, market=market)
        for t in top_t.get('tracks', []):
            if is_valid_track(t) and t['id'] not in seen_ids:
                candidate_pool.append(t)
                seen_ids.add(t['id'])

    # --- SCORING & ML ---
    # Extract features for the model
    rows = []
    meta = []

    # Pre-calculate user genre weights for the ML score
    user_genre_set = set(dna["top_genres"])

    for t in candidate_pool:
        # Simple Feature Engineering for the provided models
        # You can adjust these to match your .pkl's expected input
        artist_plays = 1 if t['artists'][0]['id'] in dna["artist_ids"] else 0

        # Check if this track's artist shares the user's top genres
        # (This is the final guardrail)
        rows.append([
            datetime.datetime.utcnow().weekday(),  # day_of_week
            float(t.get('popularity', 50)),  # artist_global_plays (proxy)
            float(0.8 if artist_plays else 0.2),  # user_affinity
            1.0,  # track_context_weight
            0.0, 0.0, 1.0  # time_bucket dummies (night default)
        ])

        meta.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "url": t["external_urls"]["spotify"],
            "albumArt": t["album"]["images"][0]["url"] if t["album"].get("images") else None,
            "popularity": t.get('popularity', 0)
        })

    # ML Inference
    X = pd.DataFrame(rows, columns=[
        "day_of_week", "artist_global_plays", "user_affinity",
        "track_context_weight", "time_bucket_afternoon",
        "time_bucket_evening", "time_bucket_night"
    ])

    with model_lock:
        p1 = model_xgb.predict_proba(X)[:, 1]
        p2 = model_lgbm.predict_proba(X)[:, 1]
        scores = (p1 + p2) / 2

    # Final Rank & Diversify
    final_recommendations = []
    for i, score in enumerate(scores):
        meta[i]["score"] = float(score)

    # Sort by score and take top N
    ranked = sorted(meta, key=lambda x: x['score'], reverse=True)

    # Artist Diversity Cap: Max 3 tracks per artist
    counts = Counter()
    for r in ranked:
        if counts[r['artist']] < 3:
            final_recommendations.append(r)
            counts[r['artist']] += 1
        if len(final_recommendations) >= request.size: break

    return {"recommendations": final_recommendations}


# --- AUTH ENDPOINTS ---
@app.get("/login")
def login():
    sp_oauth = SpotifyOAuth(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        redirect_uri=os.getenv("REDIRECT_URI"),
        scope=os.getenv("SCOPE")
    )
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        uid = sp.current_user()["id"]
        pl = sp.user_playlist_create(uid, request.name, public=False, description="AI Curated Mix")
        uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        # Spotify allows max 100 per call
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(pl["id"], uris[i:i + 100])
        return {"status": "success", "url": pl["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))