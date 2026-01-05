import os
import joblib
import datetime
import random
import logging
from collections import Counter
from typing import List, Optional
from threading import Lock

import pandas as pd
import spotipy
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

# --- CONFIG ---
SCOPE = "user-read-private user-top-read user-read-recently-played playlist-modify-private"
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")

app = FastAPI(title="Universal Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to ["https://spotify-playlist-curator.vercel.app"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
)

# --- ML MODELS ---
model_lock = Lock()
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ Multi-User ML Models Loaded")
except Exception as e:
    print(f"❌ ML Model Error: {e}")


# --- MODELS ---
class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=100)


class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


# --- CORE UTILS ---

def get_time_bucket() -> str:
    """Determine vibe context based on server time (can be adjusted for user local)."""
    hour = datetime.datetime.now().hour
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"


def get_user_top_genres(sp) -> List[str]:
    """Dynamically extract the user's current genre fingerprint."""
    try:
        # Fetch top artists to get accurate genre tags
        top_artists = sp.current_user_top_artists(limit=20, time_range='short_term')['items']
        genres = [g for a in top_artists for g in a.get('genres', [])]
        return [g for g, _ in Counter(genres).most_common(5)]
    except Exception:
        return ["pop", "chill"]  # Robust fallback


# --- ROUTES ---

@app.get("/login")
def login():
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info['access_token']
    frontend_url = f"https://spotify-playlist-curator.vercel.app/home?token={access_token}"
    return RedirectResponse(frontend_url)


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user_profile = sp.current_user()
        market = user_profile.get("country", "US")
        top_genres = get_user_top_genres(sp)
        vibe = get_time_bucket()
        day_of_week = datetime.datetime.now().weekday()

        candidates = []
        seen_ids = set()

        # --- REFINED SEARCH WITH FALLBACKS ---
        for genre in top_genres:
            # Stage 1: Specific & New (The Gold Standard)
            queries = [f'genre:"{genre}" tag:new', f'genre:"{genre}"']

            for q in queries:
                results = sp.search(q=q, type='track', limit=40, market=market)
                items = results['tracks']['items']

                if items:
                    for t in items:
                        if t['id'] not in seen_ids:
                            candidates.append(t)
                            seen_ids.add(t['id'])
                    break  # Stop if we found tracks for this genre

        # Stage 2: Universal Fallback (If the user has ultra-niche taste)
        if len(candidates) < 10:
            fallback_res = sp.search(q='tag:new', type='track', limit=50, market=market)
            candidates.extend(fallback_res['tracks']['items'])

        # --- ML SCORING (Remains the same) ---
        if not candidates:
            raise HTTPException(status_code=404, detail="Spotify returned zero tracks for these filters.")

        rows, meta = [], []
        for t in candidates:
            rows.append([
                day_of_week, float(t.get('popularity', 50)),
                0.65, 1.0,
                1.0 if vibe == "afternoon" else 0.0,
                1.0 if vibe == "evening" else 0.0,
                1.0 if vibe == "night" else 0.0
            ])
            meta.append({
                "id": t["id"], "name": t["name"], "artist": t["artists"][0]["name"],
                "url": t["external_urls"]["spotify"],
                "albumArt": t["album"]["images"][0]["url"] if t["album"].get("images") else None
            })

        X = pd.DataFrame(rows, columns=[
            "day_of_week", "artist_global_plays", "user_affinity",
            "track_context_weight", "time_bucket_afternoon",
            "time_bucket_evening", "time_bucket_night"
        ])

        with model_lock:
            p1 = model_xgb.predict_proba(X)[:, 1]
            p2 = model_lgbm.predict_proba(X)[:, 1]
            scores = (p1 + p2) / 2

        for i, score in enumerate(scores):
            meta[i]["score"] = float(score)

        return JSONResponse(content={
            "recommendations": sorted(meta, key=lambda x: x['score'], reverse=True)[:request.size],
            "vibe_detected": vibe
        })

    except Exception as e:
        logging.error(f"Pipeline Error: {str(e)}")
        # Return a clean error instead of a 500 crash
        return JSONResponse(status_code=500, content={"error": "Recommendation failed", "details": str(e)})


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(user_id, request.name, public=False)
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids]

        # Batch upload to avoid timeout
        for i in range(0, len(track_uris), 100):
            sp.playlist_add_items(playlist["id"], track_uris[i:i + 100])

        return {"status": "success", "url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))