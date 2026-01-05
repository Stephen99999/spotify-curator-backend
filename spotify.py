import os
import joblib
import datetime
import random
import logging
from collections import Counter
from typing import List

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy import SpotifyException
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

# --- APP CONFIG ---
app = FastAPI(title="Global Discovery API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
model_lock = Lock()
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ ML Models Loaded")
except Exception as e:
    print(f"❌ Model Load Error: {e}")


class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=100)


class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


# --- VIBE & TIME LOGIC ---
VIBE_MAP = {
    "morning": {"target_energy": 0.4, "target_valence": 0.6},
    "afternoon": {"target_energy": 0.8, "target_danceability": 0.7},
    "evening": {"target_energy": 0.6, "target_valence": 0.5},
    "night": {"target_energy": 0.3, "target_valence": 0.3}
}


def get_time_bucket(hour: int) -> str:
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"


# --- CORE LOGIC: THE TASTE ENGINE ---
def get_hybrid_history(sp):
    """Captures Long Term (Core) + Short Term (Recent Shifts) + Recently Played."""
    history = []
    seen = set()

    # 1. Recently Played (Instant shift - last 50 tracks)
    try:
        rp = sp.current_user_recently_played(limit=50)
        for item in rp['items']:
            tid = item['track']['id']
            if tid not in seen:
                item['track']['_origin'] = 'recent'
                history.append(item['track'])
                seen.add(tid)
    except:
        pass

    # 2. Short Term (Last 4 weeks - The 'Current Era')
    try:
        st = sp.current_user_top_tracks(limit=50, time_range='short_term')
        for t in st['items']:
            if t['id'] not in seen:
                t['_origin'] = 'short'
                history.append(t)
                seen.add(t['id'])
    except:
        pass

    # 3. Long Term (Core DNA - The 'Johnson' Anchor)
    try:
        lt = sp.current_user_top_tracks(limit=50, time_range='long_term')
        for t in lt['items']:
            if t['id'] not in seen:
                t['_origin'] = 'long'
                history.append(t)
                seen.add(t['id'])
    except:
        pass

    return history


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    sp = spotipy.Spotify(auth=request.token)

    try:
        user = sp.current_user()
        market = user.get("country", "US")
        history = get_hybrid_history(sp)

        if not history:
            raise HTTPException(status_code=400, detail="Not enough listening history.")

        # Extract User's Top Genres for strict filtering
        artist_ids = [t['artists'][0]['id'] for t in history[:50] if t.get('artists')]
        artists_data = sp.artists(artist_ids)['artists']
        genres = [g for a in artists_data for g in a.get('genres', [])]
        top_genres = [g for g, _ in Counter(genres).most_common(5)]

        # Determine "Right Now" context
        now_hour = datetime.datetime.now().hour
        bucket = get_time_bucket(now_hour)
        day_of_week = datetime.datetime.now().weekday()

        # Build Candidate Pool (Locked to User Genres + New Finds)
        candidates = []
        seen_cids = {t['id'] for t in history}

        # Strategy: Search within top genres to prevent "Algorithm Leakage"
        for genre in top_genres:
            # tag:new helps find fresh tracks in that specific genre
            q = f'genre:"{genre}"'
            results = sp.search(q=q, type='track', limit=40, market=market)
            for t in results['tracks']['items']:
                if t['id'] not in seen_cids:
                    candidates.append(t)
                    seen_cids.add(t['id'])

        if not candidates:
            # Fallback to general discovery if genre search is too narrow
            results = sp.search(q="year:2024-2025", type='track', limit=50, market=market)
            candidates.extend(results['tracks']['items'])

        # --- ML SCORING ---
        rows = []
        meta = []
        artist_play_counts = Counter([t['artists'][0]['id'] for t in history])

        for t in candidates:
            aid = t['artists'][0]['id']
            # Feature engineering for your specific .pkl models
            rows.append([
                day_of_week,
                float(t.get('popularity', 50)),  # artist_global_plays proxy
                1.0 if t['_origin'] == 'recent' else 0.5 if t['_origin'] == 'short' else 0.2,  # user_affinity
                1.0,  # track_context_weight
                1.0 if bucket == "afternoon" else 0.0,
                1.0 if bucket == "evening" else 0.0,
                1.0 if bucket == "night" else 0.0
            ])
            meta.append({
                "id": t["id"],
                "name": t["name"],
                "artist": t["artists"][0]["name"],
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

        final = sorted(meta, key=lambda x: x['score'], reverse=True)[:request.size]

        return JSONResponse(content={"recommendations": final})

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        uid = sp.current_user()["id"]
        pl = sp.user_playlist_create(uid, request.name, public=False)
        uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(pl["id"], uris[i:i + 100])
        return {"status": "success", "url": pl["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))