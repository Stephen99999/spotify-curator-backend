import os
import joblib
import datetime
import random
import logging
from collections import Counter
from typing import List
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

app = FastAPI(title="Global Discovery API")

# Enable CORS for your Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OAuth
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
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
    """Captures Long Term (Core) + Short Term (Lately) + Recently Played."""
    history = []
    seen = set()

    # 1. Recently Played (Instant shift)
    try:
        rp = sp.current_user_recently_played(limit=50)
        for item in rp['items']:
            tid = item['track']['id']
            if tid not in seen:
                item['track']['_origin'] = 'recent'
                history.append(item['track'])
                seen.add(tid)
    except: pass

    # 2. Short Term (Last 4 weeks)
    try:
        st = sp.current_user_top_tracks(limit=50, time_range='short_term')
        for t in st['items']:
            if t['id'] not in seen:
                t['_origin'] = 'short'
                history.append(t)
                seen.add(t['id'])
    except: pass

    # 3. Long Term (Natural Anchor)
    try:
        lt = sp.current_user_top_tracks(limit=50, time_range='long_term')
        for t in lt['items']:
            if t['id'] not in seen:
                t['_origin'] = 'long'
                history.append(t)
                seen.add(t['id'])
    except: pass

    return history

# --- ROUTES ---

@app.get("/login")
def login():
    """Initial step: Redirects user to Spotify Login."""
    auth_url = sp_oauth.get_authorize_url()
    return RedirectResponse(auth_url)

@app.get("/callback")
def callback(code: str):
    """Second step: Spotify sends user here, we get token and send to Frontend."""
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info['access_token']
    # Redirect back to your Vercel URL
    frontend_url = f"https://spotify-playlist-curator.vercel.app/home?token={access_token}"
    return RedirectResponse(frontend_url)

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Third step: Frontend sends token here to get AI recommendations."""
    sp = spotipy.Spotify(auth=request.token)

    try:
        user = sp.current_user()
        # DYNAMIC MARKET: Fetching from user profile for 100% accuracy
        market = user.get("country", "US")
        history = get_hybrid_history(sp)

        if not history:
            raise HTTPException(status_code=400, detail="Not enough history.")

        # Genre Locking: Extract top 5 genres from user's actual history
        artist_ids = [t['artists'][0]['id'] for t in history[:50] if t.get('artists')]
        artists_data = sp.artists(artist_ids)['artists']
        genres = [g for a in artists_data for g in a.get('genres', [])]
        top_genres = [g for g, _ in Counter(genres).most_common(5)]

        now_hour = datetime.datetime.now().hour
        bucket = get_time_bucket(now_hour)
        day_of_week = datetime.datetime.now().weekday()

        candidates = []
        seen_cids = {t['id'] for t in history}

        # Search within genres to stay relevant to user (no "Algorithm Leakage")
        for genre in top_genres:
            q = f'genre:"{genre}"'
            results = sp.search(q=q, type='track', limit=40, market=market)
            for t in results['tracks']['items']:
                if t['id'] not in seen_cids:
                    # Map origin for scoring
                    t['_origin'] = 'discovery'
                    candidates.append(t)
                    seen_cids.add(t['id'])

        # ML SCORING
        rows = []
        meta = []
        for t in candidates:
            rows.append([
                day_of_week,
                float(t.get('popularity', 50)),
                0.5, # user_affinity baseline for discovery
                1.0, # track_context_weight
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
    """Final step: Save selected tracks to user's library."""
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