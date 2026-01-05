import datetime
import logging
import os
from threading import Lock
from typing import List
from collections import Counter
import joblib
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

app = FastAPI(title="Private Style Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

model_lock = Lock()
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ ML Models Ready")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=100)

class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"

# --- HELPER: USER-CENTRIC CONTEXT ---
def get_time_bucket() -> str:
    hour = datetime.datetime.now().hour
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"

@app.post("/recommend")
async def recommend(request: RecommendRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        # 1. PROFILE FINGERPRINTING (The Style Anchor)
        user = sp.current_user()
        market = user.get("country", "US")

        # Pull personal history
        st_tracks = sp.current_user_top_tracks(limit=50, time_range='short_term')['items']
        mt_tracks = sp.current_user_top_tracks(limit=50, time_range='medium_term')['items']

        # 2. ISOLATION: Calculate User-Specific Weights
        # This prevents "leakage" because 'user_affinity' is now unique to this user's data
        style_seeds = []
        for t in st_tracks: style_seeds.append((t['artists'][0]['id'], 1.0))
        for t in mt_tracks: style_seeds.append((t['artists'][0]['id'], 0.5))

        artist_counts = Counter([s[0] for s in style_seeds])
        top_style_artists = [a for a, _ in artist_counts.most_common(12)]

        # 3. CANDIDATE SCAVENGING
        candidate_pool = []
        seen_ids = set([t['id'] for t in st_tracks] + [t['id'] for t in mt_tracks])

        for artist_id in top_style_artists:
            try:
                # Use Related Artists to find sonic neighbors
                related = sp.artist_related_artists(artist_id)['artists'][:5]
                for r_art in related:
                    r_tracks = sp.artist_top_tracks(r_art['id'], country=market)['tracks'][:3]
                    for rt in r_tracks:
                        if rt['id'] not in seen_ids:
                            # User-specific affinity score (prevents global leakage)
                            rt['_user_affinity'] = float(artist_counts[artist_id] / max(artist_counts.values()))
                            candidate_pool.append(rt)
                            seen_ids.add(rt['id'])
            except:
                continue

        # 4. ML SCORING (Fixed for Dtypes and Isolation)
        vibe = get_time_bucket()
        day = float(datetime.datetime.now().weekday())
        rows, meta = [], []

        for t in candidate_pool:
            # Explicitly casting to float to satisfy XGBoost/LGBM Requirements
            rows.append([
                day,
                float(t.get('popularity', 50)),
                float(t.get('_user_affinity', 0.5)),
                1.0,  # Personal track context weight
                1.0 if vibe == "afternoon" else 0.0,
                1.0 if vibe == "evening" else 0.0,
                1.0 if vibe == "night" else 0.0
            ])
            meta.append({
                "id": t["id"],
                "name": t["name"],
                "artist": t["artists"][0]["name"],
                "url": t["external_urls"]["spotify"],
                "albumArt": t["album"]["images"][0]["url"] if t["album"].get("images") else None
            })

        # Convert to DataFrame and FORCE float types
        X = pd.DataFrame(rows, columns=[
            "day_of_week", "artist_global_plays", "user_affinity",
            "track_context_weight", "time_bucket_afternoon",
            "time_bucket_evening", "time_bucket_night"
        ]).astype('float64')

        with model_lock:
            p1 = model_xgb.predict_proba(X)[:, 1]
            p2 = model_lgbm.predict_proba(X)[:, 1]
            scores = (p1 + p2) / 2

        for i, score in enumerate(scores):
            meta[i]["score"] = float(score)

        # 5. DIVERSITY FILTER
        final = []
        artist_cap = Counter()
        for item in sorted(meta, key=lambda x: x['score'], reverse=True):
            if artist_cap[item["artist"]] < 3:
                final.append(item)
                artist_cap[item["artist"]] += 1
            if len(final) >= request.size: break

        return JSONResponse(content={"recommendations": final, "vibe": vibe})

    except Exception as e:
        logging.error(f"Personalization Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(user_id, request.name, public=False)
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        for i in range(0, len(track_uris), 100):
            sp.playlist_add_items(playlist["id"], track_uris[i:i + 100])
        return {"status": "success", "url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))