import datetime
import logging
import os
from threading import Lock
from typing import List

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

app = FastAPI(title="Public Universal Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared OAuth object (stateless)
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE
)

# ML MODELS (Loaded once, used as read-only by multiple threads)
model_lock = Lock()
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ ML Models Ready for Multi-User Traffic")
except Exception as e:
    print(f"❌ Model Load Error: {e}")


# --- SCHEMAS ---
class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=100)


class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


# --- HELPER: NO GLOBAL LEAKAGE ---
def get_time_bucket() -> str:
    hour = datetime.datetime.now().hour
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"


# --- ROUTES ---

@app.get("/login")
def login():
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info['access_token']
    # Redirecting to Vercel with the private token
    return RedirectResponse(f"https://spotify-playlist-curator.vercel.app/home?token={access_token}")


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user_info = sp.current_user()
        market = user_info.get("country", "US")
        vibe = get_time_bucket()
        day_of_week = datetime.datetime.now().weekday()

        # 1. INDEPENDENT SEED FETCHING
        # Pull Short Term (Weight 1.0) and Mid Term (Weight 0.5)
        st_data = sp.current_user_top_tracks(limit=15, time_range='short_term')['items']
        mt_data = sp.current_user_top_tracks(limit=15, time_range='medium_term')['items']

        seeds = []
        for t in st_data: seeds.append((t['artists'][0], 1.0))
        for t in mt_data: seeds.append((t['artists'][0], 0.5))

        candidates = []
        seen_tracks = set()

        # 2. FAIL-SAFE DISCOVERY LOOP
        # We iterate through each artist seed separately.
        # If one seed hits a 404 or fails, we 'continue' to the next one.
        for artist, weight in seeds[:10]:
            try:
                # STRATEGY: Instead of 'related_artists' (404 risk),
                # we search for tracks in the same genres as the seed artist.
                artist_details = sp.artist(artist['id'])
                genres = artist_details.get('genres', [])

                if not genres:
                    continue

                # Search for 'new' tracks in this specific user's genre
                search_query = f'genre:"{genres[0]}"'
                results = sp.search(q=search_query, type='track', limit=10, market=market)

                for track in results['tracks']['items']:
                    if track['id'] not in seen_tracks:
                        # Logic check: Don't recommend the artist they already listen to
                        if track['artists'][0]['id'] != artist['id']:
                            track['_vibe_affinity'] = weight
                            candidates.append(track)
                            seen_tracks.add(track['id'])

            except Exception as seed_err:
                # Individual seed failed? Log it and move on. No crash.
                logging.warning(f"Skipping seed {artist['name']}: {seed_err}")
                continue

        # 3. ML SCORING (Remains Private to this User's Request)
        if not candidates:
            return JSONResponse(status_code=404,
                                content={"error": "Vibe pool empty. Try listening to more niche music."})

        rows, meta = [], []
        for t in candidates:
            rows.append([
                day_of_week, float(t.get('popularity', 50)),
                t.get('_vibe_affinity', 0.5), 1.0,
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
            "vibe": vibe,
            "session_user": user_info['display_name']
        })

    except Exception as e:
        logging.error(f"FATAL PIPELINE ERROR: {e}")
        return JSONResponse(status_code=500, content={"error": "System overload or API failure."})


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
