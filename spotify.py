import os

import joblib
from dotenv import load_dotenv
import datetime
import random
from collections import Counter
import pandas as pd
import spotipy
import xgboost as xgb
import lightgbm as lgb
from pydantic import BaseModel
from typing import List
from spotipy.oauth2 import SpotifyOAuth
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# --- MODELS ---
class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


app = FastAPI()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE = os.getenv("SCOPE")
# Load this once so the API doesn't have to re-read files every request
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ AI Models Loaded and Ready.")
except Exception as e:
    print(f"❌ Warning: Models not found. Ranking will be random. {e}")
    model_xgb, model_lgbm = None, None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)


# --- HELPER: FEATURE ENGINEERING ---
def extract_features(track, now, day, aft, eve, ngt, is_recent=False):
    # Feature 1: Popularity (Standardized 0-100)
    pop = track.get('popularity', 50)

    # Feature 2: Artist Global Plays (Simulated via your session)
    # If it's a 'positive' track, we treat it as a high-play artist
    artist_plays = 100 if is_recent else 10

    # Feature 3: User Affinity
    # High for recently played/liked, lower for new discoveries
    user_affinity = 1.0 if is_recent else 0.3

    # Feature 4: Track Context Weight
    # We can use the track's explicit property or duration as a proxy for 'weight'
    context_weight = 0.8 if track.get('explicit') else 0.5

    # Must match the order of your pretrained model columns!
    return [day, artist_plays, user_affinity, context_weight, aft, eve, ngt]


@app.get("/login")
def login():
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info['access_token']
    return RedirectResponse(url=f"https://spotify-playlist-curator.vercel.app/home?token={access_token}")





@app.get("/recommend")
async def recommend(token: str, size: int = Query(50, ge=30, le=50)):
    sp = spotipy.Spotify(auth=token)
    # Ensure we bypass the Google Cloud Shell proxy issue
    sp.prefix = "https://api.spotify.com/v1/"

    now = datetime.datetime.now()
    day = now.weekday()
    aft, eve, ngt = (1, 0, 0) if 12 <= now.hour < 17 else (0, 1, 0) if 17 <= now.hour < 22 else (0, 0, 1) if (
                now.hour >= 22 or now.hour < 5) else (0, 0, 0)

    try:
        # --- 1. SMART FETCHING ---
        # Get history to understand current "Vibe"
        recent = sp.current_user_recently_played(limit=20)['items']
        top_tracks = sp.current_user_top_tracks(limit=20, time_range='short_term')['items']
        liked = sp.current_user_saved_tracks(limit=20)['items']

        seen_ids = set(
            [t['track']['id'] for t in recent] + [t['id'] for t in top_tracks] + [t['track']['id'] for t in liked])

        # Identify your "Heavy Hitter" Artists from this session
        artist_ids = [t['track']['artists'][0]['id'] for t in recent] + [t['artists'][0]['id'] for t in top_tracks]
        heavy_hitters = [a_id for a_id, count in Counter(artist_ids).most_common(5)]

        # --- 2. CANDIDATE GENERATION ---
        candidates = {}
        for a_id in heavy_hitters:
            related = sp.artist_related_artists(a_id)['artists'][:3]
            for rel in related:
                # Use their top tracks as candidates
                for track in sp.artist_top_tracks(rel['id'])['tracks'][:10]:
                    if track['id'] not in seen_ids:
                        candidates[track['id']] = track

        if not candidates:
            # Emergency Fallback to Top Global so frontend doesn't get 'null'
            fallback = sp.playlist_tracks("37i9dQZEVXbMDoHDwfs2t3", limit=size)
            candidates = {t['track']['id']: t['track'] for t in fallback['items']}

        # --- 3. AI RANKING ---
        meta = []
        rows = []
        for tid, t in candidates.items():
            # Matching the 7-feature structure of your pre-trained model
            # [day, popularity, user_affinity, track_weight, is_aft, is_eve, is_ngt]
            rows.append([day, t['popularity'], 0.4, 0.5, aft, eve, ngt])
            meta.append({
                "id": t['id'], "name": t['name'], "artist": t['artists'][0]['name'],
                "url": t['external_urls']['spotify'], "pop": t['popularity'],
                "albumArt": t['album']['images'][0]['url'] if t['album']['images'] else ""
            })

        # Calculate Scores
        df_pred = pd.DataFrame(rows,
                               columns=['day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
                                        'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night'])

        if model_xgb and model_lgbm:
            scores = (model_xgb.predict_proba(df_pred)[:, 1] + model_lgbm.predict_proba(df_pred)[:, 1]) / 2
        else:
            scores = [0.5] * len(meta)  # Fallback if model files are missing

        for i in range(len(meta)):
            # Discovery Boost: Nudge up songs that aren't mainstream (pop < 70)
            discovery_boost = 1.1 if meta[i]['pop'] < 70 else 1.0
            meta[i]['score'] = float(scores[i]) * discovery_boost

        return {"recommendations": sorted(meta, key=lambda x: x['score'], reverse=True)[:size]}

    except Exception as e:
        print(f"Error: {e}")
        # ALWAYS return a valid recommendations list even if it's empty to prevent frontend crash
        return {"recommendations": []}

@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token, requests_timeout=10)
    try:
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(
            user=user_id,
            name=request.name,
            public=False,
            description="Created by AI Discovery Recommender"
        )
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)
        return {"status": "success", "playlist_url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        print(f"Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))