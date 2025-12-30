import os
import datetime
import pandas as pd
import joblib
import spotipy
from typing import List
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE = os.getenv("SCOPE")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. LOAD PRE-TRAINED MODELS ---
# These are the brains you trained with your data.
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ AI Models loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: .pkl files not found. Ensure discovery_xgb_finetuned.pkl is in the folder.")


# --- 2. DYNAMIC FEATURE ENGINEERING ---
def get_time_bucket(hour):
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    if 17 <= hour < 22: return "evening"
    return "night"


def extract_features_dynamic(track, now_dt, top_artist_names):
    """
    Translates LIVE Spotify data into the 7 features the model expects.
    """
    day = now_dt.weekday()
    hour = now_dt.hour
    bucket = get_time_bucket(hour)

    # Feature 1: Artist Popularity (Using Spotify's 0-100 scale)
    # This replaces the 'global_plays' from your JSON
    artist_pop = track['artists'][0].get('popularity', 50)

    # Feature 2: User Affinity
    # If the user already follows this artist, we give a high affinity score (0.9)
    # Otherwise, we give a baseline score (0.4)
    is_fave = 0.9 if track['artists'][0]['name'] in top_artist_names else 0.4

    # Feature 3: Track Context Weight
    # We use the track's popularity normalized to 0-1
    track_weight = track.get('popularity', 50) / 100

    # One-Hot Encoding for Time
    is_aft = 1 if bucket == 'afternoon' else 0
    is_eve = 1 if bucket == 'evening' else 0
    is_ngt = 1 if bucket == 'night' else 0

    # MUST MATCH THE 7 FEATURES IN ORDER:
    # [day_of_week, artist_global_plays, user_affinity, track_context_weight, aft, eve, ngt]
    return [day, artist_pop, is_fave, track_weight, is_aft, is_eve, is_ngt]


# --- 3. API ENDPOINTS ---

class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "AI Discovery Mix"


@app.get("/login")
def login():
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    token_info = sp_oauth.get_access_token(code)
    return RedirectResponse(url=f"https://spotify-playlist-curator.vercel.app/home?token={token_info['access_token']}")


@app.get("/recommend")
async def recommend(token: str, size: int = Query(50, ge=10, le=100)):
    sp = spotipy.Spotify(auth=token)

    try:
        # A. Get CURRENT user's preferences to use as "Context"
        # We get their top artists to see if the new songs match their taste
        user_top = sp.current_user_top_artists(limit=20, time_range='medium_term')['items']
        top_artist_names = [a['name'] for a in user_top]
        seed_ids = [a['id'] for a in user_top[:5]]  # Use top 5 as seeds for discovery

        # B. Fetch candidates from Spotify Recommendations
        # This gives us a pool of songs similar to their current favorites
        raw_recs = sp.recommendations(seed_artists=seed_ids, limit=size + 20)
        candidates = raw_recs['tracks']

        # C. Rank candidates using YOUR AI Model
        now = datetime.datetime.now()
        feature_rows = []
        meta = []

        for track in candidates:
            # Skip if we don't have enough data on the track
            if not track.get('id'): continue

            # Extract the 7 features for this specific user
            feats = extract_features_dynamic(track, now, top_artist_names)
            feature_rows.append(feats)

            meta.append({
                "id": track['id'],
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "url": track['external_urls']['spotify'],
                "albumArt": track['album']['images'][0]['url'] if track['album']['images'] else ""
            })

        # D. Run Model Prediction
        cols = ['day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
                'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night']
        X_pred = pd.DataFrame(feature_rows, columns=cols)

        # Ensemble Scores
        scores = (model_xgb.predict_proba(X_pred)[:, 1] + model_lgbm.predict_proba(X_pred)[:, 1]) / 2

        # Attach scores and sort
        for i in range(len(meta)):
            meta[i]['score'] = float(scores[i])

        final_list = sorted(meta, key=lambda x: x['score'], reverse=True)[:size]
        return {"recommendations": final_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user = sp.current_user()
        playlist = sp.user_playlist_create(user['id'], request.name, public=False)
        uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        sp.playlist_add_items(playlist['id'], uris)
        return {"status": "success", "url": playlist['external_urls']['spotify']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))