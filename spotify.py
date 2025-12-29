import os
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
    pop = track.get('popularity', 50)
    user_affinity = 1.0 if is_recent else 0.5
    return [day, pop, user_affinity, 0.5, aft, eve, ngt]


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
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_handler=None
    )

    auth_manager.token_info = {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": 3600,
    }

    sp = spotipy.Spotify(auth_manager=auth_manager, requests_timeout=10, retries=3)
    now = datetime.datetime.now()

    # Time Features
    day, aft, eve, ngt = now.weekday(), 0, 0, 0
    if 12 <= now.hour < 17:
        aft = 1
    elif 17 <= now.hour < 22:
        eve = 1
    elif (now.hour >= 22 or now.hour < 5):
        ngt = 1

    # --- 1. DATA FETCHING (IMPROVED: FREQUENCY-BASED SEEDING) ---
    positives = []
    negatives = []

    # We will track all artist IDs found in positives to count them later
    all_artist_ids = []

    try:
        # A. Gather Positives from 4 Sources
        recent_res = sp.current_user_recently_played(limit=50)
        top_short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        top_medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        liked_res = sp.current_user_saved_tracks(limit=50)

        raw_positives = []
        if recent_res: raw_positives.extend([item['track'] for item in recent_res['items']])
        if top_short: raw_positives.extend(top_short['items'])
        if top_medium: raw_positives.extend(top_medium['items'])
        if liked_res: raw_positives.extend([item['track'] for item in liked_res['items']])

        # Deduplicate Positives and Collect Artist Frequencies
        seen_pos_ids = set()
        final_positives = []

        for track in raw_positives:
            # Skip invalid tracks or local files
            if not track or not track.get('id'):
                continue

            if track['id'] not in seen_pos_ids:
                final_positives.append(track)
                seen_pos_ids.add(track['id'])

                # Collect primary artist ID for frequency analysis
                if track['artists']:
                    all_artist_ids.append(track['artists'][0]['id'])

        positives = final_positives

        if not all_artist_ids:
            raise HTTPException(status_code=400, detail="Not enough history to generate seeds.")

        # B. Identify "Heavy Hitters" (Top 5 Most Frequent Artists)
        # This replaces the generic 'sp.current_user_top_artists' call
        artist_counts = Counter(all_artist_ids)

        # Get the top 5 most common artist IDs
        top_frequent_artists = [artist_id for artist_id, count in artist_counts.most_common(5)]

        # C. Controlled Diversification (Expand Pool)
        # We start with our Heavy Hitters to ensure the core vibe is present
        expanded_artist_pool = list(top_frequent_artists)

        # We find related artists ONLY for these heavy hitters
        # This keeps the 'diversity' relevant to what you are actually listening to.
        for a_id in top_frequent_artists:
            try:
                related = sp.artist_related_artists(a_id)['artists']
                # Take top 3 related artists per heavy hitter (Don't dilute too much)
                expanded_artist_pool.extend([r['id'] for r in related[:3]])
            except:
                continue

        # Remove duplicates from the artist pool
        final_seed_artists = list(set(expanded_artist_pool))

        # D. Collect Candidate Tracks (Potential Recommendations)
        # We fetch top tracks for this curated list of artists
        for a_id in final_seed_artists:
            try:
                # We can grab up to 10 top tracks per artist
                artist_tracks = sp.artist_top_tracks(a_id)['tracks']
                negatives.extend(artist_tracks)
            except:
                continue

    except Exception as e:
        print(f"Fetch Error: {e}")
        raise HTTPException(status_code=400, detail=f"Spotify error: {str(e)}")

    if not positives or not negatives:
        raise HTTPException(status_code=400, detail="Could not build recommendation pool.")

    # ... (Step 2 and 3: Feature Engineering and Model Training remain exactly the same) ...

    # --- 2. REAL-TIME AI TRAINING ---
    training_data = []
    for p in positives:
        feats = extract_features(p, now, day, aft, eve, ngt, is_recent=True)
        training_data.append(feats + [1])

    # Increased sample to 200 to match the deeper positive data
    train_negs = random.sample(negatives, min(len(negatives), 200))
    for n in train_negs:
        feats = extract_features(n, now, day, aft, eve, ngt, is_recent=False)
        training_data.append(feats + [0])

    cols = ['day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
            'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night', 'label']
    df = pd.DataFrame(training_data, columns=cols)

    X_train = df.drop('label', axis=1)
    y_train = df['label']

    # Using 50 estimators for slightly better learning with the larger dataset
    model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss')
    model_lgbm = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1)

    model_xgb.fit(X_train, y_train)
    model_lgbm.fit(X_train, y_train)

    # --- 3. RANKING ---
    meta = []
    prediction_rows = []
    # Deduplicate candidate pool
    unique_pool = {t['id']: t for t in negatives if t and 'id' in t}.values()

    for t in unique_pool:
        # Don't recommend songs the user already has in their 'Positive' training set
        if t['id'] in seen_pos_ids:
            continue

        feats = extract_features(t, now, day, aft, eve, ngt, is_recent=False)
        prediction_rows.append(feats)
        meta.append({
            "id": t['id'],
            "name": t['name'],
            "artist": t['artists'][0]['name'],
            "pop": t['popularity'],
            "url": t['external_urls']['spotify'],
            "albumArt": t['album']['images'][0]['url'] if t['album']['images'] else ""
        })

    if not prediction_rows:
        raise HTTPException(status_code=400, detail="No new songs found to recommend.")

    X_pred = pd.DataFrame(prediction_rows, columns=cols[:-1])
    scores = (model_xgb.predict_proba(X_pred)[:, 1] + model_lgbm.predict_proba(X_pred)[:, 1]) / 2

    for i in range(len(meta)):
        meta[i]['score'] = float(scores[i])

    final_recs = sorted(meta, key=lambda x: x['score'], reverse=True)[:size]
    return {"recommendations": final_recs}

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