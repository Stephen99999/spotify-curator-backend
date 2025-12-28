import os
from dotenv import load_dotenv
import datetime
import random
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
@app.get("/recommend")
async def recommend(token: str, size: int = Query(50, ge=30, le=50)):
    sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=3)
    now = datetime.datetime.now()

    # Time Features
    day, aft, eve, ngt = now.weekday(), 0, 0, 0
    if 12 <= now.hour < 17:
        aft = 1
    elif 17 <= now.hour < 22:
        eve = 1
    elif (now.hour >= 22 or now.hour < 5):
        ngt = 1

    # --- 1. DATA FETCHING (EXPANDED POOL & DEEP MEMORY) ---
    positives = []
    negatives = []
    seed_artist_ids = set()

    try:
        # A. Gather Positives from 4 Sources (Deep Training Data)
        recent_res = sp.current_user_recently_played(limit=50)
        top_short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        top_medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        liked_res = sp.current_user_saved_tracks(limit=50)

        raw_positives = []
        if recent_res: raw_positives.extend([item['track'] for item in recent_res['items']])
        if top_short: raw_positives.extend(top_short['items'])
        if top_medium: raw_positives.extend(top_medium['items'])
        if liked_res: raw_positives.extend([item['track'] for item in liked_res['items']])

        # Deduplicate Positives and Extract Seeds
        seen_pos_ids = set()
        final_positives = []
        for track in raw_positives:
            if track['id'] not in seen_pos_ids:
                final_positives.append(track)
                seen_pos_ids.add(track['id'])
                if track['artists']:
                    seed_artist_ids.add(track['artists'][0]['id'])

        positives = final_positives

        # B. Additional Seeds from Top Artists
        top_arts = sp.current_user_top_artists(limit=20, time_range='medium_term')
        if top_arts:
            for a in top_arts['items']:
                seed_artist_ids.add(a['id'])

        if not seed_artist_ids:
            raise HTTPException(status_code=400, detail="Not enough history to generate seeds.")

        # C. Expand Pool via Related Artists (Discovery)
        discovery_seeds = random.sample(list(seed_artist_ids), min(len(seed_artist_ids), 15))
        expanded_artist_pool = list(seed_artist_ids)

        for a_id in discovery_seeds:
            try:
                related = sp.artist_related_artists(a_id)['artists']
                expanded_artist_pool.extend([r['id'] for r in related[:5]])
            except:
                continue

        # D. Collect Candidate Tracks (Potential Recommendations)
        final_artists = list(set(expanded_artist_pool))[:15]
        for a_id in final_artists:
            try:
                artist_tracks = sp.artist_top_tracks(a_id)['tracks']
                negatives.extend(artist_tracks)
            except:
                continue

    except Exception as e:
        print(f"Fetch Error: {e}")
        raise HTTPException(status_code=400, detail=f"Spotify error: {str(e)}")

    if not positives or not negatives:
        raise HTTPException(status_code=400, detail="Could not build recommendation pool.")

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