import os
import joblib
import datetime
import random
import time
from collections import Counter
import pandas as pd
import spotipy
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from spotipy.oauth2 import SpotifyOAuth
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from dotenv import load_dotenv

load_dotenv()


# --- CONFIG & MODELS ---
class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=100)


app = FastAPI(title="Discovery Playlist API")

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE = os.getenv("SCOPE")

# Load Models
model_xgb = None
model_lgbm = None
model_lock = Lock()

try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ AI Models Loaded")
except Exception as e:
    print(f"❌ Model Load Error: {e}")

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


# --- DIVERSITY ENGINE (NEW) ---
def get_jittered_params():
    """
    Returns random tunable attributes to force Spotify's API
    to look in different 'corners' of the music graph.
    """
    targets = {}

    # Roll a die to decide the 'vibe' of this specific batch
    roll = random.random()

    if roll < 0.25:
        # Mode 1: High Energy / Upbeat
        targets['min_energy'] = 0.6
        targets['target_energy'] = 0.8
    elif roll < 0.50:
        # Mode 2: Chill / Acoustic
        targets['max_energy'] = 0.6
        targets['target_acousticness'] = 0.7
    elif roll < 0.75:
        # Mode 3: Deep Cuts (Low Popularity)
        targets['max_popularity'] = 60
    else:
        # Mode 4: High Valence (Happy)
        targets['min_valence'] = 0.6

    # Always add a little bit of chaos to danceability
    targets['target_danceability'] = random.uniform(0.3, 0.9)

    return targets


# --- FEATURE HELPERS ---
def get_time_bucket(hour: int) -> str:
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"


def extract_features(artist_id: str, day: int, current_time_bucket: str, artist_play_count: int,
                     user_time_buckets: List[str], track_time_buckets: List[str]):
    def get_affinity(hist, bucket):
        return (sum(1 for h in hist if h == bucket) / len(hist)) if hist else 0.5

    return [
        day,
        float(artist_play_count),
        float(get_affinity(user_time_buckets, current_time_bucket)),
        float(get_affinity(track_time_buckets, current_time_bucket)),
        1.0 if current_time_bucket == "afternoon" else 0.0,
        1.0 if current_time_bucket == "evening" else 0.0,
        1.0 if current_time_bucket == "night" else 0.0
    ]


# --- HYBRID HISTORY FETCHING ---
def get_user_context_hybrid(sp):
    combined_tracks = []
    seen_ids = set()

    # 1. Recent History (Immediate Context)
    try:
        recent_results = sp.current_user_recently_played(limit=50)
        for item in recent_results.get('items', []):
            track = item.get("track")
            if track and track.get("id") and track["id"] not in seen_ids:
                track["_source"] = "recent"
                track["played_at"] = item.get("played_at")
                combined_tracks.append(track)
                seen_ids.add(track["id"])
    except:
        pass

    # 2. Top Tracks (Broad Context)
    try:
        top_results = sp.current_user_top_tracks(limit=50, time_range="short_term")
        for track in top_results.get('items', []):
            if track and track.get("id") and track["id"] not in seen_ids:
                track["_source"] = "top_weekly"
                track["played_at"] = None
                combined_tracks.append(track)
                seen_ids.add(track["id"])
    except:
        pass

    return combined_tracks


# --- MAIN ENDPOINT ---
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    if not model_xgb:
        raise HTTPException(status_code=500, detail="Models not active")

    sp = spotipy.Spotify(auth=request.token, requests_timeout=10, retries=2)

    # 0. Market Check
    try:
        user_info = sp.current_user()
        market = user_info.get("country", "US")
    except:
        market = "US"

    # 1. HYBRID FETCH
    context_pool = get_user_context_hybrid(sp)
    if not context_pool:
        raise HTTPException(status_code=400, detail="No listening history found.")

    # 2. Build Profiles
    seed_track_ids = []
    artist_counts = Counter()
    now_bucket = get_time_bucket(datetime.datetime.utcnow().hour)

    user_time_buckets = []
    track_bucket_map = {}

    for t in context_pool:
        tid = t.get("id")
        seed_track_ids.append(tid)

        if t.get("artists"):
            aid = t["artists"][0].get("id")
            if aid: artist_counts[aid] += 1

        ts = t.get("played_at")
        bucket = now_bucket
        if ts:
            try:
                dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                bucket = get_time_bucket(dt.hour)
            except:
                pass

        user_time_buckets.append(bucket)
        track_bucket_map.setdefault(tid, []).append(bucket)

    # 3. GENERATE CANDIDATES (WITH JITTER)
    target_pool_size = request.size * 3
    candidate_pool = []
    seen_cand_ids = set(seed_track_ids)

    unique_seeds = list(dict.fromkeys(seed_track_ids))
    # CRITICAL: Shuffle seeds so we don't always pick the same top 5
    random.shuffle(unique_seeds)

    batches = [unique_seeds[i:i + 5] for i in range(0, len(unique_seeds), 5)]
    batches = batches[:12]  # Up to 60 seeds processed

    for batch in batches:
        if len(candidate_pool) >= target_pool_size: break

        # Get random params for this specific batch
        jitter_params = get_jittered_params()

        try:
            recs = sp.recommendations(
                seed_tracks=batch,
                limit=50,
                country=market,
                **jitter_params  # <--- Inject Chaos Here
            )
            if recs and 'tracks' in recs:
                for t in recs['tracks']:
                    if t['id'] not in seen_cand_ids:
                        candidate_pool.append(t)
                        seen_cand_ids.add(t['id'])
        except:
            continue

    # Fallback
    if len(candidate_pool) < request.size:
        top_artists = [a for a, c in artist_counts.most_common(5)]
        for aid in top_artists:
            if len(candidate_pool) >= target_pool_size: break
            try:
                top_t = sp.artist_top_tracks(aid, country=market).get("tracks", [])
                for t in top_t:
                    if t['id'] not in seen_cand_ids:
                        candidate_pool.append(t)
                        seen_cand_ids.add(t['id'])
            except:
                continue

    if not candidate_pool:
        raise HTTPException(status_code=400, detail="No recs found.")

    # 4. SCORING
    meta = []
    prediction_rows = []

    # Shuffle the candidate pool before scoring to avoid "Top 50" bias from Spotify
    random.shuffle(candidate_pool)
    candidate_pool = candidate_pool[:target_pool_size]

    day = datetime.datetime.utcnow().weekday()

    for t in candidate_pool:
        tid = t.get("id")
        art = t.get("artists", [{}])[0]
        aid = art.get("id", "")

        feats = extract_features(
            aid, day, now_bucket,
            artist_counts.get(aid, 0),
            user_time_buckets,
            track_bucket_map.get(tid, [])
        )
        prediction_rows.append(feats)

        meta.append({
            "id": tid,
            "name": t.get("name"),
            "artist": art.get("name"),
            "url": t.get("external_urls", {}).get("spotify"),
            "albumArt": (t.get("album", {}).get("images") or [{}])[0].get("url"),
            "artist_plays": artist_counts.get(aid, 0),
            "score": 0.0
        })

    # Predict
    cols = ["day_of_week", "artist_global_plays", "user_affinity", "track_context_weight",
            "time_bucket_afternoon", "time_bucket_evening", "time_bucket_night"]

    X = pd.DataFrame(prediction_rows, columns=cols).fillna(0.0)

    with model_lock:
        try:
            p1 = model_xgb.predict_proba(X)[:, 1]
            p2 = model_lgbm.predict_proba(X)[:, 1]
            final_scores = (p1 + p2) / 2
        except:
            final_scores = [0.5] * len(meta)

    for i, score in enumerate(final_scores):
        boost = 1.15 if meta[i]["artist_plays"] == 0 else 1.0
        meta[i]["score"] = float(score) * boost

    results = sorted(meta, key=lambda x: x["score"], reverse=True)
    return {"recommendations": results[:request.size]}


# --- AUTH & SAVE ---
@app.get("/login")
def login():
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    token = sp_oauth.get_access_token(code)
    return RedirectResponse(f"https://spotify-playlist-curator.vercel.app/home?token={token['access_token']}")


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        uid = sp.current_user()["id"]
        pl = sp.user_playlist_create(uid, request.name, public=False)
        uris = [f"spotify:track:{id}" for id in request.track_ids]

        for i in range(0, len(uris), 100):
            sp.playlist_add_items(pl["id"], uris[i:i + 100])

        return {"status": "success", "url": pl["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))