# backend_recommend_fixed.py
import os
import joblib
import datetime
import random
import time
from collections import Counter
import pandas as pd
import spotipy
from pydantic import BaseModel, Field
from typing import List, Dict
from spotipy.oauth2 import SpotifyOAuth
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from dotenv import load_dotenv

load_dotenv()


# --- MODELS ---
class PlaylistSaveRequest(BaseModel):
    token: str
    track_ids: List[str]
    name: str = "My AI Discovery Mix"


class RecommendRequest(BaseModel):
    token: str = Field(..., min_length=10)
    size: int = Field(50, ge=1, le=200)


app = FastAPI(title="Discovery Playlist API")

# Environment / Spotify OAuth config
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE = os.getenv("SCOPE")

# Load pre-trained models
model_xgb = None
model_lgbm = None
model_lock = Lock()

try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ Pre-trained AI models loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Could not load models: {e}")
    model_xgb = None
    model_lgbm = None

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


# --- HELPERS ---
def get_time_bucket(hour: int) -> str:
    if hour < 5 or hour >= 22:
        return "night"
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "night"


def calculate_time_bucket_affinity(user_streams: List[str], current_time_bucket: str) -> float:
    if not user_streams:
        return 0.5
    total = len(user_streams)
    bucket_count = sum(1 for s in user_streams if s == current_time_bucket)
    return (bucket_count / total) if total > 0 else 0.5


def calculate_track_context_weight(track_time_buckets: List[str], current_time_bucket: str) -> float:
    if not track_time_buckets:
        return 0.5
    total = len(track_time_buckets)
    bucket_count = sum(1 for s in track_time_buckets if s == current_time_bucket)
    return (bucket_count / total) if total > 0 else 0.5


def extract_features(artist_id: str, day: int, current_time_bucket: str, artist_play_count: int,
                     user_time_buckets: List[str], track_time_buckets: List[str]):
    return [
        day,
        float(artist_play_count),
        float(calculate_time_bucket_affinity(user_time_buckets, current_time_bucket)),
        float(calculate_track_context_weight(track_time_buckets, current_time_bucket)),
        1.0 if current_time_bucket == "afternoon" else 0.0,
        1.0 if current_time_bucket == "evening" else 0.0,
        1.0 if current_time_bucket == "night" else 0.0
    ]


# --- AUTH FLOW ---
@app.get("/login")
def login():
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info.get("access_token")
    return RedirectResponse(url=f"https://spotify-playlist-curator.vercel.app/home?token={access_token}")


# --- LOGIC HELPERS ---

def get_recent_tracks_7_days(sp):
    """
    Fetches tracks played in the last 7 days.
    Iterates through 'recently_played' pages until we hit a date older than 7 days.
    """
    recent_tracks = []
    seen_ids = set()

    # 7 days ago cutoff
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=7)

    # Initial call
    results = sp.current_user_recently_played(limit=50)
    keep_fetching = True

    while keep_fetching and results:
        items = results.get('items', [])
        if not items:
            break

        for item in items:
            played_at_str = item.get("played_at")
            if not played_at_str:
                continue

            # Parse ISO date
            try:
                played_at_dt = datetime.datetime.fromisoformat(str(played_at_str).replace("Z", "+00:00")).replace(
                    tzinfo=None)
            except:
                continue

            # Check cutoff
            if played_at_dt < cutoff_date:
                keep_fetching = False
                continue

            track = item.get("track")
            if track and track.get("id"):
                # We store the full item to keep the 'played_at' context for time buckets
                recent_tracks.append(item)
                seen_ids.add(track["id"])

        # Pagination: get next set of results if we are still within time range
        if keep_fetching and results.get("next"):
            results = sp.next(results)
        else:
            break

    return recent_tracks


# --- RECOMMEND ENDPOINT (POST) ---
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    token = request.token
    size = request.size

    if model_xgb is None or model_lgbm is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Check server logs.")

    sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=2)

    try:
        sp.current_user()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")

    try:
        now = datetime.datetime.utcnow()
        day = now.weekday()
        current_time_bucket = get_time_bucket(now.hour)

        # 1. Get User History (Last 7 Days)
        recent_items = get_recent_tracks_7_days(sp)

        # fallback if history is empty (new user), try top tracks
        if not recent_items:
            try:
                top_tracks = sp.current_user_top_tracks(limit=20, time_range="short_term").get("items", [])
                # Fake the item structure
                recent_items = [{"track": t, "played_at": str(datetime.datetime.utcnow())} for t in top_tracks]
            except:
                pass

        if not recent_items:
            raise HTTPException(status_code=400, detail="Not enough listening history in the last 7 days.")

        # Process History for Features
        user_time_bucket_history = []
        artist_play_counts = Counter()
        seen_track_ids = set()
        track_time_buckets: Dict[str, List[str]] = {}
        seed_track_ids = []

        for item in recent_items:
            track = item.get("track")
            if not track: continue

            tid = track.get("id")
            if tid:
                seen_track_ids.add(tid)
                seed_track_ids.append(tid)

            # Time buckets
            played_at = item.get("played_at")
            if played_at:
                try:
                    dt = datetime.datetime.fromisoformat(str(played_at).replace("Z", "+00:00"))
                    bucket = get_time_bucket(dt.hour)
                except:
                    bucket = current_time_bucket
                user_time_bucket_history.append(bucket)
                if tid:
                    track_time_buckets.setdefault(tid, []).append(bucket)

            # Artist counts
            if track.get("artists"):
                aid = track["artists"][0].get("id")
                if aid:
                    artist_play_counts[aid] += 1

        # 2. Candidate Generation (Track-based Seeds)
        # We use sp.recommendations which takes seed_tracks.
        # Max 5 seeds per call. We make multiple calls to get a diverse pool.

        candidate_pool = []
        unique_candidate_ids = set()

        # Remove duplicates from seeds but keep order (approx) to prioritize recent
        unique_seeds = list(dict.fromkeys(seed_track_ids))

        # We will make ~4 calls with different batches of seeds
        # 1. Very recent seeds
        # 2. Random sample from last 7 days
        # 3. Random sample from last 7 days
        # 4. Top artist seeds (fallback diversity)

        seed_batches = []

        # Batch 1: Most recent 5
        if len(unique_seeds) >= 5:
            seed_batches.append(unique_seeds[:5])
        else:
            seed_batches.append(unique_seeds)

        # Batch 2 & 3: Random samples
        if len(unique_seeds) > 5:
            seed_batches.append(random.sample(unique_seeds, 5))
            seed_batches.append(random.sample(unique_seeds, 5))

        # Fetch recommendations
        for batch in seed_batches:
            if not batch: continue
            try:
                # limit=50 per call
                recs = sp.recommendations(seed_tracks=batch, limit=50)
                if recs and 'tracks' in recs:
                    for track in recs['tracks']:
                        if track['id'] not in seen_track_ids and track['id'] not in unique_candidate_ids:
                            candidate_pool.append(track)
                            unique_candidate_ids.add(track['id'])
            except Exception as e:
                print(f"Error fetching recommendations: {e}")
                continue

        # Shuffle candidates
        random.shuffle(candidate_pool)
        candidate_pool = candidate_pool[:800]

        if not candidate_pool:
            raise HTTPException(status_code=400, detail="Could not generate candidates.")

        # 3. Feature Extraction & Scoring
        prediction_rows = []
        meta = []

        for track in candidate_pool:
            tid = track.get("id")
            if not tid: continue

            # Artist info
            artist_obj = track.get("artists", [{}])[0]
            artist_id = artist_obj.get("id", "")
            artist_name = artist_obj.get("name", "")

            # Features
            artist_plays = artist_play_counts.get(artist_id, 0)
            t_buckets = track_time_buckets.get(tid, [])  # likely empty for new songs, that's okay

            feats = extract_features(artist_id, day, current_time_bucket, artist_plays, user_time_bucket_history,
                                     t_buckets)
            prediction_rows.append(feats)

            meta.append({
                "id": tid,
                "name": track.get("name", "")[:200],
                "artist": artist_name,
                "url": track.get("external_urls", {}).get("spotify", ""),
                "albumArt": (track.get("album", {}).get("images") or [{}])[0].get("url", ""),
                "artist_plays": artist_plays
            })

        # 4. Predict
        feature_cols = [
            "day_of_week",
            "artist_global_plays",
            "user_affinity",
            "track_context_weight",
            "time_bucket_afternoon",
            "time_bucket_evening",
            "time_bucket_night",
        ]

        X_pred = pd.DataFrame(prediction_rows, columns=feature_cols)
        X_pred = X_pred.fillna(0.0).astype("float32")

        with model_lock:
            scores_xgb = model_xgb.predict_proba(X_pred)[:, 1]
            scores_lgbm = model_lgbm.predict_proba(X_pred)[:, 1]

        scores = (scores_xgb + scores_lgbm) / 2.0

        for i in range(len(meta)):
            # Discovery Boost: heavier boost if the artist is new to the user
            base_score = float(scores[i])
            if meta[i]["artist_plays"] == 0:
                base_score *= 1.2  # slightly higher boost for pure discovery
            meta[i]["score"] = base_score

        final_recs = sorted(meta, key=lambda x: x["score"], reverse=True)[:size]
        return {"recommendations": final_recs}

    except HTTPException:
        raise
    except Exception as e:
        print(f"SERVER ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(user=user_id, name=request.name, public=False)
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids if tid]
        if track_uris:
            sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)
        return {"status": "success", "playlist_url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))