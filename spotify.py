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
from typing import List, Dict, Optional
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
    size: int = Field(50, ge=1, le=100)  # Max recomm limit is 100


app = FastAPI(title="Discovery Playlist API")

# Environment
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPE = os.getenv("SCOPE")

# Load models
model_xgb = None
model_lgbm = None
model_lock = Lock()

try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ Pre-trained AI models loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Could not load models: {e}")

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
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"


def calculate_time_bucket_affinity(user_streams: List[str], current_time_bucket: str) -> float:
    if not user_streams: return 0.5
    count = sum(1 for s in user_streams if s == current_time_bucket)
    return count / len(user_streams)


def calculate_track_context_weight(track_time_buckets: List[str], current_time_bucket: str) -> float:
    if not track_time_buckets: return 0.5
    count = sum(1 for s in track_time_buckets if s == current_time_bucket)
    return count / len(track_time_buckets)


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


# --- HISTORY FETCHING (FIXED) ---
def get_recent_tracks_7_days(sp):
    """
    Manually paginates through history using 'before' timestamps.
    Avoids sp.next() which can fail behind proxies.
    Strictly respects the 50 item limit per call.
    """
    recent_tracks = []
    seen_ids = set()

    # We want tracks from the last 7 days
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    cutoff_timestamp = int(cutoff_date.timestamp() * 1000)  # milliseconds

    # Initial request
    # Limit is strictly 50
    try:
        results = sp.current_user_recently_played(limit=50)
    except Exception as e:
        print(f"Error fetching initial history: {e}")
        return []

    if not results or 'items' not in results:
        return []

    items = results['items']
    if not items:
        return []

    # Add first batch
    for item in items:
        recent_tracks.append(item)
        if item.get("track", {}).get("id"):
            seen_ids.add(item["track"]["id"])

    # Manual Pagination Loop (Max 5 pages to avoid infinite loops)
    for _ in range(5):
        # Find the oldest 'played_at' in the current batch to use as cursor
        if not items:
            break

        oldest_played_at = items[-1].get("played_at")
        if not oldest_played_at:
            break

        # Check if we have already reached past the cutoff
        try:
            # Parse ISO time "2023-10-27T10:00:00.123Z" -> timestamp
            dt = datetime.datetime.fromisoformat(str(oldest_played_at).replace("Z", "+00:00"))
            ts_millis = int(dt.timestamp() * 1000)

            if ts_millis < cutoff_timestamp:
                # We have gone back far enough
                break

            # Fetch next batch BEFORE this timestamp
            results = sp.current_user_recently_played(limit=50, before=ts_millis)
            items = results.get('items', [])

            if not items:
                break

            for item in items:
                # Deduplicate based on played_at or ID to be safe
                # But simple append is usually fine for feature extraction
                recent_tracks.append(item)

        except Exception as e:
            print(f"Pagination error: {e}")
            break

    return recent_tracks


# --- ENDPOINTS ---
@app.get("/login")
def login():
    return RedirectResponse(sp_oauth.get_authorize_url())


@app.get("/callback")
def callback(code: str):
    token_info = sp_oauth.get_access_token(code)
    return RedirectResponse(url=f"https://spotify-playlist-curator.vercel.app/home?token={token_info['access_token']}")


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    if model_xgb is None:
        raise HTTPException(status_code=500, detail="Models not loaded.")

    sp = spotipy.Spotify(auth=request.token, requests_timeout=10, retries=2)

    # 1. Gather History (Last 7 Days)
    recent_items = get_recent_tracks_7_days(sp)

    # Fallback to top tracks if history is empty (new user)
    if not recent_items:
        try:
            top_tracks = sp.current_user_top_tracks(limit=20, time_range="short_term").get("items", [])
            recent_items = [{"track": t, "played_at": str(datetime.datetime.utcnow())} for t in top_tracks]
        except Exception:
            pass

    if not recent_items:
        raise HTTPException(status_code=400, detail="No listening history found.")

    # 2. Extract Seeds & User Profile
    seed_track_ids = []
    user_time_buckets = []
    artist_counts = Counter()
    track_bucket_map = {}

    now = datetime.datetime.utcnow()
    current_bucket = get_time_bucket(now.hour)

    for item in recent_items:
        t = item.get("track")
        if not t: continue

        tid = t.get("id")
        if tid:
            seed_track_ids.append(tid)

        # Parse time
        played_at = item.get("played_at")
        try:
            if played_at:
                dt = datetime.datetime.fromisoformat(str(played_at).replace("Z", "+00:00"))
                b = get_time_bucket(dt.hour)
            else:
                b = current_bucket
        except:
            b = current_bucket

        user_time_buckets.append(b)
        if tid:
            track_bucket_map.setdefault(tid, []).append(b)

        if t.get("artists"):
            aid = t["artists"][0].get("id")
            if aid: artist_counts[aid] += 1

    # 3. Generate Candidates via Seeds
    # IMPORTANT: Max 5 seeds per request.
    # We will pick 3 batches of 5 seeds to get 150 candidates max, then filter.

    unique_seeds = list(dict.fromkeys(seed_track_ids))  # preserve order (recent first)
    candidate_pool = []
    seen_cand_ids = set()

    batches = []
    # Batch A: Top 5 most recent
    if unique_seeds:
        batches.append(unique_seeds[:5])
    # Batch B: Random 5 from history
    if len(unique_seeds) > 5:
        batches.append(random.sample(unique_seeds, 5))
    # Batch C: Another Random 5
    if len(unique_seeds) > 10:
        batches.append(random.sample(unique_seeds, 5))

    for batch in batches:
        try:
            # Limit=50 per call (Spotify Max for Recs is 100, but 50 is safer for latency)
            recs = sp.recommendations(seed_tracks=batch, limit=50)
            if recs and 'tracks' in recs:
                for t in recs['tracks']:
                    if t['id'] not in seen_cand_ids:
                        candidate_pool.append(t)
                        seen_cand_ids.add(t['id'])
        except Exception as e:
            print(f"⚠️ Recommendation batch failed: {e}")
            continue

    if not candidate_pool:
        # Emergency fallback: Top Global/Viral? Or just fail.
        # Let's try one last fallback using just the #1 top artist
        try:
            if artist_counts:
                top_aid = artist_counts.most_common(1)[0][0]
                top_t = sp.artist_top_tracks(top_aid).get("tracks", [])
                candidate_pool = top_t[:20]
        except:
            pass

    if not candidate_pool:
        raise HTTPException(status_code=400, detail="Could not generate recommendations.")

    # 4. Score Candidates
    prediction_rows = []
    meta = []
    day = now.weekday()

    for t in candidate_pool:
        tid = t.get("id")
        if not tid: continue

        art = t["artists"][0]
        aid = art.get("id")
        aname = art.get("name")

        a_plays = artist_counts.get(aid, 0)
        # For new tracks, they have no specific time history, so we use empty list
        t_ctx = track_bucket_map.get(tid, [])

        feats = extract_features(aid, day, current_bucket, a_plays, user_time_buckets, t_ctx)
        prediction_rows.append(feats)

        meta.append({
            "id": tid,
            "name": t["name"],
            "artist": aname,
            "url": t["external_urls"]["spotify"],
            "albumArt": t["album"]["images"][0]["url"] if t["album"]["images"] else "",
            "score": 0.0,
            "artist_plays": a_plays
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
        # Discovery Boost: If user hasn't played this artist recently, boost slightly
        boost = 1.2 if meta[i]["artist_plays"] == 0 else 1.0
        meta[i]["score"] = float(score) * boost

    # Sort
    results = sorted(meta, key=lambda x: x["score"], reverse=True)[:request.size]
    return {"recommendations": results}


@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token)
    try:
        user = sp.current_user()
        playlist = sp.user_playlist_create(user=user["id"], name=request.name, public=False)
        uris = [f"spotify:track:{tid}" for tid in request.track_ids]

        # Batch add (Spotify limit is 100 per request)
        for i in range(0, len(uris), 100):
            batch = uris[i:i + 100]
            sp.playlist_add_items(playlist["id"], batch)

        return {"status": "success", "playlist_url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))