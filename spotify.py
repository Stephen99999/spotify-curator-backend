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
    size: int = Field(50, ge=1, le=100)  # Default 50, max 100


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


# --- FEATURE HELPERS ---
def get_time_bucket(hour: int) -> str:
    if hour < 5 or hour >= 22: return "night"
    if 5 <= hour < 12: return "morning"
    if 12 <= hour < 17: return "afternoon"
    return "evening"


def extract_features(artist_id: str, day: int, current_time_bucket: str, artist_play_count: int,
                     user_time_buckets: List[str], track_time_buckets: List[str]):
    # Helpers for density calc
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


# --- ROBUST HISTORY FETCHING ---
def get_recent_tracks_robust(sp):
    """
    Fetches up to 50 recent tracks securely.
    Manual pagination to avoid next() errors.
    """
    recent_tracks = []
    seen_ids = set()
    cutoff = int((datetime.datetime.utcnow() - datetime.timedelta(days=7)).timestamp() * 1000)

    try:
        # Initial fetch
        results = sp.current_user_recently_played(limit=50)
        items = results.get('items', [])

        # Add items
        for item in items:
            tid = item.get("track", {}).get("id")
            if tid and tid not in seen_ids:
                recent_tracks.append(item)
                seen_ids.add(tid)

        # Simple depth check: If we have fewer than 10 tracks, try one pagination step
        if len(recent_tracks) < 10 and items:
            last_ts = items[-1].get("played_at")
            if last_ts:
                try:
                    dt = datetime.datetime.fromisoformat(str(last_ts).replace("Z", "+00:00"))
                    ts_millis = int(dt.timestamp() * 1000)
                    if ts_millis > cutoff:
                        more = sp.current_user_recently_played(limit=50, before=ts_millis)
                        for item in more.get('items', []):
                            tid = item.get("track", {}).get("id")
                            if tid and tid not in seen_ids:
                                recent_tracks.append(item)
                                seen_ids.add(tid)
                except:
                    pass
    except Exception as e:
        print(f"History fetch warning: {e}")

    return recent_tracks


# --- MAIN ENDPOINT ---
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    if not model_xgb:
        raise HTTPException(status_code=500, detail="Models not active")

    sp = spotipy.Spotify(auth=request.token, requests_timeout=10, retries=2)

    # 0. User Info & Market (Important for Availability)
    try:
        user_info = sp.current_user()
        market = user_info.get("country", "US")
    except:
        market = "US"

    # 1. Get History
    recent_items = get_recent_tracks_robust(sp)

    # Fallback: Top Tracks if History Empty
    if not recent_items:
        try:
            top = sp.current_user_top_tracks(limit=50, time_range="short_term").get("items", [])
            recent_items = [{"track": t, "played_at": None} for t in top]
        except:
            pass

    if not recent_items:
        raise HTTPException(status_code=400, detail="No history found to base recommendations on.")

    # 2. Build Profiles & Seeds
    seed_track_ids = []
    artist_ids = []
    artist_counts = Counter()

    # Context data
    user_time_buckets = []
    track_bucket_map = {}
    now_bucket = get_time_bucket(datetime.datetime.utcnow().hour)
    day = datetime.datetime.utcnow().weekday()

    for item in recent_items:
        t = item.get("track")
        if not t: continue

        tid = t.get("id")
        if tid:
            seed_track_ids.append(tid)

        # Artists
        if t.get("artists"):
            aid = t["artists"][0].get("id")
            if aid:
                artist_ids.append(aid)
                artist_counts[aid] += 1

        # Time context
        ts = item.get("played_at")
        bucket = now_bucket
        if ts:
            try:
                dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                bucket = get_time_bucket(dt.hour)
            except:
                pass
        user_time_buckets.append(bucket)
        if tid:
            track_bucket_map.setdefault(tid, []).append(bucket)

    # 3. CANDIDATE GENERATION LOOP
    # We want at least 3x the requested size to allow for filtering/scoring
    target_pool_size = request.size * 3
    candidate_pool = []
    seen_cand_ids = set(seed_track_ids)  # Don't recommend what they just played

    # Strategy A: Recommendations from Track Seeds (Batched)
    unique_seeds = list(dict.fromkeys(seed_track_ids))
    random.shuffle(unique_seeds)

    # Create batches of 5 (Spotify Max)
    batches = [unique_seeds[i:i + 5] for i in range(0, len(unique_seeds), 5)]

    # Limit max batches to avoid timeout (e.g. max 10 API calls)
    batches = batches[:10]

    for batch in batches:
        if len(candidate_pool) >= target_pool_size:
            break
        try:
            # limit=50 per call is optimal
            recs = sp.recommendations(seed_tracks=batch, limit=50, country=market)
            if recs and 'tracks' in recs:
                for t in recs['tracks']:
                    if t['id'] not in seen_cand_ids:
                        candidate_pool.append(t)
                        seen_cand_ids.add(t['id'])
        except Exception as e:
            print(f"⚠️ Recs Error: {e}")

    # Strategy B: Backfill with Related Artists / Top Tracks (If pool is still small)
    if len(candidate_pool) < request.size:
        print("⚠️ Pool too small, triggering Backfill Strategy...")
        # Take top 5 artists user listened to
        top_artists = [a for a, c in artist_counts.most_common(5)]

        for aid in top_artists:
            if len(candidate_pool) >= target_pool_size: break
            try:
                top_tracks = sp.artist_top_tracks(aid, country=market).get("tracks", [])
                for t in top_tracks:
                    if t['id'] not in seen_cand_ids:
                        candidate_pool.append(t)
                        seen_cand_ids.add(t['id'])
            except:
                continue

    # Final Check
    if not candidate_pool:
        raise HTTPException(status_code=400,
                            detail="Could not generate any songs. Try listening to more music on Spotify first.")

    # 4. SCORING
    meta = []
    prediction_rows = []

    # Limit processing to target_pool_size to save CPU
    candidate_pool = candidate_pool[:target_pool_size]

    for t in candidate_pool:
        tid = t.get("id")
        art = t.get("artists", [{}])[0]
        aid = art.get("id", "")

        # Features
        a_plays = artist_counts.get(aid, 0)
        t_ctx = track_bucket_map.get(tid, [])

        feats = extract_features(aid, day, now_bucket, a_plays, user_time_buckets, t_ctx)
        prediction_rows.append(feats)

        meta.append({
            "id": tid,
            "name": t.get("name"),
            "artist": art.get("name"),
            "url": t.get("external_urls", {}).get("spotify"),
            "albumArt": (t.get("album", {}).get("images") or [{}])[0].get("url"),
            "artist_plays": a_plays,
            "score": 0.0
        })

    # AI Prediction
    cols = ["day_of_week", "artist_global_plays", "user_affinity", "track_context_weight",
            "time_bucket_afternoon", "time_bucket_evening", "time_bucket_night"]

    X = pd.DataFrame(prediction_rows, columns=cols).fillna(0.0)

    with model_lock:
        try:
            p1 = model_xgb.predict_proba(X)[:, 1]
            p2 = model_lgbm.predict_proba(X)[:, 1]
            final_scores = (p1 + p2) / 2
        except:
            # If model fails, fallback to random shuffle
            final_scores = [random.random() for _ in range(len(meta))]

    # Assign Scores & Sort
    for i, score in enumerate(final_scores):
        # Boost discovery (unheard artists)
        boost = 1.15 if meta[i]["artist_plays"] == 0 else 1.0
        meta[i]["score"] = float(score) * boost

    # 5. Return EXACTLY requested size (or as many as we found)
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

        # Batching for large saves
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(pl["id"], uris[i:i + 100])

        return {"status": "success", "url": pl["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))