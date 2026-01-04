import os
import joblib
import datetime
import random
import requests
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


# --- HELPER: IS THIS MUSIC? ---
def is_valid_seed(track: dict) -> bool:
    """Filters out Audiobooks, Podcasts, and weird metadata."""
    if not track: return False
    name = track.get("name", "").lower()

    # 1. Title Heuristics (Audiobooks often start with 'Chapter')
    bad_prefixes = ["chapter ", "episode ", "part ", "intro"]
    if any(name.startswith(p) for p in bad_prefixes):
        return False

    # 2. Duration (Audiobook chapters can be very short or very long)
    duration_ms = track.get("duration_ms", 0)
    if duration_ms < 30000:  # Skip tracks under 30s
        return False

    return True


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


# --- HYBRID HISTORY FETCHING (CLEANED) ---
def get_clean_history(sp):
    """Fetches history but aggressively removes audiobooks/duplicates."""
    combined_tracks = []
    seen_ids = set()

    # 1. Recent History
    try:
        recent_results = sp.current_user_recently_played(limit=50)
        for item in recent_results.get('items', []):
            track = item.get("track")
            if is_valid_seed(track) and track["id"] not in seen_ids:
                track["_source"] = "recent"
                track["played_at"] = item.get("played_at")
                combined_tracks.append(track)
                seen_ids.add(track["id"])
    except:
        pass

    # 2. Top Tracks (Weekly Context)
    try:
        top_results = sp.current_user_top_tracks(limit=50, time_range="short_term")
        for track in top_results.get('items', []):
            if is_valid_seed(track) and track["id"] not in seen_ids:
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

    session = requests.Session()
    session.trust_env = False  # ⛔ This stops your ISP from hijacking the request

    # Initialize Spotipy with this protected session
    sp = spotipy.Spotify(
        auth=request.token,
        requests_timeout=10,
        retries=2,
        requests_session=session  # <--- Pass the session here
    )

    try:
        market = sp.current_user().get("country", "US")
    except:
        market = "US"

    # 1. Get Clean Seeds (No Audiobooks!)
    context_pool = get_clean_history(sp)

    if len(context_pool) < 5:
        # Emergency Fallback if history is ONLY audiobooks
        top_global = sp.playlist_tracks("37i9dQZEVXbMDoHDwVN2tF", limit=10, market=market)
        context_pool = [t['track'] for t in top_global['items'] if is_valid_seed(t['track'])]

    # 2. Prepare Data
    seed_track_ids = [t['id'] for t in context_pool]
    artist_counts = Counter()
    seed_artist_ids = []

    now_bucket = get_time_bucket(datetime.datetime.utcnow().hour)
    user_time_buckets = []
    track_bucket_map = {}

    for t in context_pool:
        tid = t.get("id")
        if t.get("artists"):
            aid = t["artists"][0].get("id")
            if aid:
                artist_counts[aid] += 1
                seed_artist_ids.append(aid)

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

    # 3. GENERATE CANDIDATES (The "Related Artist Hop")
    # We need a large pool to filter out duplicates later
    target_pool_size = request.size * 4
    candidate_pool = []
    seen_cand_ids = set(seed_track_ids)

    # A. Standard Recommendations (Jittered)
    unique_seeds = list(set(seed_track_ids))
    random.shuffle(unique_seeds)
    batches = [unique_seeds[i:i + 5] for i in range(0, len(unique_seeds), 5)][:8]

    for batch in batches:
        if len(candidate_pool) >= target_pool_size // 2: break
        try:
            # High variety settings
            recs = sp.recommendations(
                seed_tracks=batch, limit=50, country=market,
                min_popularity=10, max_popularity=85  # Avoid super-mainstream
            )
            for t in recs.get('tracks', []):
                if t['id'] not in seen_cand_ids and is_valid_seed(t):
                    candidate_pool.append(t)
                    seen_cand_ids.add(t['id'])
        except:
            continue

    # B. The "Hop" Strategy (Forces Discovery)
    # Instead of "Top tracks by User's Artist", do "Top tracks by RELATED Artist"
    if seed_artist_ids:
        random.shuffle(seed_artist_ids)
        for aid in seed_artist_ids[:5]:  # Take 5 random artists user likes
            if len(candidate_pool) >= target_pool_size: break
            try:
                # Find who sounds like them
                related = sp.artist_related_artists(aid)
                if related and 'artists' in related:
                    # Pick a random related artist (Discovery!)
                    rel_art = random.choice(related['artists'][:5])

                    # Get THAT artist's top tracks
                    top = sp.artist_top_tracks(rel_art['id'], country=market)
                    for t in top.get('tracks', [])[:5]:  # Only take top 5
                        if t['id'] not in seen_cand_ids and is_valid_seed(t):
                            candidate_pool.append(t)
                            seen_cand_ids.add(t['id'])
            except:
                continue

    if not candidate_pool:
        raise HTTPException(status_code=400, detail="Could not generate songs.")

    # 4. SCORING & DIVERSIFYING
    meta = []
    prediction_rows = []

    day = datetime.datetime.utcnow().weekday()
    random.shuffle(candidate_pool)  # Shuffle to break clusters

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
            final_scores = [random.random() for _ in range(len(meta))]

    # 5. STRICT FINAL FILTERING (Artist Cap)
    final_recs = []
    final_artist_counts = Counter()

    # Sort by AI score first
    ranked_candidates = sorted(zip(meta, final_scores), key=lambda x: x[1], reverse=True)

    for item, score in ranked_candidates:
        artist_name = item["artist"]

        # RULE: Max 2 songs per artist
        if final_artist_counts[artist_name] >= 2:
            continue

        # RULE: Score Boost for new artists
        boost = 1.2 if item["artist_plays"] == 0 else 1.0
        item["score"] = float(score) * boost

        final_recs.append(item)
        final_artist_counts[artist_name] += 1

        if len(final_recs) >= request.size:
            break

    return {"recommendations": final_recs}


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