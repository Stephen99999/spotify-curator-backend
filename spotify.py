import os
import joblib
import datetime
import random
import requests
from collections import Counter
import pandas as pd
import spotipy
from pydantic import BaseModel, Field
from typing import List
from spotipy.oauth2 import SpotifyOAuth
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from dotenv import load_dotenv
import logging
from spotipy import SpotifyException

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discovery_api")


def is_spotify_id(s):
    return isinstance(s, str) and len(s) in (22, 23)


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    if not model_xgb or not model_lgbm:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # --- Spotify Client ---
    session = requests.Session()
    session.trust_env = False

    sp = spotipy.Spotify(
        auth=request.token,
        requests_timeout=10,
        retries=2,
        requests_session=session
    )

    # --- Validate Token ---
    try:
        user = sp.current_user()
        market = user.get("country", "US")
    except SpotifyException:
        raise HTTPException(status_code=401, detail="Invalid or expired Spotify token")

    # --- Fetch Clean Listening History ---
    context_pool = get_clean_history(sp)

    if len(context_pool) < 5:
        fallback = sp.playlist_tracks(
            "37i9dQZEVXbMDoHDwVN2tF", limit=10, market=market
        )
        context_pool = [
            t["track"] for t in fallback["items"]
            if t.get("track") and is_valid_seed(t["track"])
        ]

    seed_track_ids = [
        t["id"] for t in context_pool
        if t.get("id") and is_spotify_id(t["id"])
    ]

    seed_artist_ids = [
        t["artists"][0]["id"] for t in context_pool
        if t.get("artists") and t["artists"][0].get("id")
    ]

    if not seed_track_ids:
        raise HTTPException(status_code=400, detail="No valid seed tracks found")

    # --- Candidate Generation ---
    candidate_pool = []
    seen_ids = set(seed_track_ids)

    random.shuffle(seed_track_ids)
    batches = [seed_track_ids[i:i + 5] for i in range(0, len(seed_track_ids), 5)][:8]

    for batch in batches:
        try:
            recs = sp.recommendations(
                seed_tracks=batch,
                limit=50,
                market=market,
                min_popularity=10,
                max_popularity=85
            )
            for t in recs.get("tracks", []):
                if t.get("id") and t["id"] not in seen_ids and is_valid_seed(t):
                    candidate_pool.append(t)
                    seen_ids.add(t["id"])
        except SpotifyException:
            continue

    # --- Related Artist Expansion ---
    random.shuffle(seed_artist_ids)

    for aid in seed_artist_ids[:5]:
        try:
            related = sp.artist_related_artists(aid)
            artist = random.choice(related["artists"][:5])
            top_tracks = sp.artist_top_tracks(artist["id"], market=market)

            for t in top_tracks["tracks"][:5]:
                if t.get("id") and t["id"] not in seen_ids and is_valid_seed(t):
                    candidate_pool.append(t)
                    seen_ids.add(t["id"])
        except SpotifyException:
            continue

    if not candidate_pool:
        raise HTTPException(status_code=400, detail="No recommendations returned")

    # --- Feature Engineering ---
    now_bucket = get_time_bucket(datetime.datetime.utcnow().hour)
    day = datetime.datetime.utcnow().weekday()

    artist_counts = Counter(seed_artist_ids)

    rows = []
    meta = []

    for t in candidate_pool:
        aid = t["artists"][0]["id"]

        rows.append(
            extract_features(
                aid,
                day,
                now_bucket,
                artist_counts.get(aid, 0),
                [now_bucket],
                [now_bucket]
            )
        )

        meta.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "url": t["external_urls"]["spotify"],
            "albumArt": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
            "artist_plays": artist_counts.get(aid, 0),
            "score": 0.0
        })

    X = pd.DataFrame(rows, columns=[
        "day_of_week",
        "artist_global_plays",
        "user_affinity",
        "track_context_weight",
        "time_bucket_afternoon",
        "time_bucket_evening",
        "time_bucket_night"
    ]).fillna(0)

    # --- AI Scoring ---
    with model_lock:
        try:
            p1 = model_xgb.predict_proba(X)[:, 1]
            p2 = model_lgbm.predict_proba(X)[:, 1]
            scores = (p1 + p2) / 2
        except Exception:
            scores = [random.random() for _ in meta]

    # --- Final Ranking & Diversity Filter ---
    final = []
    artist_cap = Counter()

    ranked = sorted(zip(meta, scores), key=lambda x: x[1], reverse=True)

    for item, score in ranked:
        if artist_cap[item["artist"]] >= 2:
            continue

        boost = 1.2 if item["artist_plays"] == 0 else 1.0
        item["score"] = float(score) * boost

        final.append(item)
        artist_cap[item["artist"]] += 1

        if len(final) >= request.size:
            break

    return {"recommendations": final}




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