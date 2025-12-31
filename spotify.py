# backend_recommend_fixed.py
import os
import joblib
import datetime
import random
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
model_lock = Lock()  # serialize access to models to avoid race conditions

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

# SpotifyOAuth primarily used for the initial auth flow (callback)
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
    """
    Returns list of features in the exact order the model expects.
    """
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
    # note: depending on the spotipy version you use, method names may differ
    token_info = sp_oauth.get_access_token(code)
    access_token = token_info.get("access_token")
    return RedirectResponse(url=f"https://spotify-playlist-curator.vercel.app/home?token={access_token}")


# --- RECOMMEND ENDPOINT (POST) ---
@app.post("/recommend")
async def recommend(request: RecommendRequest):
    token = request.token
    size = request.size

    if model_xgb is None or model_lgbm is None:
        raise HTTPException(status_code=500, detail="Models not loaded. Check server logs.")

    # Create spotipy client using provided token; set timeouts/retries to be resilient
    sp = spotipy.Spotify(auth=token, requests_timeout=10, retries=2)

    # Validate token early: attempt a simple user call
    try:
        sp.current_user()  # will raise if token invalid
    except spotipy.SpotifyException as se:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token. Re-authenticate.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to verify token: {e}")

    try:
        now = datetime.datetime.utcnow()
        day = now.weekday()
        current_time_bucket = get_time_bucket(now.hour)

        # Build user profile
        recent_res = sp.current_user_recently_played(limit=50)
        # top tracks are optional; if rate-limited, proceed with recent only
        try:
            top_short = sp.current_user_top_tracks(limit=50, time_range="short_term")
        except Exception:
            top_short = {"items": []}

        all_played_tracks = []
        user_time_bucket_history = []
        artist_play_counts = Counter()
        seen_track_ids = set()
        track_time_buckets: Dict[str, List[str]] = {}

        if recent_res and isinstance(recent_res, dict) and "items" in recent_res:
            for item in recent_res["items"]:
                track = item.get("track")
                if not track:
                    continue
                all_played_tracks.append(track)
                track_id = track.get("id")
                if track_id:
                    seen_track_ids.add(track_id)

                # played_at may be present; try parsing robustly
                played_at = item.get("played_at") or item.get("played_at_date") or item.get("ts")
                if played_at:
                    try:
                        # Spotify uses ISO format with Z -> convert to +00:00
                        dt = datetime.datetime.fromisoformat(str(played_at).replace("Z", "+00:00"))
                        bucket = get_time_bucket(dt.hour)
                    except Exception:
                        # fallback: use current bucket if parsing fails
                        bucket = current_time_bucket
                    user_time_bucket_history.append(bucket)
                    if track_id:
                        track_time_buckets.setdefault(track_id, []).append(bucket)

                # artist counts
                if track.get("artists"):
                    first_artist = track["artists"][0]
                    if first_artist:
                        artist_id = first_artist.get("id")
                        if artist_id:
                            artist_play_counts[artist_id] += 1

        # Candidate generation
        top_artists = [a for a, _c in artist_play_counts.most_common(5)]
        candidate_pool = []
        seen_candidate_ids = set()

        for a_id in top_artists:
            related = []
            try:
                related = sp.artist_related_artists(a_id).get("artists", [])
            except spotipy.SpotifyException:
                pass  # expected for many artists

            # PRIMARY: related artists
            for rel in related[:5]:
                try:
                    top_t = sp.artist_top_tracks(rel["id"]).get("tracks", [])
                    candidate_pool.extend(top_t[:10])
                except Exception:
                    continue

            # FALLBACK: artist's own top tracks
            if not related:
                try:
                    fallback_tracks = sp.artist_top_tracks(a_id).get("tracks", [])
                    candidate_pool.extend(fallback_tracks[:50])
                except Exception:
                    pass

            for rel in related:
                try:
                    top_t = sp.artist_top_tracks(rel["id"]).get("tracks", [])[:10]
                except Exception:
                    top_t = []
                for t in top_t:
                    if not t or not t.get("id"):
                        continue
                    tid = t["id"]
                    if tid in seen_track_ids:
                        continue  # skip already-listened tracks
                    if tid in seen_candidate_ids:
                        continue
                    candidate_pool.append(t)
                    seen_candidate_ids.add(tid)

        # Shuffle and limit to a reasonable amount for model scoring
        random.shuffle(candidate_pool)
        candidate_pool = candidate_pool[:1000]  # keep an upper bound

        if not candidate_pool:
            raise HTTPException(status_code=400, detail="No candidate tracks available for recommendation.")

        # Build feature rows + metadata
        prediction_rows = []
        meta = []
        for track in candidate_pool:
            tid = track.get("id")
            if not tid:
                continue
            artist_id = track.get("artists", [{}])[0].get("id", "")
            artist_plays = artist_play_counts.get(artist_id, 0)
            # pass actual per-track time buckets if available
            t_time_buckets = track_time_buckets.get(tid, [])
            feats = extract_features(artist_id, day, current_time_bucket, artist_plays, user_time_bucket_history, t_time_buckets)
            prediction_rows.append(feats)
            meta.append({
                "id": tid,
                "name": track.get("name", "")[:200],
                "artist": track.get("artists", [{}])[0].get("name", ""),
                "url": track.get("external_urls", {}).get("spotify", ""),
                "albumArt": (track.get("album", {}).get("images") or [{}])[0].get("url", ""),
                "artist_plays": artist_plays
            })

        # Prepare DataFrame and ensure feature columns align exactly with training
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

        # Defensive: ensure columns present & dtype
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0.0
        X_pred = X_pred[feature_cols].fillna(0.0).astype("float32")

        # Predict with locking for thread-safety
        with model_lock:
            try:
                scores_xgb = model_xgb.predict_proba(X_pred)[:, 1]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"XGBoost prediction failed: {e}")
            try:
                scores_lgbm = model_lgbm.predict_proba(X_pred)[:, 1]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"LightGBM prediction failed: {e}")

        scores = (scores_xgb + scores_lgbm) / 2.0

        for i in range(len(meta)):
            meta[i]["score"] = float(scores[i])
            if meta[i].get("artist_plays", 0) == 0:
                meta[i]["score"] *= 1.15  # discovery boost

        final_recs = sorted(meta, key=lambda x: x["score"], reverse=True)[:size]
        return {"recommendations": final_recs}

    except HTTPException:
        # re-raise HTTPException so FastAPI handles it
        raise
    except Exception as e:
        # Catch-all
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")


# --- SAVE PLAYLIST ---
@app.post("/save-playlist")
async def save_playlist(request: PlaylistSaveRequest):
    sp = spotipy.Spotify(auth=request.token, requests_timeout=10, retries=2)
    try:
        # validate token
        sp.current_user()
    except spotipy.SpotifyException:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Token verification failed: {e}")

    try:
        user_id = sp.current_user()["id"]
        playlist = sp.user_playlist_create(user=user_id, name=request.name, public=False)
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids if tid]
        if not track_uris:
            raise HTTPException(status_code=400, detail="No valid track IDs supplied.")
        sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)
        return {"status": "success", "playlist_url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
