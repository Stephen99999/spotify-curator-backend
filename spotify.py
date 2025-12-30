import os
import joblib
from dotenv import load_dotenv
import datetime
import random
from collections import Counter
import pandas as pd
import numpy as np
import spotipy
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

# Load pre-trained models
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ Pre-trained AI models loaded successfully")
    print(f"   XGB features: {model_xgb.get_booster().feature_names}")
except Exception as e:
    print(f"❌ ERROR: Could not load models: {e}")
    model_xgb, model_lgbm = None, None

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


# --- FEATURE ENGINEERING (MATCHING TRAINING SCRIPT) ---
def calculate_time_bucket_affinity(user_streams, current_time_bucket):
    """
    Calculate user_affinity: user's preference for current time bucket
    Formula: (streams in this bucket) / (total streams)
    """
    if not user_streams:
        return 0.5  # Neutral if no data

    total = len(user_streams)
    bucket_count = sum(1 for s in user_streams if s == current_time_bucket)

    return bucket_count / total if total > 0 else 0.5


def calculate_track_context_weight(track_time_buckets, current_time_bucket):
    """
    Calculate track_context_weight: how often this track is played in current time bucket
    Formula: (plays in this bucket) / (total plays of track)

    For NEW tracks we haven't seen: use neutral value (0.5)
    """
    if not track_time_buckets:
        return 0.5  # New track, no history

    total = len(track_time_buckets)
    bucket_count = sum(1 for s in track_time_buckets if s == current_time_bucket)

    return bucket_count / total if total > 0 else 0.5


def get_time_bucket(hour):
    """Map hour to time bucket matching training script"""
    if hour < 5 or hour >= 22:
        return 'night'
    elif 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 22:
        return 'evening'
    return 'night'


def extract_features(artist_id, day, current_time_bucket,
                     artist_play_count, user_time_buckets, track_time_buckets=None):
    """
    Extract features EXACTLY matching training script

    Features (7 total):
    1. day_of_week: 0-6
    2. artist_global_plays: How many times YOU played this artist
    3. user_affinity: Your preference for this time bucket (0-1)
    4. track_context_weight: How often track is played in this time bucket (0-1)
    5-7. time_bucket_{afternoon/evening/night}: One-hot encoded
    """

    # Feature 1: day_of_week (0=Monday, 6=Sunday)
    day_of_week = day

    # Feature 2: artist_global_plays (YOUR play count for this artist)
    artist_global_plays = artist_play_count

    # Feature 3: user_affinity (your time bucket preference)
    user_affinity = calculate_time_bucket_affinity(user_time_buckets, current_time_bucket)

    # Feature 4: track_context_weight (track's time bucket pattern)
    track_context_weight = calculate_track_context_weight(track_time_buckets, current_time_bucket)

    # Features 5-7: One-hot encoded time buckets
    time_bucket_afternoon = 1 if current_time_bucket == 'afternoon' else 0
    time_bucket_evening = 1 if current_time_bucket == 'evening' else 0
    time_bucket_night = 1 if current_time_bucket == 'night' else 0

    return [
        day_of_week,
        artist_global_plays,
        user_affinity,
        track_context_weight,
        time_bucket_afternoon,
        time_bucket_evening,
        time_bucket_night
    ]


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
    """
    Generate personalized recommendations using your pre-trained model
    Feature engineering matches training script exactly
    """

    if not model_xgb or not model_lgbm:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Ensure discovery_xgb_finetuned.pkl and discovery_lgbm_finetuned.pkl exist."
        )

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

    # Time context
    day = now.weekday()
    hour = now.hour
    current_time_bucket = get_time_bucket(hour)

    print(
        f"⏰ Current context: {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day]} at {hour}:00 ({current_time_bucket})")

    # --- STEP 1: BUILD USER PROFILE FROM LISTENING HISTORY ---
    try:
        print("📊 Building user profile from listening history...")

        # Fetch user's listening data
        recent_res = sp.current_user_recently_played(limit=50)
        top_short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        top_medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        liked_res = sp.current_user_saved_tracks(limit=50)

        # Collect all played tracks
        all_played_tracks = []
        user_time_bucket_history = []  # Track user's time preferences
        artist_id_to_name = {}  # Mapping for later
        track_time_buckets = {}  # Track when each track was played

        # Process recent (has timestamps!)
        if recent_res:
            for item in recent_res['items']:
                track = item['track']
                if not track or not track.get('id'):
                    continue

                all_played_tracks.append(track)

                # Extract time bucket from timestamp
                try:
                    played_at = item.get('played_at')
                    if played_at:
                        played_time = datetime.datetime.fromisoformat(played_at.replace('Z', '+00:00'))
                        bucket = get_time_bucket(played_time.hour)
                        user_time_bucket_history.append(bucket)

                        # Track when this specific track was played
                        if track['id'] not in track_time_buckets:
                            track_time_buckets[track['id']] = []
                        track_time_buckets[track['id']].append(bucket)
                except:
                    pass

                # Store artist mapping
                if track['artists']:
                    artist_id_to_name[track['artists'][0]['id']] = track['artists'][0]['name']

        # Add top tracks (no timestamp, but still valuable)
        if top_short:
            all_played_tracks.extend([t for t in top_short['items'] if t])
        if top_medium:
            all_played_tracks.extend([t for t in top_medium['items'] if t])
        if liked_res:
            all_played_tracks.extend([item['track'] for item in liked_res['items'] if item.get('track')])

        # Build artist play count (artist_global_plays in training)
        artist_play_counts = Counter()
        seen_track_ids = set()

        for track in all_played_tracks:
            if track and track.get('id'):
                seen_track_ids.add(track['id'])
                if track['artists']:
                    artist_id = track['artists'][0]['id']
                    artist_play_counts[artist_id] += 1
                    artist_id_to_name[artist_id] = track['artists'][0]['name']

        if not artist_play_counts:
            raise HTTPException(status_code=400, detail="Not enough listening history to generate recommendations.")

        print(f"   ✅ Analyzed {len(all_played_tracks)} tracks from {len(artist_play_counts)} artists")
        print(f"   📊 Time bucket distribution: {Counter(user_time_bucket_history)}")
        print(f"   🎵 Top artists: {[(artist_id_to_name.get(a), c) for a, c in artist_play_counts.most_common(5)]}")

        # --- STEP 2: GENERATE CANDIDATE POOL ---
        print("🎯 Generating candidate pool from related artists...")

        # Get top artists (most played)
        top_artists = [artist_id for artist_id, count in artist_play_counts.most_common(10)]

        # Expand with related artists
        candidate_pool = []
        related_artist_cache = set()

        for artist_id in top_artists:
            try:
                # Get related artists
                related = sp.artist_related_artists(artist_id)['artists']

                for rel_artist in related[:10]:  # Top 5 related per seed
                    rel_id = rel_artist['id']
                    if rel_id not in related_artist_cache:
                        related_artist_cache.add(rel_id)
                        artist_id_to_name[rel_id] = rel_artist['name']

                        # Get top tracks from this related artist
                        try:
                            top_tracks = sp.artist_top_tracks(rel_id)['tracks']
                            candidate_pool.extend(top_tracks[:10])  # Top 10 tracks per artist
                        except:
                            continue
            except Exception as e:
                print(f"   ⚠️  Error with artist {artist_id}: {e}")
                continue

        print(f"   📈 Generated {len(candidate_pool)} candidate tracks")

    except Exception as e:
        print(f"❌ Data Collection Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Spotify error: {str(e)}")

    # --- STEP 3: SCORE CANDIDATES ---
    print("🤖 Scoring candidates with your model...")

    prediction_rows = []
    meta = []

    for track in candidate_pool:
        if not track or not track.get('id'):
            continue

        # Skip tracks user already has
        if track['id'] in seen_track_ids:
            continue

        track_id = track['id']
        artist_id = track['artists'][0]['id'] if track['artists'] else None

        if not artist_id:
            continue

        # Get artist play count (this is artist_global_plays in your training)
        artist_play_count = artist_play_counts.get(artist_id, 0)

        # Get track's time bucket history (if we've seen it before)
        track_buckets = track_time_buckets.get(track_id, None)

        # Extract features matching training script
        features = extract_features(
            artist_id=artist_id,
            day=day,
            current_time_bucket=current_time_bucket,
            artist_play_count=artist_play_count,
            user_time_buckets=user_time_bucket_history,
            track_time_buckets=track_buckets
        )

        prediction_rows.append(features)
        meta.append({
            "id": track['id'],
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "pop": track.get('popularity', 0),
            "url": track['external_urls']['spotify'],
            "albumArt": track['album']['images'][0]['url'] if track['album']['images'] else "",
            "artist_plays": artist_play_count
        })


    # Create DataFrame with exact column names
    feature_cols = [
        'day_of_week',
        'artist_global_plays',
        'user_affinity',
        'track_context_weight',
        'time_bucket_afternoon',
        'time_bucket_evening',
        'time_bucket_night'
    ]

    X_pred = pd.DataFrame(prediction_rows, columns=feature_cols)

    print(f"   🔢 Feature stats:")
    print(
        f"      - artist_global_plays: min={X_pred['artist_global_plays'].min()}, max={X_pred['artist_global_plays'].max()}, mean={X_pred['artist_global_plays'].mean():.1f}")
    print(f"      - user_affinity: mean={X_pred['user_affinity'].mean():.3f}")
    print(f"      - track_context_weight: mean={X_pred['track_context_weight'].mean():.3f}")

    # Ensemble prediction
    try:
        scores_xgb = model_xgb.predict_proba(X_pred)[:, 1]
        scores_lgbm = model_lgbm.predict_proba(X_pred)[:, 1]
        scores = (scores_xgb + scores_lgbm) / 2

        print(f"   📊 Score distribution: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    # Add scores to metadata
    for i in range(len(meta)):
        meta[i]['score'] = float(scores[i])

    # Apply diversity boost (favor discovery of new artists)
    for item in meta:
        if item['artist_plays'] == 0:  # Brand new artist
            item['score'] *= 1.12  # 12% boost
        elif item['artist_plays'] < 3:  # Rarely played
            item['score'] *= 1.06  # 6% boost

    # Sort and return top N
    final_recs = sorted(meta, key=lambda x: x['score'], reverse=True)[:size]

    print(f"✨ Returning top {len(final_recs)} recommendations")
    print(f"   📊 Score range: {final_recs[0]['score']:.3f} (best) to {final_recs[-1]['score']:.3f} (worst)")
    print(f"   🎵 Top pick: '{final_recs[0]['name']}' by {final_recs[0]['artist']}")

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
            description="🤖 AI Discovery - Trained on your listening patterns"
        )
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)
        return {"status": "success", "playlist_url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        print(f"❌ Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))