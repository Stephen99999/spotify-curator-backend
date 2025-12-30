import os
import joblib
from dotenv import load_dotenv
import datetime
import random
from collections import Counter
import pandas as pd
import numpy as np
import spotipy
import xgboost as xgb
import lightgbm as lgb
from pydantic import BaseModel
from typing import List, Optional
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

# Load pre-trained models from training data
try:
    model_xgb = joblib.load("discovery_xgb_finetuned.pkl")
    model_lgbm = joblib.load("discovery_lgbm_finetuned.pkl")
    print("✅ Pre-trained AI models loaded successfully")
except Exception as e:
    print(f"❌ ERROR: Could not load models. Ensure discovery_xgb_finetuned.pkl and discovery_lgbm_finetuned.pkl exist.")
    print(f"   Details: {e}")
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


# --- ENHANCED FEATURE ENGINEERING (ALIGNED WITH TRAINING DATA) ---
def extract_audio_features(sp, track_ids):
    """Batch fetch audio features for multiple tracks"""
    try:
        # Spotify API limit is 100 tracks per request
        all_features = {}
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            audio_features = sp.audio_features(batch)
            all_features.update({af['id']: af for af in audio_features if af})
        return all_features
    except Exception as e:
        print(f"   ⚠️  Audio features error: {e}")
        return {}


def extract_features(track, now, day, aft, eve, ngt, audio_feat=None, is_recent=False,
                     artist_freq=0, genre_match_score=0.0, recency_days=None,
                     ms_played=None, shuffle=None, skipped=None):
    """
    CRITICAL: Features must EXACTLY match training data structure

    Based on Spotify Extended Streaming History format:
    - ts, ms_played, shuffle, skipped, reason_start, reason_end

    Features (18 total - matching training):
    1. day_of_week: 0-6
    2. hour_of_day: 0-23
    3. popularity: 0-100 (normalized to 0-1)
    4. ms_played_normalized: listening duration (0-1)
    5. user_affinity: personalization signal (0-1)
    6. artist_frequency: normalized play count (0-1)
    7. genre_match_score: genre alignment (0-1)
    8. recency_weight: exponential decay (0-1)
    9. shuffle: binary (1 if shuffled)
    10. skipped: binary (1 if skipped)
    11-13. time_bucket: afternoon/evening/night
    14. danceability: 0-1
    15. energy: 0-1
    16. valence: 0-1
    17. tempo_normalized: 0-1
    18. acousticness: 0-1
    """

    # Temporal features
    hour = now.hour

    # Core features
    popularity = track.get('popularity', 50) / 100.0

    # Duration signal (from training: ms_played)
    # Normalize to 0-1 assuming max song length ~10 minutes (600,000 ms)
    if ms_played:
        duration_signal = min(ms_played / 600000.0, 1.0)
    else:
        # Estimate: if is_recent, assume full play (high signal)
        duration_signal = 0.85 if is_recent else 0.5

    # User affinity with recency decay
    if is_recent:
        base_affinity = 0.9
        if recency_days is not None:
            recency_weight = np.exp(-recency_days / 30.0)
        else:
            recency_weight = 1.0
        user_affinity = base_affinity * recency_weight
    else:
        user_affinity = 0.3
        recency_weight = 0.0

    # Artist frequency (normalized)
    artist_freq_norm = min(artist_freq / 20.0, 1.0)

    # Behavioral signals (from training data)
    shuffle_signal = 1 if shuffle else 0
    skipped_signal = 1 if skipped else 0

    # Audio features
    if audio_feat:
        danceability = audio_feat.get('danceability', 0.5)
        energy = audio_feat.get('energy', 0.5)
        valence = audio_feat.get('valence', 0.5)
        tempo = min(audio_feat.get('tempo', 120) / 200.0, 1.0)
        acousticness = audio_feat.get('acousticness', 0.5)
    else:
        danceability = 0.5
        energy = 0.5
        valence = 0.5
        tempo = 0.6
        acousticness = 0.5

    return [
        day,  # 0: day_of_week
        hour,  # 1: hour_of_day
        popularity,  # 2: popularity
        duration_signal,  # 3: ms_played_normalized
        user_affinity,  # 4: user_affinity
        artist_freq_norm,  # 5: artist_frequency
        genre_match_score,  # 6: genre_match_score
        recency_weight,  # 7: recency_weight
        shuffle_signal,  # 8: shuffle
        skipped_signal,  # 9: skipped
        aft,  # 10: time_bucket_afternoon
        eve,  # 11: time_bucket_evening
        ngt,  # 12: time_bucket_night
        danceability,  # 13: danceability
        energy,  # 14: energy
        valence,  # 15: valence
        tempo,  # 16: tempo_normalized
        acousticness  # 17: acousticness
    ]


def calculate_genre_similarity(user_genres, artist_genres):
    """Calculate genre overlap score"""
    if not user_genres or not artist_genres:
        return 0.5

    user_set = set([g.lower() for g in user_genres])
    artist_set = set([g.lower() for g in artist_genres])

    exact_matches = len(user_set & artist_set)

    partial_matches = 0
    for ug in user_set:
        for ag in artist_set:
            if ug in ag or ag in ug:
                partial_matches += 1
                break

    total_matches = exact_matches + (0.5 * partial_matches)
    max_possible = max(len(user_set), len(artist_set))

    return min(total_matches / max_possible, 1.0) if max_possible > 0 else 0.5


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
    Generate personalized recommendations using YOUR pre-trained model
    Model was trained on Spotify Extended Streaming History data
    """

    if not model_xgb or not model_lgbm:
        raise HTTPException(
            status_code=500,
            detail="Models not loaded. Please ensure discovery_xgb_finetuned.pkl and discovery_lgbm_finetuned.pkl exist."
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

    # Time Features
    day = now.weekday()
    hour = now.hour
    aft, eve, ngt = 0, 0, 0
    if 12 <= hour < 17:
        aft = 1
    elif 17 <= hour < 22:
        eve = 1
    elif (hour >= 22 or hour < 5):
        ngt = 1

    # --- STEP 1: DATA COLLECTION ---
    positives = []
    negatives = []
    all_artist_ids = []
    artist_play_times = {}
    user_genres = []

    try:
        print("📊 Fetching user listening history...")

        # Gather listening data (simulating training data structure)
        recent_res = sp.current_user_recently_played(limit=50)
        top_short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        top_medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        liked_res = sp.current_user_saved_tracks(limit=50)

        raw_positives = []

        # Recent tracks have timestamps (most valuable)
        if recent_res:
            for item in recent_res['items']:
                track = item['track']
                played_at = item.get('played_at')
                raw_positives.append({
                    'track': track,
                    'played_at': played_at,
                    'context': 'recent'
                })

        # Top tracks (treated as high-affinity but older)
        if top_short:
            for track in top_short['items']:
                raw_positives.append({
                    'track': track,
                    'played_at': None,
                    'context': 'top_short'
                })

        if top_medium:
            for track in top_medium['items']:
                raw_positives.append({
                    'track': track,
                    'played_at': None,
                    'context': 'top_medium'
                })

        # Liked tracks
        if liked_res:
            for item in liked_res['items']:
                raw_positives.append({
                    'track': item['track'],
                    'played_at': None,
                    'context': 'liked'
                })

        # Deduplicate
        seen_pos_ids = set()
        final_positives = []
        positive_track_ids = []

        for item in raw_positives:
            track = item['track']
            if not track or not track.get('id'):
                continue

            if track['id'] not in seen_pos_ids:
                final_positives.append(item)
                seen_pos_ids.add(track['id'])
                positive_track_ids.append(track['id'])

                if track['artists']:
                    artist_id = track['artists'][0]['id']
                    all_artist_ids.append(artist_id)

                    if item['played_at'] and artist_id not in artist_play_times:
                        artist_play_times[artist_id] = item['played_at']

        positives = final_positives

        if not all_artist_ids:
            raise HTTPException(status_code=400, detail="Not enough history to generate seeds.")

        print(f"   ✅ Found {len(positives)} positive tracks")

        # Extract user genre profile
        try:
            top_artists = sp.current_user_top_artists(limit=20, time_range='short_term')
            for artist in top_artists['items']:
                user_genres.extend(artist.get('genres', []))
            user_genres = list(set(user_genres))
            print(f"   🎵 Genre profile: {len(user_genres)} genres")
        except:
            user_genres = []

        # Artist frequency analysis
        artist_counts = Counter(all_artist_ids)
        top_frequent_artists = [artist_id for artist_id, count in artist_counts.most_common(10)]

        print(f"   🎯 Top artists: {len(top_frequent_artists)}")

        # Expand artist pool with genre-aware selection
        expanded_artist_pool = list(top_frequent_artists)
        artist_genre_cache = {}

        for a_id in top_frequent_artists:
            try:
                artist_info = sp.artist(a_id)
                artist_genre_cache[a_id] = artist_info.get('genres', [])

                related = sp.artist_related_artists(a_id)['artists']

                scored_related = []
                for r in related[:10]:
                    r_genres = r.get('genres', [])
                    genre_score = calculate_genre_similarity(user_genres, r_genres)
                    scored_related.append((r['id'], genre_score, r_genres))

                scored_related.sort(key=lambda x: x[1], reverse=True)
                for r_id, score, r_genres in scored_related[:4]:
                    expanded_artist_pool.append(r_id)
                    artist_genre_cache[r_id] = r_genres

            except Exception as e:
                continue

        final_seed_artists = list(set(expanded_artist_pool))
        print(f"   📈 Artist pool: {len(final_seed_artists)} artists")

        # Collect candidates
        for a_id in final_seed_artists:
            try:
                artist_tracks = sp.artist_top_tracks(a_id)['tracks']
                negatives.extend(artist_tracks)

                if a_id not in artist_genre_cache:
                    artist_info = sp.artist(a_id)
                    artist_genre_cache[a_id] = artist_info.get('genres', [])
            except:
                continue

        print(f"   🎼 Candidates: {len(negatives)} tracks")

    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        raise HTTPException(status_code=400, detail=f"Spotify error: {str(e)}")

    if not positives or not negatives:
        raise HTTPException(status_code=400, detail="Could not build recommendation pool.")

    # --- STEP 2: AUDIO FEATURES ---
    print("🎚️  Fetching audio features...")
    positive_audio_features = extract_audio_features(sp, positive_track_ids)
    negative_track_ids = [t['id'] for t in negatives if t and t.get('id')]
    negative_audio_features = extract_audio_features(sp, negative_track_ids)

    # --- STEP 3: INFERENCE ONLY (NO RETRAINING) ---
    print("🤖 Running inference with your pre-trained model...")

    # We DON'T retrain - we use the model as-is
    # Your model was trained on millions of rows from Extended Streaming History
    # Real-time data is just for context, not retraining

    # --- STEP 4: RANKING ---
    print("🎯 Ranking candidates...")

    meta = []
    prediction_rows = []
    unique_pool = {t['id']: t for t in negatives if t and 'id' in t}.values()

    for track in unique_pool:
        if track['id'] in seen_pos_ids:
            continue

        track_id = track['id']
        artist_id = track['artists'][0]['id'] if track['artists'] else None

        audio_feat = negative_audio_features.get(track_id)
        artist_freq = artist_counts.get(artist_id, 0) if artist_id else 0
        artist_genres = artist_genre_cache.get(artist_id, [])
        genre_score = calculate_genre_similarity(user_genres, artist_genres)

        # For inference: we don't have actual ms_played, shuffle, skipped
        # So we use neutral/estimated values
        feats = extract_features(
            track, now, day, aft, eve, ngt,
            audio_feat=audio_feat,
            is_recent=False,
            artist_freq=artist_freq,
            genre_match_score=genre_score,
            recency_days=None,
            ms_played=None,  # Unknown for new tracks
            shuffle=False,  # Neutral assumption
            skipped=False  # Neutral assumption
        )

        prediction_rows.append(feats)
        meta.append({
            "id": track['id'],
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "pop": track['popularity'],
            "url": track['external_urls']['spotify'],
            "albumArt": track['album']['images'][0]['url'] if track['album']['images'] else "",
            "genre_score": genre_score,
            "artist_freq": artist_freq
        })

    if not prediction_rows:
        raise HTTPException(status_code=400, detail="No new songs found to recommend.")

    # CRITICAL: Column names must match training data
    feature_cols = [
        'day_of_week', 'hour_of_day', 'popularity', 'ms_played_normalized',
        'user_affinity', 'artist_frequency', 'genre_match_score', 'recency_weight',
        'shuffle', 'skipped',
        'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night',
        'danceability', 'energy', 'valence', 'tempo_normalized', 'acousticness'
    ]

    X_pred = pd.DataFrame(prediction_rows, columns=feature_cols)

    # Ensemble prediction with YOUR models
    scores_xgb = model_xgb.predict_proba(X_pred)[:, 1]
    scores_lgbm = model_lgbm.predict_proba(X_pred)[:, 1]
    scores = (scores_xgb + scores_lgbm) / 2

    for i in range(len(meta)):
        meta[i]['score'] = float(scores[i])

    # Diversity boost for discovery
    for item in meta:
        if item['artist_freq'] < 2:
            item['score'] *= 1.08  # 8% boost for new artists

    final_recs = sorted(meta, key=lambda x: x['score'], reverse=True)[:size]

    print(f"✨ Returning {len(final_recs)} recommendations (avg score: {np.mean([r['score'] for r in final_recs]):.3f})")
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
            description="🤖 AI-Powered Discovery - Trained on your listening history"
        )
        track_uris = [f"spotify:track:{tid}" for tid in request.track_ids]
        sp.playlist_add_items(playlist_id=playlist["id"], items=track_uris)
        return {"status": "success", "playlist_url": playlist["external_urls"]["spotify"]}
    except Exception as e:
        print(f"❌ Save Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))