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


# --- FEATURE ENGINEERING (MATCHING YOUR MODEL) ---
def extract_features(track, now, day, aft, eve, ngt, is_recent=False, artist_freq=0):
    """
    EXACT features your model was trained on (7 features):

    1. day_of_week: 0-6 (Monday=0, Sunday=6)
    2. artist_global_plays: How many times user played this artist (0-100+)
    3. user_affinity: 0-1 (higher for recent/liked tracks)
    4. track_context_weight: 0-1 (context signal - popularity, explicit, etc)
    5. time_bucket_afternoon: binary (1 if 12-17h)
    6. time_bucket_evening: binary (1 if 17-22h)
    7. time_bucket_night: binary (1 if 22-5h)
    """

    # Feature 1: day_of_week (0-6)
    day_of_week = day

    # Feature 2: artist_global_plays (simulated from artist frequency)
    # This represents how many times the user has played this artist
    artist_global_plays = artist_freq

    # Feature 3: user_affinity (0-1)
    # High for tracks in user's recent/liked, lower for discoveries
    if is_recent:
        user_affinity = 0.9  # High affinity for known tracks
    else:
        # Scale based on artist familiarity
        # If we know the artist (artist_freq > 0), moderate affinity
        # If new artist, lower affinity
        if artist_freq > 5:
            user_affinity = 0.6
        elif artist_freq > 0:
            user_affinity = 0.4
        else:
            user_affinity = 0.2

    # Feature 4: track_context_weight (0-1)
    # Represents track quality/fit based on popularity and other signals
    popularity = track.get('popularity', 50)
    is_explicit = track.get('explicit', False)

    # Higher weight for popular tracks, slight boost for explicit (genre indicator)
    context_weight = (popularity / 100.0) * 0.8
    if is_explicit:
        context_weight += 0.2
    context_weight = min(context_weight, 1.0)

    # Features 5-7: time_buckets
    time_bucket_afternoon = aft
    time_bucket_evening = eve
    time_bucket_night = ngt

    return [
        day_of_week,  # 0: day_of_week
        artist_global_plays,  # 1: artist_global_plays
        user_affinity,  # 2: user_affinity
        context_weight,  # 3: track_context_weight
        time_bucket_afternoon,  # 4: time_bucket_afternoon
        time_bucket_evening,  # 5: time_bucket_evening
        time_bucket_night  # 6: time_bucket_night
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
    user_genres = []

    try:
        print("📊 Fetching user listening history...")

        # Gather listening data
        recent_res = sp.current_user_recently_played(limit=50)
        top_short = sp.current_user_top_tracks(limit=50, time_range='short_term')
        top_medium = sp.current_user_top_tracks(limit=50, time_range='medium_term')
        liked_res = sp.current_user_saved_tracks(limit=50)

        raw_positives = []

        if recent_res:
            for item in recent_res['items']:
                track = item['track']
                raw_positives.append({'track': track, 'context': 'recent'})

        if top_short:
            for track in top_short['items']:
                raw_positives.append({'track': track, 'context': 'top_short'})

        if top_medium:
            for track in top_medium['items']:
                raw_positives.append({'track': track, 'context': 'top_medium'})

        if liked_res:
            for item in liked_res['items']:
                raw_positives.append({'track': item['track'], 'context': 'liked'})

        # Deduplicate
        seen_pos_ids = set()
        final_positives = []

        for item in raw_positives:
            track = item['track']
            if not track or not track.get('id'):
                continue

            if track['id'] not in seen_pos_ids:
                final_positives.append(item)
                seen_pos_ids.add(track['id'])

                if track['artists']:
                    artist_id = track['artists'][0]['id']
                    all_artist_ids.append(artist_id)

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

        # Artist frequency analysis (CRITICAL for artist_global_plays feature)
        artist_counts = Counter(all_artist_ids)
        top_frequent_artists = [artist_id for artist_id, count in artist_counts.most_common(10)]

        print(f"   🎯 Top {len(top_frequent_artists)} artists identified")
        print(f"   📊 Artist play counts: {dict(artist_counts.most_common(5))}")

        # Expand artist pool intelligently
        expanded_artist_pool = list(top_frequent_artists)
        artist_genre_cache = {}

        for a_id in top_frequent_artists:
            try:
                artist_info = sp.artist(a_id)
                artist_genre_cache[a_id] = artist_info.get('genres', [])

                related = sp.artist_related_artists(a_id)['artists']

                # Score related artists by genre similarity
                scored_related = []
                for r in related[:10]:
                    r_genres = r.get('genres', [])
                    genre_score = calculate_genre_similarity(user_genres, r_genres)
                    scored_related.append((r['id'], genre_score, r_genres))

                # Take top 4 most genre-similar
                scored_related.sort(key=lambda x: x[1], reverse=True)
                for r_id, score, r_genres in scored_related[:4]:
                    expanded_artist_pool.append(r_id)
                    artist_genre_cache[r_id] = r_genres

            except Exception as e:
                print(f"   ⚠️  Error processing artist {a_id}: {e}")
                continue

        final_seed_artists = list(set(expanded_artist_pool))
        print(f"   📈 Expanded to {len(final_seed_artists)} seed artists")

        # Collect candidate tracks
        for a_id in final_seed_artists:
            try:
                artist_tracks = sp.artist_top_tracks(a_id)['tracks']
                negatives.extend(artist_tracks)

                if a_id not in artist_genre_cache:
                    artist_info = sp.artist(a_id)
                    artist_genre_cache[a_id] = artist_info.get('genres', [])
            except:
                continue

        print(f"   🎼 Generated {len(negatives)} candidate tracks")

    except Exception as e:
        print(f"❌ Fetch Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Spotify error: {str(e)}")

    if not positives or not negatives:
        raise HTTPException(status_code=400, detail="Could not build recommendation pool.")

    # --- STEP 2: RANKING WITH YOUR MODEL ---
    print("🤖 Running inference with pre-trained model...")

    meta = []
    prediction_rows = []
    unique_pool = {t['id']: t for t in negatives if t and 'id' in t}.values()

    for track in unique_pool:
        # Don't recommend tracks user already has
        if track['id'] in seen_pos_ids:
            continue

        track_id = track['id']
        artist_id = track['artists'][0]['id'] if track['artists'] else None

        # Get artist play count (artist_global_plays feature)
        artist_freq = artist_counts.get(artist_id, 0) if artist_id else 0

        # Extract features matching your model
        feats = extract_features(
            track, now, day, aft, eve, ngt,
            is_recent=False,
            artist_freq=artist_freq
        )

        prediction_rows.append(feats)

        # Get genre score for metadata (not used in model, but useful for user)
        artist_genres = artist_genre_cache.get(artist_id, [])
        genre_score = calculate_genre_similarity(user_genres, artist_genres)

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

    # CRITICAL: Use exact feature names from your model
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

    print(f"   🎯 Ranking {len(X_pred)} candidates...")
    print(f"   📊 Feature sample: {X_pred.iloc[0].to_dict()}")

    # Ensemble prediction
    try:
        scores_xgb = model_xgb.predict_proba(X_pred)[:, 1]
        scores_lgbm = model_lgbm.predict_proba(X_pred)[:, 1]
        scores = (scores_xgb + scores_lgbm) / 2
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    for i in range(len(meta)):
        meta[i]['score'] = float(scores[i])

    # Diversity boost: slightly favor new artists for discovery
    for item in meta:
        if item['artist_freq'] == 0:  # Brand new artist
            item['score'] *= 1.15  # 15% boost
        elif item['artist_freq'] < 2:  # Rarely played artist
            item['score'] *= 1.08  # 8% boost

    final_recs = sorted(meta, key=lambda x: x['score'], reverse=True)[:size]

    avg_score = np.mean([r['score'] for r in final_recs])
    print(f"✨ Returning {len(final_recs)} recommendations")
    print(
        f"   📊 Score range: {min([r['score'] for r in final_recs]):.3f} - {max([r['score'] for r in final_recs]):.3f}")
    print(f"   📊 Average score: {avg_score:.3f}")

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