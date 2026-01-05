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
from fastapi.responses import JSONResponse
from fastapi import Request
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
def get_time_bucket() -> str:
    hour = datetime.datetime.utcnow().hour
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


@app.post("/recommend")
async def recommend(request: RecommendRequest, req: Request):
    if not model_xgb:
        raise HTTPException(status_code=500, detail="Models not active")

    session = requests.Session()
    session.trust_env = False

    sp = spotipy.Spotify(
        auth=request.token,
        requests_timeout=10,
        retries=2,
        requests_session=session
    )

    # --- TOKEN VALIDATION ---
    try:
        user = sp.current_user()
        market = user.get("country", "US")
        logger.info(f"✓ User authenticated. Market: {market}")
    except SpotifyException as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    # --- GET USER'S LISTENING CONTEXT ---
    context_pool = get_clean_history(sp)

    if len(context_pool) < 5:
        logger.info("Limited history, fetching top tracks...")
        try:
            top_tracks = sp.current_user_top_tracks(limit=50, time_range='short_term')
            context_pool.extend([
                track for track in top_tracks.get('items', [])
                if track.get('id') and is_valid_seed(track) and
                   track['id'] not in [t['id'] for t in context_pool]
            ])
        except:
            pass

    if not context_pool:
        raise HTTPException(
            status_code=400,
            detail="No listening history found. Please listen to some music on Spotify first."
        )

    logger.info(f"Context pool size: {len(context_pool)} tracks")

    # --- EXTRACT ARTISTS AND GENRES ---
    seed_artist_ids = []
    artist_names = []
    all_genres = []

    for track in context_pool:
        if track.get('artists'):
            for artist in track['artists']:
                if artist.get('id') and artist.get('name'):
                    seed_artist_ids.append(artist['id'])
                    artist_names.append(artist['name'])

    # Get artist details for genres
    unique_artist_ids = list(set(seed_artist_ids))
    random.shuffle(unique_artist_ids)

    logger.info(f"Fetching genre data from {min(15, len(unique_artist_ids))} artists...")
    for aid in unique_artist_ids[:15]:
        try:
            artist_info = sp.artist(aid)
            genres = artist_info.get('genres', [])
            all_genres.extend(genres)
            logger.info(f"  Artist genres: {genres[:3] if genres else 'none'}")
        except Exception as e:
            logger.warning(f"  Failed to get artist {aid}: {e}")
            continue

    if not all_genres:
        logger.warning("No genres found, will rely more on artist search")

    # --- BUILD CANDIDATE POOL ---
    target_pool_size = request.size * 4
    candidate_pool = []
    seen_ids = set([t['id'] for t in context_pool])  # Excludes recently played from recommendations

    logger.info(f"Building candidate pool (target: {target_pool_size})...")

    # STRATEGY 1: Search by artists you know (75% of pool - familiar tracks)
    random.shuffle(artist_names)
    search_artists = [a for a in artist_names if a][:15]
    known_artists_set = set(search_artists)  # Artists from your history

    familiar_pool = []
    discovery_pool = []

    for i in range(0, len(search_artists), 2):
        if len(familiar_pool) >= target_pool_size * 0.75:
            break

        query_artists = search_artists[i:i + 2]
        query = ' OR '.join(query_artists)

        try:
            results = sp.search(q=query, type='track', limit=50, market=market)

            for item in results.get('tracks', {}).get('items', []):
                if not item.get('id') or item['id'] in seen_ids or not is_valid_seed(item):
                    continue

                # Separate familiar vs discovery based on artist
                item_artists = [a.get('name', '') for a in item.get('artists', [])]
                is_familiar = any(a in known_artists_set for a in item_artists)

                if is_familiar and len(familiar_pool) < target_pool_size * 0.75:
                    familiar_pool.append(item)
                    seen_ids.add(item['id'])
                elif not is_familiar and len(discovery_pool) < target_pool_size * 0.25:
                    discovery_pool.append(item)
                    seen_ids.add(item['id'])

        except Exception as e:
            logger.warning(f"Search failed for artists {query_artists}: {e}")
            continue

    candidate_pool = familiar_pool + discovery_pool
    logger.info(
        f"  After artist search: {len(familiar_pool)} familiar + {len(discovery_pool)} discovery = {len(candidate_pool)} candidates")

    # STRATEGY 2: Genre-based search (only for discovery if needed)
    if all_genres and len(discovery_pool) < target_pool_size * 0.25:
        genre_counts = Counter(all_genres)
        top_genres = [g for g, _ in genre_counts.most_common(8)]

        for genre in top_genres:
            if len(discovery_pool) >= target_pool_size * 0.25:
                break

            try:
                # Search with genre tag
                results = sp.search(
                    q=f'genre:"{genre}"',
                    type='track',
                    limit=30,
                    market=market
                )

                for item in results.get('tracks', {}).get('items', []):
                    if not (item.get('id') and item['id'] not in seen_ids and is_valid_seed(item)):
                        continue

                    # Only add if artist is NOT in known list (pure discovery)
                    item_artists = [a.get('name', '') for a in item.get('artists', [])]
                    if not any(a in known_artists_set for a in item_artists):
                        discovery_pool.append(item)
                        candidate_pool.append(item)
                        seen_ids.add(item['id'])

            except Exception as e:
                logger.warning(f"Genre search failed for '{genre}': {e}")
                continue

        logger.info(
            f"  After genre search: {len(familiar_pool)} familiar + {len(discovery_pool)} discovery = {len(candidate_pool)} candidates")

    # STRATEGY 3: Deep dive into known artists' albums (familiar content)
    for aid in unique_artist_ids[:15]:
        if len(familiar_pool) >= target_pool_size * 0.75:
            break

        try:
            albums = sp.artist_albums(
                aid,
                limit=5,
                album_type='album,single',
                market=market
            )

            for album in albums.get('items', [])[:3]:
                album_tracks = sp.album_tracks(album['id'], limit=10, market=market)

                for item in album_tracks.get('items', []):
                    if (item.get('id') and
                            item['id'] not in seen_ids):
                        # Need to check if valid (album_tracks returns simplified objects)
                        try:
                            full_track = sp.track(item['id'], market=market)
                            if is_valid_seed(full_track):
                                familiar_pool.append(full_track)
                                candidate_pool.append(full_track)
                                seen_ids.add(item['id'])
                        except:
                            continue

                        if len(familiar_pool) >= target_pool_size * 0.75:
                            break
        except Exception as e:
            logger.warning(f"Album search failed for artist {aid}: {e}")
            continue

    logger.info(
        f"  After album dive: {len(familiar_pool)} familiar + {len(discovery_pool)} discovery = {len(candidate_pool)} candidates")

    # STRATEGY 4: User's liked songs (familiar content only)
    if len(familiar_pool) < target_pool_size * 0.75:
        try:
            saved_tracks = sp.current_user_saved_tracks(limit=50, market=market)
            for item in saved_tracks.get('items', []):
                track = item.get('track')
                if (track and track.get('id') and
                        track['id'] not in seen_ids and
                        is_valid_seed(track)):
                    familiar_pool.append(track)
                    candidate_pool.append(track)
                    seen_ids.add(track['id'])

                    if len(familiar_pool) >= target_pool_size * 0.75:
                        break
        except Exception as e:
            logger.warning(f"Saved tracks fetch failed: {e}")

    logger.info(
        f"Final candidate pool: {len(familiar_pool)} familiar (75%) + {len(discovery_pool)} discovery (25%) = {len(candidate_pool)} total")

    if not candidate_pool:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate recommendations. Try following more artists or adding songs to your library."
        )

    # --- AI SCORING ---
    now_bucket = get_time_bucket()
    day = datetime.datetime.utcnow().weekday()

    valid_context_artists = [
        t['artists'][0]['id']
        for t in context_pool
        if t.get('artists') and t['artists'][0].get('id')
    ]

    artist_counts = Counter(valid_context_artists)
    user_time_buckets = [now_bucket]

    meta = []
    rows = []

    for t in candidate_pool:
        if not t.get('artists') or not t['artists'][0].get('id'):
            continue

        aid = t["artists"][0]["id"]
        rows.append(
            extract_features(
                aid,
                day,
                now_bucket,
                artist_counts.get(aid, 0),
                user_time_buckets,
                [now_bucket]
            )
        )

        meta.append({
            "id": t["id"],
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "url": t["external_urls"]["spotify"],
            "albumArt": t["album"]["images"][0]["url"] if t["album"].get("images") else None,
            "artist_plays": artist_counts.get(aid, 0),
            "score": 0.0
        })

    if not meta:
        raise HTTPException(status_code=500, detail="No valid tracks to score")

    logger.info(f"Scoring {len(meta)} tracks with ML models...")

    X = pd.DataFrame(rows, columns=[
        "day_of_week",
        "artist_global_plays",
        "user_affinity",
        "track_context_weight",
        "time_bucket_afternoon",
        "time_bucket_evening",
        "time_bucket_night"
    ]).fillna(0)

    with model_lock:
        try:
            p1 = model_xgb.predict_proba(X)[:, 1]
            p2 = model_lgbm.predict_proba(X)[:, 1]
            scores = (p1 + p2) / 2
        except Exception as e:
            logger.error(f"Model scoring failed: {e}")
            scores = [random.random() for _ in meta]

    # --- FINAL FILTERING & RANKING ---
    final = []
    artist_cap = Counter()

    ranked = sorted(zip(meta, scores), key=lambda x: x[1], reverse=True)

    for item, score in ranked:
        # Limit tracks per artist for diversity
        if artist_cap[item["artist"]] >= 5:
            continue

        # Boost discovery slightly (new artists) but keep it modest
        boost = 1.15 if item["artist_plays"] == 0 else 1.0
        item["score"] = float(score) * boost

        final.append(item)
        artist_cap[item["artist"]] += 1

        if len(final) >= request.size:
            break

    logger.info(f"✓ Returning {len(final)} recommendations")
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0, private",
        "Pragma": "no-cache",
        "Expires": "0",
        "Vary": "Authorization",
    }

    return JSONResponse(
        content={"recommendations": final},
        headers=headers
    )


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
