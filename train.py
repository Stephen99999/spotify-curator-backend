import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
import glob
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def log_step(msg):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


# ==========================================
# 1. LOAD & PREPROCESS SPOTIFY DATA
# ==========================================
log_step("Searching for Spotify Streaming History files...")

# Find all JSON files matching the pattern in your screenshot
json_files = glob.glob("Streaming_History_Audio_*.json")

if not json_files:
    raise FileNotFoundError("No 'Streaming_History_Audio_*.json' files found in the directory.")

dfs = []
for file in json_files:
    log_step(f"Loading {file}...")
    try:
        # Load JSON
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert to DataFrame
        temp_df = pd.DataFrame(data)
        dfs.append(temp_df)
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

# Combine all history files
raw_df = pd.concat(dfs, ignore_index=True)
log_step(f"Total raw streams loaded: {len(raw_df)}")

# Filter: Keep only actual plays (Spotify counts a stream after 30s)
df = raw_df[raw_df['ms_played'] > 30000].copy()

# ==========================================
# 2. FEATURE MAPPING & ENGINEERING
# ==========================================
log_step("Mapping Spotify Schema to Model Schema...")

# Map Spotify columns to your training columns
# Note: We use URIs as IDs to ensure uniqueness
df['timestamp'] = pd.to_datetime(df['ts'])
df['user_id'] = "current_user"  # Since this is your personal data, ID is static
df['artist_name'] = df['master_metadata_album_artist_name']
df['track_name'] = df['master_metadata_track_name']
df['artist_id'] = df['master_metadata_album_artist_name'].astype("category").cat.codes  # Create internal ID
df['track_id'] = df['spotify_track_uri'].astype("category").cat.codes  # Create internal ID

# Time Features
df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["time_bucket"] = pd.cut(df["hour"], bins=[-1, 4, 11, 16, 21, 24],
                           labels=["night", "morning", "afternoon", "evening", "night"],
                           ordered=False).astype("category")

# --- Discovery Features (Recalculated for Personal Data) ---

# 1. Artist Frequency (Personal)
# Note: In the original model, this was "Global Popularity".
# Here, it becomes "Your Artist Frequency".
artist_pop = df.groupby('artist_id', observed=True).size().reset_index(name='artist_global_plays')

# 2. User Affinity (Time slot preference)
user_activity = df.groupby(['user_id', 'time_bucket'], observed=True).size().reset_index(name='u_bucket_count')
user_total = df.groupby('user_id', observed=True).size().reset_index(name='u_total')
user_profile = user_activity.merge(user_total, on='user_id')
user_profile['user_affinity'] = user_profile['u_bucket_count'] / user_profile['u_total']

# 3. Track Context (Time slot preference for specific tracks)
track_pop = df.groupby('track_id', observed=True).size().reset_index(name='global_plays')
track_time = df.groupby(['track_id', 'time_bucket'], observed=True).size().reset_index(name='t_bucket_count')
track_profile = track_time.merge(track_pop, on='track_id')
track_profile['track_context_weight'] = track_profile['t_bucket_count'] / track_profile['global_plays']

# ==========================================
# 3. BUILD BALANCED DATASET (With Negatives)
# ==========================================
log_step("Building Training Data (Positives + Negatives)...")

# Positives: What you actually listened to
positives = df[['user_id', 'artist_id', 'track_id', 'day_of_week', 'time_bucket']].copy()
positives['label'] = 1

# Negatives: What you DIDN'T listen to (Random Sampling)
# We generate negatives to teach the model what you skip/ignore
all_artists = df['artist_id'].unique()
negatives = positives.sample(frac=1.0, random_state=42).copy()
negatives['artist_id'] = np.random.choice(all_artists, size=len(negatives))
negatives['label'] = 0

# Merge
data = pd.concat([positives, negatives])

# Join calculated features
data = data.merge(artist_pop, on='artist_id', how='left')
data = data.merge(user_profile[['user_id', 'time_bucket', 'user_affinity']], on=['user_id', 'time_bucket'], how='left')
data = data.merge(track_profile[['track_id', 'time_bucket', 'track_context_weight']], on=['track_id', 'time_bucket'],
                  how='left')

# One-Hot Encode Time Buckets
for bucket in ['afternoon', 'evening', 'night']:
    data[f"time_bucket_{bucket}"] = (data["time_bucket"] == bucket).astype("int8")

# Select the exact 7 features the old model expects
features = ['day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
            'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night']

X = data[features].fillna(0).astype('float32')
y = data['label']

# Split for validation to monitor overfitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
gc.collect()

# ==========================================
# 4. INCREMENTAL TRAINING (Fine-Tuning)
# ==========================================
log_step("Loading existing models for Fine-Tuning...")

try:
    # --- A. XGBoost Fine-Tuning ---
    if os.path.exists("discovery_xgb.pkl"):
        log_step("Updating XGBoost Model...")
        loaded_xgb = joblib.load("discovery_xgb.pkl")

        # We re-instantiate a classifier with the same params but LOWER learning rate for fine-tuning
        # This prevents the model from forgetting the old data too quickly (Catastrophic Forgetting)
        new_xgb = xgb.XGBClassifier(
            n_estimators=50,  # Fewer trees for update
            learning_rate=0.01,  # Lower rate to avoid overfitting
            max_depth=loaded_xgb.max_depth,
            tree_method="hist",
            n_jobs=-1
        )

        # 'xgb_model' parameter allows incremental learning
        new_xgb.fit(X_train, y_train, xgb_model=loaded_xgb)

        joblib.dump(new_xgb, "discovery_xgb_finetuned.pkl")
        print(f"   > XGBoost updated (Validation AUC: {roc_auc_score(y_val, new_xgb.predict_proba(X_val)[:, 1]):.4f})")
    else:
        print("   ! discovery_xgb.pkl not found. Skipping XGB update.")

    # --- B. LightGBM Fine-Tuning ---
    if os.path.exists("discovery_lgbm.pkl"):
        log_step("Updating LightGBM Model...")
        loaded_lgbm = joblib.load("discovery_lgbm.pkl")

        new_lgbm = lgb.LGBMClassifier(
            n_estimators=50,  # Fewer trees
            learning_rate=0.01,  # Lower rate
            num_leaves=loaded_lgbm.num_leaves,
            n_jobs=-1
        )

        # 'init_model' parameter allows incremental learning
        new_lgbm.fit(X_train, y_train, init_model=loaded_lgbm)

        joblib.dump(new_lgbm, "discovery_lgbm_finetuned.pkl")
        print(
            f"   > LightGBM updated (Validation AUC: {roc_auc_score(y_val, new_lgbm.predict_proba(X_val)[:, 1]):.4f})")
    else:
        print("   ! discovery_lgbm.pkl not found. Skipping LGBM update.")

except Exception as e:
    print(f"Error during model update: {e}")
    print("Ensure the .pkl files are in the same directory and compatible.")

log_step("Process Complete.")