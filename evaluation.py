import pandas as pd
import numpy as np
import joblib
import gc
import json
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def log_step(msg):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")


# ==========================================
# 1. LOAD & PROCESS SPOTIFY DATA
# ==========================================
log_step("Searching for Spotify Streaming History files...")
json_files = glob.glob("Streaming_History_Audio_*.json")

if not json_files:
    raise FileNotFoundError("No 'Streaming_History_Audio_*.json' files found.")

dfs = []
for file in json_files:
    try:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dfs.append(pd.DataFrame(data))
    except Exception as e:
        print(f"Skipping {file}: {e}")

raw_df = pd.concat(dfs, ignore_index=True)
# Filter for actual plays (>30s)
df = raw_df[raw_df['ms_played'] > 30000].copy()

log_step(f"Total valid streams: {len(df)}")

# Map Schema
df['timestamp'] = pd.to_datetime(df['ts'])
df['user_id'] = "current_user"
df['artist_name'] = df['master_metadata_album_artist_name']
df['track_name'] = df['master_metadata_track_name']
df['artist_id'] = df['master_metadata_album_artist_name'].astype("category").cat.codes
df['track_id'] = df['spotify_track_uri'].astype("category").cat.codes

df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["time_bucket"] = pd.cut(df["hour"], bins=[-1, 4, 11, 16, 21, 24],
                           labels=["night", "morning", "afternoon", "evening", "night"],
                           ordered=False).astype("category")

# ==========================================
# 2. SPLIT DATA (CRITICAL FOR VALIDATION)
# ==========================================
# We split 80/20. We compute features on the 80% (Train) and evaluate on the 20% (Test).
# This prevents data leakage (the model shouldn't "know" the future).
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

log_step(f"Training Context size: {len(train_df)} | Evaluation Test size: {len(test_df)}")

# ==========================================
# 3. FEATURE ENGINEERING (Based on Train Split)
# ==========================================
# We calculate "affinity" and "popularity" based strictly on the training portion
# so the test set mimics "unseen" future listening.

# A. User Discovery Profile
user_activity = train_df.groupby(['user_id', 'time_bucket'], observed=True).size().reset_index(name='u_bucket_count')
user_total = train_df.groupby('user_id', observed=True).size().reset_index(name='u_total')
user_profile = user_activity.merge(user_total, on='user_id')
user_profile['user_affinity'] = (user_profile['u_bucket_count'] / user_profile['u_total']).astype('float32')

# B. Artist & Track Discovery Profile
artist_pop = train_df.groupby('artist_id', observed=True).size().reset_index(name='artist_global_plays')
track_pop = train_df.groupby('track_id', observed=True).size().reset_index(name='global_plays')
track_time = train_df.groupby(['track_id', 'time_bucket'], observed=True).size().reset_index(name='t_bucket_count')
track_profile = track_time.merge(track_pop, on='track_id')
track_profile['track_context_weight'] = (track_profile['t_bucket_count'] / track_profile['global_plays']).astype(
    'float32')

# ==========================================
# 4. PREPARE TEST SET (Positives + Negatives)
# ==========================================
log_step("Constructing Test Set with Synthetic Negatives...")

# Positives: The actual 20% held-out data
positives = test_df[['user_id', 'artist_id', 'track_id', 'day_of_week', 'time_bucket']].copy()
positives['label'] = 1

# Negatives: Randomly swapped artists (simulate skips)
all_artists = train_df["artist_id"].unique()
negatives = positives.sample(frac=1.0, random_state=42).copy()
negatives["artist_id"] = np.random.choice(all_artists, size=len(negatives))
negatives["label"] = 0

# Combine
test_data = pd.concat([positives, negatives], ignore_index=True)

# Map Features (Left Join ensures we use the training weights)
test_data = test_data.merge(artist_pop, on='artist_id', how='left')
test_data = test_data.merge(user_profile[['user_id', 'time_bucket', 'user_affinity']], on=['user_id', 'time_bucket'],
                            how='left')
test_data = test_data.merge(track_profile[['track_id', 'time_bucket', 'track_context_weight']],
                            on=['track_id', 'time_bucket'], how='left')

# One-Hot Encoding
for b in ['afternoon', 'evening', 'night']:
    test_data[f"time_bucket_{b}"] = (test_data["time_bucket"] == b).astype("int8")

# Select Features
feature_cols = [
    'day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
    'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night'
]

X_test = test_data[feature_cols].fillna(0).astype("float32")
y_test = test_data["label"].values

# ==========================================
# 5. LOAD FINE-TUNED MODELS & EVALUATE
# ==========================================
log_step("Evaluating Models...")

try:
    # Load the fine-tuned models created in the previous step
    # Note: Use the filenames generated in your training script
    xgb_model = joblib.load("discovery_xgb_finetuned.pkl")
    lgbm_model = joblib.load("discovery_lgbm_finetuned.pkl")

    # Predict
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

    # Ensemble (Equal Weight)
    ensemble_probs = (xgb_probs + lgbm_probs) / 2


    def print_results(name, probs, true_labels):
        preds = (probs > 0.5).astype(int)
        auc = roc_auc_score(true_labels, probs)
        acc = accuracy_score(true_labels, preds)
        prec = precision_score(true_labels, preds)
        rec = recall_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)

        print(f"\n--- {name} ---")
        print(f"AUC Score: {auc:.4f}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")


    # Results
    print_results("XGBoost (Fine-Tuned)", xgb_probs, y_test)
    print_results("LightGBM (Fine-Tuned)", lgbm_probs, y_test)

    print("\n" + "=" * 40)
    print("      FINAL PERSONALIZED ENSEMBLE")
    print("=" * 40)
    print_results("Ensemble Model", ensemble_probs, y_test)

    print("\nConfusion Matrix (Ensemble):")
    cm = confusion_matrix(y_test, (ensemble_probs > 0.5).astype(int))
    print(cm)
    print(f"TP: {cm[1][1]} (Correctly Recommended)")
    print(f"FP: {cm[0][1]} (Bad Recommendation)")

except FileNotFoundError:
    print("\nERROR: Could not find fine-tuned .pkl files.")
    print("Please run the training script to generate 'discovery_xgb_finetuned.pkl' first.")