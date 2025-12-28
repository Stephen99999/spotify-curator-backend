import pandas as pd
import numpy as np
import joblib
import gc
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def log_step(msg):
    print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 1. LOAD DATA & RE-GENERATE FEATURES
# ==========================================
FILENAME = "userid-timestamp-artid-artname-traid-traname.tsv"
log_step("Loading 1K Dataset for Ensemble Evaluation")

# Load a slice for evaluation
df = pd.read_csv(
    FILENAME, sep="\t", header=None, nrows=2000000,
    names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
    usecols=["user_id", "timestamp", "artist_id", "track_id"],
    on_bad_lines="skip"
)

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna()
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["time_bucket"] = pd.cut(
    df["hour"], bins=[-1, 4, 11, 16, 21, 24],
    labels=["night", "morning", "afternoon", "evening", "night"],
    ordered=False
).astype("category")

# --- A. User Discovery Profile ---
user_activity = df.groupby(['user_id', 'time_bucket'], observed=True).size().reset_index(name='u_bucket_count')
user_total = df.groupby('user_id', observed=True).size().reset_index(name='u_total')
user_profile = user_activity.merge(user_total, on='user_id')
user_profile['user_affinity'] = (user_profile['u_bucket_count'] / user_profile['u_total']).astype('float32')

# --- B. Artist & Track Discovery Profile ---
artist_pop = df.groupby('artist_id', observed=True).size().reset_index(name='artist_global_plays')
track_pop = df.groupby('track_id', observed=True).size().reset_index(name='global_plays')
track_time = df.groupby(['track_id', 'time_bucket'], observed=True).size().reset_index(name='t_bucket_count')
track_profile = track_time.merge(track_pop, on='track_id')
track_profile['track_context_weight'] = (track_profile['t_bucket_count'] / track_profile['global_plays']).astype('float32')

# ==========================================
# 2. CREATE TEST SAMPLES
# ==========================================
log_step("Creating 50/50 Positive/Negative Split")

positives = df[['user_id', 'artist_id', 'track_id', 'day_of_week', 'time_bucket']].drop_duplicates().copy()
positives['label'] = 1

all_artists = df["artist_id"].unique()
negatives = positives.sample(frac=1.0, random_state=42).copy()
negatives["artist_id"] = np.random.choice(all_artists, size=len(negatives))
negatives["label"] = 0

# Merge Features
def merge_discovery_cols(target_df):
    target_df = target_df.merge(user_profile[['user_id', 'time_bucket', 'user_affinity']], on=['user_id', 'time_bucket'], how='left')
    target_df = target_df.merge(track_profile[['track_id', 'time_bucket', 'track_context_weight']], on=['track_id', 'time_bucket'], how='left')
    target_df = target_df.merge(artist_pop, on='artist_id', how='left')
    return target_df

test_data = pd.concat([merge_discovery_cols(positives), merge_discovery_cols(negatives)], ignore_index=True)

# Final Encoding
for b in ['afternoon', 'evening', 'night']:
    test_data[f"time_bucket_{b}"] = (test_data["time_bucket"] == b).astype("int8")

# THE EXACT 7 FEATURES MATCHING TRAIN.PY
feature_cols = [
    'day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
    'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night'
]

X_test = test_data[feature_cols].fillna(0).astype("float32")
y_test = test_data["label"].values

del df, positives, negatives, test_data
gc.collect()

# ==========================================
# 3. INDIVIDUAL & ENSEMBLE PREDICTION
# ==========================================
log_step("Loading Models and Running Evaluation...")

try:
    xgb_model = joblib.load("discovery_xgb.pkl")
    lgbm_model = joblib.load("discovery_lgbm.pkl")

    # Get individual probabilities
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

    # Calculate Ensemble average
    ensemble_probs = (xgb_probs + lgbm_probs) / 2


    def print_results(name, probs, true_labels):
        preds = (probs > 0.5).astype(int)
        print(f"\n--- {name} Results ---")
        print(f"Accuracy:  {accuracy_score(true_labels, preds):.4f}")
        print(f"Precision: {precision_score(true_labels, preds):.4f}")
        print(f"Recall:    {recall_score(true_labels, preds):.4f}")
        print(f"F1-Score:  {f1_score(true_labels, preds):.4f}")


    # 1. Evaluate XGBoost
    print_results("XGBoost Only", xgb_probs, y_test)

    # 2. Evaluate LightGBM
    print_results("LightGBM Only", lgbm_probs, y_test)

    # 3. Evaluate Ensemble
    print("\n" + "=" * 40)
    print("      FINAL ENSEMBLE PERFORMANCE")
    print("=" * 40)
    print_results("Ensemble (XGB + LGBM)", ensemble_probs, y_test)

    print("\nConfusion Matrix (Ensemble):")
    print(confusion_matrix(y_test, (ensemble_probs > 0.5).astype(int)))
    print("=" * 40)

except FileNotFoundError:
    print("Error: Model files (.pkl) not found. Please run the training script first.")

# ==========================================
# 4. OPTIONAL: THRESHOLD ANALYSIS
# ==========================================
# Sometimes 0.5 is too strict for discovery.
# Let's see what happens at 0.4 (more adventurous recommendations)
y_pred_adventurous = (ensemble_probs > 0.4).astype(int)
print(f"\nAdventurous Recall (Threshold 0.4): {recall_score(y_test, y_pred_adventurous):.4f}")