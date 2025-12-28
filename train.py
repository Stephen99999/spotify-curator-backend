import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def log_step(msg): print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")

# ==========================================
# 1. LOAD & FEATURE ENGINEERING
# ==========================================
log_step("Loading 1K Dataset")
df = pd.read_csv("userid-timestamp-artid-artname-traid-traname.tsv", sep="\t", header=None,
                 names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
                 usecols=["user_id", "timestamp", "artist_id", "track_id"], on_bad_lines="skip")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna().sample(n=3000000, random_state=42)

df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["time_bucket"] = pd.cut(df["hour"], bins=[-1, 4, 11, 16, 21, 24],
                           labels=["night", "morning", "afternoon", "evening", "night"],
                           ordered=False).astype("category")

artist_pop = df.groupby('artist_id', observed=True).size().reset_index(name='artist_global_plays')

user_profile = df.groupby(['user_id', 'time_bucket'], observed=True).size().reset_index(name='u_bucket_count')
user_total = df.groupby('user_id', observed=True).size().reset_index(name='u_total')
user_profile = user_profile.merge(user_total, on='user_id')
user_profile['user_affinity'] = user_profile['u_bucket_count'] / user_profile['u_total']

# ==========================================
# 2. BUILD BALANCED DATASET
# ==========================================
log_step("Building dataset with Artist Features")
positives = df[['user_id', 'artist_id', 'track_id', 'day_of_week', 'time_bucket']].drop_duplicates().copy()
positives['label'] = 1

all_artists = df['artist_id'].unique()
negatives = positives.sample(frac=2.0, replace=True, random_state=42).copy()
negatives['artist_id'] = np.random.choice(all_artists, size=len(negatives))
negatives['label'] = 0

data = pd.concat([positives, negatives])
data = data.merge(artist_pop, on='artist_id', how='left')
data = data.merge(user_profile[['user_id', 'time_bucket', 'user_affinity']], on=['user_id', 'time_bucket'], how='left')

for bucket in ['afternoon', 'evening', 'night']:
    data[f"time_bucket_{bucket}"] = (data["time_bucket"] == bucket).astype("int8")

features = ['day_of_week', 'artist_global_plays', 'user_affinity',
            'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night']

X = data[features].fillna(0).astype('float32')
y = data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
gc.collect()

# ==========================================
# 3. ENSEMBLE TRAINING
# ==========================================



import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def log_step(msg): print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")

log_step("Loading 1K Dataset")
df = pd.read_csv("userid-timestamp-artid-artname-traid-traname.tsv", sep="\t", header=None,
                 names=["user_id", "timestamp", "artist_id", "artist_name", "track_id", "track_name"],
                 usecols=["user_id", "timestamp", "artist_id", "track_id"], on_bad_lines="skip")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna().sample(n=3000000, random_state=42)

df["hour"] = df["timestamp"].dt.hour.astype("int8")
df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int8")
df["time_bucket"] = pd.cut(df["hour"], bins=[-1, 4, 11, 16, 21, 24],
                           labels=["night", "morning", "afternoon", "evening", "night"],
                           ordered=False).astype("category")

# --- Discovery Features ---
# 1. Artist Popularity
artist_pop = df.groupby('artist_id', observed=True).size().reset_index(name='artist_global_plays')

# 2. User Affinity (How much they like this time slot)
user_activity = df.groupby(['user_id', 'time_bucket'], observed=True).size().reset_index(name='u_bucket_count')
user_total = df.groupby('user_id', observed=True).size().reset_index(name='u_total')
user_profile = user_activity.merge(user_total, on='user_id')
user_profile['user_affinity'] = user_profile['u_bucket_count'] / user_profile['u_total']

# 3. Track Context (How much this track belongs in this time slot)
track_pop = df.groupby('track_id', observed=True).size().reset_index(name='global_plays')
track_time = df.groupby(['track_id', 'time_bucket'], observed=True).size().reset_index(name='t_bucket_count')
track_profile = track_time.merge(track_pop, on='track_id')
track_profile['track_context_weight'] = track_profile['t_bucket_count'] / track_profile['global_plays']

# ==========================================
# 2. BUILD BALANCED DATASET
# ==========================================
log_step("Building dataset")
positives = df[['user_id', 'artist_id', 'track_id', 'day_of_week', 'time_bucket']].drop_duplicates().copy()
positives['label'] = 1

all_artists = df['artist_id'].unique()
negatives = positives.sample(frac=1.0, random_state=42).copy() # 1:1 ratio for discovery
negatives['artist_id'] = np.random.choice(all_artists, size=len(negatives))
negatives['label'] = 0

data = pd.concat([positives, negatives])
data = data.merge(artist_pop, on='artist_id', how='left')
data = data.merge(user_profile[['user_id', 'time_bucket', 'user_affinity']], on=['user_id', 'time_bucket'], how='left')
data = data.merge(track_profile[['track_id', 'time_bucket', 'track_context_weight']], on=['track_id', 'time_bucket'], how='left')

for bucket in ['afternoon', 'evening', 'night']:
    data[f"time_bucket_{bucket}"] = (data["time_bucket"] == bucket).astype("int8")

# THE 7 FEATURES
features = ['day_of_week', 'artist_global_plays', 'user_affinity', 'track_context_weight',
            'time_bucket_afternoon', 'time_bucket_evening', 'time_bucket_night']

X = data[features].fillna(0).astype('float32')
y = data['label']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
gc.collect()

# ==========================================
# 3. TRAIN ENSEMBLE
# ==========================================
log_step("Training Ensemble (XGB + LGBM)")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method="hist")
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "discovery_xgb.pkl")

lgbm_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1)
lgbm_model.fit(X_train, y_train)
joblib.dump(lgbm_model, "discovery_lgbm.pkl")

# A. XGBoost
log_step("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=8, tree_method="hist", n_jobs=-1)
xgb_model.fit(X_train, y_train)

# FIXED: Use joblib instead of save_model to avoid the TypeError
joblib.dump(xgb_model, "discovery_xgb.pkl")
print("   > XGBoost saved as .pkl")

# B. LightGBM
log_step("Training LightGBM...")
lgbm_model = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=64, n_jobs=-1)
lgbm_model.fit(X_train, y_train)
joblib.dump(lgbm_model, "discovery_lgbm.pkl")
print("   > LightGBM saved as .pkl")

# ==========================================
# 4. EVALUATE ENSEMBLE
# ==========================================
xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
lgbm_probs = lgbm_model.predict_proba(X_val)[:, 1]

ensemble_probs = (0.6 * lgbm_probs) + (0.4 * xgb_probs)
print(f"\nEnsemble AUC: {roc_auc_score(y_val, ensemble_probs):.4f}")