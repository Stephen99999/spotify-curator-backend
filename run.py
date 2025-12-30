# run_once_save_artifacts.py
import pandas as pd
import json
import glob

# 1. Load Data
dfs = []
for file in glob.glob("Streaming_History_Audio_*.json"):
    with open(file, 'r', encoding='utf-8') as f:
        dfs.append(pd.DataFrame(json.load(f)))
df = pd.concat(dfs, ignore_index=True)
df = df[df['ms_played'] > 30000].copy()

# 2. Prepare Lookups
df['time_bucket'] = pd.cut(pd.to_datetime(df['ts']).dt.hour,
                           bins=[-1, 4, 11, 16, 21, 24],
                           labels=["night", "morning", "afternoon", "evening", "night"],
                           ordered=False)
df['artist_id'] = df['master_metadata_album_artist_name'] # Using Name as ID for simplicity in API
df['user_id'] = "current_user"

# A. Artist Plays
artist_counts = df['artist_id'].value_counts().to_dict()

# B. User Affinity (Time of Day Preference)
total_plays = len(df)
bucket_counts = df['time_bucket'].value_counts()
user_affinity = {bucket: count / total_plays for bucket, count in bucket_counts.items()}

# 3. Save
with open("artifacts_artist_pop.json", "w") as f: json.dump(artist_counts, f)
with open("artifacts_user_affinity.json", "w") as f: json.dump(user_affinity, f)
print("Artifacts saved!")