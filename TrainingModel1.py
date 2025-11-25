"""
train_and_save_artifacts.py

Creates model_artifacts containing:
 - xgb_popularity_model.joblib
 - scaler_num.joblib
 - emotion_ohe.joblib
 - embedder_info.txt

Usage:
  python train_and_save_artifacts.py

Requirements:
  pip install pandas numpy scikit-learn sentence-transformers xgboost joblib tqdm nltk

Notes:
 - Update BASE_PATH to your CSV folder if needed.
 - Expected CSV filenames (in BASE_PATH):
     happy_tracks.csv, sad_tracks.csv, energetic_tracks.csv, love_tracks.csv
"""

import os
import random
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import joblib

# NLP / embedding
from sentence_transformers import SentenceTransformer

# Preprocessing & modeling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Local candidate generator (optional)
import nltk
from nltk.corpus import wordnet
import re
from collections import Counter

# Ensure NLTK wordnet exists for local candidate generator (optional use)
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')

# ---------------------------
# CONFIG
# ---------------------------
# Change this to match your folder if different
BASE_PATH = r"C:\Users\Winte\OneDrive\Desktop\Spotify6"  # Updated to new enhanced dataset folder

# Attempt to auto-detect enhanced vs final naming patterns.
_ENHANCED_PATTERN = {
    "happy": "happy_tracks_enhanced.csv",
    "sad": "sad_tracks_enhanced.csv",
    "energetic": "energetic_tracks_enhanced.csv",
    "love": "love_tracks_enhanced.csv"
}
_FINAL_PATTERN = {
    "happy": "happy_tracks_final.csv",
    "sad": "sad_tracks_final.csv",
    "energetic": "energetic_tracks_final.csv",
    "love": "love_tracks_final.csv"
}

# Choose whichever pattern actually exists in BASE_PATH (prefer enhanced)
def _choose_pattern(base):
    enhanced_ok = all(os.path.exists(os.path.join(base, fn)) for fn in _ENHANCED_PATTERN.values())
    if enhanced_ok:
        print("Detected enhanced CSV files (artist + album metadata). Using *_tracks_enhanced.csv files.")
        return _ENHANCED_PATTERN
    final_ok = all(os.path.exists(os.path.join(base, fn)) for fn in _FINAL_PATTERN.values())
    if final_ok:
        print("Enhanced files not found; falling back to *_tracks_final.csv files.")
        return _FINAL_PATTERN
    # Fallback: build pattern dynamically by probing for either name
    resolved = {}
    for emotion, enh_name in _ENHANCED_PATTERN.items():
        enh_path = os.path.join(base, enh_name)
        fin_name = _FINAL_PATTERN[emotion]
        fin_path = os.path.join(base, fin_name)
        if os.path.exists(enh_path):
            resolved[emotion] = enh_name
        elif os.path.exists(fin_path):
            resolved[emotion] = fin_name
        else:
            resolved[emotion] = enh_name  # default (will raise later if missing)
    print("Mixed presence of files; using detected/emergent pattern.")
    return resolved

EMOTION_FILES = _choose_pattern(BASE_PATH)

EMBED_MODEL_NAME = "all-mpnet-base-v2"   # good-quality sentence-transformer
OUT_DIR = "model_artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

XGB_MODEL_PATH = os.path.join(OUT_DIR, "xgb_popularity_model.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "scaler_num.joblib")
OHE_PATH = os.path.join(OUT_DIR, "emotion_ohe.joblib")
EMBED_INFO_PATH = os.path.join(OUT_DIR, "embedder_info.txt")
ALBUM_TYPE_OHE_PATH = os.path.join(OUT_DIR, "album_type_ohe.joblib")

# ---------------------------
# 1) LOAD & CLEAN DATA
# ---------------------------
def read_csv_safely(path):
    """Try multiple encodings to read CSV"""
    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue
    # Last resort: read with errors='ignore'
    return pd.read_csv(path, encoding='utf-8', errors='ignore')

dfs = []
missing_files = []
for emotion_label, fname in EMOTION_FILES.items():
    fpath = os.path.join(BASE_PATH, fname)
    if not os.path.exists(fpath):
        missing_files.append(fpath)
        continue
    print(f"Loading {emotion_label} from {fname} ...")
    df_e = read_csv_safely(fpath)
    df_e['emotion'] = emotion_label
    dfs.append(df_e)

if missing_files:
    print("WARNING: The following expected files were not found:")
    for mf in missing_files:
        print("  -", mf)
    if not dfs:
        raise FileNotFoundError("No emotion CSVs could be loaded. Aborting.")

df = pd.concat(dfs, ignore_index=True)
print(f"Loaded total rows: {len(df)}")

# Ensure required columns exist
required_cols = ['name', 'artists', 'popularity', 'duration_ms', 'album_release_date', 'emotion']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in input CSVs: {missing}")

# Drop rows with nulls in required fields
df = df.dropna(subset=required_cols).reset_index(drop=True)

# Track-level deduplication to avoid leakage (keep first occurrence)
if 'track_id' in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=['track_id']).reset_index(drop=True)
    after = len(df)
    print(f"Track-level dedup: removed {before - after} duplicates by track_id; remaining {after} rows")
print(f"Rows after dropping missing required fields: {len(df)}")

# Clip/clean fields
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).clip(0, 100)
df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce').fillna(0).clip(lower=10000)  # at least 10s

# ---------------------------
# 2) DATE & ENGINEERED FEATURES
# ---------------------------
def add_release_features(df_in, current_year=None):
    df2 = df_in.copy()
    df2['album_release_date'] = pd.to_datetime(df2['album_release_date'], errors='coerce')
    if current_year is None:
        current_year = datetime.utcnow().year
    df2['release_year'] = df2['album_release_date'].dt.year.fillna(current_year).astype(int)
    df2['release_age'] = current_year - df2['release_year']
    df2['release_month'] = df2['album_release_date'].dt.month.fillna(0).astype(int)
    
    # Days since release (more granular than year)
    df2['days_since_release'] = (pd.Timestamp.now() - df2['album_release_date']).dt.days
    df2['days_since_release'] = df2['days_since_release'].fillna(df2['release_age'] * 365).clip(lower=0)
    
    # Release recency categories
    df2['is_very_recent'] = (df2['days_since_release'] < 30).astype(int)  # Last month
    df2['is_recent'] = (df2['days_since_release'] < 90).astype(int)  # Last 3 months
    df2['is_new'] = (df2['days_since_release'] < 365).astype(int)  # Last year
    
    # Decade features
    df2['decade'] = (df2['release_year'] // 10) * 10
    df2['is_2020s'] = (df2['decade'] == 2020).astype(int)
    df2['is_2010s'] = (df2['decade'] == 2010).astype(int)
    df2['is_2000s'] = (df2['decade'] == 2000).astype(int)
    df2['is_classic'] = (df2['release_year'] < 2000).astype(int)
    
    return df2

def add_text_features(df_in):
    """Add features derived from track and artist names"""
    df2 = df_in.copy()
    
    # Title features
    df2['title_length'] = df2['name'].astype(str).str.len()
    df2['title_word_count'] = df2['name'].astype(str).str.split().str.len()
    df2['title_has_numbers'] = df2['name'].astype(str).str.contains(r'\d', regex=True).astype(int)
    df2['title_has_special'] = df2['name'].astype(str).str.contains(r'[!?@#$%^&*()]', regex=True).astype(int)
    df2['title_all_caps_ratio'] = df2['name'].astype(str).apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    
    # Common keywords in titles
    df2['title_has_remix'] = df2['name'].astype(str).str.lower().str.contains(r'remix|mix|edit', regex=True).astype(int)
    df2['title_has_feat'] = df2['name'].astype(str).str.lower().str.contains(r'feat|ft\.|featuring', regex=True).astype(int)
    df2['title_has_live'] = df2['name'].astype(str).str.lower().str.contains(r'live|concert|tour', regex=True).astype(int)
    df2['title_has_version'] = df2['name'].astype(str).str.lower().str.contains(r'version|remaster|deluxe', regex=True).astype(int)
    
    # Artist features
    df2['artist_name_length'] = df2['artists'].astype(str).str.len()
    df2['num_artists'] = df2['artists'].astype(str).str.count(',') + 1  # Count commas + 1
    df2['is_collaboration'] = (df2['num_artists'] > 1).astype(int)
    df2['has_feat_in_artist'] = df2['artists'].astype(str).str.contains(r'feat|ft\.|&', regex=True).astype(int)
    
    return df2

def add_duration_features(df_in):
    """Add duration-based features"""
    df2 = df_in.copy()
    
    # Duration in seconds
    df2['duration_s'] = df2['duration_ms'] / 1000.0
    
    # Duration categories (in seconds)
    df2['is_very_short'] = (df2['duration_s'] < 120).astype(int)  # < 2 min
    df2['is_short'] = ((df2['duration_s'] >= 120) & (df2['duration_s'] < 180)).astype(int)  # 2-3 min
    df2['is_medium'] = ((df2['duration_s'] >= 180) & (df2['duration_s'] < 240)).astype(int)  # 3-4 min
    df2['is_long'] = ((df2['duration_s'] >= 240) & (df2['duration_s'] < 300)).astype(int)  # 4-5 min
    df2['is_very_long'] = (df2['duration_s'] >= 300).astype(int)  # > 5 min
    
    # Typical pop song duration (3-4 minutes)
    df2['is_typical_duration'] = df2['is_medium'].copy()
    
    # Duration z-score within each emotion (normalized by emotion)
    for emotion in df2['emotion'].unique():
        mask = df2['emotion'] == emotion
        emotion_mean = df2.loc[mask, 'duration_s'].mean()
        emotion_std = df2.loc[mask, 'duration_s'].std()
        if emotion_std > 0:
            df2.loc[mask, 'duration_z_emotion'] = (df2.loc[mask, 'duration_s'] - emotion_mean) / emotion_std
        else:
            df2.loc[mask, 'duration_z_emotion'] = 0
    
    return df2

print("Adding engineered features...")
df = add_release_features(df)
df = add_text_features(df)
df = add_duration_features(df)
print(f"Total features after engineering: {len(df.columns)}")

# ---------------------------
# 2b) GENRE MULTI-HOT (from artist_genres)
# ---------------------------
def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:40]

genre_cols = []
TOP_N_GENRES = 40
if 'artist_genres' in df.columns:
    # Parse and count genre frequencies
    genre_lists = df['artist_genres'].fillna("").astype(str).apply(
        lambda x: [g.strip().lower() for g in x.split(',') if g.strip()]
    )
    counts = Counter([g for lst in genre_lists for g in lst])
    top_genres = [g for g, _ in counts.most_common(TOP_N_GENRES)]
    # Create binary columns for top genres
    for g in top_genres:
        col = f"genre_{_slug(g)}"
        df[col] = genre_lists.apply(lambda lst, gg=g: int(gg in lst))
        genre_cols.append(col)
    if genre_cols:
        print(f"Added {len(genre_cols)} genre multi-hot features.")

# ---------------------------
# 3) EMBEDDINGS (Title + Artist + Emotion joint text)
# ---------------------------
print("Loading embedding model:", EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)

joint_texts = (
    "Title: " + df['name'].astype(str)
    + " | Artist: " + df['artists'].astype(str)
    + " | Emotion: " + df['emotion'].astype(str)
    + (" | AlbumType: " + df['album_type'].astype(str) if 'album_type' in df.columns else "")
    + (" | Genres: " + df['artist_genres'].astype(str) if 'artist_genres' in df.columns else "")
)

print("Computing embeddings (this may take a few minutes on first run)...")
embeddings = embedder.encode(joint_texts.tolist(), batch_size=128, show_progress_bar=True, convert_to_numpy=True)
print("Embeddings computed. Shape:", embeddings.shape)

# ---------------------------
# 4) NUMERIC & CATEGORICAL FEATURES
# ---------------------------
# Numeric features (continuous values)
BASE_NUM_COLS = [
    'duration_s', 'release_age', 'days_since_release',
    'title_length', 'title_word_count', 'title_all_caps_ratio',
    'artist_name_length', 'num_artists', 'duration_z_emotion'
]
ADDITIONAL_NUM_CANDIDATES = [
    'artist_popularity', 'artist_followers', 'album_total_tracks',
    'available_markets', 'avg_artist_popularity'
]
num_cols = [c for c in BASE_NUM_COLS if c in df.columns]
for nc in ADDITIONAL_NUM_CANDIDATES:
    if nc in df.columns:
        num_cols.append(nc)

# Binary features (0/1 indicators)
BASE_BINARY_COLS = [
    'is_very_recent', 'is_recent', 'is_new',
    'is_2020s', 'is_2010s', 'is_2000s', 'is_classic',
    'title_has_numbers', 'title_has_special', 'title_has_remix',
    'title_has_feat', 'title_has_live', 'title_has_version',
    'is_collaboration', 'has_feat_in_artist',
    'is_very_short', 'is_short', 'is_medium', 'is_long', 'is_very_long',
    'is_typical_duration'
]
binary_cols = [c for c in BASE_BINARY_COLS if c in df.columns]
if 'explicit' in df.columns:
    binary_cols.append('explicit')
if 'genre_cols' in locals() and genre_cols:
    binary_cols.extend(genre_cols)

# Fill any NaN values
for col in num_cols + binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

X_num = df[num_cols].astype(float).values
X_binary = df[binary_cols].astype(float).values

scaler_num = StandardScaler()
X_num_scaled = scaler_num.fit_transform(X_num)
print(f"Numeric features scaled. Shape: {X_num_scaled.shape}")
print(f"Binary features. Shape: {X_binary.shape}")

# One-hot encode emotion
emotion_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
emotion_feat = emotion_ohe.fit_transform(df[['emotion']].astype(str))
print("Emotion one-hot shape:", emotion_feat.shape)

# Album type one-hot (optional)
album_type_feat = None
album_type_ohe = None
if 'album_type' in df.columns:
    album_type_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    album_type_feat = album_type_ohe.fit_transform(df[['album_type']].astype(str))
    print("Album type one-hot shape:", album_type_feat.shape)

# Optional: add cyclic month features if desired
# df['release_month_sin'] = np.sin(2*np.pi*df['release_month']/12)
# df['release_month_cos'] = np.cos(2*np.pi*df['release_month']/12)
# month_feat = df[['release_month_sin','release_month_cos']].values

# ---------------------------
# 5) FINAL FEATURE MATRIX
# ---------------------------
feature_blocks = [embeddings, X_num_scaled, X_binary, emotion_feat]
if album_type_feat is not None:
    feature_blocks.append(album_type_feat)
X = np.hstack(feature_blocks)
print("Final feature matrix X shape:", X.shape)
print(f"  - Embeddings: {embeddings.shape[1]} features")
print(f"  - Numeric: {X_num_scaled.shape[1]} features")
print(f"  - Binary: {X_binary.shape[1]} features")
print(f"  - Emotion: {emotion_feat.shape[1]} features")
if album_type_feat is not None:
    print(f"  - AlbumType: {album_type_feat.shape[1]} features")

# TARGET: log(1 + popularity)
y_raw = df['popularity'].astype(float).values
y_log = np.log1p(y_raw)

# ---------------------------
# 6) TRAIN/TEST SPLIT (grouped by track_id when available)
# ---------------------------
if 'track_id' in df.columns:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df['track_id'].astype(str).values
    train_idx, test_idx = next(gss.split(X, y_log, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_log, y_test_log = y_log[train_idx], y_log[test_idx]
    df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
else:
    X_train, X_test, y_train_log, y_test_log, df_train, df_test = train_test_split(
        X, y_log, df, test_size=0.2, random_state=42
    )
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------------------------
# 7) HYPERPARAMETER SEARCH (manual randomized) + TRAIN
# ---------------------------
def sample_params(rng):
    return {
        'n_estimators': int(rng.integers(900, 1801)),
        'max_depth': int(rng.integers(4, 11)),
        'learning_rate': float(10 ** rng.uniform(-2.1, -0.9)),  # ~0.008 to 0.125
        'subsample': float(rng.uniform(0.6, 1.0)),
        'colsample_bytree': float(rng.uniform(0.6, 1.0)),
        'min_child_weight': int(rng.integers(1, 9)),
        'reg_alpha': float(rng.uniform(0.0, 0.5)),
        'reg_lambda': float(rng.uniform(0.5, 2.0)),
    }

def eval_params(params, X_tr, y_tr, X_val, y_val):
    model = XGBRegressor(
        tree_method="hist",
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
        **params
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    return rmse, model

print("Tuning XGBoost hyperparameters (random search)...")
rng = np.random.default_rng(42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train_log, test_size=0.2, random_state=42)
best_rmse = float('inf')
best_params = None
N_TRIALS = 20
for i in range(1, N_TRIALS + 1):
    p = sample_params(rng)
    rmse, _ = eval_params(p, X_tr, y_tr, X_val, y_val)
    if rmse < best_rmse:
        best_rmse = rmse
        best_params = p
    if i % 5 == 0:
        print(f"  trial {i}/{N_TRIALS}: best_val_rmse={best_rmse:.4f}")

if best_params is None:
    best_params = sample_params(rng)
print("Best params from search:", best_params)

xgb = XGBRegressor(
    tree_method="hist",
    random_state=42,
    verbosity=1,
    early_stopping_rounds=50,
    **best_params
)

print("Training final XGBoost with best params...")
xgb.fit(
    X_train, y_train_log,
    eval_set=[(X_test, y_test_log)],
    verbose=50
)
print("Training complete. Best iteration:", getattr(xgb, 'best_iteration', None))

# ---------------------------
# 8) EVALUATION
# ---------------------------
y_pred_log = xgb.predict(X_test)
y_pred_raw = np.clip(np.expm1(y_pred_log), 0, 100)  # inverse transform and clip

y_true_raw = df_test['popularity'].astype(float).values

r2_raw = r2_score(y_true_raw, y_pred_raw)
mae_raw = mean_absolute_error(y_true_raw, y_pred_raw)
rmse_raw = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))

r2_log = r2_score(y_test_log, y_pred_log)
mae_log = mean_absolute_error(y_test_log, y_pred_log)

print("\n=== Evaluation Results ===")
print(f"Raw scale:   R² = {r2_raw:.4f} | MAE = {mae_raw:.4f} | RMSE = {rmse_raw:.4f}")
print(f"Log scale:   R² = {r2_log:.4f} | MAE = {mae_log:.4f}")

# Show sample predictions
print("\nSample predictions (true_popularity -> predicted_popularity):")
for i in range(min(10, len(y_true_raw))):
    print(f"  {y_true_raw[i]:.1f} -> {y_pred_raw[i]:.2f}")

# ---------------------------
# 9) SAVE ARTIFACTS
# ---------------------------
joblib.dump(xgb, XGB_MODEL_PATH)
joblib.dump(scaler_num, SCALER_PATH)
joblib.dump(emotion_ohe, OHE_PATH)
if album_type_ohe is not None:
    joblib.dump(album_type_ohe, ALBUM_TYPE_OHE_PATH)
with open(EMBED_INFO_PATH, "w") as f:
    f.write(EMBED_MODEL_NAME + "\n")

print(f"\nSaved artifacts to '{OUT_DIR}':")
print(" -", os.path.basename(XGB_MODEL_PATH))
print(" -", os.path.basename(SCALER_PATH))
print(" -", os.path.basename(OHE_PATH))
print(" -", os.path.basename(EMBED_INFO_PATH))

# ---------------------------
# 10) OPTIONAL: local title suggestion example
# ---------------------------
def generate_candidates_local(title, n=8, seed=42):
    random.seed(seed)
    tokens = title.split()
    candidates = set()
    attempts = 0
    max_attempts = n * 8
    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        new_tokens = tokens.copy()
        num_replace = random.choice([1,1,2])
        idxs = random.sample(range(len(tokens)), min(len(tokens), num_replace))
        replaced = False
        for idx in idxs:
            word = tokens[idx].strip(".,!?;:\"'()[]").lower()
            syns = wordnet.synsets(word)
            lemmas = []
            for s in syns:
                for l in s.lemmas():
                    lem = l.name().replace('_', ' ')
                    if lem.lower() != word and lem.isalpha():
                        lemmas.append(lem)
            if lemmas:
                new_tokens[idx] = random.choice(lemmas)
                replaced = True
        cand = " ".join(new_tokens)
        if replaced and cand.lower() != title.lower():
            candidates.add(cand)
    if not candidates:
        candidates.add(title)
    return list(candidates)

def score_titles_with_model(titles, artist, duration_ms, release_year, emotion_label, top_k=5):
    feats = []
    for t in titles:
        joint = f"Title: {t} | Artist: {artist} | Emotion: {emotion_label}"
        emb = embedder.encode([joint], convert_to_numpy=True)[0]
        # Build numeric vector matching training numeric columns (fill unknowns with 0)
        dur_s = duration_ms / 1000.0
        age = datetime.utcnow().year - int(release_year)
        numeric_values = []
        for col in num_cols:
            if col == 'duration_s':
                numeric_values.append(dur_s)
            elif col == 'release_age':
                numeric_values.append(age)
            elif col == 'days_since_release':
                # Approximate days since release using age*365 (no exact date)
                numeric_values.append(age * 365)
            else:
                # Unknown for candidate context; use 0
                numeric_values.append(0.0)
        num = np.array([numeric_values], dtype=float)
        try:
            num_scaled = scaler_num.transform(num)[0]
        except Exception:
            # Fallback: if scaler shape mismatch, pad/truncate
            target_len = getattr(scaler_num, 'n_features_in_', len(numeric_values))
            vec = numeric_values[:target_len] + [0.0] * max(0, target_len - len(numeric_values))
            num_scaled = np.array(vec, dtype=float)
        try:
            emo_enc = emotion_ohe.transform([[emotion_label]])[0]
        except Exception:
            emo_enc = np.zeros(len(emotion_ohe.categories_[0]))
        # Binary placeholders to match training shape
        binary_placeholder = np.zeros(len(binary_cols), dtype=float)
        block_list = [emb, num_scaled, binary_placeholder, emo_enc]
        if album_type_ohe is not None:
            # We don't know album type for generated titles; use a zero vector of correct length
            block_list.append(np.zeros(album_type_feat.shape[1], dtype=float))
        feat = np.hstack(block_list)
        feats.append(feat)
    feats = np.vstack(feats)
    y_pred_log = xgb.predict(feats)
    y_pred_raw = np.clip(np.expm1(y_pred_log), 0, 100)
    ranked = sorted(zip(titles, y_pred_raw.tolist()), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

def score_titles_batch(titles, artist, duration_ms, release_year, emotion_label):
    # Build joint texts
    joints = [f"Title: {t} | Artist: {artist} | Emotion: {emotion_label}" for t in titles]
    embs = embedder.encode(joints, convert_to_numpy=True, batch_size=128, show_progress_bar=False)

    # Numeric placeholders aligned to num_cols
    dur_s = duration_ms / 1000.0
    age = datetime.utcnow().year - int(release_year)
    numeric_values = []
    for col in num_cols:
        if col == 'duration_s':
            numeric_values.append(dur_s)
        elif col == 'release_age':
            numeric_values.append(age)
        elif col == 'days_since_release':
            numeric_values.append(age * 365)
        else:
            numeric_values.append(0.0)
    num_mat = np.tile(np.array(numeric_values, dtype=float), (len(titles), 1))
    try:
        num_scaled = scaler_num.transform(num_mat)
    except Exception:
        target_len = getattr(scaler_num, 'n_features_in_', num_mat.shape[1])
        pad = target_len - num_mat.shape[1]
        if pad > 0:
            num_mat = np.hstack([num_mat, np.zeros((len(titles), pad), dtype=float)])
        num_scaled = scaler_num.transform(num_mat[:, :target_len])

    # Emotion one-hot
    try:
        emo_enc = emotion_ohe.transform([[emotion_label]])[0]
    except Exception:
        emo_enc = np.zeros(len(emotion_ohe.categories_[0]))
    emo_mat = np.tile(emo_enc, (len(titles), 1))

    # Binary placeholders and optional album type
    bin_mat = np.zeros((len(titles), len(binary_cols)), dtype=float)
    blocks = [embs, num_scaled, bin_mat, emo_mat]
    if album_type_ohe is not None and album_type_feat is not None:
        blocks.append(np.zeros((len(titles), album_type_feat.shape[1]), dtype=float))
    feats = np.hstack(blocks)
    y_pred_log = xgb.predict(feats)
    y_pred_raw = np.clip(np.expm1(y_pred_log), 0, 100)
    return y_pred_raw

# Example suggestion run (prints to console)
if len(df_test) > 0:
    sample = df_test.iloc[0]
    title0 = sample['name']
    artist0 = sample['artists']
    dur0 = sample['duration_ms']
    ry0 = sample['release_year']
    emo0 = sample['emotion']
    print("\nExample title-suggestion (local fallback):")
    cands = generate_candidates_local(title0, n=8)
    ranked = score_titles_with_model(cands + [title0], artist0, dur0, ry0, emo0, top_k=5)
    for t, p in ranked:
        print(f"  {t}  -> predicted_popularity: {p:.2f}")

print("\nTraining + artifact creation complete.")

# ---------------------------
# 11) TOP TITLE WORDS BY EMOTION (model-scored)
# ---------------------------
def extract_top_words(df_in, emotion, max_vocab=500, min_freq=10):
    sub = df_in[df_in['emotion'] == emotion]
    tokens = (
        sub['name'].astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.split()
    )
    freq = {}
    for lst in tokens:
        for w in lst:
            if len(w) >= 3 and (w.isalpha() or w.isalnum()):
                freq[w] = freq.get(w, 0) + 1
    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)
    vocab = [w for w, _ in items[:max_vocab]]
    return vocab

try:
    emotions_list = sorted(df['emotion'].astype(str).unique().tolist())
    top_words_output = []
    for emo in emotions_list:
        vocab = extract_top_words(df, emo, max_vocab=500, min_freq=15)
        if not vocab:
            continue
        # Score words as standalone titles with typical defaults
        preds = score_titles_batch(vocab, artist="Sample Artist", duration_ms=210_000, release_year=datetime.utcnow().year, emotion_label=emo)
        top_pairs = sorted(zip(vocab, preds.tolist()), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 title words for emotion '{emo}' (model-scored):")
        for w, p in top_pairs:
            print(f"  {w:20s} -> predicted_popularity: {p:.2f}")
        for w, p in top_pairs:
            top_words_output.append({"emotion": emo, "word": w, "predicted_popularity": float(p)})
    if top_words_output:
        out_csv = os.path.join(OUT_DIR, "top_title_words_by_emotion.csv")
        pd.DataFrame(top_words_output).to_csv(out_csv, index=False)
        print(f"\nSaved top title words per emotion -> {out_csv}")
except Exception as e:
    print(f"\n(Non-fatal) Could not compute top title words: {e}")
