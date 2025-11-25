"""
predict_popularity.py

Usage:
- Edit ARTIFACT_DIR if needed (where model_artifacts/ lives).
- Call predict_popularity(...) in Python, or run the script and follow the example at the bottom.

Requirements:
pip install sentence-transformers joblib numpy pandas scikit-learn xgboost
"""

import os
import joblib
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
import pandas as pd

# -----------------------
# CONFIG (edit if needed)
# -----------------------
ARTIFACT_DIR = "model_artifacts"
XGB_MODEL_PATH = os.path.join(ARTIFACT_DIR, "xgb_popularity_model.joblib")
SCALER_PATH = os.path.join(ARTIFACT_DIR, "scaler_num.joblib")
OHE_PATH = os.path.join(ARTIFACT_DIR, "emotion_ohe.joblib")
EMBED_INFO_PATH = os.path.join(ARTIFACT_DIR, "embedder_info.txt")

# -----------------------
# LOAD ARTIFACTS
# -----------------------
if not os.path.exists(XGB_MODEL_PATH):
    raise FileNotFoundError(f"XGBoost model not found at {XGB_MODEL_PATH}. Run training or update ARTIFACT_DIR.")

xgb = joblib.load(XGB_MODEL_PATH)
scaler_num = joblib.load(SCALER_PATH)
emotion_ohe = joblib.load(OHE_PATH)

# read embedding model name (saved during training)
if os.path.exists(EMBED_INFO_PATH):
    with open(EMBED_INFO_PATH, "r") as f:
        embed_model_name = f.read().strip().splitlines()[0]
else:
    # fallback default if file missing
    embed_model_name = "all-mpnet-base-v2"

# Load the same sentence-transformer
embedder = SentenceTransformer(embed_model_name)

# -----------------------
# PREDICTION FUNCTION
# -----------------------
def _parse_release_year(release_date_str):
    """
    Try common date formats; return release year as int.
    If parse fails, return current year.
    """
    if pd.isna(release_date_str):
        return datetime.utcnow().year
    # try pandas to_datetime (handles many formats)
    try:
        dt = pd.to_datetime(release_date_str, errors='coerce')
        if pd.isna(dt):
            return datetime.utcnow().year
        return int(dt.year)
    except Exception:
        return datetime.utcnow().year

def predict_popularity(title: str,
                       artist: str,
                       duration_ms: float,
                       album_release_date: str = None,
                       emotion: str = "energetic"):
    """
    Returns a dict:
      - predicted_popularity: float (0-100)
      - predicted_log: float (model's prediction in log1p space)
      - details: dict with intermediate values (embedding dim, scaled numeric)
    """
    # 1) prepare joint text and embedding
    joint_text = f"Title: {title} | Artist: {artist} | Emotion: {emotion}"
    emb = embedder.encode([joint_text], convert_to_numpy=True)[0]  # shape (D,)

    # 2) numeric features -> duration_s and release_age
    try:
        dur_s = float(duration_ms) / 1000.0
    except Exception:
        # fallback if bad input
        dur_s = float(duration_ms or 0) / 1000.0

    release_year = _parse_release_year(album_release_date)
    release_age = datetime.utcnow().year - int(release_year)

    num = np.array([[dur_s, release_age]], dtype=float)
    num_scaled = scaler_num.transform(num)[0]  # shape (2,)

    # 3) emotion encoding
    try:
        emo_enc = emotion_ohe.transform([[emotion]])[0]
    except Exception:
        # if an unknown emotion appears, fallback to zeros of appropriate length
        emo_enc = np.zeros(len(emotion_ohe.categories_[0]), dtype=float)

    # 4) combine into feature vector (must match training order)
    feat = np.hstack([emb, num_scaled, emo_enc]).reshape(1, -1)  # shape (1, D+2+K)

    # 5) predict (log-space) and inverse-transform
    pred_log = xgb.predict(feat)[0]                  # model predicts log1p(popularity)
    pred_raw = np.expm1(pred_log)                    # inverse of np.log1p
    pred_raw = float(np.clip(pred_raw, 0.0, 100.0))  # ensure bounds

    return {
        "predicted_popularity": pred_raw,
        "predicted_log1p": float(pred_log),
        "details": {
            "embedding_dim": emb.shape[0],
            "duration_seconds": dur_s,
            "release_year": int(release_year),
            "release_age": release_age,
            "num_scaled": num_scaled.tolist(),
            "emotion_onehot": emo_enc.tolist()
        }
    }

# -----------------------
# HELPER: pretty CLI example
# -----------------------
if __name__ == "__main__":
    # example interactive usage
    print("Example: predict a song's popularity from title, artist, duration_ms, release_date, emotion")
    # You can change these values to test new songs
    example = {
        "title": "Dancing in the Rain",
        "artist": "Example Artist",
        "duration_ms": 210000,            # 3.5 minutes
        "album_release_date": "2024-07-15",
        "emotion": "energetic"
    }
    print("Inputs:")
    for k,v in example.items():
        print(f"  {k}: {v}")
    res = predict_popularity(**example)
    print("\nPrediction:")
    print(f"  Predicted popularity (0-100): {res['predicted_popularity']:.2f}")
    print(f"  Predicted model output (log1p space): {res['predicted_log1p']:.4f}")
    print("\nDetails:", res['details'])
