# --- Import required libraries ---
import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from math import atanh, tanh

# Attempt to import sklearn; supply a lightweight fallback if unavailable
try:
    from sklearn.feature_extraction.text import CountVectorizer
except ModuleNotFoundError:
    print("⚠️ scikit-learn not installed. Using a simple fallback CountVectorizer. Install scikit-learn for better results: 'pip install scikit-learn'.")
    class CountVectorizer:
        def __init__(self, stop_words="english", max_features=3000):
            self.stop_words = set([
                "the","a","and","or","to","of","in","on","for","with","at","by","an","be","is","it","this","that","from","as"
            ]) if stop_words == "english" else set()
            self.max_features = max_features
            self.vocabulary_ = []
        def fit_transform(self, docs):
            freq = {}
            tokenized_docs = []
            for doc in docs:
                tokens = [t for t in re.findall(r'[a-z0-9]+', str(doc).lower()) if t not in self.stop_words]
                tokenized_docs.append(tokens)
                for t in tokens:
                    freq[t] = freq.get(t,0)+1
            # Select top max_features
            self.vocabulary_ = [w for w,_ in sorted(freq.items(), key=lambda x: -x[1])[:self.max_features]]
            # Build matrix
            matrix = []
            for tokens in tokenized_docs:
                row = [tokens.count(w) for w in self.vocabulary_]
                matrix.append(row)
            import numpy as _np
            return _np.array(matrix)
        def get_feature_names_out(self):
            return self.vocabulary_

try:
    from scipy.stats import spearmanr
except ModuleNotFoundError:
    print("⚠️ scipy not installed. Spearman correlation will be approximated using Pearson.")
    def spearmanr(x, y):
        import numpy as _np
        # Fallback Pearson correlation
        x = _np.array(x); y = _np.array(y)
        if x.size != y.size or x.size == 0:
            return _np.nan, None
        if _np.std(x)==0 or _np.std(y)==0:
            return 0.0, None
        corr = _np.corrcoef(x, y)[0,1]
        return corr, None

# Try to import Kendall's tau; if unavailable, we'll skip it gracefully
try:
    from scipy.stats import kendalltau
except ModuleNotFoundError:
    kendalltau = None

# --- Set your input and output directories ---
# Inputs are read from this folder (existing final CSVs)
INPUT_DIR = r"C:\Users\Winte\OneDrive\Desktop\Final Data UNIC COMP 399"
# Outputs will be written to a folder inside the current workspace to avoid permission issues
OUTPUT_DIR = os.path.join(os.getcwd(), "cleaned_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Define your input files and emotions ---
emotion_files = {
    "happy": "happy_tracks_final.csv",
    "sad": "sad_tracks_final.csv",
    "energetic": "energetic_tracks_final.csv",
    "love": "love_tracks_final.csv"
}

# --- Define cleaning function for titles ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    # keep letters, numbers, and spaces (preserve numbers in titles)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Robust CSV loader with encoding fallbacks ---
def _looks_like_excel(path):
    p = Path(path)
    try:
        with open(path, 'rb') as f:
            sig = f.read(4)
        # xlsx/xlsm are ZIP files starting with PK\x03\x04; old xls starts with D0 CF 11 E0
        return p.suffix.lower() in {'.xlsx', '.xls', '.xlsm', '.xlsb'} or sig.startswith(b'PK\x03\x04') or sig.startswith(b'\xD0\xCF\x11\xE0')
    except Exception:
        return p.suffix.lower() in {'.xlsx', '.xls', '.xlsm', '.xlsb'}

def read_table_safely(path):
    # If file is actually an Excel workbook (common when header shows PK...), try Excel reader
    if _looks_like_excel(path):
        try:
            return pd.read_excel(path)
        except ModuleNotFoundError:
            print("⚠️ Detected Excel file but 'openpyxl' is not installed. Run: pip install openpyxl")
            raise
        except Exception as e:
            print(f"⚠️ Failed to read Excel file '{path}' via pandas.read_excel: {e}")
            # fall through to CSV attempts in case it's a mislabelled text file
            pass

    # CSV attempts with encodings and tolerant parsing
    return read_csv_safely(path)

def read_csv_safely(path):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        # Try fast path
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
        except Exception as e:
            last_err = e
        # Try with python engine and forgiving line handling
        try:
            return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
        except Exception as e:
            last_err = e
        # Try delimiter inference
        try:
            return pd.read_csv(path, encoding=enc, engine="python", sep=None, on_bad_lines="skip")
        except Exception as e:
            last_err = e
        continue
    # Final attempt without specifying encoding may still work
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        raise last_err if last_err else RuntimeError(f"Failed to read CSV: {path}")

# --- Correlation helper supporting multiple methods ---
def compute_correlation(x, y, method: str = 'spearman'):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size != y.size or x.size < 2:
        return np.nan
    if method == 'pearson':
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    elif method == 'spearman':
        corr, _ = spearmanr(x, y)
        try:
            return float(corr)
        except Exception:
            return np.nan
    elif method == 'kendall':
        if kendalltau is None:
            return np.nan
        corr, _ = kendalltau(x, y)
        try:
            return float(corr)
        except Exception:
            return np.nan
    else:
        return np.nan

def pearson_ci(r: float, n: int, alpha: float = 0.05):
    """Approximate two-sided CI for Pearson r via Fisher z-transform."""
    try:
        if n is None or n <= 3 or np.isnan(r):
            return np.nan, np.nan
        # Clamp r to avoid atanh inf
        r = max(min(float(r), 0.999999), -0.999999)
        z = atanh(r)
        se = 1.0 / np.sqrt(n - 3.0)
        # 1.96 ~ standard normal 97.5th percentile
        z_crit = 1.959963984540054
        lo = z - z_crit * se
        hi = z + z_crit * se
        return float(tanh(lo)), float(tanh(hi))
    except Exception:
        return np.nan, np.nan

def bootstrap_corr_ci(x, y, method: str = 'spearman', B: int = 500, alpha: float = 0.05, random_state: int = 42):
    """Bootstrap CI for correlation methods (Spearman/Kendall/Pearson)."""
    x = np.asarray(x); y = np.asarray(y)
    n = x.size
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        xi = x[idx]; yi = y[idx]
        vals.append(compute_correlation(xi, yi, method))
    vals = np.asarray(vals)
    lo = float(np.nanpercentile(vals, 100 * (alpha / 2)))
    hi = float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi

# --- Loop through each emotion dataset ---
summary_rows = []
for emotion, filename in emotion_files.items():
    print(f"\nProcessing {emotion} dataset...")

    # Load CSV
    filepath = os.path.join(INPUT_DIR, filename)
    df = read_table_safely(filepath)

    # Normalize column names aggressively: lowercase, strip, remove BOM, collapse non-alphanumerics to underscore
    def _norm_col(c):
        c = str(c).replace('\ufeff', '').strip().lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)
        return c.strip("_")
    original_cols = list(df.columns)
    df.columns = [_norm_col(c) for c in df.columns]

    # Basic column normalization for malformed / minimal files
    # Attempt to standardize expected columns via common aliases
    rename_map = {}
    if 'track_id' not in df.columns:
        for alt in ['id', 'trackid', 'spotify_track_id', 'track_uri', 'uri']:
            if alt in df.columns:
                rename_map[alt] = 'track_id'
                break
    if 'name' not in df.columns:
        for alt in ['track_name', 'title', 'song', 'track']:
            if alt in df.columns:
                rename_map[alt] = 'name'
                break
    if 'popularity' not in df.columns:
        for alt in ['track_popularity', 'pop', 'pop_score', 'popularity_0_100', 'popularity_0100', 'popularity_100']:
            if alt in df.columns:
                rename_map[alt] = 'popularity'
                break
    # duration column mapping (optional but recommended)
    if 'duration_ms' not in df.columns:
        for alt in ['duration', 'track_duration_ms', 'length_ms', 'durationms', 'duration_ms_x', 'duration_ms_y', 'ms', 'duration_in_ms']:
            if alt in df.columns:
                rename_map[alt] = 'duration_ms'
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    # If required columns are still missing, skip this emotion gracefully
    required_cols = {'track_id', 'name', 'popularity'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"⚠️ Skipping {emotion}: missing columns {missing} in {filename}")
        print(f"   Columns found after normalization: {list(df.columns)}")
        if original_cols:
            print(f"   Original header row: {original_cols}")
        summary_rows.append({
            'emotion': emotion,
            'track_count': 0,
            'mean_popularity': np.nan,
            'std_popularity': np.nan,
            'var_popularity': np.nan
        })
        continue

    # Drop duplicates and missing titles or popularity
    df = df.drop_duplicates(subset=["track_id"])
    df = df.dropna(subset=["name", "popularity"])
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df = df.dropna(subset=["popularity"])
    # Ensure duration numeric if present
    if 'duration_ms' in df.columns:
        df['duration_ms'] = pd.to_numeric(df['duration_ms'], errors='coerce')

    # --- Compute summary stats for this emotion ---
    # Compute duration stats if available
    mean_dur = std_dur = var_dur = np.nan
    dur_pop_spearman = np.nan
    dur_pop_pearson = np.nan
    dur_pop_kendall = np.nan
    # CI placeholders
    dur_pop_spear_lo = np.nan; dur_pop_spear_hi = np.nan
    dur_pop_pear_lo = np.nan; dur_pop_pear_hi = np.nan
    dur_pop_kend_lo = np.nan; dur_pop_kend_hi = np.nan
    if 'duration_ms' in df.columns:
        dur_series = df['duration_ms'].dropna()
        if dur_series.shape[0] > 0:
            mean_dur = float(dur_series.mean())
            std_dur = float(dur_series.std())
            var_dur = float(dur_series.var())
        # duration vs popularity correlation (Spearman or Pearson fallback)
        both = df[['duration_ms','popularity']].dropna()
        if both.shape[0] > 1:
            x_d = both['duration_ms'].to_numpy()
            y_p = both['popularity'].to_numpy()
            n_pairs = x_d.shape[0]
            dur_pop_spearman = compute_correlation(x_d, y_p, 'spearman')
            dur_pop_pearson = compute_correlation(x_d, y_p, 'pearson')
            dur_pop_kendall = compute_correlation(x_d, y_p, 'kendall')
            # CIs: Pearson analytic, others via bootstrap
            dur_pop_pear_lo, dur_pop_pear_hi = pearson_ci(dur_pop_pearson, n_pairs)
            B = 400 if n_pairs > 5000 else (800 if n_pairs > 1000 else 1000)
            dur_pop_spear_lo, dur_pop_spear_hi = bootstrap_corr_ci(x_d, y_p, 'spearman', B=B)
            if not np.isnan(dur_pop_kendall):
                dur_pop_kend_lo, dur_pop_kend_hi = bootstrap_corr_ci(x_d, y_p, 'kendall', B=B)

    summary_rows.append({
        'emotion': emotion,
        'track_count': int(df["popularity"].shape[0]),
        'mean_popularity': float(df["popularity"].mean()),
        'std_popularity': float(df["popularity"].std()),
        'var_popularity': float(df["popularity"].var()),
        'mean_duration_ms': mean_dur,
        'std_duration_ms': std_dur,
        'var_duration_ms': var_dur,
        'spearman_corr_duration_vs_popularity': dur_pop_spearman,
        'spearman_ci_low': dur_pop_spear_lo,
        'spearman_ci_high': dur_pop_spear_hi,
        'pearson_corr_duration_vs_popularity': dur_pop_pearson,
        'pearson_ci_low': dur_pop_pear_lo,
        'pearson_ci_high': dur_pop_pear_hi,
        'kendall_corr_duration_vs_popularity': dur_pop_kendall,
        'kendall_ci_low': dur_pop_kend_lo,
        'kendall_ci_high': dur_pop_kend_hi
    })

    # Clean titles
    df["clean_title"] = df["name"].apply(clean_text)

    # Bag-of-Words representation
    vectorizer = CountVectorizer(stop_words="english", max_features=3000)
    X = vectorizer.fit_transform(df["clean_title"])
    words = vectorizer.get_feature_names_out()
    word_df = pd.DataFrame(X.toarray(), columns=words)

    # Correlation between words and popularity using multiple methods
    methods = ['spearman', 'pearson'] + (['kendall'] if kendalltau is not None else [])
    corr_tables = {}
    for m in methods:
        vals = []
        for word in words:
            vals.append((word, compute_correlation(word_df[word], df["popularity"], m)))
        m_df = pd.DataFrame(vals, columns=["word", f"{m}_corr"]).sort_values(by=f"{m}_corr", ascending=False)
        corr_tables[m] = m_df

    # Use Spearman for printed Top 10 for continuity
    corr_df = corr_tables['spearman'] if 'spearman' in corr_tables else corr_tables[methods[0]]

    # Merge top/bottom correlated words into summary stats
    top_words = corr_df.head(10)
    bottom_words = corr_df.tail(10)

    print(f"Top 10 positively correlated words for {emotion}:")
    print(top_words)
    print(f"Top 10 negatively correlated words for {emotion}:")
    print(bottom_words)
    # Also print duration/popularity correlations
    msg_parts = []
    if not np.isnan(dur_pop_spearman):
        ci_txt = f" [{dur_pop_spear_lo:.4f}, {dur_pop_spear_hi:.4f}]" if not np.isnan(dur_pop_spear_lo) else ""
        msg_parts.append(f"Spearman={dur_pop_spearman:.4f}{ci_txt}")
    if not np.isnan(dur_pop_pearson):
        ci_txt = f" [{dur_pop_pear_lo:.4f}, {dur_pop_pear_hi:.4f}]" if not np.isnan(dur_pop_pear_lo) else ""
        msg_parts.append(f"Pearson={dur_pop_pearson:.4f}{ci_txt}")
    if not np.isnan(dur_pop_kendall):
        ci_txt = f" [{dur_pop_kend_lo:.4f}, {dur_pop_kend_hi:.4f}]" if not np.isnan(dur_pop_kend_lo) else ""
        msg_parts.append(f"Kendall={dur_pop_kendall:.4f}{ci_txt}")
    print("Duration vs Popularity correlation for {}: {}".format(
        emotion, ", ".join(msg_parts) if msg_parts else "N/A"
    ))

    # Save per-method word correlation tables to CSVs
    for m, table in corr_tables.items():
        m_out = os.path.join(OUTPUT_DIR, f"{emotion}_words_{m}_correlations.csv")
        table.to_csv(m_out, index=False)

    # Save cleaned dataset
    output_path = os.path.join(OUTPUT_DIR, f"{emotion}_tracks_final_clean.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned file saved as: {output_path}")

# --- Write summary CSV for all emotions ---
summary_df = pd.DataFrame(summary_rows, columns=[
    'emotion', 'track_count',
    'mean_popularity', 'std_popularity', 'var_popularity',
    'mean_duration_ms', 'std_duration_ms', 'var_duration_ms',
    'spearman_corr_duration_vs_popularity', 'spearman_ci_low', 'spearman_ci_high',
    'pearson_corr_duration_vs_popularity', 'pearson_ci_low', 'pearson_ci_high',
    'kendall_corr_duration_vs_popularity', 'kendall_ci_low', 'kendall_ci_high'
])
summary_output_path = os.path.join(OUTPUT_DIR, "emotion_popularity_summary.csv")
summary_df.to_csv(summary_output_path, index=False)
print(f"\n📊 Summary saved as: {summary_output_path}")

print("\nAll emotion datasets processed successfully.")
