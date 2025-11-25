"""
Flask Web Application for Spotify Popularity Predictor
Beautiful, modern interface for track popularity predictions
"""

from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from collections import Counter
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# Paths - use relative paths for deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_artifacts")
SPOTIFY6_PATH = os.path.join(BASE_DIR, "Spotify6")

# Load model and artifacts
print("Loading trained model and artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, "xgb_popularity_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_num.joblib"))
emotion_ohe = joblib.load(os.path.join(MODEL_DIR, "emotion_ohe.joblib"))
album_type_ohe = joblib.load(os.path.join(MODEL_DIR, "album_type_ohe.joblib"))
sentence_model = SentenceTransformer("all-mpnet-base-v2")

# Load top genres from training data (optional - use defaults if files not found)
print("Loading genre database...")
all_genres = []
if os.path.exists(SPOTIFY6_PATH):
    for emotion_file in ["energetic_tracks_enhanced.csv", "happy_tracks_enhanced.csv", 
                         "love_tracks_enhanced.csv", "sad_tracks_enhanced.csv"]:
        path = os.path.join(SPOTIFY6_PATH, emotion_file)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
                if 'artist_genres' in df.columns:
                    for genres_str in df['artist_genres'].dropna():
                        try:
                            genres = eval(genres_str) if isinstance(genres_str, str) else genres_str
                            if isinstance(genres, list):
                                all_genres.extend(genres)
                        except:
                            pass
            except Exception as e:
                print(f"Warning: Could not load {emotion_file}: {e}")

def _slug(text):
    """Convert genre to slug format."""
    return text.lower().strip().replace(' ', '_').replace('-', '_')

# Use loaded genres or fallback to common genres
if all_genres:
    genre_counts = Counter(_slug(g) for g in all_genres)
    top_genres = [g for g, _ in genre_counts.most_common(40)]
else:
    print("Using default genre list...")
    top_genres = ['pop', 'hip_hop', 'rock', 'dance_pop', 'rap', 'r_b', 'edm', 'trap', 
                  'indie', 'alternative', 'country', 'latin', 'electronic', 'house', 
                  'soul', 'funk', 'reggaeton', 'jazz', 'metal', 'folk', 'blues', 'punk',
                  'disco', 'techno', 'dubstep', 'ambient', 'classical', 'reggae', 'indie_pop',
                  'synth_pop', 'electro', 'garage', 'drum_and_bass', 'trance', 'hardcore',
                  'lo_fi', 'k_pop', 'afrobeat', 'grime', 'progressive']

# Load average statistics
print("Computing baseline statistics...")
all_data = []
for emotion_file in ["energetic_tracks_enhanced.csv", "happy_tracks_enhanced.csv", 
                     "love_tracks_enhanced.csv", "sad_tracks_enhanced.csv"]:
    path = os.path.join(SPOTIFY6_PATH, emotion_file)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    avg_stats = {
        'artist_popularity': combined_df['artist_popularity'].mean() if 'artist_popularity' in combined_df.columns else 50,
        'artist_followers': combined_df['artist_followers'].mean() if 'artist_followers' in combined_df.columns else 1000000,
        'album_total_tracks': combined_df['album_total_tracks'].median() if 'album_total_tracks' in combined_df.columns else 10,
        'available_markets': combined_df['available_markets'].mean() if 'available_markets' in combined_df.columns else 150,
        'duration_ms': combined_df['duration_ms'].mean() if 'duration_ms' in combined_df.columns else 200000,
        'explicit': combined_df['explicit'].mean() if 'explicit' in combined_df.columns else 0.14,
        'num_artists': combined_df['num_artists'].median() if 'num_artists' in combined_df.columns else 1,
        'is_collaboration': combined_df['is_collaboration'].mean() if 'is_collaboration' in combined_df.columns else 0.33,
        'avg_artist_popularity': combined_df['avg_artist_popularity'].mean() if 'avg_artist_popularity' in combined_df.columns else 50,
    }
    top_stats = {
        'artist_popularity': combined_df['artist_popularity'].quantile(0.75) if 'artist_popularity' in combined_df.columns else 70,
        'artist_followers': combined_df['artist_followers'].quantile(0.75) if 'artist_followers' in combined_df.columns else 5000000,
        'available_markets': combined_df['available_markets'].quantile(0.75) if 'available_markets' in combined_df.columns else 180,
    }
else:
    avg_stats = {
        'artist_popularity': 50, 'artist_followers': 1000000, 'album_total_tracks': 10,
        'available_markets': 150, 'duration_ms': 200000, 'explicit': 0.14,
        'num_artists': 1, 'is_collaboration': 0.33, 'avg_artist_popularity': 50
    }
    top_stats = {
        'artist_popularity': 70, 'artist_followers': 5000000, 'available_markets': 180
    }

# Spotify API client
spotify_client = None

def init_spotify(client_id, client_secret):
    """Initialize Spotify API client."""
    global spotify_client
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        spotify_client.search(q="test", limit=1)
        return True
    except Exception as e:
        print(f"Spotify API error: {e}")
        return False

def fetch_spotify_data(title, artist):
    """Fetch track data from Spotify."""
    if spotify_client is None:
        return None
    
    try:
        query = f"track:{title} artist:{artist}"
        results = spotify_client.search(q=query, type='track', limit=5)
        
        if not results['tracks']['items']:
            return None
        
        track = results['tracks']['items'][0]
        artist_ids = [artist['id'] for artist in track['artists']]
        artists_data = spotify_client.artists(artist_ids)['artists']
        
        actual_data = {
            'popularity': track['popularity'],
            'name': track['name'],
            'artists': ', '.join([a['name'] for a in track['artists']]),
            'artist_popularity': artists_data[0]['popularity'] if artists_data else 0,
            'artist_followers': artists_data[0]['followers']['total'] if artists_data else 0,
            'artist_genres': artists_data[0]['genres'] if artists_data else [],
            'album_type': track['album']['album_type'],
            'album_total_tracks': track['album']['total_tracks'],
            'duration_ms': track['duration_ms'],
            'explicit': 1 if track['explicit'] else 0,
            'available_markets': len(track.get('available_markets', [])),
            'num_artists': len(track['artists']),
            'is_collaboration': 1 if len(track['artists']) > 1 else 0,
            'spotify_link': track['external_urls']['spotify'],
            'track_id': track['id'],
            'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None
        }
        
        if len(artists_data) > 0:
            avg_artist_pop = sum(a['popularity'] for a in artists_data) / len(artists_data)
            actual_data['avg_artist_popularity'] = avg_artist_pop
        else:
            actual_data['avg_artist_popularity'] = 0
        
        return actual_data
    except Exception as e:
        print(f"Error fetching Spotify data: {e}")
        return None

# Import feature building functions from predict_popularity.py
def extract_title_features(title):
    """Extract features from title."""
    title_str = str(title)
    word_count = len(title_str.split())
    char_count = len(title_str)
    has_feat = 1 if any(x in title_str.lower() for x in ['feat', 'ft.', 'featuring']) else 0
    has_parentheses = 1 if '(' in title_str or '[' in title_str else 0
    has_numbers = 1 if any(c.isdigit() for c in title_str) else 0
    has_special = 1 if any(c in title_str for c in ['!', '?', '#', '$', '%', '&', '*']) else 0
    has_remix = 1 if any(x in title_str.lower() for x in ['remix', 'mix', 'edit']) else 0
    has_live = 1 if 'live' in title_str.lower() else 0
    has_version = 1 if any(x in title_str.lower() for x in ['version', 'ver.', 'remaster']) else 0
    all_caps_ratio = sum(1 for c in title_str if c.isupper()) / max(len(title_str), 1)
    
    return {
        'word_count': word_count, 'char_count': char_count, 'has_feat': has_feat,
        'has_parentheses': has_parentheses, 'has_numbers': has_numbers,
        'has_special': has_special, 'has_remix': has_remix, 'has_live': has_live,
        'has_version': has_version, 'all_caps_ratio': all_caps_ratio
    }

def compute_duration_features(duration_ms, emotion='happy'):
    """Compute duration features."""
    duration_s = duration_ms / 1000.0
    is_very_short = 1 if duration_s < 120 else 0
    is_short = 1 if 120 <= duration_s < 180 else 0
    is_medium = 1 if 180 <= duration_s < 240 else 0
    is_long = 1 if 240 <= duration_s < 300 else 0
    is_very_long = 1 if duration_s >= 300 else 0
    is_typical_duration = 1 if 180 <= duration_s <= 240 else 0
    
    emotion_duration_stats = {
        'happy': (3.33 * 60, 1.19 * 60), 'sad': (3.00 * 60, 1.19 * 60),
        'energetic': (3.14 * 60, 0.98 * 60), 'love': (3.49 * 60, 1.13 * 60)
    }
    mean_dur, std_dur = emotion_duration_stats.get(emotion, (200, 60))
    duration_z_emotion = (duration_s - mean_dur) / max(std_dur, 1)
    
    return {
        'duration_s': duration_s, 'is_very_short': is_very_short, 'is_short': is_short,
        'is_medium': is_medium, 'is_long': is_long, 'is_very_long': is_very_long,
        'is_typical_duration': is_typical_duration, 'duration_z_emotion': duration_z_emotion
    }

def compute_release_features():
    """Compute release date features."""
    from datetime import datetime
    current_date = datetime.now()
    release_age = 0.5
    days_since_release = 180
    is_very_recent = 1 if days_since_release < 30 else 0
    is_recent = 1 if 30 <= days_since_release < 180 else 0
    is_new = 1 if days_since_release < 365 else 0
    current_year = current_date.year
    release_year = current_year
    is_2020s = 1 if release_year >= 2020 else 0
    is_2010s = 1 if 2010 <= release_year < 2020 else 0
    is_2000s = 1 if 2000 <= release_year < 2010 else 0
    is_classic = 1 if release_year < 2000 else 0
    
    return {
        'release_age': release_age, 'days_since_release': days_since_release,
        'is_very_recent': is_very_recent, 'is_recent': is_recent, 'is_new': is_new,
        'is_2020s': is_2020s, 'is_2010s': is_2010s, 'is_2000s': is_2000s, 'is_classic': is_classic
    }

def build_feature_vector(title, artist, emotion='happy', artist_popularity=None, 
                         artist_followers=None, genres=None, album_type='single',
                         album_total_tracks=None, explicit=0, available_markets=None,
                         duration_ms=None, num_artists=None, is_collaboration=None,
                         avg_artist_popularity=None):
    """Build feature vector for prediction."""
    # Use defaults if not provided
    if artist_popularity is None:
        artist_popularity = avg_stats['artist_popularity']
    if artist_followers is None:
        artist_followers = avg_stats['artist_followers']
    if album_total_tracks is None:
        album_total_tracks = avg_stats['album_total_tracks']
    if available_markets is None:
        available_markets = avg_stats['available_markets']
    if duration_ms is None:
        duration_ms = avg_stats['duration_ms']
    if num_artists is None:
        num_artists = avg_stats['num_artists']
    if is_collaboration is None:
        is_collaboration = avg_stats['is_collaboration']
    if avg_artist_popularity is None:
        avg_artist_popularity = avg_stats['avg_artist_popularity']
    
    title_feats = extract_title_features(title)
    duration_feats = compute_duration_features(duration_ms, emotion)
    release_feats = compute_release_features()
    
    genres_str = ', '.join(genres) if genres else ''
    text = f"Title: {title} | Artist: {artist} | Emotion: {emotion}"
    if album_type:
        text += f" | AlbumType: {album_type}"
    if genres_str:
        text += f" | Genres: {genres_str}"
    
    embedding = sentence_model.encode([text])[0]
    
    numeric_features = np.array([
        duration_feats['duration_s'], release_feats['release_age'], release_feats['days_since_release'],
        title_feats['char_count'], title_feats['word_count'], title_feats['all_caps_ratio'],
        len(artist), num_artists, duration_feats['duration_z_emotion'],
        artist_popularity, artist_followers, album_total_tracks, available_markets, avg_artist_popularity
    ])
    
    has_feat_in_artist = 1 if any(x in artist.lower() for x in ['feat', 'ft.', '&']) else 0
    
    binary_features = [
        release_feats['is_very_recent'], release_feats['is_recent'], release_feats['is_new'],
        release_feats['is_2020s'], release_feats['is_2010s'], release_feats['is_2000s'], release_feats['is_classic'],
        title_feats['has_numbers'], title_feats['has_special'], title_feats['has_remix'],
        title_feats['has_feat'], title_feats['has_live'], title_feats['has_version'],
        is_collaboration, has_feat_in_artist,
        duration_feats['is_very_short'], duration_feats['is_short'], duration_feats['is_medium'],
        duration_feats['is_long'], duration_feats['is_very_long'], duration_feats['is_typical_duration'],
        explicit
    ]
    
    genre_binary = np.zeros(40)
    if genres:
        genre_slugs = [_slug(g) for g in genres]
        for i, top_genre in enumerate(top_genres[:40]):
            if top_genre in genre_slugs:
                genre_binary[i] = 1
    
    binary_features.extend(genre_binary)
    binary_features = np.array(binary_features)
    
    emotion_encoded = emotion_ohe.transform([[emotion]])
    if hasattr(emotion_encoded, 'toarray'):
        emotion_encoded = emotion_encoded.toarray()[0]
    else:
        emotion_encoded = emotion_encoded[0] if len(emotion_encoded.shape) > 1 else emotion_encoded
    
    album_type_encoded = album_type_ohe.transform([[album_type]])
    if hasattr(album_type_encoded, 'toarray'):
        album_type_encoded = album_type_encoded.toarray()[0]
    else:
        album_type_encoded = album_type_encoded[0] if len(album_type_encoded.shape) > 1 else album_type_encoded
    
    numeric_features_scaled = scaler.transform([numeric_features])[0]
    
    feature_vector = np.concatenate([
        embedding, numeric_features_scaled, binary_features, emotion_encoded, album_type_encoded
    ])
    
    return feature_vector.reshape(1, -1)

def predict_popularity(title, artist, **kwargs):
    """Predict popularity score."""
    feature_vec = build_feature_vector(title, artist, **kwargs)
    log_pred = model.predict(feature_vec)[0]
    popularity = np.expm1(log_pred)
    return max(0, min(100, popularity))

# Flask routes
@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    data = request.json
    
    title = data.get('title', '')
    artist = data.get('artist', '')
    emotion = data.get('emotion', 'happy')
    use_spotify = data.get('use_spotify', False)
    spotify_client_id = data.get('spotify_client_id', '')
    spotify_client_secret = data.get('spotify_client_secret', '')
    
    if not title or not artist:
        return jsonify({'error': 'Title and artist are required'}), 400
    
    result = {
        'title': title,
        'artist': artist,
        'emotion': emotion
    }
    
    # Try to fetch Spotify data if requested
    actual_data = None
    if use_spotify and spotify_client_id and spotify_client_secret:
        if init_spotify(spotify_client_id, spotify_client_secret):
            actual_data = fetch_spotify_data(title, artist)
    
    # Make predictions
    if actual_data:
        # Prediction with real data
        kwargs_real = {
            'emotion': emotion,
            'artist_popularity': actual_data['artist_popularity'],
            'artist_followers': actual_data['artist_followers'],
            'genres': actual_data['artist_genres'],
            'album_type': actual_data['album_type'],
            'album_total_tracks': actual_data['album_total_tracks'],
            'explicit': actual_data['explicit'],
            'available_markets': actual_data['available_markets'],
            'duration_ms': actual_data['duration_ms'],
            'num_artists': actual_data['num_artists'],
            'is_collaboration': actual_data['is_collaboration'],
            'avg_artist_popularity': actual_data['avg_artist_popularity']
        }
        predicted_real = predict_popularity(title, artist, **kwargs_real)
        
        # Prediction with defaults
        kwargs_baseline = {'emotion': emotion, 'album_type': 'single'}
        predicted_baseline = predict_popularity(title, artist, **kwargs_baseline)
        
        result['actual_popularity'] = int(actual_data['popularity'])
        result['predicted_with_data'] = float(round(predicted_real, 1))
        result['predicted_baseline'] = float(round(predicted_baseline, 1))
        result['data_impact'] = float(round(predicted_real - predicted_baseline, 1))
        result['error'] = float(round(abs(predicted_real - actual_data['popularity']), 1))
        result['error_pct'] = float(round((result['error'] / max(actual_data['popularity'], 1)) * 100, 1))
        result['metadata'] = {
            'artist_popularity': int(actual_data['artist_popularity']),
            'followers': int(actual_data['artist_followers']),
            'genres': actual_data['artist_genres'][:5],
            'album_type': actual_data['album_type'],
            'markets': int(actual_data['available_markets']),
            'duration': float(round(actual_data['duration_ms'] / 1000, 1)),
            'explicit': bool(actual_data['explicit']),
            'spotify_link': actual_data['spotify_link'],
            'album_art': actual_data.get('album_art')
        }
    else:
        # Prediction with defaults only
        kwargs = {'emotion': emotion, 'album_type': 'single'}
        predicted = predict_popularity(title, artist, **kwargs)
        result['predicted'] = float(round(predicted, 1))
    
    # Category
    pred_score = result.get('predicted_with_data', result.get('predicted', 0))
    if pred_score >= 70:
        result['category'] = 'Hit Potential'
        result['category_icon'] = '🔥'
    elif pred_score >= 50:
        result['category'] = 'Solid Performer'
        result['category_icon'] = '✨'
    elif pred_score >= 30:
        result['category'] = 'Moderate Reach'
        result['category_icon'] = '📈'
    else:
        result['category'] = 'Niche Appeal'
        result['category_icon'] = '🌱'
    
    return jsonify(result)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SPOTIFY POPULARITY PREDICTOR - WEB INTERFACE")
    print("="*70)
    print("\nStarting Flask web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
