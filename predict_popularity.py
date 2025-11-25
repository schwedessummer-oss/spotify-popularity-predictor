"""
Interactive Popularity Predictor with Improvement Suggestions

This script allows you to input a song name and artist, then predicts its popularity
using the trained XGBoost model. It also provides actionable suggestions for improving
popularity based on the model's learned patterns.
"""

import os
import sys
import re
import joblib
import numpy as np
import pandas as pd
import getpass
from sentence_transformers import SentenceTransformer
from collections import Counter
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Paths
MODEL_DIR = r"C:\Users\Winte\OneDrive\Desktop\Spotify Data Collection 4\model_artifacts"
SPOTIFY6_PATH = r"C:\Users\Winte\OneDrive\Desktop\Spotify6"

# Initialize Spotify API (optional - for accuracy checking)
spotify_client = None

def init_spotify():
    """Initialize Spotify API client with user credentials."""
    global spotify_client
    
    if spotify_client is not None:
        return True
    
    print("\n" + "="*70)
    print("SPOTIFY API SETUP (Optional)")
    print("="*70)
    print("To fetch actual popularity and compare with predictions,")
    print("enter your Spotify Developer credentials.")
    print("Press Enter on both prompts to skip and use prediction-only mode.\n")
    
    client_id = getpass.getpass("Spotify Client ID: ").strip()
    if not client_id:
        print("⚠️  Skipping Spotify API - predictions only mode.")
        return False
    
    client_secret = getpass.getpass("Spotify Client Secret: ").strip()
    if not client_secret:
        print("⚠️  Skipping Spotify API - predictions only mode.")
        return False
    
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        )
        spotify_client = spotipy.Spotify(auth_manager=auth_manager)
        # Test connection
        spotify_client.search(q="test", limit=1)
        print("✅ Spotify API connected successfully!\n")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Spotify API: {e}")
        print("⚠️  Continuing in prediction-only mode.\n")
        return False

def fetch_actual_track_data(title, artist):
    """
    Search Spotify for track and return actual data.
    Returns: (popularity, artist_data, genres, album_type, etc.) or None
    """
    if spotify_client is None:
        return None
    
    try:
        # Search for track
        query = f"track:{title} artist:{artist}"
        results = spotify_client.search(q=query, type='track', limit=5)
        
        if not results['tracks']['items']:
            return None
        
        # Get best match (first result usually most relevant)
        track = results['tracks']['items'][0]
        
        # Fetch artist details for full metadata
        artist_ids = [artist['id'] for artist in track['artists']]
        artists_data = spotify_client.artists(artist_ids)['artists']
        
        # Extract data
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
            'track_id': track['id']
        }
        
        # Calculate average artist popularity
        if len(artists_data) > 0:
            avg_artist_pop = sum(a['popularity'] for a in artists_data) / len(artists_data)
            actual_data['avg_artist_popularity'] = avg_artist_pop
        else:
            actual_data['avg_artist_popularity'] = 0
        
        return actual_data
        
    except Exception as e:
        print(f"⚠️  Error fetching from Spotify: {e}")
        return None

# Load model and artifacts
print("Loading trained model and artifacts...")
model = joblib.load(os.path.join(MODEL_DIR, "xgb_popularity_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_num.joblib"))
emotion_ohe = joblib.load(os.path.join(MODEL_DIR, "emotion_ohe.joblib"))
album_type_ohe = joblib.load(os.path.join(MODEL_DIR, "album_type_ohe.joblib"))
sentence_model = SentenceTransformer("all-mpnet-base-v2")

# Load top genres from training data
print("Loading genre database from training data...")
all_genres = []
for emotion_file in ["energetic_tracks_enhanced.csv", "happy_tracks_enhanced.csv", 
                     "love_tracks_enhanced.csv", "sad_tracks_enhanced.csv"]:
    path = os.path.join(SPOTIFY6_PATH, emotion_file)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        if 'artist_genres' in df.columns:
            for genres_str in df['artist_genres'].dropna():
                try:
                    genres = eval(genres_str) if isinstance(genres_str, str) else genres_str
                    if isinstance(genres, list):
                        all_genres.extend(genres)
                except:
                    pass

def _slug(text):
    """Convert genre to slug format."""
    return text.lower().strip().replace(' ', '_').replace('-', '_')

# Get top 40 genres
genre_counts = Counter(_slug(g) for g in all_genres)
top_genres = [g for g, _ in genre_counts.most_common(40)]
print(f"Loaded {len(top_genres)} top genres for encoding.")

# Load average statistics from training data
print("Computing baseline statistics from training data...")
all_data = []
for emotion_file in ["energetic_tracks_enhanced.csv", "happy_tracks_enhanced.csv", 
                     "love_tracks_enhanced.csv", "sad_tracks_enhanced.csv"]:
    path = os.path.join(SPOTIFY6_PATH, emotion_file)
    if os.path.exists(path):
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        all_data.append(df)

if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Compute averages
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
    
    # Top performing values (75th percentile)
    top_stats = {
        'artist_popularity': combined_df['artist_popularity'].quantile(0.75) if 'artist_popularity' in combined_df.columns else 70,
        'artist_followers': combined_df['artist_followers'].quantile(0.75) if 'artist_followers' in combined_df.columns else 5000000,
        'available_markets': combined_df['available_markets'].quantile(0.75) if 'available_markets' in combined_df.columns else 180,
    }
    
    print("Baseline statistics computed.")
else:
    print("Warning: Could not load training data. Using defaults.")
    avg_stats = {
        'artist_popularity': 50, 'artist_followers': 1000000, 'album_total_tracks': 10,
        'available_markets': 150, 'duration_ms': 200000, 'explicit': 0.14,
        'num_artists': 1, 'is_collaboration': 0.33, 'avg_artist_popularity': 50
    }
    top_stats = {
        'artist_popularity': 70, 'artist_followers': 5000000, 'available_markets': 180
    }


def extract_title_features(title):
    """Extract basic features from title."""
    title_str = str(title)
    word_count = len(title_str.split())
    char_count = len(title_str)
    has_feat = 1 if any(x in title_str.lower() for x in ['feat', 'ft.', 'featuring']) else 0
    has_parentheses = 1 if '(' in title_str or '[' in title_str else 0
    has_numbers = 1 if any(c.isdigit() for c in title_str) else 0
    
    # Additional features to match training
    has_special = 1 if any(c in title_str for c in ['!', '?', '#', '$', '%', '&', '*']) else 0
    has_remix = 1 if any(x in title_str.lower() for x in ['remix', 'mix', 'edit']) else 0
    has_live = 1 if 'live' in title_str.lower() else 0
    has_version = 1 if any(x in title_str.lower() for x in ['version', 'ver.', 'remaster']) else 0
    all_caps_ratio = sum(1 for c in title_str if c.isupper()) / max(len(title_str), 1)
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'has_feat': has_feat,
        'has_parentheses': has_parentheses,
        'has_numbers': has_numbers,
        'has_special': has_special,
        'has_remix': has_remix,
        'has_live': has_live,
        'has_version': has_version,
        'all_caps_ratio': all_caps_ratio
    }


def compute_duration_features(duration_ms, emotion='happy'):
    """Compute duration-related features to match training."""
    duration_s = duration_ms / 1000.0
    
    # Duration categories (based on training script thresholds)
    is_very_short = 1 if duration_s < 120 else 0
    is_short = 1 if 120 <= duration_s < 180 else 0
    is_medium = 1 if 180 <= duration_s < 240 else 0
    is_long = 1 if 240 <= duration_s < 300 else 0
    is_very_long = 1 if duration_s >= 300 else 0
    is_typical_duration = 1 if 180 <= duration_s <= 240 else 0
    
    # Duration z-score per emotion (using dataset averages)
    emotion_duration_stats = {
        'happy': (3.33 * 60, 1.19 * 60),      # mean, std in seconds
        'sad': (3.00 * 60, 1.19 * 60),
        'energetic': (3.14 * 60, 0.98 * 60),
        'love': (3.49 * 60, 1.13 * 60)
    }
    mean_dur, std_dur = emotion_duration_stats.get(emotion, (200, 60))
    duration_z_emotion = (duration_s - mean_dur) / max(std_dur, 1)
    
    return {
        'duration_s': duration_s,
        'is_very_short': is_very_short,
        'is_short': is_short,
        'is_medium': is_medium,
        'is_long': is_long,
        'is_very_long': is_very_long,
        'is_typical_duration': is_typical_duration,
        'duration_z_emotion': duration_z_emotion
    }


def compute_release_features():
    """Compute release date features (current date context)."""
    from datetime import datetime
    current_date = datetime.now()
    
    # Assume track is recent (new release) if not specified
    release_age = 0.5  # years
    days_since_release = 180  # days
    
    is_very_recent = 1 if days_since_release < 30 else 0
    is_recent = 1 if 30 <= days_since_release < 180 else 0
    is_new = 1 if days_since_release < 365 else 0
    
    current_year = current_date.year
    release_year = current_year  # Assume current year
    
    is_2020s = 1 if release_year >= 2020 else 0
    is_2010s = 1 if 2010 <= release_year < 2020 else 0
    is_2000s = 1 if 2000 <= release_year < 2010 else 0
    is_classic = 1 if release_year < 2000 else 0
    
    return {
        'release_age': release_age,
        'days_since_release': days_since_release,
        'is_very_recent': is_very_recent,
        'is_recent': is_recent,
        'is_new': is_new,
        'is_2020s': is_2020s,
        'is_2010s': is_2010s,
        'is_2000s': is_2000s,
        'is_classic': is_classic
    }


def build_feature_vector(title, artist, emotion='happy', artist_popularity=None, 
                         artist_followers=None, genres=None, album_type='single',
                         album_total_tracks=None, explicit=0, available_markets=None,
                         duration_ms=None, num_artists=None, is_collaboration=None,
                         avg_artist_popularity=None):
    """
    Build complete feature vector matching training format EXACTLY.
    
    Training feature order:
    1. Embeddings (768)
    2. Numeric features (scaled)
    3. Binary features
    4. Emotion one-hot (4)
    5. Album type one-hot (3)
    """
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
    
    # Extract features
    title_feats = extract_title_features(title)
    duration_feats = compute_duration_features(duration_ms, emotion)
    release_feats = compute_release_features()
    
    # Generate embedding for title + artist + emotion + album_type + genres
    # MUST match training format exactly!
    genres_str = ', '.join(genres) if genres else ''
    text = (
        f"Title: {title} | "
        f"Artist: {artist} | "
        f"Emotion: {emotion}"
    )
    if album_type:
        text += f" | AlbumType: {album_type}"
    if genres_str:
        text += f" | Genres: {genres_str}"
    
    embedding = sentence_model.encode([text])[0]  # 768 dims
    
    # NUMERIC FEATURES (must match training order exactly)
    # BASE_NUM_COLS = ['duration_s', 'release_age', 'days_since_release',
    #                  'title_length', 'title_word_count', 'title_all_caps_ratio',
    #                  'artist_name_length', 'num_artists', 'duration_z_emotion']
    # ADDITIONAL = ['artist_popularity', 'artist_followers', 'album_total_tracks',
    #               'available_markets', 'avg_artist_popularity']
    
    numeric_features = np.array([
        duration_feats['duration_s'],
        release_feats['release_age'],
        release_feats['days_since_release'],
        title_feats['char_count'],  # title_length
        title_feats['word_count'],  # title_word_count
        title_feats['all_caps_ratio'],  # title_all_caps_ratio
        len(artist),  # artist_name_length
        num_artists,
        duration_feats['duration_z_emotion'],
        artist_popularity,
        artist_followers,
        album_total_tracks,
        available_markets,
        avg_artist_popularity
    ])
    
    # BINARY FEATURES (must match training order)
    # BASE_BINARY_COLS = ['is_very_recent', 'is_recent', 'is_new',
    #                     'is_2020s', 'is_2010s', 'is_2000s', 'is_classic',
    #                     'title_has_numbers', 'title_has_special', 'title_has_remix',
    #                     'title_has_feat', 'title_has_live', 'title_has_version',
    #                     'is_collaboration', 'has_feat_in_artist',
    #                     'is_very_short', 'is_short', 'is_medium', 'is_long', 'is_very_long',
    #                     'is_typical_duration']
    # + explicit + genre columns
    
    has_feat_in_artist = 1 if any(x in artist.lower() for x in ['feat', 'ft.', '&']) else 0
    
    binary_features = [
        release_feats['is_very_recent'],
        release_feats['is_recent'],
        release_feats['is_new'],
        release_feats['is_2020s'],
        release_feats['is_2010s'],
        release_feats['is_2000s'],
        release_feats['is_classic'],
        title_feats['has_numbers'],
        title_feats['has_special'],
        title_feats['has_remix'],
        title_feats['has_feat'],
        title_feats['has_live'],
        title_feats['has_version'],
        is_collaboration,
        has_feat_in_artist,
        duration_feats['is_very_short'],
        duration_feats['is_short'],
        duration_feats['is_medium'],
        duration_feats['is_long'],
        duration_feats['is_very_long'],
        duration_feats['is_typical_duration'],
        explicit
    ]
    
    # Genre multi-hot encoding (40 top genres)
    genre_binary = np.zeros(40)
    if genres:
        genre_slugs = [_slug(g) for g in genres]
        for i, top_genre in enumerate(top_genres[:40]):  # Only top 40
            if top_genre in genre_slugs:
                genre_binary[i] = 1
    
    # Combine all binary features
    binary_features.extend(genre_binary)
    binary_features = np.array(binary_features)
    
    # Emotion one-hot (4 dims)
    emotion_encoded = emotion_ohe.transform([[emotion]])
    if hasattr(emotion_encoded, 'toarray'):
        emotion_encoded = emotion_encoded.toarray()[0]
    else:
        emotion_encoded = emotion_encoded[0] if len(emotion_encoded.shape) > 1 else emotion_encoded
    
    # Album type one-hot (3 dims)
    album_type_encoded = album_type_ohe.transform([[album_type]])
    if hasattr(album_type_encoded, 'toarray'):
        album_type_encoded = album_type_encoded.toarray()[0]
    else:
        album_type_encoded = album_type_encoded[0] if len(album_type_encoded.shape) > 1 else album_type_encoded
    
    # Scale numeric features (MUST match training scaler)
    numeric_features_scaled = scaler.transform([numeric_features])[0]
    
    # Assemble: embedding (768) + numeric_scaled (14) + binary (22+1+40=63) + emotion (4) + album_type (3)
    # Total: 768 + 14 + 63 + 4 + 3 = 852
    feature_vector = np.concatenate([
        embedding,
        numeric_features_scaled,
        binary_features,
        emotion_encoded,
        album_type_encoded
    ])
    
    return feature_vector.reshape(1, -1)


def predict_popularity(title, artist, **kwargs):
    """Predict popularity for a song."""
    feature_vec = build_feature_vector(title, artist, **kwargs)
    log_pred = model.predict(feature_vec)[0]
    popularity = np.expm1(log_pred)  # Inverse of log1p
    return max(0, min(100, popularity))  # Clamp to 0-100


def generate_suggestions(title, artist, base_popularity, **kwargs):
    """Generate actionable suggestions to improve popularity."""
    suggestions = []
    
    # Create a copy of kwargs without emotion for testing different emotions
    kwargs_no_emotion = {k: v for k, v in kwargs.items() if k != 'emotion'}
    
    # Test different emotions
    print("\n  Testing emotion impact...")
    emotion_impacts = {}
    for emotion in ['happy', 'energetic', 'love', 'sad']:
        pred = predict_popularity(title, artist, emotion=emotion, **kwargs_no_emotion)
        emotion_impacts[emotion] = pred
    
    best_emotion = max(emotion_impacts, key=emotion_impacts.get)
    if emotion_impacts[best_emotion] > base_popularity + 2:
        suggestions.append({
            'category': 'Emotion/Vibe',
            'suggestion': f"Position as '{best_emotion}' track",
            'impact': f"+{emotion_impacts[best_emotion] - base_popularity:.1f} points",
            'explanation': f"'{best_emotion.capitalize()}' tracks in your genre tend to perform better"
        })
    
    # Test collaboration
    print("  Testing collaboration impact...")
    current_collab = kwargs.get('is_collaboration', avg_stats['is_collaboration'])
    kwargs_test_collab = kwargs.copy()
    if current_collab < 0.5:
        kwargs_test_collab['is_collaboration'] = 1
        kwargs_test_collab['num_artists'] = 2
        collab_pred = predict_popularity(title, artist, **kwargs_test_collab)
        if collab_pred > base_popularity + 2:
            suggestions.append({
                'category': 'Artist Strategy',
                'suggestion': "Add featured artist collaboration",
                'impact': f"+{collab_pred - base_popularity:.1f} points",
                'explanation': "Collaborations increase discovery and cross-audience appeal"
            })
    
    # Test higher artist popularity
    print("  Testing artist popularity impact...")
    current_artist_pop = kwargs.get('artist_popularity', avg_stats['artist_popularity'])
    if current_artist_pop < top_stats['artist_popularity']:
        high_artist_pred = predict_popularity(
            title, artist,
            artist_popularity=top_stats['artist_popularity'],
            avg_artist_popularity=top_stats['artist_popularity'],
            **{k: v for k, v in kwargs.items() if k not in ['artist_popularity', 'avg_artist_popularity']}
        )
        if high_artist_pred > base_popularity + 5:
            suggestions.append({
                'category': 'Artist Growth',
                'suggestion': f"Grow artist popularity to {top_stats['artist_popularity']:.0f}+",
                'impact': f"+{high_artist_pred - base_popularity:.1f} points",
                'explanation': "Higher artist popularity strongly predicts track success"
            })
    
    # Test market availability
    print("  Testing market availability...")
    current_markets = kwargs.get('available_markets', avg_stats['available_markets'])
    if current_markets < top_stats['available_markets']:
        market_pred = predict_popularity(
            title, artist,
            available_markets=top_stats['available_markets'],
            **{k: v for k, v in kwargs.items() if k != 'available_markets'}
        )
        if market_pred > base_popularity + 3:
            suggestions.append({
                'category': 'Distribution',
                'suggestion': f"Expand to {int(top_stats['available_markets'])}+ markets",
                'impact': f"+{market_pred - base_popularity:.1f} points",
                'explanation': "Wider distribution increases potential audience reach"
            })
    
    # Test title variations
    print("  Testing title optimizations...")
    title_tests = []
    
    # Test with feat
    if 'feat' not in title.lower():
        feat_title = f"{title} (feat. Artist)"
        feat_pred = predict_popularity(feat_title, artist, **kwargs)
        title_tests.append(('Add featured artist to title', feat_pred))
    
    # Test shorter title
    if len(title.split()) > 3:
        short_title = ' '.join(title.split()[:3])
        short_pred = predict_popularity(short_title, artist, **kwargs)
        title_tests.append(('Shorten title (first 3 words)', short_pred))
    
    # Test with parenthetical
    if '(' not in title:
        paren_title = f"{title} (Radio Edit)"
        paren_pred = predict_popularity(paren_title, artist, **kwargs)
        title_tests.append(('Add version descriptor', paren_pred))
    
    best_title_test = max(title_tests, key=lambda x: x[1]) if title_tests else None
    if best_title_test and best_title_test[1] > base_popularity + 1:
        suggestions.append({
            'category': 'Title Optimization',
            'suggestion': best_title_test[0],
            'impact': f"+{best_title_test[1] - base_popularity:.1f} points",
            'explanation': "Title format affects discoverability and click-through"
        })
    
    # Test genre changes
    print("  Testing top genre impacts...")
    current_genres = kwargs.get('genres', [])
    top_test_genres = ['pop', 'rap', 'edm', 'r&b', 'rock']
    
    genre_tests = []
    for test_genre in top_test_genres:
        if test_genre not in [_slug(g) for g in current_genres]:
            test_genres = current_genres + [test_genre]
            kwargs_test_genre = kwargs.copy()
            kwargs_test_genre['genres'] = test_genres
            genre_pred = predict_popularity(title, artist, **kwargs_test_genre)
            genre_tests.append((test_genre, genre_pred))
    
    if genre_tests:
        best_genre = max(genre_tests, key=lambda x: x[1])
        if best_genre[1] > base_popularity + 2:
            suggestions.append({
                'category': 'Genre Strategy',
                'suggestion': f"Incorporate '{best_genre[0]}' elements",
                'impact': f"+{best_genre[1] - base_popularity:.1f} points",
                'explanation': f"'{best_genre[0].capitalize()}' genre signals improve algorithmic placement"
            })
    
    # Test album type
    print("  Testing release format...")
    current_album_type = kwargs.get('album_type', 'single')
    for test_type in ['album', 'single', 'compilation']:
        if test_type != current_album_type:
            kwargs_test_album = kwargs.copy()
            kwargs_test_album['album_type'] = test_type
            type_pred = predict_popularity(title, artist, **kwargs_test_album)
            if type_pred > base_popularity + 2:
                suggestions.append({
                    'category': 'Release Format',
                    'suggestion': f"Release as '{test_type}'",
                    'impact': f"+{type_pred - base_popularity:.1f} points",
                    'explanation': f"'{test_type.capitalize()}' format may perform better for this track"
                })
                break
    
    # Sort by impact
    suggestions.sort(key=lambda x: float(x['impact'].replace('+', '').replace(' points', '')), reverse=True)
    
    return suggestions


def interactive_mode():
    """Run interactive prediction session."""
    print("\n" + "="*70)
    print("SPOTIFY POPULARITY PREDICTOR")
    print("="*70)
    print("\nPredict track popularity and get improvement suggestions!")
    print("Type 'quit' to exit.\n")
    
    # Initialize Spotify API
    has_spotify = init_spotify()
    
    while True:
        print("-" * 70)
        
        # Get user input
        title = input("\nEnter song title: ").strip()
        if title.lower() == 'quit':
            break
        
        artist = input("Enter artist name: ").strip()
        if artist.lower() == 'quit':
            break
        
        # Try to fetch actual data from Spotify
        actual_data = None
        use_actual_data = False
        
        if has_spotify:
            print("\n🔍 Searching Spotify for track data...")
            actual_data = fetch_actual_track_data(title, artist)
            
            if actual_data:
                print(f"✅ Found: '{actual_data['name']}' by {actual_data['artists']}")
                use_input = input("Use actual Spotify data for prediction? (y/n) [y]: ").strip().lower()
                use_actual_data = use_input != 'n'
            else:
                print("⚠️  Track not found on Spotify. Using manual inputs.")
        
        # If using actual data, build kwargs from it
        if use_actual_data and actual_data:
            kwargs = {
                'emotion': 'happy',  # Default, can still be overridden
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
            
            # Still ask for emotion
            emotion = input(f"  Emotion (happy/sad/energetic/love) [happy]: ").strip().lower() or 'happy'
            if emotion in ['happy', 'sad', 'energetic', 'love']:
                kwargs['emotion'] = emotion
                
        else:
            # Manual input mode
            print("\nOptional - press Enter to use dataset averages:")
            
            emotion = input("  Emotion (happy/sad/energetic/love) [happy]: ").strip().lower() or 'happy'
            if emotion not in ['happy', 'sad', 'energetic', 'love']:
                emotion = 'happy'
            
            artist_pop_input = input(f"  Artist popularity 0-100 [{avg_stats['artist_popularity']:.0f}]: ").strip()
            artist_popularity = float(artist_pop_input) if artist_pop_input else None
            
            followers_input = input(f"  Artist followers [{int(avg_stats['artist_followers']):,}]: ").strip()
            artist_followers = float(followers_input.replace(',', '')) if followers_input else None
            
            genres_input = input("  Genres (comma-separated): ").strip()
            genres = [g.strip() for g in genres_input.split(',')] if genres_input else []
            
            album_type = input("  Album type (single/album/compilation) [single]: ").strip().lower() or 'single'
            if album_type not in ['single', 'album', 'compilation']:
                album_type = 'single'
            
            explicit_input = input("  Explicit content? (y/n) [n]: ").strip().lower()
            explicit = 1 if explicit_input == 'y' else 0
            
            # Build kwargs
            kwargs = {
                'emotion': emotion,
                'album_type': album_type,
                'explicit': explicit
            }
            if artist_popularity is not None:
                kwargs['artist_popularity'] = artist_popularity
                kwargs['avg_artist_popularity'] = artist_popularity
            if artist_followers is not None:
                kwargs['artist_followers'] = artist_followers
            if genres:
                kwargs['genres'] = genres
        
        # Predict
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        # Always make TWO predictions: with actual data and with defaults
        print(f"\n🎵 '{title}' by {artist}")
        
        # Prediction 1: With actual Spotify data (if available)
        if use_actual_data and actual_data:
            predicted_pop_actual = predict_popularity(title, artist, **kwargs)
            
            print(f"\n📊 ACTUAL Popularity:    {actual_data['popularity']:.0f}/100")
            print(f"🤖 PREDICTED (with real data): {predicted_pop_actual:.1f}/100")
            
            error = abs(predicted_pop_actual - actual_data['popularity'])
            error_pct = (error / max(actual_data['popularity'], 1)) * 100
            
            print(f"\n📈 Model Accuracy:")
            print(f"   Absolute Error: {error:.1f} points")
            print(f"   Relative Error: {error_pct:.1f}%")
            
            if error <= 5:
                accuracy_msg = "🎯 Excellent prediction!"
            elif error <= 10:
                accuracy_msg = "✅ Good prediction"
            elif error <= 15:
                accuracy_msg = "👍 Acceptable prediction"
            else:
                accuracy_msg = "⚠️  Large deviation - consider model limitations"
            print(f"   {accuracy_msg}")
            
            # Show actual metadata used
            print(f"\n📋 Actual Track Metadata Used:")
            print(f"   Artist Popularity: {actual_data['artist_popularity']}")
            print(f"   Followers: {actual_data['artist_followers']:,}")
            print(f"   Genres: {', '.join(actual_data['artist_genres'][:5]) if actual_data['artist_genres'] else 'None'}")
            print(f"   Album Type: {actual_data['album_type']}")
            print(f"   Markets: {actual_data['available_markets']}")
            print(f"   Duration: {actual_data['duration_ms']/1000:.1f}s")
            print(f"   Explicit: {'Yes' if actual_data['explicit'] else 'No'}")
            print(f"   🔗 {actual_data['spotify_link']}")
            
            # Prediction 2: With dataset averages (baseline comparison)
            print(f"\n{'─'*70}")
            print("📉 BASELINE PREDICTION (using dataset averages)")
            print(f"{'─'*70}")
            
            # Build kwargs with defaults only
            kwargs_baseline = {
                'emotion': kwargs.get('emotion', 'happy'),
                'album_type': 'single'
            }
            predicted_pop_baseline = predict_popularity(title, artist, **kwargs_baseline)
            print(f"🤖 PREDICTED (with defaults): {predicted_pop_baseline:.1f}/100")
            
            # Show the impact of using real data
            data_impact = predicted_pop_actual - predicted_pop_baseline
            print(f"\n💎 Real Data Impact: {data_impact:+.1f} points")
            if abs(data_impact) > 5:
                print(f"   {'↗️ Real metadata significantly boosts prediction!' if data_impact > 0 else '↘️ Real metadata lowers prediction (unusual)'}")
            else:
                print(f"   Track characteristics matter more than artist metadata")
            
            # Use actual data prediction for suggestions
            predicted_pop = predicted_pop_actual
            
        else:
            # No Spotify data - single prediction with defaults
            predicted_pop = predict_popularity(title, artist, **kwargs)
            print(f"\n📊 Predicted Popularity: {predicted_pop:.1f}/100")
            print(f"   (Using dataset average values)")
        
        # Categorize
        if predicted_pop >= 70:
            category = "🔥 HIT POTENTIAL"
        elif predicted_pop >= 50:
            category = "✨ SOLID PERFORMER"
        elif predicted_pop >= 30:
            category = "📈 MODERATE REACH"
        else:
            category = "🌱 NICHE APPEAL"
        print(f"\n   Category: {category}")
        
        # Generate suggestions
        print("\n" + "-"*70)
        print("💡 IMPROVEMENT SUGGESTIONS")
        print("-"*70)
        print("\nAnalyzing optimization opportunities...")
        
        suggestions = generate_suggestions(title, artist, predicted_pop, **kwargs)
        
        if suggestions:
            print(f"\nFound {len(suggestions)} actionable recommendations:\n")
            for i, sug in enumerate(suggestions, 1):
                print(f"{i}. [{sug['category']}] {sug['suggestion']}")
                print(f"   Expected Impact: {sug['impact']}")
                print(f"   Why: {sug['explanation']}\n")
        else:
            print("\n✅ Track is well-optimized! No major improvements detected.")
        
        print("\n" + "="*70)
        
        # Ask to continue
        continue_input = input("\nPredict another track? (y/n): ").strip().lower()
        if continue_input != 'y':
            break
    
    print("\n👋 Thanks for using the Popularity Predictor!\n")


def batch_mode(csv_file):
    """Process batch predictions from CSV file."""
    print(f"\nProcessing batch predictions from: {csv_file}")
    
    df = pd.read_csv(csv_file)
    required_cols = ['title', 'artist']
    
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return
    
    results = []
    
    for idx, row in df.iterrows():
        title = row['title']
        artist = row['artist']
        
        # Optional columns
        kwargs = {}
        if 'emotion' in df.columns and pd.notna(row['emotion']):
            kwargs['emotion'] = row['emotion']
        if 'artist_popularity' in df.columns and pd.notna(row['artist_popularity']):
            kwargs['artist_popularity'] = row['artist_popularity']
            kwargs['avg_artist_popularity'] = row['artist_popularity']
        if 'artist_followers' in df.columns and pd.notna(row['artist_followers']):
            kwargs['artist_followers'] = row['artist_followers']
        if 'genres' in df.columns and pd.notna(row['genres']):
            kwargs['genres'] = [g.strip() for g in str(row['genres']).split(',')]
        if 'album_type' in df.columns and pd.notna(row['album_type']):
            kwargs['album_type'] = row['album_type']
        if 'explicit' in df.columns and pd.notna(row['explicit']):
            kwargs['explicit'] = int(row['explicit'])
        
        predicted_pop = predict_popularity(title, artist, **kwargs)
        
        results.append({
            'title': title,
            'artist': artist,
            'predicted_popularity': predicted_pop
        })
        
        print(f"  {idx+1}/{len(df)}: '{title}' by {artist} -> {predicted_pop:.1f}")
    
    # Save results
    output_file = csv_file.replace('.csv', '_predictions.csv')
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\n✅ Predictions saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Batch mode with CSV input
        batch_mode(sys.argv[1])
    else:
        # Interactive mode
        interactive_mode()
