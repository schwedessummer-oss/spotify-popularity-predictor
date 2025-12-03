"""
Spotify Popularity Predictor - Streamlit Web App
Predicts track popularity and provides improvement suggestions
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Page config
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1DB954;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-value {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
    }
    .prediction-label {
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .suggestion-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: #f8f9fa;
        border-left: 4px solid #1DB954;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #1ed760;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.emotion_ohe = None
    st.session_state.album_type_ohe = None
    st.session_state.embedder = None

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        model_dir = "model_artifacts"
        
        # Load model
        model_path = os.path.join(model_dir, "xgb_popularity_model.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "light_model.joblib")
        model = joblib.load(model_path)
        
        # Load scaler
        scaler = joblib.load(os.path.join(model_dir, "scaler_num.joblib"))
        
        # Load encoders
        emotion_ohe = joblib.load(os.path.join(model_dir, "emotion_ohe.joblib"))
        
        album_type_ohe = None
        album_ohe_path = os.path.join(model_dir, "album_type_ohe.joblib")
        if os.path.exists(album_ohe_path):
            album_type_ohe = joblib.load(album_ohe_path)
        
        # Load embedder
        embed_info_path = os.path.join(model_dir, "embedder_info.txt")
        embed_model = "all-mpnet-base-v2"
        if os.path.exists(embed_info_path):
            with open(embed_info_path, 'r') as f:
                embed_model = f.read().strip()
        
        embedder = SentenceTransformer(embed_model)
        
        return model, scaler, emotion_ohe, album_type_ohe, embedder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None

# Initialize Spotify client (optional)
@st.cache_resource
def init_spotify_client():
    """Initialize Spotify API client if credentials available"""
    try:
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        
        if client_id and client_secret:
            auth_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            return spotipy.Spotify(auth_manager=auth_manager)
    except:
        pass
    return None

# Feature engineering functions
def extract_title_features(title):
    """Extract features from song title"""
    title_str = str(title)
    features = {
        'title_length': len(title_str),
        'title_word_count': len(title_str.split()),
        'title_all_caps_ratio': sum(1 for c in title_str if c.isupper()) / len(title_str) if len(title_str) > 0 else 0,
        'title_has_numbers': int(any(c.isdigit() for c in title_str)),
        'title_has_special': int(any(c in '!?@#$%^&*()' for c in title_str)),
        'title_has_remix': int(any(word in title_str.lower() for word in ['remix', 'mix', 'edit'])),
        'title_has_feat': int(any(word in title_str.lower() for word in ['feat', 'ft.', 'featuring'])),
        'title_has_live': int(any(word in title_str.lower() for word in ['live', 'concert', 'tour'])),
        'title_has_version': int(any(word in title_str.lower() for word in ['version', 'remaster', 'deluxe']))
    }
    return features

def extract_artist_features(artist):
    """Extract features from artist name"""
    artist_str = str(artist)
    features = {
        'artist_name_length': len(artist_str),
        'num_artists': artist_str.count(',') + 1,
        'is_collaboration': int(artist_str.count(',') > 0),
        'has_feat_in_artist': int(any(word in artist_str.lower() for word in ['feat', 'ft.', '&']))
    }
    return features

def compute_duration_features(duration_ms, emotion_avg_duration=210000):
    """Compute duration-based features"""
    duration_s = duration_ms / 1000.0
    features = {
        'duration_s': duration_s,
        'is_very_short': int(duration_s < 120),
        'is_short': int(120 <= duration_s < 180),
        'is_medium': int(180 <= duration_s < 240),
        'is_long': int(240 <= duration_s < 300),
        'is_very_long': int(duration_s >= 300),
        'is_typical_duration': int(180 <= duration_s < 240),
        'duration_z_emotion': (duration_s - (emotion_avg_duration / 1000.0)) / 30.0  # Approximate z-score
    }
    return features

def compute_release_features(release_date):
    """Compute release date features"""
    try:
        if isinstance(release_date, str):
            release_dt = pd.to_datetime(release_date)
        else:
            release_dt = release_date
    except:
        release_dt = pd.Timestamp.now()
    
    current_year = datetime.now().year
    release_year = release_dt.year
    days_since = (pd.Timestamp.now() - release_dt).days
    
    features = {
        'release_age': current_year - release_year,
        'days_since_release': max(0, days_since),
        'is_very_recent': int(days_since < 30),
        'is_recent': int(days_since < 90),
        'is_new': int(days_since < 365),
        'decade': (release_year // 10) * 10,
        'is_2020s': int(release_year >= 2020),
        'is_2010s': int(2010 <= release_year < 2020),
        'is_2000s': int(2000 <= release_year < 2010),
        'is_classic': int(release_year < 2000)
    }
    return features

def build_feature_vector(title, artist, duration_ms, release_date, emotion, album_type, 
                         embedder, scaler, emotion_ohe, album_type_ohe):
    """Build complete feature vector for prediction"""
    # Get embedding
    joint_text = f"Title: {title} | Artist: {artist} | Emotion: {emotion}"
    if album_type:
        joint_text += f" | AlbumType: {album_type}"
    
    embedding = embedder.encode([joint_text], convert_to_numpy=True)[0]
    
    # Extract all features
    title_feats = extract_title_features(title)
    artist_feats = extract_artist_features(artist)
    duration_feats = compute_duration_features(duration_ms)
    release_feats = compute_release_features(release_date)
    
    # Combine numeric features
    numeric_features = [
        duration_feats['duration_s'],
        release_feats['release_age'],
        release_feats['days_since_release'],
        title_feats['title_length'],
        title_feats['title_word_count'],
        title_feats['title_all_caps_ratio'],
        artist_feats['artist_name_length'],
        artist_feats['num_artists'],
        duration_feats['duration_z_emotion']
    ]
    
    # Scale numeric features
    numeric_scaled = scaler.transform([numeric_features])[0]
    
    # Binary features
    binary_features = [
        release_feats['is_very_recent'],
        release_feats['is_recent'],
        release_feats['is_new'],
        release_feats['is_2020s'],
        release_feats['is_2010s'],
        release_feats['is_2000s'],
        release_feats['is_classic'],
        title_feats['title_has_numbers'],
        title_feats['title_has_special'],
        title_feats['title_has_remix'],
        title_feats['title_has_feat'],
        title_feats['title_has_live'],
        title_feats['title_has_version'],
        artist_feats['is_collaboration'],
        artist_feats['has_feat_in_artist'],
        duration_feats['is_very_short'],
        duration_feats['is_short'],
        duration_feats['is_medium'],
        duration_feats['is_long'],
        duration_feats['is_very_long'],
        duration_feats['is_typical_duration']
    ]
    
    # One-hot encode emotion
    emotion_encoded = emotion_ohe.transform([[emotion]])[0]
    
    # Combine all features
    feature_blocks = [embedding, numeric_scaled, binary_features, emotion_encoded]
    
    # Add album type if encoder exists
    if album_type_ohe and album_type:
        album_encoded = album_type_ohe.transform([[album_type]])[0]
        feature_blocks.append(album_encoded)
    
    return np.hstack(feature_blocks).reshape(1, -1)

def predict_popularity(feature_vector, model):
    """Make prediction and inverse transform"""
    y_pred_log = model.predict(feature_vector)[0]
    y_pred_raw = np.clip(np.expm1(y_pred_log), 0, 100)
    return y_pred_raw

def generate_suggestions(title, artist, duration_ms, release_date, emotion, album_type,
                        base_prediction, embedder, scaler, emotion_ohe, album_type_ohe, model):
    """Generate improvement suggestions"""
    suggestions = []
    
    # Try different emotions
    emotions_to_try = ['happy', 'energetic', 'love', 'sad']
    emotion_scores = {}
    
    for emo in emotions_to_try:
        if emo != emotion:
            vec = build_feature_vector(title, artist, duration_ms, release_date, emo, 
                                      album_type, embedder, scaler, emotion_ohe, album_type_ohe)
            pred = predict_popularity(vec, model)
            emotion_scores[emo] = pred
    
    best_emotion = max(emotion_scores.items(), key=lambda x: x[1])
    if best_emotion[1] > base_prediction + 3:
        suggestions.append({
            'type': 'Emotion',
            'suggestion': f"Try categorizing as '{best_emotion[0].title()}' emotion",
            'impact': f"+{best_emotion[1] - base_prediction:.1f} popularity",
            'reason': f"This emotion category tends to perform better for similar tracks"
        })
    
    # Try with collaboration
    if ' feat. ' not in artist.lower() and ',' not in artist:
        collab_artist = f"{artist} feat. Artist Name"
        vec = build_feature_vector(title, collab_artist, duration_ms, release_date, emotion,
                                   album_type, embedder, scaler, emotion_ohe, album_type_ohe)
        pred = predict_popularity(vec, model)
        if pred > base_prediction + 2:
            suggestions.append({
                'type': 'Collaboration',
                'suggestion': "Consider featuring another artist",
                'impact': f"+{pred - base_prediction:.1f} popularity",
                'reason': "Collaborations often reach wider audiences"
            })
    
    # Try optimal duration
    optimal_durations = [180000, 210000, 240000]  # 3, 3.5, 4 minutes
    for dur in optimal_durations:
        if abs(dur - duration_ms) > 30000:
            vec = build_feature_vector(title, artist, dur, release_date, emotion,
                                      album_type, embedder, scaler, emotion_ohe, album_type_ohe)
            pred = predict_popularity(vec, model)
            if pred > base_prediction + 1.5:
                suggestions.append({
                    'type': 'Duration',
                    'suggestion': f"Adjust duration to ~{dur//60000}:{(dur%60000)//1000:02d}",
                    'impact': f"+{pred - base_prediction:.1f} popularity",
                    'reason': f"This duration ({dur//1000}s) tends to perform better"
                })
                break
    
    # Recent release bonus
    recent_date = pd.Timestamp.now() - pd.Timedelta(days=30)
    vec = build_feature_vector(title, artist, duration_ms, recent_date, emotion,
                               album_type, embedder, scaler, emotion_ohe, album_type_ohe)
    pred = predict_popularity(vec, model)
    if pred > base_prediction + 2:
        suggestions.append({
            'type': 'Timing',
            'suggestion': "Release during a trending period or re-release",
            'impact': f"+{pred - base_prediction:.1f} popularity",
            'reason': "Recent releases get algorithmic boost"
        })
    
    return suggestions[:4]  # Top 4 suggestions

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Spotify Popularity Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict your track\'s popularity and get AI-powered improvement suggestions</p>', unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model, scaler, emotion_ohe, album_type_ohe, embedder = load_model_artifacts()
            if model is not None:
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.emotion_ohe = emotion_ohe
                st.session_state.album_type_ohe = album_type_ohe
                st.session_state.embedder = embedder
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("Failed to load model. Please check that model_artifacts folder exists.")
                return
    
    # Initialize Spotify client
    sp = init_spotify_client()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.write("""
        This AI model predicts Spotify track popularity (0-100) using:
        - 🎯 **XGBoost** algorithm
        - 🧠 **768-dim embeddings** (title/artist/emotion)
        - 📈 **850+ engineered features**
        - 📚 Trained on **thousands of tracks**
        """)
        
        st.header("🔍 How to Use")
        st.write("""
        1. Enter your track details
        2. Click "Predict Popularity"
        3. View prediction & suggestions
        4. (Optional) Search Spotify for actual popularity
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Track Details")
        
        # Input form
        with st.form("prediction_form"):
            title = st.text_input("🎵 Song Title", placeholder="Enter song title...", help="The name of your track")
            artist = st.text_input("🎤 Artist Name", placeholder="Enter artist name...", help="Artist or band name")
            
            col_a, col_b = st.columns(2)
            with col_a:
                duration_min = st.number_input("⏱️ Duration (minutes)", min_value=0, max_value=60, value=3, help="Track length in minutes")
                duration_sec = st.number_input("⏱️ Duration (seconds)", min_value=0, max_value=59, value=30, help="Additional seconds")
            
            with col_b:
                emotion = st.selectbox("😊 Emotion", ['happy', 'sad', 'energetic', 'love'], help="Primary emotion/mood of the track")
                album_type = st.selectbox("💿 Album Type", ['single', 'album', 'compilation'], help="Type of release")
            
            release_date = st.date_input("📅 Release Date", value=datetime.now(), help="Track release date")
            
            submit = st.form_submit_button("🔮 Predict Popularity", use_container_width=True)
        
        if submit:
            if not title or not artist:
                st.error("⚠️ Please enter both title and artist name")
            else:
                # Calculate duration in ms
                duration_ms = (duration_min * 60 + duration_sec) * 1000
                
                with st.spinner("Analyzing track..."):
                    # Build feature vector
                    feature_vec = build_feature_vector(
                        title, artist, duration_ms, release_date, emotion, album_type,
                        st.session_state.embedder, st.session_state.scaler,
                        st.session_state.emotion_ohe, st.session_state.album_type_ohe
                    )
                    
                    # Predict
                    prediction = predict_popularity(feature_vec, st.session_state.model)
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p class="prediction-value">{prediction:.1f}</p>
                        <p class="prediction-label">Predicted Popularity Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate suggestions
                    st.subheader("💡 Improvement Suggestions")
                    suggestions = generate_suggestions(
                        title, artist, duration_ms, release_date, emotion, album_type,
                        prediction, st.session_state.embedder, st.session_state.scaler,
                        st.session_state.emotion_ohe, st.session_state.album_type_ohe,
                        st.session_state.model
                    )
                    
                    if suggestions:
                        for i, sugg in enumerate(suggestions, 1):
                            st.markdown(f"""
                            <div class="suggestion-card">
                                <strong>#{i} {sugg['type']}: {sugg['suggestion']}</strong><br>
                                <span style="color: #1DB954; font-weight: bold;">{sugg['impact']}</span><br>
                                <em>{sugg['reason']}</em>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Your track is already optimized! 🎉")
                    
                    # Spotify lookup (if available)
                    if sp:
                        st.subheader("🔎 Compare with Spotify")
                        if st.button("Search on Spotify", use_container_width=True):
                            try:
                                query = f"track:{title} artist:{artist}"
                                results = sp.search(q=query, type='track', limit=1)
                                if results['tracks']['items']:
                                    track = results['tracks']['items'][0]
                                    actual_pop = track['popularity']
                                    st.metric("Actual Spotify Popularity", actual_pop, 
                                             delta=f"{actual_pop - prediction:.1f} vs prediction")
                                    st.success(f"Found: {track['name']} by {track['artists'][0]['name']}")
                                else:
                                    st.warning("Track not found on Spotify")
                            except Exception as e:
                                st.error(f"Error searching Spotify: {e}")
    
    with col2:
        st.header("📈 Understanding Scores")
        st.metric("Excellent", "70-100", help="Highly popular tracks")
        st.metric("Good", "50-69", help="Above average popularity")
        st.metric("Average", "30-49", help="Moderate reach")
        st.metric("Low", "0-29", help="Limited visibility")
        
        st.markdown("---")
        st.info("""
        **💡 Tips for Higher Popularity:**
        - Release recent tracks
        - Collaborate with other artists
        - Optimize track duration (3-4 min)
        - Choose trending emotions
        - Build consistent release schedule
        """)

if __name__ == "__main__":
    main()
