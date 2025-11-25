# Spotify Popularity Predictor - User Guide

## Features

✅ **Predict Track Popularity**: Enter song title + artist → get 0-100 popularity score  
✅ **Real Accuracy Metrics**: Fetch actual Spotify data to compare prediction vs reality  
✅ **Smart Suggestions**: Get 7+ actionable recommendations to boost popularity  
✅ **Two Modes**: Interactive (step-by-step) or Batch (CSV processing)

## Quick Start

### Interactive Mode (Recommended)

```powershell
python predict_popularity.py
```

**Steps:**
1. **Spotify API Setup** (optional but recommended):
   - Enter your Spotify Client ID and Secret
   - Or press Enter twice to skip (prediction-only mode)
   
2. **Enter Track Details**:
   - Song title
   - Artist name
   
3. **If Spotify API enabled**:
   - Tool automatically searches for the track
   - Uses real metadata (artist popularity, followers, genres, etc.)
   - Shows **actual popularity** vs **predicted popularity**
   - Displays **accuracy metrics** (absolute error, % error)
   
4. **Get Results**:
   - Predicted popularity score (0-100)
   - Category rating (Hit Potential, Solid Performer, etc.)
   - Improvement suggestions ranked by impact

### Batch Mode

Create a CSV file with columns: `title`, `artist`, and optionally: `emotion`, `artist_popularity`, `artist_followers`, `genres`, `album_type`, `explicit`

```powershell
python predict_popularity.py your_tracks.csv
```

Output: `your_tracks_predictions.csv`

## Example Output (With Spotify API)

```
🔍 Searching Spotify for track data...
✅ Found: 'Blinding Lights' by The Weeknd
Use actual Spotify data for prediction? (y/n) [y]: y

======================================================================
PREDICTION RESULTS
======================================================================

🎵 'Blinding Lights' by The Weeknd

📊 ACTUAL Popularity:    95/100
🤖 PREDICTED Popularity: 92.3/100

📈 Model Accuracy:
   Absolute Error: 2.7 points
   Relative Error: 2.8%
   🎯 Excellent prediction!

📋 Actual Track Metadata:
   Artist Popularity: 97
   Followers: 117,583,920
   Genres: pop, canadian pop, dance pop
   Album Type: album
   Markets: 184
   Duration: 200.5s
   Explicit: No
   🔗 https://open.spotify.com/track/...

   Category: 🔥 HIT POTENTIAL

----------------------------------------------------------------------
💡 IMPROVEMENT SUGGESTIONS
----------------------------------------------------------------------

1. [Distribution] Expand to 185+ markets
   Expected Impact: +1.2 points
   Why: Wider distribution increases potential audience reach

2. [Emotion/Vibe] Position as 'energetic' track
   Expected Impact: +0.8 points
   Why: 'Energetic' tracks in your genre tend to perform better
```

## Example Output (Without Spotify API)

```
======================================================================
PREDICTION RESULTS
======================================================================

🎵 'Summer Dreams' by New Artist

📊 Predicted Popularity: 38.5/100

   Category: 📈 MODERATE REACH

----------------------------------------------------------------------
💡 IMPROVEMENT SUGGESTIONS
----------------------------------------------------------------------

1. [Artist Growth] Grow artist popularity to 70+
   Expected Impact: +18.5 points
   Why: Higher artist popularity strongly predicts track success

2. [Distribution] Expand to 180+ markets
   Expected Impact: +8.2 points
   Why: Wider distribution increases potential audience reach

3. [Artist Strategy] Add featured artist collaboration
   Expected Impact: +5.3 points
   Why: Collaborations increase discovery and cross-audience appeal

4. [Genre Strategy] Incorporate 'pop' elements
   Expected Impact: +3.7 points
   Why: 'Pop' genre signals improve algorithmic placement

5. [Title Optimization] Add featured artist to title
   Expected Impact: +2.1 points
   Why: Title format affects discoverability and click-through
```

## Optimization Categories

The tool tests and suggests improvements across:

1. **Emotion/Vibe**: Position track as happy/sad/energetic/love
2. **Artist Strategy**: Add collaborations, grow artist base
3. **Distribution**: Expand to more markets
4. **Title Optimization**: Add feat., shorten, add descriptors
5. **Genre Strategy**: Incorporate high-performing genre elements
6. **Release Format**: Single vs album vs compilation
7. **Artist Growth**: Target artist popularity benchmarks

## Model Performance

- **R² Score**: 0.8050 (80.5% variance explained)
- **MAE**: 7.35 points
- **Training Set**: 38,022 unique tracks across 4 emotions
- **Features**: 851 total (embeddings + metadata + genres)

## Getting Spotify API Credentials

1. Go to https://developer.spotify.com/dashboard
2. Log in with your Spotify account
3. Click "Create App"
4. Fill in app name and description
5. Copy your **Client ID** and **Client Secret**
6. Use these when running the predictor

## Tips

- **With Spotify API**: Most accurate predictions using real artist/track metadata
- **Without Spotify API**: Still useful for hypothetical tracks or when API unavailable
- **Manual mode**: Good for unreleased tracks or "what-if" scenarios
- **Batch mode**: Process multiple tracks at once for efficiency

## Technical Details

- Uses trained XGBoost model from `model_artifacts/`
- Text embeddings: SentenceTransformer (all-mpnet-base-v2)
- Feature engineering: Multi-hot genres, scaled numerics, one-hot categoricals
- Prediction range clamped to 0-100 (inverse log1p transformation)
