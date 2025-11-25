# Spotify Popularity Predictor - Web Interface

## 🚀 Quick Start

### 1. Install Flask
```powershell
pip install flask
```

### 2. Run the Web App
```powershell
python web_app.py
```

### 3. Open Your Browser
Go to: **http://localhost:5000**

---

## ✨ Features

### Beautiful Modern Interface
- **Gradient design** with smooth animations
- **Responsive layout** works on desktop and mobile
- **Real-time predictions** with loading states
- **Visual comparisons** between predictions

### Two Prediction Modes

#### 1. Basic Mode (No API needed)
- Enter song title and artist
- Get instant prediction using dataset averages
- Perfect for hypothetical tracks

#### 2. Enhanced Mode (With Spotify API)
- Check "Fetch real data from Spotify API"
- Enter your Spotify credentials
- Get actual track data automatically
- See **TWO predictions**:
  - With real metadata (accurate)
  - With defaults (baseline)
- View **data impact** (how much real data matters)
- Display **accuracy metrics** vs actual popularity
- Show **album art** and metadata
- Link directly to **Spotify player**

---

## 🎨 What You'll See

### With Spotify API Enabled:

```
🎵 Rise Up by Andra Day
[Album Art Image]

🤖 Prediction with Real Data
    72.5/100
    ✨ Solid Performer

📈 Model Accuracy
    Actual Spotify Popularity: 72/100
    Absolute Error: 0.5 pts
    Relative Error: 0.7%

┌─────────────────┬─────────────────┐
│  With Real Data │  With Defaults  │
│      72.5       │      45.2       │
└─────────────────┴─────────────────┘

💎 Real Data Impact
    +27.3 points
    ↗️ Real metadata significantly boosts prediction!

📋 Track Metadata
    Artist Popularity: 57/100
    Followers: 469,105
    Genres: soft pop
    Album Type: album
    Markets: 171
    Duration: 253.3s
    Explicit: No

[🎧 Listen on Spotify Button]
```

### Without Spotify API:

```
🎵 Summer Dreams by New Artist

🤖 Predicted Popularity
    42.3/100
    📈 Moderate Reach

Using dataset average values.
Enable Spotify API for more accurate predictions.
```

---

## 🔑 Getting Spotify API Credentials

1. Go to https://developer.spotify.com/dashboard
2. Log in with your Spotify account
3. Click "Create App"
4. Fill in:
   - **App Name**: My Popularity Predictor
   - **App Description**: Track popularity prediction tool
5. Click "Create"
6. Copy your **Client ID** and **Client Secret**
7. Paste them into the web form

**Note**: Credentials are only used for that prediction session, not stored.

---

## 📊 How It Works

1. **Enter track info** → song title, artist, emotion
2. **Optional**: Enable Spotify API to fetch real data
3. **Click Predict** → model analyzes 850+ features
4. **View results**:
   - Predicted popularity score (0-100)
   - Category rating
   - If using API: accuracy, impact analysis, metadata
   - Album artwork
   - Spotify streaming link

---

## 🎯 Benefits Over Command Line

### Visual Appeal
- Beautiful gradient design
- Smooth animations and transitions
- Album artwork display
- Color-coded results

### Easier Input
- Clean form fields
- Dropdown for emotions
- Toggle for Spotify API
- No typing commands

### Better Results Display
- Side-by-side comparisons
- Visual impact indicators
- Organized metadata grids
- Interactive elements

### User-Friendly
- No terminal knowledge needed
- Works in any browser
- Shareable (can access from other devices)
- Mobile responsive

---

## 🛠️ Advanced Usage

### Run on Network
To access from other devices on your network:

```powershell
python web_app.py
```

Then from another device go to: `http://YOUR_IP:5000`

### Keep Running in Background
Use PowerShell:
```powershell
Start-Process python -ArgumentList "web_app.py" -WindowStyle Hidden
```

### Stop the Server
In the terminal where it's running, press: **Ctrl+C**

---

## 🎨 Customization

The design uses:
- **Primary Colors**: Purple gradient (#667eea to #764ba2)
- **Spotify Green**: #1DB954
- **Clean whites** and subtle grays
- **Rounded corners** (10-20px radius)
- **Soft shadows** for depth

To customize colors, edit `templates/index.html` CSS section.

---

## 📱 Screenshots

### Main Form
- Clean input fields for title and artist
- Emotion dropdown with emojis
- Spotify API toggle
- Big purple predict button

### Results (With Spotify)
- Large prediction score display
- Accuracy comparison
- Baseline vs real data comparison
- Beautiful metadata grid
- Green Spotify button

### Results (Without Spotify)
- Simple prediction display
- Category badge
- Hint to enable API

---

## 🔧 Troubleshooting

**"Module not found" error**:
```powershell
pip install flask spotipy sentence-transformers
```

**Port already in use**:
Edit `web_app.py` line at the bottom:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to 5001
```

**Spotify API not working**:
- Check credentials are correct
- Ensure app is created in Spotify Dashboard
- Check internet connection

---

## 🎉 Enjoy!

You now have a beautiful, professional-grade web interface for your ML model!

Perfect for:
- **Demos** to friends and colleagues
- **Portfolio** projects
- **Production use** for artists/labels
- **Research presentations**
