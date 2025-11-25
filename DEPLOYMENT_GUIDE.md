# Spotify Popularity Predictor - Deployment Guide

## 🚀 Quick Deploy to Render (Free)

Your app is now ready to deploy! Follow these steps:

### Step 1: Create a GitHub Repository

1. Go to https://github.com and sign in
2. Click "New repository"
3. Name it: `spotify-popularity-predictor`
4. Make it **Public**
5. Click "Create repository"

### Step 2: Upload Your Files

Open PowerShell in your project folder and run:

```powershell
cd "C:\Users\Winte\OneDrive\Desktop\Spotify Data Collection 4"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Spotify Popularity Predictor"

# Link to GitHub (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/spotify-popularity-predictor.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Render

1. Go to https://render.com and sign up (free, no credit card)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - **Name**: `spotify-popularity-predictor`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn web_app:app`
   - **Plan**: `Free`

5. Click "Create Web Service"

### Step 4: Wait for Deployment (5-10 minutes)

Render will:
- Install all dependencies
- Load your ML model (450MB+)
- Start the web server

You'll get a URL like: `https://spotify-popularity-predictor.onrender.com`

## ⚠️ Important Notes

### Free Tier Limitations:
- **Cold starts**: App sleeps after 15 min of inactivity (takes 30-60 sec to wake up)
- **750 hours/month**: Limited monthly usage
- **No custom domain**: Gets a `.onrender.com` URL

### Large Model Size:
Your model artifacts are **~450MB**. This might:
- Take 5-10 minutes to deploy
- Use significant bandwidth on free tier
- Work fine but be patient during first load

### Spotify API Credentials:
Users will need to provide their own Spotify API credentials in the web form.

## 🎯 Alternative Options

### Option 1: Ngrok (Quickest for Testing)
```powershell
# Download ngrok from https://ngrok.com/download
# Extract and run:
.\ngrok http 5000
```
You'll get a temporary URL like `https://abc123.ngrok.io`

### Option 2: PythonAnywhere
- Go to https://www.pythonanywhere.com
- Free tier: Always-on, no cold starts
- Limited to 512MB storage (your model is 450MB - tight fit!)

### Option 3: Heroku
- Similar to Render but requires credit card verification
- More reliable but paid tiers start at $7/month

## 📝 Files Created for Deployment

- ✅ `requirements.txt` - Python dependencies
- ✅ `Procfile` - Tells Render how to start your app
- ✅ `web_app.py` - Updated with relative paths and PORT support

## 🔧 Troubleshooting

**If deployment fails:**
1. Check build logs on Render dashboard
2. Model files must be in `model_artifacts/` folder
3. Training data must be in `Spotify6/` folder

**If app is slow:**
- First load after sleep takes 30-60 seconds
- Model loading adds 10-20 seconds
- This is normal for free tier

## 🌟 Your Public URL

Once deployed, your URL will be:
`https://spotify-popularity-predictor.onrender.com`

Share this link with anyone!
