# Deploying to Streamlit Community Cloud

This guide will help you deploy your Spotify Popularity Predictor to **Streamlit Community Cloud** for free public access.

---

## Prerequisites

1. ✅ GitHub account
2. ✅ Your repository pushed to GitHub (already done: `schwedessummer-oss/spotify-popularity-predictor`)
3. ✅ Model artifacts in `model_artifacts/` folder (already committed)

---

## Deployment Steps

### 1. Sign Up for Streamlit Community Cloud

1. Go to https://streamlit.io/cloud
2. Click **"Sign up"** and authenticate with your GitHub account
3. Grant Streamlit access to your repositories

### 2. Create New App

1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository**: `schwedessummer-oss/spotify-popularity-predictor`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g., `spotify-popularity-predictor`)

### 3. Advanced Settings (Optional)

Click **"Advanced settings"** if you want to add:

- **Python version**: 3.11 (recommended)
- **Secrets**: If you want Spotify API integration:
  ```toml
  SPOTIPY_CLIENT_ID = "your_client_id"
  SPOTIPY_CLIENT_SECRET = "your_client_secret"
  ```

### 4. Deploy

1. Click **"Deploy!"**
2. Wait 5-10 minutes for initial build
3. Your app will be live at: `https://your-app-name.streamlit.app`

---

## Troubleshooting

### Build Fails - Memory Issues

If the build fails due to memory limits (sentence-transformers is large):

**Option 1: Use smaller embedding model**
Edit `app.py` and change:
```python
embed_model = "all-MiniLM-L6-v2"  # Smaller, faster model
```

Then retrain with the smaller model or update `embedder_info.txt`.

**Option 2: Request more resources**
Streamlit Community Cloud has resource limits. You may need to:
- Optimize model size
- Use Streamlit's paid tier for more resources
- Consider alternative deployment (see below)

### Deployment is Slow

First deploy takes 5-15 minutes due to:
- Installing PyTorch and sentence-transformers
- Downloading embedding models
- Loading large XGBoost model

Subsequent updates are faster (cached dependencies).

### App URL Not Loading

1. Check **"Manage app"** → **"Logs"** for errors
2. Verify all files are committed to GitHub
3. Ensure `model_artifacts/` folder is in the repo

---

## Alternative: Deploy Without Model Artifacts in Repo

If your model artifacts are too large for GitHub (>100MB):

### Option A: Git LFS (Large File Storage)

1. Install Git LFS:
   ```bash
   git lfs install
   git lfs track "*.joblib"
   git add .gitattributes
   git add model_artifacts/*.joblib
   git commit -m "Add model artifacts with LFS"
   git push
   ```

2. Streamlit Cloud supports Git LFS automatically.

### Option B: Download from External Storage

1. Upload model artifacts to Google Drive, Dropbox, or cloud storage
2. Add download code in `app.py`:
   ```python
   import gdown
   
   @st.cache_resource
   def download_model():
       gdown.download("YOUR_GOOGLE_DRIVE_LINK", "model_artifacts/xgb_model.joblib")
   ```

3. Add `gdown` to requirements.

---

## Updating Your Deployed App

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```
3. Streamlit auto-deploys on every push to `main` branch
4. Changes go live in 1-2 minutes

---

## Monitoring

- **Logs**: Click "Manage app" → "Logs" to see real-time output
- **Usage**: View visitor analytics in Streamlit dashboard
- **Reboot**: Click "Reboot app" if something goes wrong

---

## Custom Domain (Optional)

Streamlit Community Cloud provides:
- Free subdomain: `your-app.streamlit.app`
- Custom domain: Available on paid plans

---

## Cost

**Free tier includes:**
- ✅ 1 app deployed
- ✅ Unlimited public access
- ✅ Auto-redeploy on git push
- ✅ Community support

**Paid tier ($25/month):**
- Multiple apps
- Custom domains
- More resources (CPU/RAM)
- Priority support

---

## Test Your App Locally First

Before deploying, test locally:

```bash
cd "C:\Users\Winte\OneDrive\Desktop\Spotify Data Collection 4"
pip install -r requirements_streamlit.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Your Public URL

After deployment, your app will be accessible at:

**https://spotify-popularity-predictor.streamlit.app**

(or your custom subdomain)

Share this link with anyone - no login required!

---

## Support

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Community Forum: https://discuss.streamlit.io
- Report issues in your GitHub repo
