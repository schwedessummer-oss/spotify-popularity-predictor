# Deploy to PythonAnywhere - Step by Step Guide

## Why PythonAnywhere?
- Free tier, no credit card needed
- Always-on (no cold starts)
- Built specifically for Python web apps
- Much simpler than Render

## Step-by-Step Instructions:

### 1. Sign Up (2 minutes)
1. Go to: https://www.pythonanywhere.com/registration/register/beginner/
2. Choose a username (e.g., `schwedessummer`)
3. Enter email and password
4. Click "Register"

### 2. Upload Your Files (5 minutes)

**Option A - Via Web Interface:**
1. Click "Files" tab
2. Click "Upload a file"
3. Upload these files one by one:
   - `web_app.py`
   - `requirements.txt` 
   - `render.yaml` (not needed but won't hurt)
4. Create folder: `templates`
5. Upload `templates/index.html`
6. Create folder: `model_artifacts`
7. Upload all 4 files from `model_artifacts/`:
   - `xgb_popularity_model.joblib`
   - `scaler_num.joblib`
   - `emotion_ohe.joblib`
   - `album_type_ohe.joblib`

**Option B - Via Git (Faster):**
1. Click "Consoles" → "Bash"
2. Run:
```bash
git clone https://github.com/schwedessummer-oss/spotify-popularity-predictor.git
cd spotify-popularity-predictor
```

### 3. Install Dependencies (2 minutes)
1. Click "Consoles" → "Bash"
2. Run:
```bash
cd spotify-popularity-predictor
pip3.10 install --user flask spotipy joblib numpy pandas scikit-learn xgboost
```
(Skip sentence-transformers/torch - too big, we have fallback)

### 4. Set Up Web App (3 minutes)
1. Click "Web" tab
2. Click "Add a new web app"
3. Click "Next" (choose free domain)
4. Select "Flask"
5. Select "Python 3.10"
6. For path, enter: `/home/YOUR_USERNAME/spotify-popularity-predictor/web_app.py`
7. Click "Next"

### 5. Configure WSGI (1 minute)
1. On Web tab, click WSGI configuration file link
2. Delete all content
3. Replace with:
```python
import sys
path = '/home/YOUR_USERNAME/spotify-popularity-predictor'
if path not in sys.path:
    sys.path.append(path)

from web_app import app as application
```
4. Save (Ctrl+S)

### 6. Go Live!
1. Go back to "Web" tab
2. Click green "Reload" button
3. Your URL: `https://YOUR_USERNAME.pythonanywhere.com`

## Troubleshooting:

**If you see errors:**
1. Click "Error log" on Web tab
2. Check "Server log" on Web tab

**Common fixes:**
- Import errors: Install missing packages in Bash console
- Model not found: Check file paths match your username

## Your Public URL:
Once setup: `https://schwedessummer.pythonanywhere.com`

Share this with anyone!

---

**Note:** Free tier limits:
- 512 MB disk space (your app uses ~450MB - tight but works)
- One web app
- Always-on (no sleeping like Render)
