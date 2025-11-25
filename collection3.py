#!/usr/bin/env python3
"""
Spotify Emotion Track Collector - Links Only
Collect 100 unique tracks per emotion from playlists (>=250k followers)
and save per-emotion CSVs with Spotify links.
"""

import os, sys, json, time, argparse, getpass
from pathlib import Path
from itertools import islice
from dotenv import load_dotenv
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# ---------------- CONFIG ----------------
DEFAULT_MIN_FOLLOWERS = 250_000
DEFAULT_TARGET = 100
SEARCH_LIMIT = 50
MIN_PLAYLISTS = 10  # Minimum number of playlists to find before starting collection
SLEEP = 0.3
OUT_DIR = Path("output_links")
COLLECTED_FILE = OUT_DIR / "collected_ids.json"

EMOTION_KEYWORDS = {
    "happy": [
        "happy vibes", "happy hits", "happy mood", "feel good", "good mood",
        "upbeat", "good vibes", "sunny day", "joy", "cheerful", "smile",
        "happiness", "happy songs", "summer vibes", "positive vibes",
        "feel good music", "mood booster", "happy playlist"
    ],
    "sad": [
        "sad vibes", "sad songs", "heartbreak", "melancholy", "crying",
        "lonely nights", "tears", "sad playlist", "breakup songs", "emotional",
        "feeling blue", "in my feelings", "sad hits", "rainy day", "missing you",
        "sad hours", "depressed", "broken heart"
    ],
    "energetic": [
        "workout hits", "party playlist", "hype music", "gym motivation",
        "energetic vibes", "pump up", "dance hits", "beast mode", "cardio",
        "power workout", "training", "fitness motivation", "high energy",
        "running playlist", "party anthems", "club hits", "adrenaline",
        "motivation mix"
    ],
    "love": [
        "love songs", "romantic hits", "romance playlist", "wedding songs",
        "slow dance", "valentine's day", "date night", "romantic evening",
        "love hits", "couples playlist", "romantic dinner", "sweet romance",
        "love ballads", "romantic classics", "romantic vibes", "soulmate songs",
        "passion", "forever love"
    ],
}

# ---------------- HELPERS ----------------
def chunked(it, size):
    it = iter(it)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk

def load_ids():
    if COLLECTED_FILE.exists():
        try:
            return json.loads(COLLECTED_FILE.read_text())
        except Exception:
            return {}
    return {}

def save_ids(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COLLECTED_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ---------------- AUTH BLOCK ----------------
def authenticate():
    """Obtain Spotify client credentials (from .env or prompt)."""
    load_dotenv()
    cid = os.getenv("SPOTIPY_CLIENT_ID") or os.getenv("CLIENT_ID")
    secret = os.getenv("SPOTIPY_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

    # Prompt interactively if missing
    if not cid:
        cid = input("Enter SPOTIPY_CLIENT_ID: ").strip() or None
    if not secret:
        try:
            secret = getpass.getpass("Enter SPOTIPY_CLIENT_SECRET (hidden): ").strip() or None
        except Exception:
            secret = input("Enter SPOTIPY_CLIENT_SECRET: ").strip() or None

    if not cid or not secret:
        print("❌ Missing Spotify credentials. Set them in .env or enter manually.")
        sys.exit(1)

    # Offer to save credentials
    try:
        save_choice = input("Save credentials to .env for future runs? [y/N]: ").strip().lower()
    except Exception:
        save_choice = "n"
    if save_choice == "y":
        Path(".env").write_text(f"SPOTIPY_CLIENT_ID={cid}\nSPOTIPY_CLIENT_SECRET={secret}\n", encoding="utf-8")
        print("✅ Credentials saved to .env (keep private).")

    try:
        auth_mgr = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        sp = spotipy.Spotify(auth_manager=auth_mgr, requests_timeout=15, retries=3)
        # Test the connection with a simple search
        sp.search(q="test", limit=1, type="playlist")
        print("✅ Authenticated to Spotify API")
        return sp
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        sys.exit(1)

# ---------------- PLAYLIST SEARCH ----------------
def search_playlists(sp, keywords, min_followers):
    found = {}
    total_attempts = 0
    max_total_attempts = len(keywords) * 3  # Allow multiple search types per keyword

    for kw in keywords:
        if total_attempts >= max_total_attempts:
            print(f"\n⚠️ Reached maximum total search attempts ({max_total_attempts})")
            break

        print(f"\nSearching for playlists with keyword '{kw}'...")
        try:
            # 1. First try exact phrase in title (with quotes)
            total_attempts += 1
            res = sp.search(q=f'"{kw}"', type="playlist", limit=SEARCH_LIMIT, market="US")
            if res and "playlists" in res:
                for p in res["playlists"]["items"]:
                    if not p or "id" not in p or p["id"] in found:
                        continue
                    
                    try:
                        playlist = sp.playlist(p["id"], fields="id,name,description,followers,owner")
                        name = playlist["name"].lower()
                        followers = playlist["followers"]["total"]
                        
                        if followers >= min_followers and kw.lower() in name:
                            found[p["id"]] = {
                                "id": p["id"],
                                "name": playlist["name"],
                                "followers": followers,
                                "owner": playlist["owner"]["display_name"],
                                "priority": 1  # Highest priority for exact phrase matches
                            }
                            print(f"Found playlist (exact phrase): {playlist['name']} ({followers:,} followers)")
                    except Exception as e:
                        print(f"Error getting playlist {p['id']}: {str(e)}")
            time.sleep(SLEEP)
            
            # 2. If needed, try partial matches in title
            if len(found) < MIN_PLAYLISTS:
                total_attempts += 1
                # Search without quotes for more flexible matching
                res = sp.search(q=kw, type="playlist", limit=SEARCH_LIMIT, market="US")
                if res and "playlists" in res:
                    for p in res["playlists"]["items"]:
                        if not p or "id" not in p or p["id"] in found:
                            continue
                            
                        try:
                            playlist = sp.playlist(p["id"], fields="id,name,description,followers,owner")
                            name = playlist["name"].lower()
                            followers = playlist["followers"]["total"]
                            
                            # Check if any word from the keyword is in the title
                            if followers >= min_followers and any(word.lower() in name for word in kw.split()):
                                found[p["id"]] = {
                                    "id": p["id"],
                                    "name": playlist["name"],
                                    "followers": followers,
                                    "owner": playlist["owner"]["display_name"],
                                    "priority": 2  # Medium priority for partial matches
                                }
                                print(f"Found playlist (partial match): {playlist['name']} ({followers:,} followers)")
                        except Exception as e:
                            print(f"Error getting playlist {p['id']}: {str(e)}")
                time.sleep(SLEEP)
                
                # 3. If still not enough, try description search
                if len(found) < MIN_PLAYLISTS:
                    total_attempts += 1
                    for p in res["playlists"]["items"]:
                        if not p or "id" not in p or p["id"] in found:
                            continue
                            
                        try:
                            playlist = sp.playlist(p["id"], fields="id,name,description,followers,owner")
                            followers = playlist["followers"]["total"]
                            description = (playlist.get("description") or "").lower()
                            
                            # Check if keyword appears in description
                            if followers >= min_followers and kw.lower() in description:
                                found[p["id"]] = {
                                    "id": p["id"],
                                    "name": playlist["name"],
                                    "followers": followers,
                                    "owner": playlist["owner"]["display_name"],
                                    "priority": 3  # Lower priority for description matches
                                }
                                print(f"Found playlist (description match): {playlist['name']} ({followers:,} followers)")
                        except Exception as e:
                            print(f"Error getting playlist {p['id']}: {str(e)}")
                    time.sleep(SLEEP)
                
        except Exception as e:
            print(f"⚠️ Search error for '{kw}': {str(e)}")
            
    # Sort by priority (1=exact title, 2=partial title, 3=description) then by followers
    return sorted(found.values(), key=lambda x: (-x.get("priority", 3), -x["followers"]))

# ---------------- TRACKS ----------------
def iter_tracks(sp, playlist_id):
    try:
        res = sp.playlist_items(
            playlist_id,
            fields="items.track.id,items.track.name,items.track.artists,items.track.album,items.track.popularity,items.track.duration_ms,items.track.is_local,next",
            additional_types=["track"],
            limit=100
        )
        if not res:
            print("⚠️ No response from playlist_items")
            return
    except Exception as e:
        print("⚠️ playlist_items error:", e)
        return

    while res:
        items = res.get("items", [])
        if not items:
            break

        for item in items:
            if item and item.get("track"):
                yield item

        try:
            if res.get("next"):
                res = sp.next(res)
                time.sleep(SLEEP)
            else:
                break
        except Exception as e:
            print("⚠️ Error fetching next page:", e)
            break

# ---------------- COLLECTOR ----------------
def collect_emotion(sp, emotion, target, min_followers, collected_map):
    kws = EMOTION_KEYWORDS[emotion]
    print(f"\n🎧 Collecting '{emotion}' ({len(kws)} keywords, target {target})...")
    
    # First try with original follower count
    playlists = search_playlists(sp, kws, min_followers)
    
    if not playlists:
        print(f"⚠️ No playlists found with {min_followers:,} followers, trying with lower threshold...")
        playlists = search_playlists(sp, kws, min_followers // 2)
        
    if not playlists:
        print(f"❌ No suitable playlists found for {emotion}")
        return pd.DataFrame()
    
    print(f" → {len(playlists)} playlists found")
    
    # Track IDs already used in any emotion
    global_used = {tid for ids in collected_map.values() for tid in ids}
    emotion_used = set(collected_map.get(emotion, []))
    collected = []
    track_info = {}  # Store track info for deduplication
    
    # Try up to 20 playlists to find enough tracks
    max_attempts = min(len(playlists), 20)
    attempts = 0
    
    for pl in playlists:
        if len(collected) >= target:
            break
            
        if attempts >= max_attempts:
            print(f"⚠️ Reached maximum playlist attempts ({max_attempts})")
            break
            
        attempts += 1
        pid, name = pl["id"], pl["name"]
        print(f"Scanning playlist: {name} ({pl['followers']:,} followers)")
        new = 0
        
        try:
            for item in iter_tracks(sp, pid):
                if len(collected) >= target:
                    break
                    
                t = item.get("track")
                if not t or t.get("is_local"):
                    continue
                    
                tid = t.get("id")
                if not tid or tid in global_used or tid in emotion_used:
                    continue
                
                # Check for duplicate tracks by name and artist
                track_name = t.get("name", "").lower()
                artists = [a.get("name", "").lower() for a in t.get("artists", [])]
                track_key = f"{track_name}|{'|'.join(sorted(artists))}"
                
                # Skip if we've seen this track before
                if track_key in track_info:
                    continue
                    
                collected.append(tid)
                emotion_used.add(tid)
                track_info[track_key] = tid
                new += 1
                
            print(f"   +{new} new tracks")
            
        except Exception as e:
            print(f"⚠️ Error processing playlist: {str(e)}")
            continue
            
    if not collected:
        print(f"❌ No tracks found for {emotion}")
        return pd.DataFrame()

    collected = collected[:target]
    print(f"Fetching metadata for {len(collected)} tracks...")
    data = []
    
    for chunk in chunked(collected, 50):
        try:
            res = sp.tracks(chunk)
            for t in res.get("tracks", []) or []:
                if not t:
                    continue
                tid = t.get("id")
                data.append({
                    "track_id": tid,
                    "name": t.get("name"),
                    "artists": ", ".join([a["name"] for a in t.get("artists", [])]),
                    "album": t.get("album", {}).get("name"),
                    "album_release_date": t.get("album", {}).get("release_date"),
                    "popularity": t.get("popularity"),
                    "duration_ms": t.get("duration_ms"),
                    "spotify_link": f"https://open.spotify.com/track/{tid}",
                    "emotion": emotion
                })
        except Exception as e:
            print("⚠️ Error fetching metadata:", e)
        time.sleep(SLEEP)

    df = pd.DataFrame(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{emotion}_tracks.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} tracks → {out}")

    collected_map.setdefault(emotion, [])
    collected_map[emotion] = list(set(collected_map[emotion] + collected))
    return df

# ---------------- MAIN ----------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", choices=["happy","sad","energetic","love","all"], default="all")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--min-followers", type=int, default=DEFAULT_MIN_FOLLOWERS)
    args = parser.parse_args(argv)

    sp = authenticate()
    collected = load_ids()
    emotions = [args.emotion] if args.emotion != "all" else list(EMOTION_KEYWORDS)
    for e in emotions:
        try:
            collect_emotion(sp, e, args.target, args.min_followers, collected)
        except Exception as ex:
            print("Error collecting", e, ":", ex)
    save_ids(collected)
    print("\n🎉 All done. CSVs saved in 'output_links'.")