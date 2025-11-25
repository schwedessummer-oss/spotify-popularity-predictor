#!/usr/bin/env python3
"""
Spotify Emotion Track Collector — Links Only
Collect 100 unique tracks per emotion from playlists (≥250k followers)
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
DEFAULT_TARGET = 10_000  # Target tracks per emotion (maximum of range)
SEARCH_LIMIT = 50     # Spotify API maximum per request
SEARCH_PAGES = 6      # Number of search pages per keyword (increased for more tracks)
MAX_PLAYLISTS = 150   # Maximum playlists to process per emotion (increased for more tracks)
SLEEP = 0.3          # Delay between API calls
MIN_PLAYLISTS = 20    # Minimum playlists before starting collection
OUT_DIR = Path("output_links")
COLLECTED_FILE = OUT_DIR / "collected_ids.json"

EMOTION_KEYWORDS = {
    "happy": [
        "happy", "happy hits", "happy mood", "feel good", "good mood",
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
        # Verify credentials were saved correctly
        load_dotenv()
        saved_cid = os.getenv("SPOTIPY_CLIENT_ID")
        saved_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        if saved_cid != cid or saved_secret != secret:
            print("⚠️ Warning: Credentials may not have saved correctly")

    try:
        auth_mgr = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        sp = spotipy.Spotify(auth_manager=auth_mgr, requests_timeout=15, retries=3)
        # Test the connection with a simple search
        sp.search(q="test", limit=1, type="playlist")
        print("✅ Authenticated to Spotify API")
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        sys.exit(1)
    return sp

# ---------------- PLAYLIST SEARCH ----------------
def search_playlists(sp, keywords, min_followers):
    found = {}

    for kw in keywords:
        print(f"\nSearching for playlists with keyword '{kw}'...")
        offset = 0
        
        for page in range(SEARCH_PAGES):
            try:
                # Search without quotes to get broader matches
                res = sp.search(
                    q=kw, 
                    type="playlist", 
                    limit=SEARCH_LIMIT, 
                    offset=offset,
                    market="US"
                )
                
                if not res or "playlists" not in res:
                    break
                    
                playlists = res["playlists"]["items"]
                if not playlists:
                    break
                    
                for p in playlists:
                    if not p or "id" not in p or p["id"] in found:
                        continue
                        
                    try:
                        # Get full playlist details including description
                        playlist = sp.playlist(p["id"], fields="id,name,description,followers,owner")
                        name = playlist.get("name", "").lower()
                        description = playlist.get("description", "").lower()
                        followers = playlist.get("followers", {}).get("total", 0)
                        
                        # Skip official Spotify playlists
                        if playlist.get("owner", {}).get("id") == "spotify":
                            continue
                            
                        # Accept if keyword is in name OR description and has enough followers
                        kw_lower = kw.lower()
                        if followers >= min_followers and (
                            kw_lower in name or 
                            kw_lower in description or
                            any(k.lower() in description for k in keywords)
                        ):
                            # Priority based on where the match was found
                            priority = 1 if kw_lower in name else 2
                            
                            found[p["id"]] = {
                                "id": p["id"],
                                "name": playlist["name"],
                                "followers": followers,
                                "owner": playlist["owner"]["display_name"],
                                "priority": priority
                            }
                            match_type = "title" if kw_lower in name else "description"
                            print(f"Found playlist (match in {match_type}): {playlist['name']} ({followers:,} followers)")
                            
                    except Exception as e:
                        print(f"⚠️ Error getting playlist {p['id']}: {str(e)}")
                        continue
                
                offset += len(playlists)
                time.sleep(SLEEP)
                
            except Exception as e:
                print(f"⚠️ Search error for '{kw}': {str(e)}")
                continue
                
    # Sort by priority (title matches first) then by followers
    return sorted(found.values(), key=lambda x: (-x.get("priority", 2), -x["followers"]))
                
        except Exception as e:
            print(f"⚠️ Search error for '{kw}': {str(e)}")
            
    # Sort by priority (exact matches first) then by followers
    return sorted(found.values(), key=lambda x: (-x.get("priority", 2), -x["followers"]))

# ---------------- TRACKS ----------------
def iter_tracks(sp, playlist_id):
    try:
        res = sp.playlist_items(
            playlist_id,
            fields="items.track.id,items.track.name,items.track.artists,items.track.album,items.track.popularity,items.track.duration_ms,items.track.is_local,next",
            additional_types=["track"], limit=100)
        if not res:
            print("⚠️ No response from playlist_items")
            return
    except Exception as e:
        print("⚠️ playlist_items error:", e)
        return
    
    while res:
        items = res.get("items")
        if not items:
            print("⚠️ No items in playlist response")
            break
            
        for item in items:
            if item and isinstance(item, dict):
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
    """Collect tracks for a specific emotion up to the target amount."""
    kws = EMOTION_KEYWORDS[emotion]
    print(f"\n🎧 Collecting '{emotion}' ({len(kws)} keywords, target {target:,} tracks)...")
    start_time = time.time()
    
    # First try with original follower count
    playlists = search_playlists(sp, kws, min_followers)
    total_searched = len(playlists)
    
    # If we don't have enough playlists, try with lower follower count
    if len(playlists) < MIN_PLAYLISTS:
        print(f"⚠️ Only found {len(playlists)} playlists with {min_followers:,} followers, trying with lower threshold...")
        more_playlists = search_playlists(sp, kws, min_followers // 2)
        playlists.extend(p for p in more_playlists if p["id"] not in {pl["id"] for pl in playlists})
        total_searched += len(more_playlists)
        
    if not playlists:
        print(f"❌ No suitable playlists found for {emotion}")
        return pd.DataFrame()
    
    print(f" → Found {len(playlists)} unique playlists from {total_searched} total results")
    
    # Track IDs already used in any emotion
    global_used = {tid for ids in collected_map.values() for tid in ids}
    emotion_used = set(collected_map.get(emotion, []))
    collected = []
    track_info = {}  # Store track info for deduplication
    
    # Process more playlists to get enough tracks
    max_attempts = min(len(playlists), MAX_PLAYLISTS)
    attempts = 0
    last_progress = 0
    progress_interval = max(1, target // 20)  # Show progress every 5%
    
    for pl in playlists:
        if len(collected) >= target:
            break
            
        if attempts >= max_attempts:
            print(f"\n⚠️ Reached maximum playlist attempts ({max_attempts})")
            break
            
        attempts += 1
        pid, name = pl["id"], pl["name"]
        print(f"\nScanning playlist {attempts}/{max_attempts}: {name} ({pl['followers']:,} followers)")
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
                
                # Check for duplicate tracks by name and artist within this emotion
                track_name = t.get("name", "").lower()
                artists = [a.get("name", "").lower() for a in t.get("artists", [])]
                track_key = f"{track_name}|{'|'.join(sorted(artists))}"
                
                # Skip if we've seen this track before in this emotion
                if track_key in track_info:
                    continue
                    
                collected.append(tid)
                emotion_used.add(tid)
                track_info[track_key] = tid
                new += 1
                
                # Show progress every 5%
                if len(collected) - last_progress >= progress_interval:
                    elapsed = time.time() - start_time
                    tracks_per_sec = len(collected) / elapsed
                    eta = (target - len(collected)) / tracks_per_sec if tracks_per_sec > 0 else 0
                    print(f"\rProgress: {len(collected):,}/{target:,} tracks ({len(collected)/target*100:.1f}%) - ETA: {eta/60:.1f} minutes", end="")
                    last_progress = len(collected)
                
            if new > 0:
                print(f"\n   +{new:,} new tracks from this playlist")
            
        except Exception as e:
            print(f"\n⚠️ Error processing playlist: {str(e)}")
            continue
            
    print("\n")  # Clear progress line
    
    if not collected:
        print(f"❌ No tracks found for {emotion}")
        return pd.DataFrame()

    print(f"Fetching metadata for {len(collected):,} tracks...")
    data = []
    metadata_progress = 0
    
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
                metadata_progress += 1
                if metadata_progress % 500 == 0:
                    print(f"\rFetched metadata for {metadata_progress:,}/{len(collected):,} tracks", end="")
        except Exception as e:
            print(f"\n⚠️ Error fetching metadata: {str(e)}")
        time.sleep(SLEEP)

    print("\n")  # Clear progress line
    
    df = pd.DataFrame(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{emotion}_tracks.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    
    elapsed = time.time() - start_time
    tracks_per_min = len(df) / (elapsed / 60)
    print(f"✅ Saved {len(df):,} tracks → {out}")
    print(f"   Collection rate: {tracks_per_min:.1f} tracks/minute")

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

if __name__ == "__main__":
    main()
