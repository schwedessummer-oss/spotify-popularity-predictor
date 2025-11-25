#!/usr/bin/env python3
"""
Spotify Emotion Track Collector — Links Only
Collect tracks per emotion from playlists and save per-emotion CSVs with Spotify links.
"""

import os, sys, json, time, argparse, getpass, random
from pathlib import Path
from itertools import islice
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import deque

# ---------------- CONFIG ----------------
DEFAULT_MIN_FOLLOWERS = 75_000  # Minimum playlist follower count
DEFAULT_TARGET = 5_000  # Target tracks per emotion
SEARCH_LIMIT = 50     # Spotify API maximum per request
SEARCH_PAGES = 4      # Number of search pages per keyword
MAX_PLAYLISTS = 100   # Maximum playlists to process per emotion
SLEEP = 0.5          # Base delay between API calls (can be raised dynamically)
BURST_LIMIT = 25     # Max requests before taking a longer break
BURST_COOLDOWN = 5   # Seconds to pause after hitting burst limit
MIN_PLAYLISTS = 20   # Minimum playlists before starting collection
REQUEST_COUNT = 0    # Global request counter
OUT_DIR = Path("output_links")
COLLECTED_FILE = OUT_DIR / "collected_ids.json"

# Per-minute safeguard: limit API calls in a rolling window
PER_MIN_LIMIT = 100              # Max requests per 60 seconds (adjustable)
REQUEST_TIMES = deque(maxlen=PER_MIN_LIMIT * 2)  # store timestamps of recent requests

# Jitter to avoid burst alignment
JITTER_MAX = 0.15

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

# ---------------- API SAFEGUARDS ----------------
def polite_sleep(base: float | None = None):
    """Sleep with a small random jitter to avoid thundering herd.
    If base is None, use SLEEP.
    """
    d = SLEEP if base is None else base
    time.sleep(d + random.uniform(0, JITTER_MAX))

def _cooldown_if_burst():
    """If we've made many requests quickly, take a short cooldown."""
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    if REQUEST_COUNT % BURST_LIMIT == 0:
        print(f"⏳ Burst limit reached ({REQUEST_COUNT} requests). Cooling down {BURST_COOLDOWN}s...")
        polite_sleep(BURST_COOLDOWN)

def _gate_per_minute():
    """Ensure we don't exceed PER_MIN_LIMIT in a rolling 60-second window."""
    now = time.time()
    # Drop timestamps older than 60 seconds
    while REQUEST_TIMES and now - REQUEST_TIMES[0] > 60:
        REQUEST_TIMES.popleft()
    # If at or above limit, wait until the window opens
    if len(REQUEST_TIMES) >= PER_MIN_LIMIT:
        wait_for = 60 - (now - REQUEST_TIMES[0])
        if wait_for > 0:
            print(f"⏳ Per-minute cap hit ({PER_MIN_LIMIT}/min). Sleeping {wait_for:.1f}s...")
            polite_sleep(wait_for)
    # Record this request
    REQUEST_TIMES.append(time.time())

def with_api(call, desc: str = "API call", max_retries: int = 6):
    """Execute an API call with throttling, jitter, and exponential backoff.
    Handles 429 rate limits using Retry-After when available.
    """
    backoff = 1.0
    throttled = 0
    for attempt in range(1, max_retries + 1):
        try:
            _gate_per_minute()
            result = call()
            _cooldown_if_burst()
            return result
        except spotipy.SpotifyException as e:
            status = getattr(e, "http_status", None)
            # Try to extract Retry-After if present
            retry_after = None
            headers = getattr(e, "http_headers", None) or getattr(e, "headers", None)
            if headers and isinstance(headers, dict):
                ra = headers.get("Retry-After") or headers.get("retry-after")
                try:
                    retry_after = float(ra) if ra is not None else None
                except Exception:
                    retry_after = None

            if status == 429:
                wait = max(2.0, (retry_after or backoff)) + random.uniform(0, JITTER_MAX)
                print(f"⚠️ 429 rate limited during {desc}. Waiting {wait:.1f}s (attempt {attempt}/{max_retries})")
                polite_sleep(wait)
                throttled += 1
                # After consecutive throttles, widen our safety margins temporarily
                if throttled >= 2:
                    # slow the baseline sleep and reduce per-minute window for a while
                    extra = min(30, 2 ** throttled)
                    print(f"🛡️ Increasing backoff proactively by {extra}s due to repeated throttling")
                    polite_sleep(extra)
                    # Nudge global pacing upwards for the remainder of the run
                    global SLEEP, PER_MIN_LIMIT
                    SLEEP = min(1.0, SLEEP + 0.1)
                    PER_MIN_LIMIT = max(60, PER_MIN_LIMIT - 5)
            else:
                wait = backoff + random.uniform(0, JITTER_MAX)
                print(f"⚠️ {status or 'Error'} on {desc}. Retrying in {wait:.1f}s (attempt {attempt}/{max_retries})")
                polite_sleep(wait)
            backoff = min(backoff * 2, 60)  # cap backoff growth
        except Exception as e:
            wait = backoff + random.uniform(0, JITTER_MAX)
            print(f"⚠️ Error on {desc}: {e}. Retrying in {wait:.1f}s (attempt {attempt}/{max_retries})")
            polite_sleep(wait)
            backoff = min(backoff * 2, 60)
    # Final attempt outside loop to raise if still failing
    _gate_per_minute()
    result = call()
    _cooldown_if_burst()
    return result

def sp_search(sp, *args, **kwargs):
    return with_api(lambda: sp.search(*args, **kwargs), desc="search")

def sp_playlist(sp, *args, **kwargs):
    return with_api(lambda: sp.playlist(*args, **kwargs), desc="playlist")

def sp_playlist_items(sp, *args, **kwargs):
    return with_api(lambda: sp.playlist_items(*args, **kwargs), desc="playlist_items")

def sp_next(sp, *args, **kwargs):
    return with_api(lambda: sp.next(*args, **kwargs), desc="next")

def sp_tracks(sp, *args, **kwargs):
    return with_api(lambda: sp.tracks(*args, **kwargs), desc="tracks")

# ---------------- AUTH BLOCK ----------------
def authenticate():
    """Obtain Spotify client credentials through manual input."""
    print("\n🔑 Spotify API Credentials")
    print("Please enter your Spotify API credentials:")
    
    # Always prompt for credentials
    cid = input("Enter Client ID: ").strip()
    try:
        secret = getpass.getpass("Enter Client Secret (hidden): ").strip()
    except Exception:
        secret = input("Enter Client Secret: ").strip()

    if not cid or not secret:
        print("❌ Both Client ID and Client Secret are required.")
        sys.exit(1)

    try:
        auth_mgr = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        sp = spotipy.Spotify(
            auth_manager=auth_mgr,
            requests_timeout=30,  # Increased timeout
            retries=5,           # More retries
            status_retries=5,    # Retry on rate limit (429)
            backoff_factor=2,    # Exponential backoff: 2, 4, 8, 16, 32 seconds
            status_forcelist=(429, 500, 502, 503, 504)  # Retry on these status codes
        )
        # Test the connection with a simple search
        sp_search(sp, q="test", limit=1, type="playlist")
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
        
        for _ in range(SEARCH_PAGES):
            try:
                # Search without quotes for broader matches
                res = sp_search(sp, q=kw, type="playlist", limit=SEARCH_LIMIT, offset=offset, market="US")
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
                        playlist = sp_playlist(sp, p["id"], fields="id,name,description,followers,owner")
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
                polite_sleep()
                
            except Exception as e:
                print(f"⚠️ Search error for '{kw}': {str(e)}")
                continue
    
    # Sort by priority (title matches first) then by followers
    return sorted(found.values(), key=lambda x: (-x.get("priority", 2), -x["followers"]))

def iter_tracks(sp, playlist_id):
    """Iterate through all tracks in a playlist."""
    try:
        res = sp_playlist_items(
            sp,
            playlist_id,
            fields="items.track.id,items.track.name,items.track.artists,items.track.album,items.track.popularity,items.track.duration_ms,items.track.is_local,next",
            additional_types=["track"], 
            limit=100)
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
                res = sp_next(sp, res)
                polite_sleep()
            else:
                break
        except Exception as e:
            print("⚠️ Error fetching next page:", e)
            break

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
    
    checkpoint_every = 1000
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
                
                # Skip if we've seen this track before
                if track_key in track_info:
                    continue
                    
                collected.append(tid)
                emotion_used.add(tid)
                track_info[track_key] = tid
                new += 1

                # Periodic checkpoint of raw collected IDs (not metadata yet)
                if len(collected) % checkpoint_every == 0:
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    tmp_file = OUT_DIR / f"{emotion}_checkpoint_ids.json"
                    with tmp_file.open("w", encoding="utf-8") as fh:
                        json.dump({"emotion": emotion, "ids": collected}, fh)
                    print(f"💾 Checkpoint saved ({len(collected)} IDs) -> {tmp_file}")
                
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
            res = sp_tracks(sp, chunk)
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
    polite_sleep()

    print("\n")  # Clear progress line
    
    df = pd.DataFrame(data)

    # Final checkpoint metadata save if large collection
    if len(df) >= checkpoint_every:
        meta_tmp = OUT_DIR / f"{emotion}_checkpoint_metadata.csv"
        df.to_csv(meta_tmp, index=False, encoding="utf-8-sig")
        print(f"💾 Metadata checkpoint saved -> {meta_tmp}")
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
    parser.add_argument("--emotion", choices=["happy","sad","energetic","love","all"], default=None,
                        help="Emotion to collect. If omitted, you'll be prompted (default: all)")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--min-followers", type=int, default=DEFAULT_MIN_FOLLOWERS)
    parser.add_argument("--safe-mode", action="store_true", help="Enable conservative pacing to avoid API limits")
    parser.add_argument("--per-minute", type=int, help="Override max requests per minute (default 100)")
    parser.add_argument("--sleep", type=float, help="Override base sleep between requests in seconds (default 0.5)")
    args = parser.parse_args(argv)

    # Prompt for emotion if not provided; default is 'all'
    if args.emotion is None:
        try:
            choice = input("Select emotion to collect [all/happy/sad/energetic/love] (default: all): ").strip().lower()
        except Exception:
            choice = "all"
        if choice not in {"all", "happy", "sad", "energetic", "love", ""}:
            print("Unrecognized choice; defaulting to 'all'.")
            choice = "all"
        args.emotion = "all" if choice == "" else choice

    sp = authenticate()
    collected = load_ids()
    # Apply rate overrides / safe mode
    global PER_MIN_LIMIT, SLEEP, MAX_PLAYLISTS, BURST_LIMIT
    if args.safe_mode:
        PER_MIN_LIMIT = 60
        SLEEP = max(0.8, SLEEP)
        MAX_PLAYLISTS = min(MAX_PLAYLISTS, 60)
        BURST_LIMIT = min(BURST_LIMIT, 15)
        print(f"🛡️ Safe mode: PER_MIN_LIMIT={PER_MIN_LIMIT}, SLEEP={SLEEP}, MAX_PLAYLISTS={MAX_PLAYLISTS}, BURST_LIMIT={BURST_LIMIT}")
    if args.per_minute:
        PER_MIN_LIMIT = max(30, int(args.per_minute))
    if args.sleep:
        SLEEP = max(0.3, float(args.sleep))

    emotions = [args.emotion] if args.emotion != "all" else list(EMOTION_KEYWORDS)
    
    print("\n🔍 Configuration:")
    print(f"Target tracks per emotion: {args.target:,}")
    print(f"Minimum playlist followers: {args.min_followers:,}")
    print(f"Rate: PER_MIN_LIMIT={PER_MIN_LIMIT}/min, SLEEP={SLEEP}s, BURST_LIMIT={BURST_LIMIT}")
    print(f"Emotions to collect: {', '.join(emotions)}")
    print("-" * 50)
    
    for e in emotions:
        try:
            df = collect_emotion(sp, e, args.target, args.min_followers, collected)
            if not df.empty:
                print(f"\n📊 Statistics for {e}:")
                print(f"Tracks collected: {len(df):,}")
                print(f"Unique artists: {df['artists'].nunique():,}")
                print(f"Average popularity: {df['popularity'].mean():.1f}")
                print(f"Average duration: {df['duration_ms'].mean() / 60000:.1f} minutes")
                print("-" * 50)
        except Exception as ex:
            print(f"❌ Error collecting {e}: {str(ex)}")
            
    save_ids(collected)
    print("\n🎉 All done! CSVs saved in 'output_links' directory.")

if __name__ == "__main__":
    main()