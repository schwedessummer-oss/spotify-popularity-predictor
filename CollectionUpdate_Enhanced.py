#!/usr/bin/env python3
"""
Spotify Emotion Track Collector — Enhanced with Artist & Album Metadata
Collects additional features to improve model accuracy:
- Artist popularity, followers, genres
- Album type, total tracks
- Track explicit flag, available markets count
"""
import os, sys, json, time, argparse, getpass, random
from pathlib import Path
from itertools import islice
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import deque

# ---------------- CONFIG ----------------
DEFAULT_MIN_FOLLOWERS = 50_000
DEFAULT_TARGET = 10_000
SEARCH_LIMIT = 50
SEARCH_PAGES = 3
MAX_PLAYLISTS = 100
SLEEP = 0.5
BURST_LIMIT = 25
BURST_COOLDOWN = 5
MIN_PLAYLISTS = 20
REQUEST_COUNT = 0
OUT_DIR = Path(r"C:\Users\Winte\OneDrive\Desktop\Spotify6")
COLLECTED_FILE = OUT_DIR / "collected_ids.json"

# Per-minute safeguard: token bucket implementation
PER_MIN_LIMIT = 100
_token_bucket = {
    "capacity": PER_MIN_LIMIT,
    "tokens": PER_MIN_LIMIT,
    "last_refill": time.time(),
    "refill_per_sec": PER_MIN_LIMIT / 60.0
}

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
        "love", "romance", "romantic", "passion", "intimate",
        "crush", "falling in love", "soulmate", "lovers", "devotion",
        "affection", "adore", "sweetheart", "beloved", "together",
        "forever love", "true love", "hearts", "chemistry"
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

# ---------------- RATE LIMITING (token bucket) ----------------
def configure_token_bucket(per_minute: int):
    global _token_bucket
    per_minute = max(1, int(per_minute))
    _token_bucket = {
        "capacity": per_minute,
        "tokens": per_minute,
        "last_refill": time.time(),
        "refill_per_sec": per_minute / 60.0
    }

def _refill_tokens():
    now = time.time()
    tb = _token_bucket
    elapsed = now - tb["last_refill"]
    if elapsed <= 0:
        return
    added = elapsed * tb["refill_per_sec"]
    if added > 0:
        tb["tokens"] = min(tb["capacity"], tb["tokens"] + added)
        tb["last_refill"] = now

def _acquire_token(block=True):
    _refill_tokens()
    tb = _token_bucket
    if tb["tokens"] >= 1:
        tb["tokens"] -= 1
        return True
    if not block:
        return False
    secs = (1 - tb["tokens"]) / tb["refill_per_sec"]
    if secs < 0:
        secs = 0.1
    sleep_for = secs + random.uniform(0, JITTER_MAX)
    print(f"⏳ Token bucket empty. Sleeping {sleep_for:.2f}s to respect {tb['capacity']}/min limit")
    time.sleep(sleep_for)
    _refill_tokens()
    if tb["tokens"] >= 1:
        tb["tokens"] -= 1
        return True
    return False

def polite_sleep(base: float | None = None):
    d = SLEEP if base is None else base
    time.sleep(d + random.uniform(0, JITTER_MAX))

def _cooldown_if_burst():
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    if REQUEST_COUNT % BURST_LIMIT == 0:
        print(f"⏳ Burst limit reached ({REQUEST_COUNT} requests). Cooling down {BURST_COOLDOWN}s...")
        polite_sleep(BURST_COOLDOWN)

# ---------------- API WRAPPER WITH BACKOFF ----------------
def with_api(call, desc: str = "API call", max_retries: int = 6):
    backoff = 1.0
    throttled = 0
    for attempt in range(1, max_retries + 1):
        try:
            _acquire_token(block=True)
            result = call()
            _cooldown_if_burst()
            return result
        except spotipy.SpotifyException as e:
            status = getattr(e, "http_status", None)
            retry_after = None
            headers = getattr(e, "http_headers", None) or getattr(e, "headers", None)
            if headers and isinstance(headers, dict):
                ra = headers.get("Retry-After") or headers.get("retry-after")
                try:
                    retry_after = float(ra) if ra is not None else None
                except Exception:
                    retry_after = None

            if status == 429:
                wait = max(1.5, (retry_after or backoff)) + random.uniform(0, JITTER_MAX)
                print(f"⚠️ 429 rate limited during {desc}. Waiting {wait:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                throttled += 1
                tb = _token_bucket
                new_cap = max(10, int(tb["capacity"] * 0.8))
                if new_cap < tb["capacity"]:
                    print(f"🛡️ Reducing token bucket capacity from {tb['capacity']} to {new_cap} due to throttling")
                    configure_token_bucket(new_cap)
            else:
                wait = backoff + random.uniform(0, JITTER_MAX)
                print(f"⚠️ {status or 'Error'} on {desc}. Retrying in {wait:.1f}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
            backoff = min(backoff * 2, 60)
        except Exception as e:
            wait = backoff + random.uniform(0, JITTER_MAX)
            print(f"⚠️ Error on {desc}: {e}. Retrying in {wait:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)
            backoff = min(backoff * 2, 60)
    _acquire_token(block=True)
    result = call()
    _cooldown_if_burst()
    return result

# Wrapped sp functions
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

def sp_artists(sp, *args, **kwargs):
    return with_api(lambda: sp.artists(*args, **kwargs), desc="artists")

# ---------------- AUTH BLOCK ----------------
def authenticate():
    print("\n🔑 Spotify API Credentials")
    print("Please enter your Spotify API credentials:")
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
            requests_timeout=30,
            retries=5,
            status_retries=5,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504)
        )
        sp_search(sp, q="test", limit=1, type="playlist")
        print("✅ Authenticated to Spotify API")
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        sys.exit(1)
    return sp

# ---------------- PLAYLIST SEARCH ----------------
def search_playlists(sp, keywords, min_followers, max_playlists=MAX_PLAYLISTS):
    found = {}
    playlist_details_cache = {}

    for kw in keywords:
        if len(found) >= max_playlists:
            break
        print(f"\nSearching for playlists with keyword '{kw}'...")
        offset = 0

        for page in range(SEARCH_PAGES):
            if len(found) >= max_playlists:
                break
            try:
                res = sp_search(sp, q=kw, type="playlist", limit=SEARCH_LIMIT, offset=offset, market="US")
                if not res or "playlists" not in res:
                    break
                playlists = res["playlists"]["items"] or []
                if not playlists:
                    break

                for p in playlists:
                    if not p or "id" not in p:
                        continue
                    pid = p["id"]
                    if pid in found:
                        continue
                    owner_id = (p.get("owner") or {}).get("id")
                    if owner_id == "spotify":
                        continue

                    if len(found) >= max_playlists:
                        break

                    if pid in playlist_details_cache:
                        playlist = playlist_details_cache[pid]
                    else:
                        try:
                            playlist = sp_playlist(sp, pid, fields="id,name,description,followers,owner")
                            playlist_details_cache[pid] = playlist
                        except Exception as e:
                            print(f"⚠️ Skipping playlist {pid} due to error fetching details: {e}")
                            continue

                    followers = playlist.get("followers", {}).get("total", 0)
                    name = playlist.get("name", "") or ""
                    description = (playlist.get("description") or "") or ""
                    kw_lower = kw.lower()

                    if followers >= min_followers and (kw_lower in name.lower() or kw_lower in description.lower()):
                        priority = 1 if kw_lower in name.lower() else 2
                        found[pid] = {
                            "id": pid,
                            "name": playlist.get("name", ""),
                            "followers": followers,
                            "owner": playlist.get("owner", {}).get("display_name"),
                            "priority": priority
                        }
                        match_type = "title" if kw_lower in name.lower() else "description"
                        print(f"Found playlist (match in {match_type}): {playlist['name']} ({followers:,} followers)")

                    if len(found) >= max_playlists:
                        break

                offset += len(playlists)
                polite_sleep()
            except Exception as e:
                print(f"⚠️ Search error for '{kw}': {str(e)}")
                polite_sleep(1.0)
                continue

    return sorted(found.values(), key=lambda x: (x.get("priority", 2), -x["followers"]))

def iter_tracks(sp, playlist_id):
    try:
        res = sp_playlist_items(
            sp,
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
        items = res.get("items")
        if not items:
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

# ---------------- ENHANCED METADATA COLLECTION ----------------
def get_artist_metadata(sp, artist_ids):
    """Fetch artist popularity, followers, and genres in batches"""
    artist_data = {}
    unique_ids = list(set(artist_ids))
    
    for chunk in chunked(unique_ids, 50):
        try:
            res = sp_artists(sp, chunk)
            for artist in res.get("artists", []) or []:
                if artist:
                    artist_data[artist["id"]] = {
                        "popularity": artist.get("popularity", 0),
                        "followers": artist.get("followers", {}).get("total", 0),
                        "genres": ",".join(artist.get("genres", [])[:3])  # Top 3 genres
                    }
        except Exception as e:
            print(f"⚠️ Error fetching artist metadata: {e}")
            polite_sleep(1.0)
    
    return artist_data

# ---------------- COLLECTION ----------------
def collect_emotion(sp, emotion, target, min_followers, collected_map, allow_cross_emotion_duplicates=False):
    kws = EMOTION_KEYWORDS[emotion]
    print(f"\n🎧 Collecting '{emotion}' ({len(kws)} keywords, target {target:,} tracks)...")
    start_time = time.time()

    playlists = search_playlists(sp, kws, min_followers, max_playlists=MAX_PLAYLISTS)
    total_searched = len(playlists)

    if len(playlists) < MIN_PLAYLISTS:
        print(f"⚠️ Only found {len(playlists)} playlists with {min_followers:,} followers, trying with lower threshold...")
        more_playlists = search_playlists(sp, kws, max(100, min_followers // 2), max_playlists=MAX_PLAYLISTS)
        existing_ids = {p["id"] for p in playlists}
        for p in more_playlists:
            if p["id"] not in existing_ids and len(playlists) < MAX_PLAYLISTS:
                playlists.append(p)
        total_searched += len(more_playlists)

    if not playlists:
        print(f"❌ No suitable playlists found for {emotion}")
        return pd.DataFrame()

    print(f" → Found {len(playlists)} unique playlists from {total_searched} total results")

    # If cross-emotion duplicates are allowed, do not exclude tracks from other emotions
    global_used = set() if allow_cross_emotion_duplicates else {tid for ids in collected_map.values() for tid in ids}
    emotion_used = set(collected_map.get(emotion, []))
    collected = []
    track_info = {}
    max_attempts = min(len(playlists), MAX_PLAYLISTS)
    attempts = 0
    last_progress = 0
    progress_interval = max(1, target // 20)
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
                track_name = t.get("name", "").lower()
                artists = [a.get("name", "").lower() for a in t.get("artists", [])]
                track_key = f"{track_name}|{'|'.join(sorted(artists))}"
                if track_key in track_info:
                    continue
                collected.append(tid)
                emotion_used.add(tid)
                track_info[track_key] = tid
                new += 1

                if len(collected) % checkpoint_every == 0:
                    OUT_DIR.mkdir(parents=True, exist_ok=True)
                    tmp_file = OUT_DIR / f"{emotion}_checkpoint_ids.json"
                    with tmp_file.open("w", encoding="utf-8") as fh:
                        json.dump({"emotion": emotion, "ids": collected}, fh)
                    print(f"💾 Checkpoint saved ({len(collected)} IDs) -> {tmp_file}")

                if len(collected) - last_progress >= progress_interval:
                    elapsed = time.time() - start_time
                    tracks_per_sec = len(collected) / elapsed if elapsed > 0 else 0
                    eta = (target - len(collected)) / tracks_per_sec if tracks_per_sec > 0 else 0
                    print(f"\rProgress: {len(collected):,}/{target:,} tracks ({len(collected)/target*100:.1f}%) - ETA: {eta/60:.1f} minutes", end="")
                    last_progress = len(collected)

            if new > 0:
                print(f"\n   +{new:,} new tracks from this playlist")
        except Exception as e:
            print(f"\n⚠️ Error processing playlist: {str(e)}")
            continue

    print("\n")
    if not collected:
        print(f"❌ No tracks found for {emotion}")
        return pd.DataFrame()

    print(f"Fetching metadata for {len(collected):,} tracks...")
    data = []
    all_artist_ids = []
    metadata_progress = 0
    
    # First pass: collect track data and artist IDs
    for chunk in chunked(collected, 50):
        try:
            res = sp_tracks(sp, chunk)
            for t in res.get("tracks", []) or []:
                if not t:
                    continue
                tid = t.get("id")
                album = t.get("album", {})
                artists_list = t.get("artists", [])
                
                # Collect artist IDs for batch fetch
                artist_ids = [a["id"] for a in artists_list if a.get("id")]
                all_artist_ids.extend(artist_ids)
                
                # Store track data with artist IDs for later enrichment
                data.append({
                    "track_id": tid,
                    "name": t.get("name"),
                    "artists": ", ".join([a["name"] for a in artists_list]),
                    "artist_ids": ",".join(artist_ids),  # Store for artist data lookup
                    "album": album.get("name"),
                    "album_type": album.get("album_type", ""),  # single/album/compilation
                    "album_total_tracks": album.get("total_tracks", 0),
                    "album_release_date": album.get("release_date"),
                    "popularity": t.get("popularity"),
                    "duration_ms": t.get("duration_ms"),
                    "explicit": t.get("explicit", False),
                    "available_markets": len(t.get("available_markets", [])),
                    "spotify_link": f"https://open.spotify.com/track/{tid}",
                    "emotion": emotion
                })
                metadata_progress += 1
                if metadata_progress % 500 == 0:
                    print(f"\rFetched metadata for {metadata_progress:,}/{len(collected):,} tracks", end="")
        except Exception as e:
            print(f"\n⚠️ Error fetching metadata: {e}")
            polite_sleep(1.0)
    
    print(f"\n\n✨ Fetching artist data for {len(set(all_artist_ids)):,} unique artists...")
    artist_metadata = get_artist_metadata(sp, all_artist_ids)
    
    # Second pass: enrich track data with artist metadata
    print("Enriching track data with artist information...")
    for track in data:
        artist_ids = track["artist_ids"].split(",") if track["artist_ids"] else []
        
        if artist_ids:
            # Get primary artist (first one) data
            primary_artist_id = artist_ids[0]
            primary_data = artist_metadata.get(primary_artist_id, {})
            
            track["artist_popularity"] = primary_data.get("popularity", 0)
            track["artist_followers"] = primary_data.get("followers", 0)
            track["artist_genres"] = primary_data.get("genres", "")
            
            # Additional artist features
            track["num_artists"] = len(artist_ids)
            track["is_collaboration"] = len(artist_ids) > 1
            
            # Average popularity across all artists
            all_pops = [artist_metadata.get(aid, {}).get("popularity", 0) for aid in artist_ids]
            track["avg_artist_popularity"] = sum(all_pops) / len(all_pops) if all_pops else 0
        else:
            track["artist_popularity"] = 0
            track["artist_followers"] = 0
            track["artist_genres"] = ""
            track["num_artists"] = 0
            track["is_collaboration"] = False
            track["avg_artist_popularity"] = 0
        
        # Remove temporary artist_ids field
        del track["artist_ids"]
    
    polite_sleep()

    df = pd.DataFrame(data)
    if len(df) >= checkpoint_every:
        meta_tmp = OUT_DIR / f"{emotion}_checkpoint_metadata.csv"
        df.to_csv(meta_tmp, index=False, encoding="utf-8-sig")
        print(f"💾 Metadata checkpoint saved -> {meta_tmp}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{emotion}_tracks_enhanced.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_time
    tracks_per_min = len(df) / (elapsed / 60) if elapsed > 0 else 0
    print(f"✅ Saved {len(df):,} enhanced tracks → {out}")
    print(f"   Collection rate: {tracks_per_min:.1f} tracks/minute")
    print(f"   New features: artist_popularity, artist_followers, artist_genres,")
    print(f"                 album_type, explicit, available_markets, num_artists")

    collected_map.setdefault(emotion, [])
    collected_map[emotion] = list(set(collected_map[emotion] + collected))
    return df

# ---------------- MAIN ----------------
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", choices=["happy","sad","energetic","love","all"], default=None)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--min-followers", type=int, default=DEFAULT_MIN_FOLLOWERS)
    parser.add_argument("--safe-mode", action="store_true")
    parser.add_argument("--per-minute", type=int, help="Override max requests per minute (default 100)")
    parser.add_argument("--sleep", type=float, help="Override base sleep between requests in seconds")
    parser.add_argument("--ignore-previous", action="store_true", help="Do not de-duplicate against previous runs (collected_ids.json)")
    parser.add_argument("--allow-cross-emotion-duplicates", action="store_true", help="Allow the same track to appear in multiple emotions in this run")
    args = parser.parse_args(argv)

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
    # Control de-duplication against previous runs
    collected = {} if args.ignore_previous else load_ids()

    global PER_MIN_LIMIT, SLEEP, MAX_PLAYLISTS, BURST_LIMIT
    if args.safe_mode:
        PER_MIN_LIMIT = 60
        SLEEP = max(0.8, SLEEP)
        MAX_PLAYLISTS = min(MAX_PLAYLISTS, 60)
        BURST_LIMIT = min(BURST_LIMIT, 15)
        print(f"🛡️ Safe mode: PER_MIN_LIMIT={PER_MIN_LIMIT}, SLEEP={SLEEP}, MAX_PLAYLISTS={MAX_PLAYLISTS}, BURST_LIMIT={BURST_LIMIT}")
    if args.per_minute:
        PER_MIN_LIMIT = max(10, int(args.per_minute))
    if args.sleep:
        SLEEP = max(0.0, float(args.sleep))

    configure_token_bucket(PER_MIN_LIMIT)

    emotions = [args.emotion] if args.emotion != "all" else list(EMOTION_KEYWORDS)

    print("\n🔍 Configuration:")
    print(f"Target tracks per emotion: {args.target:,}")
    print(f"Minimum playlist followers: {args.min_followers:,}")
    print(f"Rate: PER_MIN_LIMIT={PER_MIN_LIMIT}/min, SLEEP={SLEEP}s, BURST_LIMIT={BURST_LIMIT}")
    print(f"Emotions to collect: {', '.join(emotions)}")
    print(f"Output directory: {OUT_DIR}")
    print(f"📊 Enhanced features: artist_popularity, artist_followers, artist_genres,")
    print(f"                      album_type, explicit, available_markets, collaborations")
    print("-" * 50)

    for e in emotions:
        try:
            df = collect_emotion(
                sp,
                e,
                args.target,
                args.min_followers,
                collected,
                allow_cross_emotion_duplicates=args.allow_cross_emotion_duplicates,
            )
            if not df.empty:
                print(f"\n📊 Statistics for {e}:")
                print(f"Tracks collected: {len(df):,}")
                print(f"Unique artists: {df['artists'].nunique():,}")
                print(f"Average popularity: {df['popularity'].mean():.1f}")
                print(f"Average artist popularity: {df['artist_popularity'].mean():.1f}")
                print(f"Average artist followers: {df['artist_followers'].mean():,.0f}")
                print(f"Collaborations: {df['is_collaboration'].sum():,} ({df['is_collaboration'].mean()*100:.1f}%)")
                print(f"Explicit tracks: {df['explicit'].sum():,} ({df['explicit'].mean()*100:.1f}%)")
                print(f"Average duration: {df['duration_ms'].mean() / 60000:.1f} minutes")
                print("-" * 50)
        except Exception as ex:
            print(f"❌ Error collecting {e}: {str(ex)}")

    save_ids(collected)
    print(f"\n🎉 All done! Enhanced CSVs saved in '{OUT_DIR}'.")
    print("New columns added: artist_popularity, artist_followers, artist_genres, album_type,")
    print("                   explicit, available_markets, num_artists, is_collaboration, avg_artist_popularity")

if __name__ == "__main__":
    main()
