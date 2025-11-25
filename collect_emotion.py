# Modified version for large-scale collection
def collect_emotion(sp, emotion, target, min_followers, collected_map):
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
    
    # Only track duplicates within the same emotion
    emotion_used = set(collected_map.get(emotion, []))
    collected = []
    track_info = {}  # Store track info for deduplication within this emotion
    
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
                if not tid or tid in emotion_used:
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