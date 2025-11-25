#!/usr/bin/env python3
"""Calculate API usage for enhanced collector with 10K target"""

target = 10000
emotions = 4

# API calls per emotion
playlist_search = 18 * 3        # 18 keywords × 3 pages
playlist_details = 100          # max 100 playlists
playlist_items = 100 * 15       # 100 playlists × ~15 pages average
tracks_metadata = (10000 // 50) + 1    # batches of 50
artists_metadata = (int(10000 * 1.5) // 50) + 1  # ~1.5 artists/track

calls_per_emotion = (playlist_search + playlist_details + 
                     playlist_items + tracks_metadata + artists_metadata)
total_calls = calls_per_emotion * emotions
time_100 = total_calls / 100
time_60 = total_calls / 60

print("API Usage Analysis (10K target, 50K min followers)")
print("=" * 60)
print(f"\nPer Emotion ({target:,} tracks):")
print(f"  Playlist search:     {playlist_search:>6,} calls")
print(f"  Playlist details:    {playlist_details:>6,} calls")
print(f"  Playlist items:      {playlist_items:>6,} calls")
print(f"  Track metadata:      {tracks_metadata:>6,} calls")
print(f"  Artist metadata:     {int(artists_metadata):>6,} calls")
print(f"  {'─' * 40}")
print(f"  Subtotal:            {int(calls_per_emotion):>6,} calls/emotion")

print(f"\nTotal for {emotions} emotions:  {int(total_calls):>6,} calls")

print(f"\nTime Estimates:")
print(f"  At 100 req/min:  {time_100:>6.0f} min ({time_100/60:>4.1f} hours)")
print(f"  At 60 req/min:   {time_60:>6.0f} min ({time_60/60:>4.1f} hours)")

print(f"\n{'=' * 60}")
if total_calls > 12000:
    print("⚠️  STATUS: LONG COLLECTION")
    print(f"\nRecommendation: Run one emotion at a time")
    print(f"  Time per emotion: ~{calls_per_emotion/100:.0f} min (100/min)")
    print(f"                    ~{calls_per_emotion/60:.0f} min (60/min safe)")
    print(f"\nCommands:")
    print(f"  python CollectionUpdate_Enhanced.py --emotion happy --target 10000")
    print(f"  python CollectionUpdate_Enhanced.py --emotion sad --target 10000")
    print(f"  python CollectionUpdate_Enhanced.py --emotion energetic --target 10000")
    print(f"  python CollectionUpdate_Enhanced.py --emotion love --target 10000")
else:
    print("✅ STATUS: SAFE - Can run all emotions together")
    print(f"\nCommand:")
    print(f"  python CollectionUpdate_Enhanced.py --emotion all --target 10000")

print(f"\nWith 50K min followers:")
print(f"  ✅ More curated/popular playlists")
print(f"  ✅ Better quality tracks")
print(f"  ⚠️  May find fewer playlists (will auto-lower threshold if needed)")
