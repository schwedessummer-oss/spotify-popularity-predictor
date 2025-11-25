#!/usr/bin/env python3
"""
Export Spotify playlist tracks to CSV.
Collects: items.track.id, items.track.name, items.track.artists, items.track.album,
items.track.popularity, items.track.duration_ms, items.track.is_local, and adds
spotify_link plus the playlist name and playlist follower count for each row.
"""

import re
import sys
import csv
import time
import argparse
import getpass
from pathlib import Path

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# Throttling (kept simple and conservative)
SLEEP = 0.4  # seconds between requests

FIELDS = (
    "items.track.id,items.track.name,items.track.artists,items.track.album,"
    "items.track.popularity,items.track.duration_ms,items.track.is_local,next"
)


def extract_playlist_id(url_or_id: str) -> str:
    """Extract playlist ID from URL or return as-is if already an ID."""
    s = url_or_id.strip()
    # Match full URLs like https://open.spotify.com/playlist/<id>
    m = re.search(r"spotify\.com/playlist/([a-zA-Z0-9]+)", s)
    if m:
        return m.group(1)
    # Match spotify:playlist:<id>
    m = re.search(r"spotify:playlist:([a-zA-Z0-9]+)", s)
    if m:
        return m.group(1)
    # Assume it's already an ID
    return s


def prompt_credentials():
    print("\n🔑 Spotify API Credentials (manual input)")
    cid = input("Client ID: ").strip()
    try:
        secret = getpass.getpass("Client Secret (hidden): ").strip()
    except Exception:
        secret = input("Client Secret: ").strip()
    if not cid or not secret:
        print("❌ Both Client ID and Client Secret are required.")
        sys.exit(1)
    return cid, secret


def get_client(cid: str, secret: str) -> spotipy.Spotify:
    auth_mgr = SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(auth_manager=auth_mgr, requests_timeout=30, retries=5)
    # Light test
    sp.search(q="test", limit=1, type="playlist")
    return sp


def iter_playlist_tracks(sp: spotipy.Spotify, playlist_id: str):
    """Yield playlist items, paging until exhausted."""
    resp = sp.playlist_items(
        playlist_id,
        fields=FIELDS,
        additional_types=["track"],
        limit=100,
    )
    if not resp:
        return

    while True:
        items = resp.get("items") or []
        for it in items:
            yield it
        nxt = resp.get("next")
        if not nxt:
            break
        # Spotipy supports following next via sp.next
        resp = sp.next(resp)
        time.sleep(SLEEP)


def get_playlist_meta(sp: spotipy.Spotify, playlist_id: str) -> dict:
    """Return a small dict with playlist metadata (name, followers, owner name and id)."""
    try:
        p = sp.playlist(playlist_id, fields="id,name,followers,owner")
        name = p.get("name")
        followers = p.get("followers", {}).get("total")
        owner = p.get("owner") or {}
        owner_name = owner.get("display_name")
        owner_id = owner.get("id")
        return {
            "playlist_name": name,
            "playlist_followers": followers,
            "playlist_owner_name": owner_name,
            "playlist_owner_id": owner_id
        }
    except Exception:
        return {
            "playlist_name": None,
            "playlist_followers": None,
            "playlist_owner_name": None,
            "playlist_owner_id": None
        }


def normalize_row(item: dict) -> dict | None:
    t = (item or {}).get("track")
    if not t:
        return None
    tid = t.get("id")
    if not tid:
        return None
    name = t.get("name")
    artists = ", ".join(a.get("name", "") for a in t.get("artists", []) if a)
    album_obj = t.get("album") or {}
    album_name = album_obj.get("name")
    popularity = t.get("popularity")
    duration_ms = t.get("duration_ms")
    is_local = t.get("is_local")
    link = f"https://open.spotify.com/track/{tid}"

    return {
        "track_id": tid,
        "name": name,
        "artists": artists,
        "album": album_name,
        "popularity": popularity,
        "duration_ms": duration_ms,
        "is_local": is_local,
        "spotify_link": link,
    }


def export_playlist(sp: spotipy.Spotify, playlist_id: str, out_path: Path) -> Path:
    meta = get_playlist_meta(sp, playlist_id)
    rows: list[dict] = []
    count = 0
    for item in iter_playlist_tracks(sp, playlist_id):
        row = normalize_row(item)
        if row:
            # augment with playlist metadata
            row["playlist_name"] = meta.get("playlist_name")
            row["playlist_followers"] = meta.get("playlist_followers")
            row["playlist_owner_name"] = meta.get("playlist_owner_name")
            row["playlist_owner_id"] = meta.get("playlist_owner_id")
            rows.append(row)
            count += 1
            if count % 200 == 0:
                print(f"  …fetched {count} tracks")
    if not rows:
        print("⚠️ No tracks found or playlist is empty.")
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} tracks → {out_path}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Export a Spotify playlist's tracks to CSV")
    p.add_argument("--playlist", "-p", required=True, help="Playlist URL or ID")
    p.add_argument("--out", "-o", help="Output CSV path (default: playlist_<id>.csv)")
    args = p.parse_args(argv)

    playlist_id = extract_playlist_id(args.playlist)
    out = Path(args.out) if args.out else Path(f"playlist_{playlist_id}.csv")

    cid, secret = prompt_credentials()
    sp = get_client(cid, secret)

    print(f"\n▶ Exporting playlist {playlist_id} → {out}")
    export_playlist(sp, playlist_id, out)


if __name__ == "__main__":
    main()
