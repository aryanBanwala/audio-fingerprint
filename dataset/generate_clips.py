"""
Auto-generate random clips from remix files using ffmpeg.
Runs locally (not inside Docker).

Usage:
  python dataset/generate_clips.py              # process all songs
  python dataset/generate_clips.py Jai_Ho       # process one song
"""

import os
import sys
import random
import subprocess
import json

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CLIPS = 12          # clips per remix
MIN_DURATION = 5        # minimum clip length in seconds
MAX_DURATION = 10       # maximum clip length in seconds
SEED = 42               # for reproducibility

random.seed(SEED)


def get_audio_duration(filepath):
    """Get duration of audio file in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def generate_clips(remix_path, clips_dir, num_clips=NUM_CLIPS):
    """Generate random clips from a remix file."""
    duration = get_audio_duration(remix_path)

    # Need at least MAX_DURATION seconds to make clips
    if duration < MAX_DURATION:
        print(f"    [WARN] File too short ({duration:.1f}s), skipping")
        return 0

    os.makedirs(clips_dir, exist_ok=True)
    generated = 0

    for i in range(1, num_clips + 1):
        clip_duration = random.randint(MIN_DURATION, MAX_DURATION)
        max_start = duration - clip_duration - 1
        if max_start <= 0:
            continue
        start_time = random.uniform(1, max_start)  # avoid very start

        clip_path = os.path.join(clips_dir, f"clip_{i}.mp3")
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-ss", str(start_time),
            "-i", remix_path,
            "-t", str(clip_duration),
            "-acodec", "libmp3lame",
            "-q:a", "2",
            clip_path
        ]
        subprocess.run(cmd, capture_output=True)
        generated += 1

    return generated


def process_song(song_dir):
    """Process all remixes in a song directory."""
    song_name = os.path.basename(song_dir)
    remix_dir = os.path.join(song_dir, "remix")

    if not os.path.isdir(remix_dir):
        print(f"  [{song_name}] No remix/ folder, skipping")
        return

    # Find all remix files
    remix_files = sorted([
        f for f in os.listdir(remix_dir)
        if f.startswith("remix_") and f.endswith(".mp3")
    ])

    if not remix_files:
        print(f"  [{song_name}] No remix files found, skipping")
        return

    for remix_file in remix_files:
        remix_num = remix_file.replace("remix_", "").replace(".mp3", "")
        clips_dir = os.path.join(remix_dir, f"clips_{remix_num}")
        remix_path = os.path.join(remix_dir, remix_file)

        # Skip if clips already generated for this remix
        if os.path.isdir(clips_dir) and any(f.endswith(".mp3") for f in os.listdir(clips_dir)):
            existing = len([f for f in os.listdir(clips_dir) if f.endswith(".mp3")])
            print(f"  [{song_name}] {remix_file} -> clips_{remix_num}/ (already has {existing} clips, skipping)")
            continue

        print(f"  [{song_name}] {remix_file} -> clips_{remix_num}/")
        count = generate_clips(remix_path, clips_dir)
        print(f"    Generated {count} clips")


if __name__ == "__main__":
    # Optional: process specific song
    if len(sys.argv) > 1:
        song_name = sys.argv[1]
        song_dir = os.path.join(DATASET_DIR, song_name)
        if os.path.isdir(song_dir):
            process_song(song_dir)
        else:
            print(f"Song folder not found: {song_dir}")
        sys.exit(0)

    # Process all songs
    print("Generating clips from all remixes...\n")
    songs = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    for song in songs:
        process_song(os.path.join(DATASET_DIR, song))

    print("\nDone!")
