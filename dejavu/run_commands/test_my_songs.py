"""
Test Dejavu with your own songs.
Step 1: Fingerprint upload files (load into database)
Step 2: Recognize test clips against the database

Run INSIDE Docker container:
  python run_commands/test_my_songs.py
"""

import os
import sys
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer

config = {
    "database": {
        "host": "db",
        "user": "postgres",
        "password": "password",
        "database": "dejavu"
    },
    "database_type": "postgres"
}

SONGS = {
    "Dil_Ka_Jo_Haal_Hai": {
        "uploads": "music/Dil_Ka_Jo_Haal_Hai/upload_*.mp3",
        "clips": "music/Dil_Ka_Jo_Haal_Hai/test/clip_*.mp3",
    },
    "Hale_Dil": {
        "uploads": "music/Hale_Dil/upload_*.mp3",
        "clips": "music/Hale_Dil/test/clip_*.mp3",
    },
    "Lag_Ja_Gale": {
        "uploads": "music/Lag_Ja_Gale/upload_*.mp3",
        "clips": "music/Lag_Ja_Gale/test/clip_*.mp3",
    },
    "One_Love": {
        "uploads": "music/One_Love/upload_*.mp3",
        "clips": "music/One_Love/test/clip_*.mp3",
    },
}

if __name__ == '__main__':
    djv = Dejavu(config)

    # ========== STEP 1: FINGERPRINT upload files ==========
    print("=" * 60)
    print("STEP 1: FINGERPRINTING (loading songs into database)")
    print("=" * 60)

    for song_name, paths in SONGS.items():
        upload_files = sorted(glob.glob(paths["uploads"]))
        if not upload_files:
            print(f"  [SKIP] {song_name}: no upload files found")
            continue
        print(f"\n  [{song_name}] Fingerprinting {len(upload_files)} files...")
        for f in upload_files:
            print(f"    -> {f}")
            djv.fingerprint_file(f)

    print("\nFingerprinting complete!")

    # ========== STEP 2: RECOGNIZE test clips ==========
    print("\n" + "=" * 60)
    print("STEP 2: RECOGNIZING (matching test clips)")
    print("=" * 60)

    total_clips = 0
    correct = 0
    failed = 0

    for song_name, paths in SONGS.items():
        clip_files = sorted(glob.glob(paths["clips"]))
        if not clip_files:
            print(f"  [SKIP] {song_name}: no test clips found")
            continue

        print(f"\n  [{song_name}] Testing {len(clip_files)} clips...")
        for clip in clip_files:
            total_clips += 1
            clip_basename = os.path.basename(clip)
            try:
                result = djv.recognize(FileRecognizer, clip)
                if result and result.get("results") and len(result["results"]) > 0:
                    best = result["results"][0]
                    matched_name = best["song_name"]
                    if isinstance(matched_name, bytes):
                        matched_name = matched_name.decode()
                    confidence = best["fingerprinted_confidence"]
                    time_taken = result["total_time"]
                    print(f"    {clip_basename} -> {matched_name} (confidence: {confidence:.2f}, time: {time_taken:.2f}s)")
                    correct += 1
                else:
                    print(f"    {clip_basename} -> NO MATCH")
                    failed += 1
            except Exception as e:
                print(f"    {clip_basename} -> ERROR: {e}")
                failed += 1

    # ========== RESULTS ==========
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Total clips tested: {total_clips}")
    print(f"  Matched: {correct}")
    print(f"  Failed/No match: {failed}")
    if total_clips > 0:
        print(f"  Accuracy: {correct/total_clips*100:.1f}%")
