"""
Test Dejavu with dataset/ songs.
Step 1: Fingerprint original songs (load into database)
Step 2: Recognize remix files + clips against the database
Step 3: Save results to JSON

Run INSIDE Docker container:
  python run_commands/test_my_songs.py
"""

import os
import sys
import glob
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dejavu import Dejavu
from dejavu.logic.recognizer.file_recognizer import FileRecognizer
from dejavu.logic import decoder

config = {
    "database": {
        "host": "db",
        "user": "postgres",
        "password": "password",
        "database": "dejavu"
    },
    "database_type": "postgres"
}

DATASET_DIR = "dataset"
RESULTS_FILE = "dataset/results.json"


def discover_songs():
    """Auto-discover songs from dataset/ folder structure."""
    songs = {}
    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: {DATASET_DIR}/ folder not found!")
        sys.exit(1)

    for song_name in sorted(os.listdir(DATASET_DIR)):
        song_dir = os.path.join(DATASET_DIR, song_name)
        if not os.path.isdir(song_dir):
            continue

        original = os.path.join(song_dir, "original.mp3")
        remixes = sorted(glob.glob(os.path.join(song_dir, "remix", "remix_*.mp3")))
        clips = sorted(glob.glob(os.path.join(song_dir, "remix", "clips_*", "clip_*.mp3")))

        songs[song_name] = {
            "original": original if os.path.isfile(original) else None,
            "remixes": remixes,
            "clips": clips,
        }

    return songs


def parse_match(match):
    """Extract clean fields from a dejavu match result."""
    name = match["song_name"]
    if isinstance(name, bytes):
        name = name.decode()
    return {
        "song_name": name,
        "confidence": round(match["fingerprinted_confidence"], 4),
        "input_confidence": round(match["input_confidence"], 4),
        "hashes_matched": match["hashes_matched_in_input"],
        "offset_seconds": round(match["offset_seconds"], 2),
    }


def recognize_file(djv, filepath, expected_song):
    """Recognize a file and return result dict with top 2 matches."""
    try:
        result = djv.recognize(FileRecognizer, filepath)
        time_taken = round(result.get("total_time", 0), 2) if result else 0
        matches = result.get("results", []) if result else []

        if matches:
            # Top 1 (best match)
            top1 = parse_match(matches[0])
            # Top 2 (second best, if exists)
            top2 = parse_match(matches[1]) if len(matches) > 1 else None

            return {
                "file": filepath,
                "expected": expected_song,
                "matched": top1["song_name"],
                "confidence": top1["confidence"],
                "hashes_matched": top1["hashes_matched"],
                "top2": top2,
                "time": time_taken,
                "status": "PASS" if top1["song_name"] == expected_song else "FAIL"
            }
        else:
            # No match found at all — dejavu returned empty results
            return {
                "file": filepath,
                "expected": expected_song,
                "matched": None,
                "confidence": None,
                "hashes_matched": 0,
                "top2": None,
                "time": time_taken,
                "status": "NO_MATCH"
            }
    except Exception as e:
        return {
            "file": filepath,
            "expected": expected_song,
            "matched": None,
            "confidence": None,
            "hashes_matched": 0,
            "top2": None,
            "time": 0,
            "error": str(e),
            "status": "ERROR"
        }


if __name__ == '__main__':
    djv = Dejavu(config)
    songs = discover_songs()

    print(f"Found {len(songs)} songs in {DATASET_DIR}/\n")

    # JSON results structure
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "total_songs": len(songs),
        "fingerprinted": [],
        "remix_tests": [],
        "clip_tests": [],
        "summary": {}
    }

    # ========== STEP 1: FINGERPRINT originals ==========
    print("=" * 60)
    print("STEP 1: FINGERPRINTING (loading originals into database)")
    print("=" * 60)

    for song_name, data in songs.items():
        if not data["original"]:
            print(f"  [SKIP] {song_name}: no original.mp3 found")
            continue

        # Check if already fingerprinted (by file hash) — skip if yes
        file_hash = decoder.unique_hash(data["original"])
        if file_hash in djv.songhashes_set:
            print(f"  [SKIP] {song_name}: already in database")
            all_results["fingerprinted"].append(song_name)
            continue

        print(f"\n  [{song_name}] Fingerprinting original...")
        print(f"    -> {data['original']}")
        djv.fingerprint_file(data["original"], song_name=song_name)
        all_results["fingerprinted"].append(song_name)

    print("\nFingerprinting complete!")

    # ========== VERIFY: Check DB for all fingerprinted songs ==========
    print("\n  Verifying database...")
    db_songs = djv.get_fingerprinted_songs()
    db_song_names = []
    for s in db_songs:
        name = s.get("song_name", "")
        if isinstance(name, bytes):
            name = name.decode()
        db_song_names.append(name)

    # Check which songs from dataset are in DB and which are missing
    expected_songs = [name for name in songs if songs[name]["original"]]
    missing_from_db = [name for name in expected_songs if name not in db_song_names]
    extra_in_db = [name for name in db_song_names if name not in expected_songs]

    all_results["db_verification"] = {
        "songs_in_db": len(db_song_names),
        "expected": len(expected_songs),
        "missing_from_db": missing_from_db,
        "extra_in_db": extra_in_db,
        "all_uploaded": len(missing_from_db) == 0,
        "db_song_list": sorted(db_song_names),
    }

    if missing_from_db:
        print(f"  WARNING: {len(missing_from_db)} songs NOT in DB: {missing_from_db}")
    else:
        print(f"  All {len(db_song_names)} songs verified in database.")

    # ========== STEP 2: RECOGNIZE remixes (whole files) ==========
    print("\n" + "=" * 60)
    print("STEP 2: RECOGNIZING REMIXES (whole cover files)")
    print("=" * 60)

    for song_name, data in songs.items():
        if not data["remixes"]:
            continue

        # expected = folder name (e.g. "Jai_Ho") — same as what we stored in DB
        expected_song = song_name

        print(f"\n  [{song_name}] Testing {len(data['remixes'])} remix(es)...")
        for remix in data["remixes"]:
            r = recognize_file(djv, remix, expected_song)
            r["song"] = song_name
            r["type"] = "remix"
            all_results["remix_tests"].append(r)

            remix_basename = os.path.basename(remix)
            top2_info = ""
            if r["top2"]:
                top2_info = f" | #2: {r['top2']['song_name']} ({r['top2']['confidence']:.4f})"

            if r["status"] == "PASS":
                print(f"    {remix_basename} -> {r['matched']} [PASS] (conf: {r['confidence']:.4f}, hashes: {r['hashes_matched']}, time: {r['time']:.2f}s){top2_info}")
            elif r["status"] == "NO_MATCH":
                print(f"    {remix_basename} -> NO MATCH [FAIL] (no hashes matched any song)")
            elif r["status"] == "ERROR":
                print(f"    {remix_basename} -> ERROR: {r.get('error', 'unknown')}")
            else:
                print(f"    {remix_basename} -> {r['matched']} [FAIL] expected: {expected_song} (conf: {r['confidence']:.4f}){top2_info}")

    # ========== STEP 3: RECOGNIZE clips ==========
    print("\n" + "=" * 60)
    print("STEP 3: RECOGNIZING CLIPS (short segments from remixes)")
    print("=" * 60)

    for song_name, data in songs.items():
        if not data["clips"]:
            continue

        expected_song = song_name

        print(f"\n  [{song_name}] Testing {len(data['clips'])} clips...")
        for clip in data["clips"]:
            r = recognize_file(djv, clip, expected_song)
            r["song"] = song_name
            r["type"] = "clip"
            all_results["clip_tests"].append(r)

            clip_basename = os.path.basename(clip)
            top2_info = ""
            if r["top2"]:
                top2_info = f" | #2: {r['top2']['song_name']} ({r['top2']['confidence']:.4f})"

            if r["status"] == "PASS":
                print(f"    {clip_basename} -> {r['matched']} [PASS] (conf: {r['confidence']:.4f}, hashes: {r['hashes_matched']}, time: {r['time']:.2f}s){top2_info}")
            elif r["status"] == "NO_MATCH":
                print(f"    {clip_basename} -> NO MATCH [FAIL] (no hashes matched any song)")
            elif r["status"] == "ERROR":
                print(f"    {clip_basename} -> ERROR: {r.get('error', 'unknown')}")
            else:
                print(f"    {clip_basename} -> {r['matched']} [FAIL] expected: {expected_song} (conf: {r['confidence']:.4f}){top2_info}")

    # ========== BUILD PER-SONG METRICS ==========
    per_song = {}
    for song_name in songs:
        per_song[song_name] = {
            "remix_pass": 0, "remix_fail": 0, "remix_nomatch": 0,
            "clip_pass": 0, "clip_fail": 0, "clip_nomatch": 0,
        }

    for r in all_results["remix_tests"]:
        s = r["song"]
        if r["status"] == "PASS":
            per_song[s]["remix_pass"] += 1
        elif r["status"] == "NO_MATCH":
            per_song[s]["remix_nomatch"] += 1
        else:
            per_song[s]["remix_fail"] += 1

    for r in all_results["clip_tests"]:
        s = r["song"]
        if r["status"] == "PASS":
            per_song[s]["clip_pass"] += 1
        elif r["status"] == "NO_MATCH":
            per_song[s]["clip_nomatch"] += 1
        else:
            per_song[s]["clip_fail"] += 1

    # Add total per song
    for s in per_song:
        ps = per_song[s]
        ps["total_tests"] = (ps["remix_pass"] + ps["remix_fail"] + ps["remix_nomatch"]
                             + ps["clip_pass"] + ps["clip_fail"] + ps["clip_nomatch"])
        ps["total_pass"] = ps["remix_pass"] + ps["clip_pass"]
        ps["total_fail"] = ps["total_tests"] - ps["total_pass"]

    # ========== SORT: failed tests first, passed last ==========
    all_results["remix_tests"].sort(key=lambda r: (r["status"] == "PASS", r["song"]))
    all_results["clip_tests"].sort(key=lambda r: (r["status"] == "PASS", r["song"]))

    # ========== OVERALL COUNTS ==========
    def count_status(tests, status):
        return len([r for r in tests if r["status"] == status])

    remix_pass = count_status(all_results["remix_tests"], "PASS")
    remix_fail = count_status(all_results["remix_tests"], "FAIL")
    remix_nomatch = count_status(all_results["remix_tests"], "NO_MATCH")
    remix_total = len(all_results["remix_tests"])

    clip_pass = count_status(all_results["clip_tests"], "PASS")
    clip_fail = count_status(all_results["clip_tests"], "FAIL")
    clip_nomatch = count_status(all_results["clip_tests"], "NO_MATCH")
    clip_total = len(all_results["clip_tests"])

    total_pass = remix_pass + clip_pass
    total_tests = remix_total + clip_total

    all_results["per_song"] = per_song
    all_results["summary"] = {
        "remix": {"total": remix_total, "pass": remix_pass, "fail": remix_fail, "no_match": remix_nomatch},
        "clips": {"total": clip_total, "pass": clip_pass, "fail": clip_fail, "no_match": clip_nomatch},
        "overall": {"total": total_tests, "pass": total_pass, "fail": total_tests - total_pass},
        "remix_accuracy": f"{remix_pass/remix_total*100:.1f}%" if remix_total > 0 else "N/A",
        "clip_accuracy": f"{clip_pass/clip_total*100:.1f}%" if clip_total > 0 else "N/A",
        "overall_accuracy": f"{total_pass/total_tests*100:.1f}%" if total_tests > 0 else "N/A",
    }

    # ========== PRINT: Per-song breakdown ==========
    print("\n" + "=" * 60)
    print("PER-SONG BREAKDOWN")
    print("=" * 60)
    # Sort: worst songs first (most failures)
    sorted_songs = sorted(per_song.items(), key=lambda x: x[1]["total_fail"], reverse=True)
    for s, ps in sorted_songs:
        r_total = ps["remix_pass"] + ps["remix_fail"] + ps["remix_nomatch"]
        c_total = ps["clip_pass"] + ps["clip_fail"] + ps["clip_nomatch"]
        tag = "PASS" if ps["total_fail"] == 0 else "FAIL"
        print(f"\n  [{tag}] {s}")
        if r_total > 0:
            print(f"    Remix:  {ps['remix_pass']}/{r_total} pass", end="")
            if ps["remix_fail"]: print(f", {ps['remix_fail']} wrong match", end="")
            if ps["remix_nomatch"]: print(f", {ps['remix_nomatch']} no match", end="")
            print()
        if c_total > 0:
            print(f"    Clips:  {ps['clip_pass']}/{c_total} pass", end="")
            if ps["clip_fail"]: print(f", {ps['clip_fail']} wrong match", end="")
            if ps["clip_nomatch"]: print(f", {ps['clip_nomatch']} no match", end="")
            print()

    # ========== PRINT: Final summary ==========
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Songs fingerprinted: {len(all_results['fingerprinted'])}")
    print(f"  Remixes:  {remix_pass} PASS / {remix_fail} WRONG MATCH / {remix_nomatch} NO MATCH  ({all_results['summary']['remix_accuracy']})")
    print(f"  Clips:    {clip_pass} PASS / {clip_fail} WRONG MATCH / {clip_nomatch} NO MATCH  ({all_results['summary']['clip_accuracy']})")
    print(f"  Overall:  {total_pass}/{total_tests} matched correctly  ({all_results['summary']['overall_accuracy']})")

    # Save to JSON — db_verification and summary at top, then per_song, then tests
    ordered_results = {
        "timestamp": all_results["timestamp"],
        "db_verification": all_results["db_verification"],
        "summary": all_results["summary"],
        "per_song": all_results["per_song"],
        "fingerprinted": all_results["fingerprinted"],
        "remix_tests": all_results["remix_tests"],
        "clip_tests": all_results["clip_tests"],
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(ordered_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
