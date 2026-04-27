"""
Dejavu test runner — runs INSIDE the dejavu docker container.

Discovers songs from /code/music, fingerprints all upload_*.mp3 per song
(under the song folder name), then queries test.mp3 + test/clip_*.mp3.
Writes /code/scripts/results/dejavu_results.json.
"""

import os
import sys
import time

# /code is the repo root inside the container; dejavu code is mounted there.
sys.path.insert(0, "/code")
sys.path.insert(0, "/code/scripts")

from dejavu import Dejavu  # noqa: E402
from dejavu.logic.recognizer.file_recognizer import FileRecognizer  # noqa: E402
from dejavu.logic import decoder  # noqa: E402

import common  # noqa: E402

CONFIG = {
    "database": {
        "host": "db",
        "user": "postgres",
        "password": "password",
        "database": "dejavu",
    },
    "database_type": "postgres",
}


def parse_match(match):
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


def recognize(djv, filepath, expected):
    t0 = time.time()
    try:
        result = djv.recognize(FileRecognizer, filepath)
    except Exception as e:
        return common.make_test_record(
            filepath, expected, status="ERROR", error=e,
            time_sec=time.time() - t0,
        )

    elapsed = round(result.get("total_time", time.time() - t0), 3) if result else round(time.time() - t0, 3)
    matches = result.get("results", []) if result else []
    if not matches:
        return common.make_test_record(
            filepath, expected, status="NO_MATCH", time_sec=elapsed,
        )

    top1 = parse_match(matches[0])
    top2 = parse_match(matches[1]) if len(matches) > 1 else None
    return common.make_test_record(
        filepath, expected,
        matched=top1["song_name"],
        confidence=top1["confidence"],
        hashes_matched=top1["hashes_matched"],
        top2=top2,
        time_sec=elapsed,
        status=common.classify(top1["song_name"], expected),
        extra={"offset_seconds": top1["offset_seconds"],
               "input_confidence": top1["input_confidence"]},
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dejavu test runner (runs inside container)")
    ap.add_argument("--data-dir", default="dataset",
                    help="folder name under /code (e.g. dataset, music). Default: dataset")
    args = ap.parse_args()

    # Inside container, /code is the dejavu root; data folders are mounted there.
    data_dir = os.path.join("/code", os.path.basename(args.data_dir))
    songs = common.discover_songs(data_dir)

    total_uploads = sum(len(v["uploads"]) for v in songs.values())
    total_full = sum(1 for v in songs.values() if v["test_full"])
    total_clips = sum(len(v["test_clips"]) for v in songs.values())

    print(f"\nDiscovered {len(songs)} songs in {data_dir}")
    print(f"  uploads = {total_uploads}, full = {total_full}, clips = {total_clips}\n")

    djv = Dejavu(CONFIG)

    # --- Step 1: upload all upload_*.mp3 ---
    print("=" * 70)
    print("  STEP 1 — Fingerprinting uploads")
    print("=" * 70)
    uploaded = []
    for song, data in songs.items():
        for up in data["uploads"]:
            label = f"{song}::{os.path.basename(up).replace('.mp3', '')}"
            try:
                file_hash = decoder.unique_hash(up)
                if file_hash in djv.songhashes_set:
                    print(f"  [SKIP exists] {label}")
                    uploaded.append(label)
                    continue
                djv.fingerprint_file(up, song_name=song)
                uploaded.append(label)
                print(f"  [OK] {label}  ({os.path.basename(up)})")
            except Exception as e:
                print(f"  [ERR] {label}: {e}")

    print(f"\n  Uploaded {len(uploaded)}/{total_uploads}")

    # --- Step 2: query test.mp3 ---
    print("\n" + "=" * 70)
    print("  STEP 2 — Recognizing test.mp3 (full remix)")
    print("=" * 70)
    full_records = []
    idx = 0
    for song, data in songs.items():
        if not data["test_full"]:
            continue
        idx += 1
        rec = recognize(djv, data["test_full"], song)
        full_records.append(rec)
        common.print_progress(
            idx, total_full, song, "test.mp3",
            rec["status"], rec["matched"], song,
            conf=rec["confidence"], t=rec["time"], top2=rec["top2"],
        )

    # --- Step 3: query clips ---
    print("\n" + "=" * 70)
    print("  STEP 3 — Recognizing clips")
    print("=" * 70)
    clip_records = []
    idx = 0
    for song, data in songs.items():
        for clip in data["test_clips"]:
            idx += 1
            rec = recognize(djv, clip, song)
            clip_records.append(rec)
            common.print_progress(
                idx, total_clips, song, os.path.basename(clip),
                rec["status"], rec["matched"], song,
                conf=rec["confidence"], t=rec["time"], top2=rec["top2"],
            )

    path, summary = common.write_results(
        "dejavu", songs, uploaded, full_records, clip_records,
        data_dir=data_dir,
    )
    common.print_final_report(
        "dejavu", summary,
        common.build_per_song(songs, full_records, clip_records),
        path,
    )


if __name__ == "__main__":
    main()
