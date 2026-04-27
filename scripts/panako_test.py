"""
Panako test runner — runs on host (no docker).

Stores each upload_*.mp3 file under its filesystem path; song name is recovered
from the parent directory of the matched file path.
"""

import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

PANAKO_DIR = os.path.join(common.ROOT, "Panako")
PANAKO_DB_DIR = os.path.expanduser("~/.panako")

JAVA_OPTS = [
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
]


def find_jar():
    libs = os.path.join(PANAKO_DIR, "build", "libs")
    if not os.path.isdir(libs):
        return None
    for f in sorted(os.listdir(libs)):
        if f.endswith("-all.jar"):
            return os.path.join(libs, f)
    return None


def ensure_jar():
    jar = find_jar()
    if jar:
        return jar
    print("[build] panako jar not found, running ./gradlew shadowJar")
    subprocess.run(["./gradlew", "shadowJar"], cwd=PANAKO_DIR, check=True)
    jar = find_jar()
    if not jar:
        raise RuntimeError("panako jar build failed")
    return jar


def wipe_db():
    if os.path.isdir(PANAKO_DB_DIR):
        shutil.rmtree(PANAKO_DB_DIR)
    os.makedirs(os.path.join(PANAKO_DB_DIR, "dbs"), exist_ok=True)


def run_java(jar, *args, timeout=120):
    cmd = ["java", *JAVA_OPTS, "-jar", jar, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def store(jar, mp3_path):
    res = run_java(jar, "store", mp3_path)
    return res.returncode == 0, (res.stdout + res.stderr)


def parse_query_lines(stdout):
    """Parse Panako's semicolon-separated output rows (skip the header).

    Format:
      Index ; Total ; Query path ; Q start ; Q stop ; Match path ; Match id ;
      M start ; M stop ; Match score ; Time factor ; Freq factor ; Sec with match
    """
    rows = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 13:
            continue
        # header has "Index" as 1st field; data has digit
        if not parts[0].isdigit():
            continue
        try:
            match_score = int(parts[9])
        except ValueError:
            continue
        match_path = parts[5]
        if not match_path or match_score <= 0:
            continue
        rows.append({
            "index": int(parts[0]),
            "query_start": float(parts[3]),
            "query_stop": float(parts[4]),
            "match_path": match_path,
            "match_id": parts[6],
            "match_start": float(parts[7]),
            "match_stop": float(parts[8]),
            "score": match_score,
            "time_factor": parts[10].rstrip("%").strip(),
            "freq_factor": parts[11].rstrip("%").strip(),
            "sec_with_match": float(parts[12]),
        })
    return rows


def song_from_path(p):
    """ /...music/Hale_Dil/upload_1.mp3 -> Hale_Dil """
    parent = os.path.basename(os.path.dirname(p)) if p else ""
    return parent or p


def aggregate(rows):
    by_song = defaultdict(lambda: {"max_score": 0, "total_score": 0, "best": None})
    for r in rows:
        song = song_from_path(r["match_path"])
        slot = by_song[song]
        slot["total_score"] += r["score"]
        if r["score"] > slot["max_score"]:
            slot["max_score"] = r["score"]
            slot["best"] = r
    ranked = sorted(
        by_song.items(),
        key=lambda kv: (kv[1]["max_score"], kv[1]["total_score"]),
        reverse=True,
    )
    return ranked


def query(jar, mp3_path, expected):
    t0 = time.time()
    try:
        res = run_java(jar, "query", mp3_path)
    except subprocess.TimeoutExpired as e:
        return common.make_test_record(
            mp3_path, expected, status="ERROR",
            error=f"timeout: {e}", time_sec=time.time() - t0,
        )
    elapsed = time.time() - t0
    if res.returncode != 0:
        return common.make_test_record(
            mp3_path, expected, status="ERROR",
            error=(res.stderr or "")[-300:],
            time_sec=elapsed,
        )

    rows = parse_query_lines(res.stdout)
    ranked = aggregate(rows)
    if not ranked:
        return common.make_test_record(
            mp3_path, expected, status="NO_MATCH", time_sec=elapsed,
        )

    top1_song, top1 = ranked[0]
    top2 = None
    if len(ranked) > 1:
        s, d = ranked[1]
        top2 = {
            "song_name": s,
            "score": d["max_score"],
            "total_score": d["total_score"],
        }
    return common.make_test_record(
        mp3_path, expected,
        matched=top1_song,
        confidence=top1["best"]["sec_with_match"] if top1["best"] else None,
        hashes_matched=top1["max_score"],
        top2=top2,
        time_sec=elapsed,
        status=common.classify(top1_song, expected),
        extra={
            "best_alignment": top1["best"],
            "candidates": len(ranked),
        },
    )


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Panako test runner")
    ap.add_argument("--data-dir", default=common.DEFAULT_DATA_DIR,
                    help="root data folder (e.g. dataset or music). Default: dataset")
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    jar = ensure_jar()
    print(f"[jar] {jar}")
    print(f"[db]  wiping {PANAKO_DB_DIR}")
    wipe_db()

    songs = common.discover_songs(data_dir)
    total_uploads = sum(len(v["uploads"]) for v in songs.values())
    total_full = sum(1 for v in songs.values() if v["test_full"])
    total_clips = sum(len(v["test_clips"]) for v in songs.values())
    print(f"\nDiscovered {len(songs)} songs in {data_dir}")
    print(f"  uploads = {total_uploads}, full = {total_full}, clips = {total_clips}\n")

    # --- Step 1: store all uploads ---
    print("=" * 70)
    print("  STEP 1 — Storing fingerprints")
    print("=" * 70)
    uploaded = []
    for song, data in songs.items():
        for up in data["uploads"]:
            label = f"{song}::{os.path.basename(up)}"
            ok, log = store(jar, up)
            if ok:
                uploaded.append(label)
                print(f"  [OK]  {label}")
            else:
                tail = log.strip().splitlines()[-3:] if log else []
                print(f"  [ERR] {label}\n    " + "\n    ".join(tail))
    print(f"\n  Stored {len(uploaded)}/{total_uploads}")

    # --- Step 2: query test.mp3 ---
    print("\n" + "=" * 70)
    print("  STEP 2 — Recognizing test.mp3")
    print("=" * 70)
    full_records = []
    idx = 0
    for song, data in songs.items():
        if not data["test_full"]:
            continue
        idx += 1
        rec = query(jar, data["test_full"], song)
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
            rec = query(jar, clip, song)
            clip_records.append(rec)
            common.print_progress(
                idx, total_clips, song, os.path.basename(clip),
                rec["status"], rec["matched"], song,
                conf=rec["confidence"], t=rec["time"], top2=rec["top2"],
            )

    path, summary = common.write_results(
        "panako", songs, uploaded, full_records, clip_records,
        data_dir=data_dir,
        extra_meta={"jar": jar, "strategy": "OLAF (default)"},
    )
    common.print_final_report(
        "panako", summary,
        common.build_per_song(songs, full_records, clip_records),
        path,
    )


if __name__ == "__main__":
    main()
