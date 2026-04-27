"""
Olaf test runner — runs on host (no docker).

Per-upload store ID: "<SongName>::<upload_basename>"  (e.g. "Hale_Dil::upload_1")
Query results are grouped by ref_path; song_name = part before "::".
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

OLAF_BIN = os.path.join(common.ROOT, "olaf", "bin", "olaf_c")
OLAF_DB = os.path.expanduser("~/.olaf/db")


def ensure_olaf_built():
    if os.path.isfile(OLAF_BIN):
        return
    print(f"[build] olaf binary not found, building -> {OLAF_BIN}")
    subprocess.run(["make"], cwd=os.path.join(common.ROOT, "olaf"), check=True)


def wipe_db():
    """Olaf stores its LMDB at ~/.olaf/db/ — clear it for a fresh run."""
    if os.path.isdir(OLAF_DB):
        shutil.rmtree(OLAF_DB)
    os.makedirs(OLAF_DB, exist_ok=True)


def to_raw(mp3_path, raw_path):
    """ffmpeg: mono, 16kHz, 32-bit float LE — Olaf's expected raw format."""
    subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", mp3_path,
            "-ac", "1", "-ar", "16000",
            "-f", "f32le", "-acodec", "pcm_f32le",
            raw_path,
        ],
        check=True,
    )


def store(raw_path, store_id):
    res = subprocess.run(
        [OLAF_BIN, "store", raw_path, store_id],
        capture_output=True, text=True,
    )
    return res.returncode == 0, (res.stdout + res.stderr)


def query(raw_path, query_id):
    """Returns list of dicts: [{count, q_start, q_stop, ref_path, ref_id, ref_start, ref_stop}, ...]"""
    res = subprocess.run(
        [OLAF_BIN, "query", raw_path, query_id],
        capture_output=True, text=True,
    )
    rows = []
    for line in (res.stdout or "").splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            count = int(parts[0])
        except ValueError:
            continue
        if count <= 0:
            continue
        ref_path = parts[3]
        if not ref_path:
            continue
        rows.append({
            "count": count,
            "q_start": float(parts[1]),
            "q_stop": float(parts[2]),
            "ref_path": ref_path,
            "ref_id": parts[4],
            "ref_start": float(parts[5]),
            "ref_stop": float(parts[6]),
        })
    return rows


def song_from_ref(ref_path):
    """'Hale_Dil::upload_1' -> 'Hale_Dil'"""
    return ref_path.split("::", 1)[0] if "::" in ref_path else ref_path


def aggregate(rows):
    """Group by song; song's score = max single-alignment count, also keep total count."""
    by_song = defaultdict(lambda: {"max_count": 0, "total_count": 0, "best": None})
    for r in rows:
        song = song_from_ref(r["ref_path"])
        slot = by_song[song]
        slot["total_count"] += r["count"]
        if r["count"] > slot["max_count"]:
            slot["max_count"] = r["count"]
            slot["best"] = r
    ranked = sorted(
        by_song.items(),
        key=lambda kv: (kv[1]["max_count"], kv[1]["total_count"]),
        reverse=True,
    )
    return ranked


def recognize(mp3_path, expected, query_id, tmpdir):
    raw = os.path.join(tmpdir, "q.raw")
    t0 = time.time()
    try:
        to_raw(mp3_path, raw)
        rows = query(raw, query_id)
    except Exception as e:
        return common.make_test_record(
            mp3_path, expected, status="ERROR", error=e,
            time_sec=time.time() - t0,
        )
    elapsed = time.time() - t0

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
            "match_count": d["max_count"],
            "total_count": d["total_count"],
        }

    return common.make_test_record(
        mp3_path, expected,
        matched=top1_song,
        confidence=top1["max_count"],  # use match count as a proxy for confidence
        hashes_matched=top1["total_count"],
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
    ap = argparse.ArgumentParser(description="Olaf test runner")
    ap.add_argument("--data-dir", default=common.DEFAULT_DATA_DIR,
                    help="root data folder (e.g. dataset or music). Default: dataset")
    args = ap.parse_args()
    data_dir = os.path.abspath(args.data_dir)

    ensure_olaf_built()
    print(f"[db] wiping {OLAF_DB}")
    wipe_db()

    songs = common.discover_songs(data_dir)
    total_uploads = sum(len(v["uploads"]) for v in songs.values())
    total_full = sum(1 for v in songs.values() if v["test_full"])
    total_clips = sum(len(v["test_clips"]) for v in songs.values())
    print(f"\nDiscovered {len(songs)} songs in {data_dir}")
    print(f"  uploads = {total_uploads}, full = {total_full}, clips = {total_clips}\n")

    tmpdir = tempfile.mkdtemp(prefix="olaf_test_")

    # --- Step 1: store all uploads ---
    print("=" * 70)
    print("  STEP 1 — Storing fingerprints")
    print("=" * 70)
    uploaded = []
    try:
        for song, data in songs.items():
            for up in data["uploads"]:
                base = os.path.splitext(os.path.basename(up))[0]
                store_id = f"{song}::{base}"
                raw = os.path.join(tmpdir, f"{song}_{base}.raw")
                try:
                    to_raw(up, raw)
                    ok, log = store(raw, store_id)
                    if ok:
                        uploaded.append(store_id)
                        print(f"  [OK] {store_id}")
                    else:
                        print(f"  [ERR] {store_id}\n    {log.strip()}")
                except Exception as e:
                    print(f"  [ERR] {store_id}: {e}")
                finally:
                    if os.path.exists(raw):
                        os.remove(raw)
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
            rec = recognize(data["test_full"], song, f"{song}_test_full", tmpdir)
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
                rec = recognize(clip, song, f"{song}_clip_{idx}", tmpdir)
                clip_records.append(rec)
                common.print_progress(
                    idx, total_clips, song, os.path.basename(clip),
                    rec["status"], rec["matched"], song,
                    conf=rec["confidence"], t=rec["time"], top2=rec["top2"],
                )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    path, summary = common.write_results(
        "olaf", songs, uploaded, full_records, clip_records,
        data_dir=data_dir,
    )
    common.print_final_report(
        "olaf", summary,
        common.build_per_song(songs, full_records, clip_records),
        path,
    )


if __name__ == "__main__":
    main()
