"""
Shared helpers for the 3 audio-fingerprint test scripts (Dejavu, Olaf, Panako).

Two folder layouts are auto-detected:

  music/<Song>/                       dataset/<Song>/
    upload_*.mp3   (uploads)            original.mp3              (upload)
    test.mp3       (full remix)         remix/remix_1.mp3         (full remix)
    test/clip_*    (clips)              remix/clips_1/clip_*.mp3  (clips)

Per-song:
  PASS     = top-1 matched song_name == folder name
  FAIL     = top-1 matched some other song
  NO_MATCH = library returned nothing
  ERROR    = exception during recognition
"""

import glob
import json
import os
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MUSIC_DIR = os.path.join(ROOT, "music")
DATASET_DIR = os.path.join(ROOT, "dataset")
DEFAULT_DATA_DIR = DATASET_DIR  # dataset/ is the larger benchmark
RESULTS_DIR = os.path.join(ROOT, "scripts", "results")


def _discover_dataset_layout(song_dir):
    """dataset/<Song>/{original.mp3, remix/remix_1.mp3, remix/clips_1/*.mp3}"""
    original = os.path.join(song_dir, "original.mp3")
    if not os.path.isfile(original):
        return None
    remix_dir = os.path.join(song_dir, "remix")
    remix_files = sorted(glob.glob(os.path.join(remix_dir, "remix_*.mp3")))
    clip_files = sorted(glob.glob(os.path.join(remix_dir, "clips_*", "clip_*.mp3")))
    return {
        "uploads": [original],
        "test_full": remix_files[0] if remix_files else None,
        "test_clips": clip_files,
    }


def _discover_music_layout(song_dir):
    """music/<Song>/{upload_*.mp3, test.mp3, test/clip_*.mp3}"""
    uploads = sorted(glob.glob(os.path.join(song_dir, "upload_*.mp3")))
    if not uploads:
        return None
    test_full = os.path.join(song_dir, "test.mp3")
    return {
        "uploads": uploads,
        "test_full": test_full if os.path.isfile(test_full) else None,
        "test_clips": sorted(glob.glob(os.path.join(song_dir, "test", "clip_*.mp3"))),
    }


def discover_songs(data_dir=DEFAULT_DATA_DIR):
    """Walk data_dir, auto-detect layout per song, return uniform dict.

    Returns: { song_name: {"uploads": [...], "test_full": str|None, "test_clips": [...]} }
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    songs = {}
    for song_name in sorted(os.listdir(data_dir)):
        song_dir = os.path.join(data_dir, song_name)
        if not os.path.isdir(song_dir):
            continue
        record = _discover_dataset_layout(song_dir) or _discover_music_layout(song_dir)
        if record is None:
            continue
        songs[song_name] = record
    return songs


def results_path(library, data_dir):
    """e.g. scripts/results/dejavu_dataset_results.json"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = os.path.basename(os.path.normpath(data_dir))
    return os.path.join(RESULTS_DIR, f"{library}_{tag}_results.json")


def make_test_record(filepath, expected, matched=None, confidence=None,
                     hashes_matched=None, top2=None, time_sec=0.0,
                     status="FAIL", error=None, extra=None):
    """Uniform per-test record across all 3 libraries."""
    rec = {
        "file": os.path.relpath(filepath, ROOT) if os.path.isabs(filepath) else filepath,
        "expected": expected,
        "matched": matched,
        "confidence": confidence,
        "hashes_matched": hashes_matched,
        "top2": top2,
        "time": round(float(time_sec), 3),
        "status": status,
    }
    if error is not None:
        rec["error"] = str(error)
    if extra:
        rec.update(extra)
    return rec


def classify(matched, expected, no_match=False, errored=False):
    if errored:
        return "ERROR"
    if no_match or matched is None:
        return "NO_MATCH"
    return "PASS" if matched == expected else "FAIL"


def _count(records, status):
    return sum(1 for r in records if r["status"] == status)


def _avg_time(records):
    times = [r["time"] for r in records if isinstance(r.get("time"), (int, float))]
    return round(sum(times) / len(times), 3) if times else 0.0


def build_per_song(songs, full_records, clip_records):
    per_song = {}
    for s in songs:
        per_song[s] = {
            "full_pass": 0, "full_fail": 0, "full_nomatch": 0, "full_error": 0,
            "clip_pass": 0, "clip_fail": 0, "clip_nomatch": 0, "clip_error": 0,
        }
    for r in full_records:
        bucket = {"PASS": "full_pass", "FAIL": "full_fail",
                  "NO_MATCH": "full_nomatch", "ERROR": "full_error"}[r["status"]]
        per_song.setdefault(r["expected"], per_song.get(r["expected"], {}))[bucket] += 1
    for r in clip_records:
        bucket = {"PASS": "clip_pass", "FAIL": "clip_fail",
                  "NO_MATCH": "clip_nomatch", "ERROR": "clip_error"}[r["status"]]
        per_song.setdefault(r["expected"], per_song.get(r["expected"], {}))[bucket] += 1

    for s, d in per_song.items():
        d["total_tests"] = (d["full_pass"] + d["full_fail"] + d["full_nomatch"] + d["full_error"]
                            + d["clip_pass"] + d["clip_fail"] + d["clip_nomatch"] + d["clip_error"])
        d["total_pass"] = d["full_pass"] + d["clip_pass"]
        d["total_fail"] = d["total_tests"] - d["total_pass"]
    return per_song


def build_summary(full_records, clip_records):
    def block(records):
        total = len(records)
        p = _count(records, "PASS")
        f = _count(records, "FAIL")
        nm = _count(records, "NO_MATCH")
        er = _count(records, "ERROR")
        return {
            "total": total,
            "pass": p,
            "fail": f,
            "no_match": nm,
            "error": er,
            "accuracy": f"{p/total*100:.1f}%" if total else "N/A",
            "avg_query_time_sec": _avg_time(records),
        }

    full_block = block(full_records)
    clip_block = block(clip_records)
    overall_total = full_block["total"] + clip_block["total"]
    overall_pass = full_block["pass"] + clip_block["pass"]
    overall_fail = overall_total - overall_pass

    return {
        "test_full": full_block,
        "test_clips": clip_block,
        "overall": {
            "total": overall_total,
            "pass": overall_pass,
            "fail": overall_fail,
            "accuracy": f"{overall_pass/overall_total*100:.1f}%" if overall_total else "N/A",
        },
    }


def write_results(library, songs, uploaded, full_records, clip_records,
                  data_dir=None, output_path=None, extra_meta=None):
    """Sort tests (failed first), build summary, write JSON."""
    full_records.sort(key=lambda r: (r["status"] == "PASS", r["expected"]))
    clip_records.sort(key=lambda r: (r["status"] == "PASS", r["expected"]))

    expected_uploads = sum(len(v["uploads"]) for v in songs.values())
    db_verification = {
        "songs_in_dataset": len(songs),
        "expected_uploads": expected_uploads,
        "uploaded": len(uploaded),
        "all_uploaded": len(uploaded) == expected_uploads,
        "uploads_per_song": {s: len(v["uploads"]) for s, v in songs.items()},
    }

    summary = build_summary(full_records, clip_records)
    per_song = build_per_song(songs, full_records, clip_records)

    out = {
        "library": library,
        "data_dir": os.path.basename(os.path.normpath(data_dir)) if data_dir else None,
        "timestamp": datetime.now().isoformat(),
        "db_verification": db_verification,
        "summary": summary,
        "per_song": per_song,
        "uploaded": sorted(uploaded),
        "test_full": full_records,
        "test_clips": clip_records,
    }
    if extra_meta:
        out["meta"] = extra_meta

    if output_path is None:
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        output_path = results_path(library, data_dir)

    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    return output_path, summary


def print_progress(idx, total, song, basename, status, matched, expected, conf=None, t=None, top2=None):
    """Pretty per-test line for stdout."""
    tag = {
        "PASS": "PASS",
        "FAIL": "FAIL",
        "NO_MATCH": "NO MATCH",
        "ERROR": "ERROR",
    }.get(status, status)

    parts = [f"[{idx}/{total}] {song}/{basename} -> "]
    if status == "PASS":
        parts.append(f"{matched} [PASS]")
    elif status == "FAIL":
        parts.append(f"{matched} [FAIL] expected: {expected}")
    elif status == "NO_MATCH":
        parts.append("NO MATCH [FAIL]")
    else:
        parts.append(f"ERROR [{matched}]")

    extras = []
    if conf is not None:
        extras.append(f"conf={conf}")
    if t is not None:
        extras.append(f"time={t}s")
    if top2:
        n = top2.get("song_name") if isinstance(top2, dict) else top2
        extras.append(f"#2={n}")
    if extras:
        parts.append("  (" + ", ".join(extras) + ")")
    print("".join(parts))


def print_final_report(library, summary, per_song, output_path):
    print("\n" + "=" * 70)
    print(f"  {library.upper()} — FINAL RESULTS")
    print("=" * 70)
    f = summary["test_full"]
    c = summary["test_clips"]
    o = summary["overall"]
    print(f"  test.mp3:  {f['pass']}/{f['total']} pass  ({f['accuracy']})  avg {f['avg_query_time_sec']}s")
    print(f"  clips:     {c['pass']}/{c['total']} pass  ({c['accuracy']})  avg {c['avg_query_time_sec']}s")
    print(f"  overall:   {o['pass']}/{o['total']} pass  ({o['accuracy']})")
    print("\n  Per-song breakdown:")
    sorted_songs = sorted(per_song.items(), key=lambda x: x[1]["total_fail"], reverse=True)
    for s, d in sorted_songs:
        flag = "PASS" if d["total_fail"] == 0 else "FAIL"
        print(f"    [{flag}] {s}: full {d['full_pass']}/{d['full_pass']+d['full_fail']+d['full_nomatch']+d['full_error']}, "
              f"clips {d['clip_pass']}/{d['clip_pass']+d['clip_fail']+d['clip_nomatch']+d['clip_error']}")
    print(f"\n  Results: {output_path}")
