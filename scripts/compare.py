"""
Compare results across all libraries.

Run:
  python3 scripts/compare.py                  # default: dataset
  python3 scripts/compare.py --data-dir music # override
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

LIBS = ["dejavu", "olaf", "panako", "neuralfp"]


def load(lib, data_tag):
    path = os.path.join(common.RESULTS_DIR, f"{lib}_{data_tag}_results.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_row(label, cells, widths):
    return "  " + label.ljust(widths[0]) + "".join(c.rjust(w) for c, w in zip(cells, widths[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset",
                    help="Which dataset's results to compare. Default: dataset")
    args = ap.parse_args()
    data_tag = os.path.basename(os.path.normpath(args.data_dir))

    data = {lib: load(lib, data_tag) for lib in LIBS}
    available = [lib for lib in LIBS if data[lib]]

    if not available:
        print(f"No '*_{data_tag}_results.json' results found in {common.RESULTS_DIR}")
        print("Run the test scripts first:")
        for lib in LIBS:
            print(f"  bash scripts/{lib}_test.sh --data-dir {data_tag}")
        return 1

    print("\n" + "=" * 78)
    print(f"  ACOUSTIC FINGERPRINT — {len(available)}-WAY COMPARISON  ({data_tag}/)")
    print("=" * 78)

    widths = [22] + [14] * len(available)
    header = fmt_row("Metric", available, widths)
    print(header)
    print("  " + "-" * (sum(widths) - 2))

    def row(label, getter):
        cells = []
        for lib in available:
            try:
                cells.append(str(getter(data[lib])))
            except Exception:
                cells.append("-")
        print(fmt_row(label, cells, widths))

    row("test.mp3 pass", lambda d: f"{d['summary']['test_full']['pass']}/{d['summary']['test_full']['total']}")
    row("test.mp3 acc", lambda d: d['summary']['test_full']['accuracy'])
    row("test.mp3 avg time", lambda d: f"{d['summary']['test_full']['avg_query_time_sec']}s")
    row("clip pass", lambda d: f"{d['summary']['test_clips']['pass']}/{d['summary']['test_clips']['total']}")
    row("clip acc", lambda d: d['summary']['test_clips']['accuracy'])
    row("clip avg time", lambda d: f"{d['summary']['test_clips']['avg_query_time_sec']}s")
    row("clip no_match", lambda d: d['summary']['test_clips']['no_match'])
    row("clip wrong match", lambda d: d['summary']['test_clips']['fail'])
    row("OVERALL pass", lambda d: f"{d['summary']['overall']['pass']}/{d['summary']['overall']['total']}")
    row("OVERALL acc", lambda d: d['summary']['overall']['accuracy'])
    row("uploads stored", lambda d: f"{d['db_verification']['uploaded']}/{d['db_verification']['expected_uploads']}")

    # Per-song side-by-side accuracy
    songs = sorted({s for d in data.values() if d for s in d["per_song"]})
    print("\n  Per-song clip accuracy:")
    print(fmt_row("song", available, widths))
    print("  " + "-" * (sum(widths) - 2))
    for s in songs:
        cells = []
        for lib in available:
            ps = data[lib]["per_song"].get(s, {})
            tot = ps.get("clip_pass", 0) + ps.get("clip_fail", 0) + ps.get("clip_nomatch", 0) + ps.get("clip_error", 0)
            if tot == 0:
                cells.append("-")
            else:
                cells.append(f"{ps['clip_pass']}/{tot}")
        print(fmt_row(s, cells, widths))

    print("\n  Results files:")
    for lib in available:
        print(f"    {lib}: scripts/results/{lib}_{data_tag}_results.json")
    missing = [lib for lib in LIBS if lib not in available]
    if missing:
        print(f"  Missing: {', '.join(missing)} — "
              f"run bash scripts/<lib>_test.sh --data-dir {data_tag}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
