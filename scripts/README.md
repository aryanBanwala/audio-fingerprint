# scripts/ — 3-way audio fingerprint benchmark

Uniform test harness for **Dejavu**, **Olaf**, and **Panako** against the
`music/` dataset at the repo root. Each library script:

1. Wipes its DB (fresh state every run)
2. Fingerprints **every** `upload_*.mp3` per song folder under that song name
3. Queries the song's `test.mp3` (full remix) and `test/clip_*.mp3` (short clips)
4. Writes `scripts/results/<library>_results.json` with a uniform schema

## Music folder layout (input)

```
music/<Song_Name>/
  upload_1.mp3, upload_2.mp3, ...   # all uploaded under <Song_Name>
  test.mp3                          # 1 full-length remix to recognize
  test/clip_1.mp3, clip_2.mp3, ...  # short clips of the remix to recognize
```

PASS = top-1 matched song equals folder name. FAIL = matched a different song.
NO_MATCH = library returned nothing. ERROR = exception during recognition.

## Run

From the repo root:

```bash
bash scripts/dejavu_test.sh   # uses Docker (postgres + python container)
bash scripts/olaf_test.sh     # builds olaf C binary if missing
bash scripts/panako_test.sh   # builds panako shadow jar if missing
python3 scripts/compare.py    # 3-way summary table once results exist
```

You can run them in any order. Results are written to `scripts/results/`.

## Prerequisites

- `ffmpeg` (audio decoding for all 3)
- **Dejavu**: Docker
- **Olaf**: `gcc`, `make` (auto-builds `olaf/bin/olaf_c`)
- **Panako**: Java 11+, internet on first build (auto-builds `Panako/build/libs/panako-*-all.jar`)

## Output schema (`scripts/results/<lib>_results.json`)

```jsonc
{
  "library": "olaf",
  "timestamp": "...",
  "db_verification": {
    "songs_in_dataset": 4,
    "expected_uploads": 14,
    "uploaded": 14,
    "all_uploaded": true,
    "uploads_per_song": { "Hale_Dil": 4, ... }
  },
  "summary": {
    "test_full":  { "total": 4,  "pass", "fail", "no_match", "error",
                     "accuracy": "75.0%", "avg_query_time_sec": 0.38 },
    "test_clips": { "total": 75, ... },
    "overall":    { "total": 79, "pass", "fail", "accuracy" }
  },
  "per_song": {
    "Hale_Dil": { "full_pass", "clip_pass", "clip_fail", "clip_nomatch", ... }
  },
  "uploaded": [ "Hale_Dil::upload_1", ... ],
  "test_full":  [ /* per-test records, FAILed first */ ],
  "test_clips": [ /* per-test records, FAILed first */ ]
}
```

Each per-test record has: `file, expected, matched, confidence, hashes_matched,
top2, time, status`.

## Library-specific notes

| Library | DB location | Wipe strategy |
|---------|-------------|---------------|
| Dejavu  | postgres in `dejavu` container | `docker compose down -v` |
| Olaf    | `~/.olaf/db/` (LMDB)           | `rm -rf ~/.olaf/db` |
| Panako  | `~/.panako/dbs/` (LMDB)        | `rm -rf ~/.panako` |

**Olaf** stores each upload under id `<Song>::<upload_basename>`; the song name
is recovered by splitting the returned `ref_path` on `::`.
**Panako** stores by file path; song name = parent directory of the matched path.
**Dejavu** stores under `song_name = <Song>` for every upload (postgres groups
by name natively).

The "confidence" field is library-specific and not directly comparable across
libraries — use it for relative ranking within one library only:

- Dejavu: `fingerprinted_confidence` (0.0–1.0)
- Olaf:   max single-alignment match count
- Panako: `Seconds with match (%)` (0.0–1.0)

The `hashes_matched` field is similarly library-specific. Use the `pass/fail/
no_match/accuracy` metrics in `summary` for cross-library comparison.
