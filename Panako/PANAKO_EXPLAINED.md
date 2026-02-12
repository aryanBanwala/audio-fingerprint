# Panako - Audio Fingerprinting System

## What is Panako?

Panako is a **Shazam-like audio fingerprinting system** written in Java. It can identify audio by extracting unique "fingerprints" from sound and matching them against a stored database. Think of it like a digital version of recognizing a song by hearing a few notes — except it works by analyzing the mathematical structure of the audio signal.

The default algorithm is called **OLAF** (Overly Lightweight Acoustic Fingerprinting), based on the original Shazam paper by Avery Wang (2003).

---

## How Does It Work?

The entire system works in two phases: **Store** (indexing songs) and **Query** (finding matches).

### Phase 1: Store (Building the Database)

```
Audio File (MP3/WAV/FLAC/etc.)
        |
        v
   FFmpeg Decode (converts to 16kHz mono PCM)
        |
        v
   FFT Spectrogram (time-frequency representation)
        |
        v
   Peak Detection (find loud/dominant points)
        |
        v
   Fingerprint Construction (group peaks into triples)
        |
        v
   Hash & Store in LMDB Database
```

#### Step 1: Audio Decoding

Panako uses **FFmpeg** as a universal audio decoder. Any format FFmpeg supports (MP3, WAV, FLAC, OGG, AAC, even video files) gets converted to:
- **16,000 Hz** sample rate (down from typical 44,100 Hz)
- **Mono** channel
- **16-bit signed** PCM

This simplification reduces data while keeping enough information for fingerprinting.

#### Step 2: Spectrogram via FFT

The audio signal is split into overlapping windows and analyzed using **Fast Fourier Transform (FFT)**:

- **FFT Size**: 1024 samples (~64ms at 16kHz)
- **Hop Size**: 128 samples (~8ms)
- **Window**: Hamming window (reduces spectral leakage)

This produces a **spectrogram** — a 2D grid where:
- X-axis = time (frames)
- Y-axis = frequency (512 bins, 0 to 8kHz)
- Value = magnitude (how loud that frequency is at that moment)

#### Step 3: Event Point Extraction

From the spectrogram, **event points** (spectral peaks) are detected — these are the loudest, most distinctive moments in the audio:

1. **Frequency filter**: Find local maxima across nearby frequency bins (using logarithmic MIDI-based spacing to match human hearing)
2. **Time filter**: Find local maxima across nearby time frames (25-frame window)
3. **2D peak**: A point must be a peak in BOTH dimensions simultaneously

Each event point is described as **(t, f, m)** — time frame, frequency bin, magnitude.

#### Step 4: Fingerprint Construction

Three event points are grouped together to form one **fingerprint**:

```
Fingerprint = (EventPoint1, EventPoint2, EventPoint3)
```

Constraints:
- Points must be ordered in time: t1 < t2 <= t3
- Frequency distance between points: 1 to 128 bins
- Time distance between points: 5 to 40 frames

#### Step 5: Hashing

Each fingerprint is compressed into a **64-bit hash** that encodes:
- Time differences between the three points
- Relative frequency orderings (which point is higher/lower)
- Relative magnitude orderings
- Quantized frequency values

This hash captures the **"shape"** of the spectral pattern — not the exact values, making it tolerant to small distortions.

#### Step 6: Storage in LMDB

Fingerprints are stored in **LMDB** (Lightning Memory-mapped Database), a fast B-Tree key-value store:

| Table | Key | Value |
|---|---|---|
| `olaf_fingerprints` | 64-bit hash | (resourceID, timestamp) |
| `olaf_resource_map` | resourceID | (path, duration, num_fingerprints) |

One hash can map to multiple songs (collisions are expected and handled).

---

### Phase 2: Query (Finding a Match)

```
Query Audio Clip
        |
        v
   Same extraction pipeline (decode -> FFT -> peaks -> fingerprints -> hashes)
        |
        v
   For each hash: search database with range [hash-2, hash+2]
        |
        v
   Collect all hits, group by song ID
        |
        v
   Time alignment filtering (histogram of time offsets)
        |
        v
   Score, validate, return results
```

#### How Matching Works

1. **Hash Lookup with Fuzzy Range**: For each query fingerprint hash, Panako searches the database for hashes within +-2 of the query hash. This fuzzy matching tolerates small audio distortions.

2. **Hit Grouping**: All matching database entries are grouped by their resource (song) ID.

3. **Time Alignment**: For each candidate song, Panako calculates `deltaT = matchTime - queryTime` for every hit. If a match is real, most hits should have the **same time offset** (the query clip came from a consistent position in the song). Random false matches produce scattered offsets.

4. **Scoring**: The number of time-aligned hits = the match score. Higher = more confident match.

5. **Validation Thresholds**:
   - Minimum 10 raw hits before filtering
   - Minimum 5 hits after time alignment
   - Match duration >= 5 seconds
   - At least 20% of query seconds must have matching fingerprints

---

## Our Test Setup

### Songs Used

We tested with 4 songs, each having multiple versions (original, covers, instrumentals, acapellas):

| Song | Stored Versions (uploads) | Validation (test) |
|---|---|---|
| **Dil Ka Jo Haal Hai** | Instrumental, Acapella, Electric Guitar Cover | Lyrical Original (Besharam) |
| **Hale Dil** | Piano Cover, Vocals Cover, Acapella, Electric Guitar | Lyrical Original (Murder 2) |
| **Lag Ja Gale** | Instrumental, Violin Cover, Harmonium Cover | Lata Mangeshkar Original |
| **One Love** | Drum Cover, Piano Karaoke, Vocals Only, Instrumental | Shubh Official Audio |

**14 files stored** = 131,098 fingerprints from ~49 minutes of audio.

### Test: Full Song Query

Querying the full original test.mp3 against the database:

| Song | Result | Matched With | Score | Coverage |
|---|---|---|---|---|
| Dil Ka Jo Haal Hai | MATCH | upload_1 (Instrumental) + upload_2 (Acapella) | 1531 + 963 | 62% + 49% |
| Hale Dil | Not tested (full) | - | - | - |
| Lag Ja Gale | Not tested (full) | - | - | - |
| One Love | Not tested (full) | - | - | - |

### Test: 10-Second Clip Query

Each test file was split into 10-second clips. Random clips were queried:

| Song | Clip | Result | Matched With | Score | Coverage |
|---|---|---|---|---|---|
| Dil Ka Jo Haal Hai | clip_7 | **MATCH** | upload_2 (Acapella) | 157 | 100% |
| Hale Dil | clip_11 | **MATCH** | upload_3 (Acapella) | 39 | 100% |
| Lag Ja Gale | clip_19 | **NO MATCH** | null | -1 | 0% |
| One Love | clip_4 | **MATCH** | upload_3 (Vocals Only) | 13 | 57% |

---

## Key Observations

### What Worked
- **3 out of 4 songs matched** even with just 10-second clips
- Matches were found primarily against **vocal versions** (acapella/vocals only) — because the original songs contain vocals, and vocal fingerprints overlap between the original and vocal-extracted versions
- Panako is extremely fast: queries complete in **<20ms** for hash matching

### What Didn't Work
- **Lag Ja Gale failed** — the stored versions were all pure instrumentals (violin, harmonium, generic instrumental) with no vocal overlap to the Lata Mangeshkar original
- **One Love had low confidence** (13 hits, 57% coverage) — barely crossed the threshold

### Why This Happens

Panako uses **exact audio fingerprinting** — it matches the actual acoustic signal, not the musical content. Two recordings of the same song by different artists produce completely different spectrograms and therefore different fingerprints.

It works well for:
- Same recording with added noise
- Same recording with different bitrate/format
- Same recording pitch-shifted or speed-changed (using Panako strategy, not Olaf)
- Clips from a longer recording

It does NOT work for:
- Cover versions (different recording, different instruments)
- Different arrangements of the same melody
- Instrumental vs vocal versions (unless significant audio overlap exists)

---

## Architecture Overview

```
be.panako/
├── cli/                    # Command-line interface
│   ├── Panako.java         # Main entry point
│   ├── Store.java          # Store subcommand
│   ├── Query.java          # Query subcommand
│   ├── Monitor.java        # Real-time monitoring
│   ├── Stats.java          # Database statistics
│   └── ...
├── strategy/
│   ├── Strategy.java       # Abstract interface
│   └── olaf/               # OLAF algorithm (default)
│       ├── OlafStrategy.java          # Main orchestrator
│       ├── OlafEventPoint.java        # Spectral peak (t, f, m)
│       ├── OlafEventPointProcessor.java # FFT + peak detection
│       ├── OlafFingerprint.java       # 3-point fingerprint + hash
│       └── storage/
│           ├── OlafStorage.java       # Storage interface
│           ├── OlafStorageKV.java     # LMDB implementation
│           ├── OlafStorageFile.java   # File-based fallback
│           └── OlafStorageMemory.java # In-memory (testing)
└── util/                   # Utilities and configuration
```

## Dependencies

| Dependency | Purpose |
|---|---|
| **FFmpeg** | Universal audio decoding |
| **LMDB** | Fast key-value database for fingerprint storage |
| **JNR-FFI** | Java native interface for LMDB C library |
| **TarsosDSP** | Audio signal processing (FFT, windowing) |
| **Java 11+** | Runtime |

---

## Conclusion

Panako (OLAF) is an efficient, production-grade **exact audio matching** system. It excels at identifying recordings even from short, noisy clips — but it cannot detect cover songs or different arrangements of the same melody. For cover song detection, ML-based approaches like **Neural Audio Fingerprinting** would be needed, which learn higher-level musical features rather than raw acoustic patterns.
