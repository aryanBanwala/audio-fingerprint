# Panako - How It Works

Panako is a **Shazam-like audio fingerprinting system** in Java. It identifies audio by extracting unique fingerprints from sound and matching them against a database. The default algorithm is **OLAF** (Overly Lightweight Acoustic Fingerprinting), based on the Shazam paper (Wang, 2003).

---

## The Pipeline

```
Audio File → FFmpeg (16kHz mono) → FFT Spectrogram → Peak Detection → Fingerprints → 64-bit Hash → LMDB Database
```

### 1. Decode
FFmpeg converts any audio/video format to 16kHz, mono, 16-bit PCM.

### 2. Spectrogram
FFT (1024-sample window, 128-sample hop) converts audio into a time-frequency grid — showing which frequencies are loud at each moment.

### 3. Peak Detection
Local maxima are found in both time and frequency dimensions. These **event points** `(time, frequency, magnitude)` represent the most distinctive moments in the audio.

### 4. Fingerprint
Three event points are grouped into one fingerprint. Constraints ensure the points are spaced apart in time (5-40 frames) and frequency (1-128 bins).

### 5. Hash
Each fingerprint is compressed into a **64-bit hash** encoding the relative shape — time differences, frequency orderings, magnitude orderings. This makes it tolerant to minor distortions.

### 6. Store
Hashes go into **LMDB** (a fast B-Tree key-value store):
- Key = hash, Value = (songID, timestamp)
- One hash can map to multiple songs

---

## Querying

Same pipeline extracts hashes from the query clip, then:

1. **Fuzzy lookup** — search database for hashes within +-2 of each query hash
2. **Group by song** — collect all hits per song ID
3. **Time alignment** — correct matches have consistent time offsets; random matches don't
4. **Score** — count of time-aligned hits. Thresholds: min 5 filtered hits, min 5s duration, min 20% coverage

---

## Our Test Results

**14 files stored** (covers, instrumentals, acapellas) across 4 songs. Queried with 10-second clips of the originals:

| Song | Match? | Matched With | Score | Coverage |
|---|---|---|---|---|
| Dil Ka Jo Haal Hai | **YES** | Acapella version | 157 | 100% |
| Hale Dil | **YES** | Acapella version | 39 | 100% |
| Lag Ja Gale | **NO** | - | - | 0% |
| One Love | **YES** | Vocals Only version | 13 | 57% |

**Pattern**: Matches happen when stored audio shares the same vocal track as the original. Pure instrumentals/covers with different recordings don't match.

---

## Limitations

**Works for**: Same recording with noise, different bitrate/format, pitch-shift, clips from a longer file.

**Doesn't work for**: Cover songs, different arrangements, instrumental vs vocal (unless audio overlaps). Panako matches the **acoustic signal**, not the **melody**.

For cover song detection, ML-based approaches (like Neural Audio Fingerprinting) are needed — they learn musical features rather than raw waveform patterns.
