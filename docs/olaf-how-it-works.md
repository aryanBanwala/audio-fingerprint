# Olaf - How It Works

Olaf (Overly Lightweight Acoustic Fingerprinting) is a landmark-based acoustic fingerprinting system written in C. It identifies audio by extracting compact "fingerprints" from sound and matching them against a database of known tracks.

This document explains the algorithm and architecture behind Olaf.

## The Fingerprinting Pipeline

Olaf processes audio through four stages:

```
Audio Input -> Spectrogram (FFT) -> Event Points (Peaks) -> Fingerprints (Hashes) -> Store / Match
```

### 1. Audio Input

Olaf expects mono audio at **16 kHz**, 32-bit float format. All audio is converted to this format externally using `ffmpeg` before being fed to Olaf. This keeps the C core simple — it doesn't handle decoding or resampling.

The audio is processed in overlapping blocks:
- **Block size**: 1024 samples (~64ms at 16kHz)
- **Step size**: 128 samples (~8ms) — each block overlaps the previous by 896 samples

### 2. Spectrogram via FFT

Each audio block is transformed into the frequency domain using a **Fast Fourier Transform (FFT)**, powered by the PFFFT library. This produces a magnitude spectrum showing which frequencies are present and how loud they are.

The sequence of FFT frames forms a **spectrogram** — a 2D representation of the audio with time on one axis and frequency on the other. Each cell holds a magnitude value.

**Key parameters** (from `olaf_config.c`):
- Frequencies below bin 9 (~140 Hz) are ignored to filter out low rumble
- A minimum magnitude threshold (`0.001`) filters out silence

### 3. Event Point Extraction

Event points (EPs) are spectral **peaks** — the loudest points in local neighborhoods of the spectrogram. They represent the most distinctive moments in the audio.

Olaf finds peaks using a **max filter** (Van Herk/Gil-Werman algorithm) applied across both time and frequency:
- **Frequency filter**: 103 bins wide
- **Time filter**: 24 frames wide (~200ms)

A point is an event point only if it is the maximum in its local neighborhood. Each event point records:
- **Frequency bin** (which frequency)
- **Time index** (when it occurred)
- **Magnitude** (how loud)

Up to **60 event points** are extracted per audio block.

**Source**: `src/olaf_ep_extractor.c`

### 4. Fingerprint Generation

Individual event points aren't distinctive enough on their own. Olaf combines **3 event points** into a single fingerprint (configurable to 2 on embedded platforms for efficiency).

The fingerprint extractor pairs nearby event points with constraints:
- **Time distance**: between 2 and 33 frames (16ms–264ms apart)
- **Frequency distance**: between 1 and 128 bins apart

Each fingerprint is hashed into a compact **64-bit integer** that encodes the frequency and time relationships between its component event points. This hash is what gets stored and searched.

Up to **300 fingerprints** are generated per audio block.

**Source**: `src/olaf_fp_extractor.c`

### 5. Storage (LMDB Database)

When storing a track, Olaf writes each fingerprint hash to an **LMDB** (Lightning Memory-Mapped Database) B+-tree. Each entry maps:

```
fingerprint hash -> (audio_id, time_offset)
```

- **audio_id**: A Jenkins hash of the original filename, uniquely identifying the track
- **time_offset**: When in the track this fingerprint occurs

LMDB provides fast, persistent, memory-mapped storage with low overhead. The database lives at `~/.olaf/db/`.

On embedded platforms (ESP32) and in the browser (WASM), fingerprints are stored in a compiled-in header file instead of LMDB, since those environments can't use a filesystem database.

**Source**: `src/olaf_db.c` (LMDB), `src/olaf_db_mem.c` (in-memory)

### 6. Matching (Query)

When querying, Olaf extracts fingerprints from the query audio and looks up each hash in the database. For each hash match, it retrieves the stored `(audio_id, time_offset)`.

The matcher then looks for **time-aligned clusters** of matches:

1. For each candidate audio_id, collect all matching fingerprints
2. Compute the time difference: `ref_time - query_time` for each match
3. If many matches share the same time difference, they form an **alignment** — meaning the query and reference are playing the same audio at a consistent offset
4. Report matches with at least **6 aligned fingerprints** (`minMatchCount`)

A search range of **5 bins** around each hash allows for slight variations in the audio.

**Source**: `src/olaf_fp_matcher.c`

## Architecture

### Shared Core Design

Olaf's C core is shared across three platforms:

```
                    +-----------------+
                    |   C Core        |
                    |  (fingerprint   |
                    |   algorithm)    |
                    +-----------------+
                   /        |          \
          ESP32           Desktop        Browser
       (Arduino)       (Linux/macOS)     (WASM)
     in-memory DB       LMDB DB        in-memory DB
     mic input        ffmpeg input     Web Audio API
```

The only differences between platforms are:
- **How audio enters** the system (mic, file, Web Audio API)
- **How fingerprints are stored** (LMDB vs in-memory array)

### Module Responsibilities

| Module | File | Role |
|--------|------|------|
| Stream Processor | `olaf_stream_processor.c` | Reads audio blocks, runs the pipeline |
| EP Extractor | `olaf_ep_extractor.c` | Finds spectral peaks in FFT output |
| FP Extractor | `olaf_fp_extractor.c` | Combines peaks into fingerprint hashes |
| FP Matcher | `olaf_fp_matcher.c` | Matches query hashes against database |
| DB (LMDB) | `olaf_db.c` | Persistent B+-tree storage |
| DB (Memory) | `olaf_db_mem.c` | In-memory storage for embedded/WASM |
| Runner | `olaf_runner.c` | Orchestrates store/query/delete modes |
| Config | `olaf_config.c` | Compile-time configuration parameters |
| FFT | `pffft.c` | Fast Fourier Transform (PFFFT library) |

### Data Flow for a Query

```
1. ffmpeg converts audio to 16kHz mono f32le
2. olaf_c reads raw audio file
3. Stream processor splits into 1024-sample blocks (128-sample steps)
4. Each block -> FFT -> magnitude spectrum
5. EP extractor finds peaks in the spectrum
6. FP extractor combines peaks into fingerprint hashes
7. FP matcher looks up each hash in LMDB
8. Matcher clusters time-aligned hits per audio_id
9. Results with >= 6 aligned matches are reported
```

## Performance

Olaf is designed for speed:

- **Indexing**: ~27 fingerprints/second, ~2700x realtime
- **Querying**: ~2100x realtime (a 10-second clip is matched in ~5ms)
- **Database**: 100,000 tracks (340+ days of audio) fit in a 15GB database
- **Query at scale**: 80x realtime on the 100K track dataset

The B+-tree structure of LMDB gives logarithmic lookup complexity, so query performance scales well with database size.

## References

1. Wang, Avery L. "An Industrial-Strength Audio Search Algorithm" (2003) — the foundational landmark fingerprinting paper
2. Six, Joren and Leman, Marc. "Panako - A Scalable Acoustic Fingerprinting System Handling Time-Scale and Pitch Modification" (2014)
