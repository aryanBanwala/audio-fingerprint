# Olaf - Setup and Usage Guide

This guide covers how to install, build, and use Olaf for audio fingerprinting on macOS/Linux.

## Prerequisites

| Tool | Purpose | Install (macOS) | Install (Ubuntu/Debian) |
|------|---------|-----------------|------------------------|
| gcc/clang | Compile Olaf | `xcode-select --install` | `sudo apt install gcc` |
| make | Build system | Included with Xcode tools | `sudo apt install make` |
| ffmpeg | Audio decoding & conversion | `brew install ffmpeg` | `sudo apt install ffmpeg` |
| ruby | CLI wrapper script | Pre-installed on macOS | `sudo apt install ruby` |

## Building

```bash
cd olaf
make
```

This compiles the C source into `bin/olaf_c`. Build variants:

| Command | Output | Use case |
|---------|--------|----------|
| `make` | `bin/olaf_c` | Standard build with LMDB database |
| `make mem` | `bin/olaf_mem` | In-memory build (for testing embedded/WASM behavior) |
| `make web` | WASM files in `wasm/` | Browser build (requires Emscripten) |
| `make lib` | `bin/libolaf.so` | Shared library for Python wrapper |
| `make test` | `bin/olaf_tests` | Unit tests |

## Two Ways to Run Olaf

### Option A: Ruby CLI (recommended)

Install system-wide so the Ruby wrapper can find the binary:

```bash
sudo make install
```

This copies `olaf_c` to `/usr/local/bin/` and the `olaf` Ruby script alongside it. Then use the `olaf` command directly:

```bash
olaf store song.mp3
olaf query clip.mp3
olaf stats
```

### Option B: Direct binary (no install)

Use `bin/olaf_c` directly. You must manually convert audio to raw format first:

```bash
# Convert audio to raw format (mono, 16kHz, 32-bit float)
ffmpeg -i song.mp3 -ac 1 -ar 16000 -f f32le -acodec pcm_f32le song.raw

# Store
./bin/olaf_c store song.raw "song.mp3"

# Query
ffmpeg -i clip.mp3 -ac 1 -ar 16000 -f f32le -acodec pcm_f32le clip.raw
./bin/olaf_c query clip.raw "clip.mp3"

# Stats
./bin/olaf_c stats
```

The second argument (e.g., `"song.mp3"`) is the original filename — Olaf uses it to generate a unique ID and reports it back in query results.

## Commands

### Store (index audio)

Add audio to the fingerprint database:

```bash
olaf store song.mp3                    # Single file
olaf store /path/to/music/             # All audio in a folder (recursive)
olaf store playlist.txt                # List of file paths (one per line)
```

Supported formats: `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, `.mp4`, `.wv`, `.ape`, `.wma`

Olaf automatically skips files that are already indexed.

### Query (search for matches)

Find which stored track matches a given audio clip:

```bash
olaf query clip.mp3
```

Output format:
```
match_count, query_start(s), query_stop(s), ref_path, ref_id, ref_start(s), ref_stop(s)
257, 1.49, 8.82, song.mp3, 584065814, 1.49, 8.82
```

Options:
- `--threads n` — Use multiple threads for faster matching
- `--fragmented` — Split the query into 30s fragments and match each separately (useful for long unsegmented audio)
- `--no-identity-match` — Don't report a file matching itself (for deduplication)

### Query from microphone (live matching)

```bash
# macOS
ffmpeg -f avfoundation -i "none:default" -ac 1 -ar 16000 -f f32le -acodec pcm_f32le pipe:1 | olaf query
```

Olaf will continuously print matches as it recognizes audio from the mic.

### Delete (remove from index)

```bash
olaf delete song.mp3
```

### Dedup (find duplicates)

Stores all files then queries each one, reporting non-self matches:

```bash
olaf dedup /path/to/music/
olaf dedup --fragmented --threads 4 /path/to/music/
```

### Stats (database info)

```bash
olaf stats
```

Shows B-tree depth, number of entries, database size, and a per-track breakdown of fingerprint counts and durations.

### Clear (reset database)

```bash
olaf clear          # Asks for confirmation
olaf clear -f       # Force delete without confirmation
```

### Cache (parallel fingerprint extraction)

For large collections, extract fingerprints in parallel then store them:

```bash
gem install threach                    # Required for multi-threading
olaf cache --threads 8 /path/to/music/
olaf store_cached
```

## Database Location

Olaf stores its database at `~/.olaf/db/` and cached fingerprints at `~/.olaf/cache/`. These directories are created automatically on first use.

## Configuration

Olaf's parameters are set at compile time in `src/olaf_config.c`. Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audioSampleRate` | 16000 | Expected input sample rate (Hz) |
| `audioBlockSize` | 1024 | FFT window size (samples) |
| `audioStepSize` | 128 | Hop size between FFT frames (samples) |
| `numberOfEPsPerFP` | 3 | Event points per fingerprint (2 on ESP32) |
| `minMatchCount` | 6 | Minimum aligned matches to report a result |
| `maxDBCollisions` | 2000 | Max hash collisions to check per query |
| `searchRange` | 5 | Hash search range for fuzzy matching |

The Ruby script `olaf.rb` also has settings at the top of the file:

| Setting | Default | Description |
|---------|---------|-------------|
| `TARGET_SAMPLE_RATE` | 16000 | Resampling target for ffmpeg |
| `FRAGMENT_DURATION_IN_SECONDS` | 30 | Fragment size for `--fragmented` queries |
| `SKIP_DUPLICATES` | true | Skip already-indexed files on store |
| `CHECK_INCOMING_AUDIO` | true | Validate audio duration before storing |

## Docker

```bash
# Build
docker build -t olaf:1.0 .

# Store audio (mount your audio directory)
docker run -v $HOME/.olaf/docker_dbs:/root/.olaf -v $PWD:/root/audio olaf:1.0 olaf store song.mp3

# Query
docker run -v $HOME/.olaf/docker_dbs:/root/.olaf -v $PWD:/root/audio olaf:1.0 olaf query clip.mp3

# Stats
docker run -v $HOME/.olaf/docker_dbs:/root/.olaf olaf:1.0 olaf stats
```

## Python Wrapper

Use Olaf from Python via CFFI bindings:

```bash
make lib                                          # Build shared library
pip install -r python-wrapper/requirements.txt     # Install CFFI
python python-wrapper/setup.py                     # Build bindings
export LD_LIBRARY_PATH=$(pwd)/bin                  # Point to libolaf.so
python python-wrapper/spectrogram_example.py       # Run example
```

## Testing

```bash
# Unit tests
make test
./bin/olaf_tests

# Functional tests (requires ffmpeg, ffprobe, ruby)
ruby eval/olaf_functional_tests.rb

# Evaluation (requires SoX)
ruby eval/olaf_evaluation.rb /folder/with/music

# Benchmark
ruby eval/olaf_benchmark/olaf_benchmark.rb /folder/with/music
```

## Troubleshooting

**"No such file or directory - /usr/local/bin/olaf_c"**
The Ruby wrapper expects `olaf_c` at `/usr/local/bin/`. Either run `sudo make install` or use `bin/olaf_c` directly with raw audio files.

**Query returns no matches**
- Ensure the reference audio was stored first (`olaf store`)
- Check `olaf stats` to verify the database has entries
- Make sure the query clip is long enough (at least a few seconds)

**ffmpeg not found**
Install with `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux).
