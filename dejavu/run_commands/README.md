# Dejavu Run Commands

## How it works (2 steps)
1. **FINGERPRINT** = Load songs into database (one time per song)
2. **RECOGNIZE** = Match unknown audio against database

## Quick Start (run from dejavu/ root folder)

### Step 0: Start Docker containers
```bash
docker compose up -d
docker compose run python /bin/bash
```
You are now inside the container. All commands below run INSIDE the container.

### Step 1: Fingerprint (load) songs
```bash
# Fingerprint all mp3 files in the mp3/ folder
python dejavu.py -c dejavu_docker.cnf --fingerprint mp3 mp3

# Fingerprint all wav files in the test/ folder
python dejavu.py -c dejavu_docker.cnf --fingerprint test wav

# Fingerprint a single file
python dejavu.py -c dejavu_docker.cnf --fingerprint path/to/song.mp3
```

### Step 2: Recognize (match) a song
```bash
# Recognize from a file
python dejavu.py -c dejavu_docker.cnf --recognize file path/to/unknown_clip.mp3

# Recognize from microphone (10 seconds)
python dejavu.py -c dejavu_docker.cnf --recognize mic 10
```

### Step 3: When done
```bash
exit
docker compose down
```

## Adding your own songs
Put your mp3/wav files in the `mp3/` folder (or any folder), then fingerprint them.

## Important Notes
- Fingerprinting is ONE TIME per song — Dejavu remembers what it already fingerprinted
- You can keep adding songs anytime without losing previous fingerprints
- Microphone recognition does NOT work inside Docker (use file recognition instead)
- The database persists as long as you don't run `docker compose down -v`
