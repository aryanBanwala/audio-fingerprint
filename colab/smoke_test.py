# %% [markdown]
# # Colab Smoke Test — Verify env before committing to full pipeline
#
# **Goal:** In ~5-10 min, verify:
# 1. ✅ A100 GPU available
# 2. ✅ Drive mount works
# 3. ✅ kaggle.json uploadable + Kaggle CLI works
# 4. ✅ Indian dataset downloadable (~100 MB)
# 5. ✅ TF + kapre + librosa + faiss imports
# 6. ✅ NeuralFP repo clones + imports
# 7. ✅ Random-init model produces inference (no training needed)
# 8. ✅ Result accuracy ≈ 27% (random init baseline matches Kaggle)
#
# **NO training**, **NO 11 GB FMA download**. Just confirm pipeline runs.

# %% [markdown]
# ## CELL 1 — Mount Drive + verify GPU

# %%
import subprocess, os, sys, time
from google.colab import drive

# Drive mount with retry (credential propagation sometimes fails first try)
for attempt in range(3):
    try:
        drive.mount('/content/drive', force_remount=True)
        if os.path.isdir('/content/drive/MyDrive'):
            print(f"✅ Drive mounted (attempt {attempt+1})")
            break
    except Exception as e:
        print(f"⚠️ Drive mount attempt {attempt+1} failed: {e}")
        time.sleep(3)
else:
    raise RuntimeError("Drive mount failed 3 times. Try: Runtime → Disconnect and delete runtime → Run all")

nvidia = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
gpu_line = [l for l in nvidia.split('\n') if 'A100' in l or 'V100' in l or 'T4' in l or 'L4' in l]
print(f"GPU: {gpu_line[0].strip() if gpu_line else 'unknown'}")

DRIVE_BASE = '/content/drive/MyDrive/btp'
os.makedirs(f'{DRIVE_BASE}/secrets', exist_ok=True)
print(f"✅ Drive ready at {DRIVE_BASE}")

# %% [markdown]
# ## CELL 2 — Kaggle auth (upload kaggle.json first time)

# %%
KAGGLE_JSON_DRIVE = f'{DRIVE_BASE}/secrets/kaggle.json'
KAGGLE_JSON_LOCAL = os.path.expanduser('~/.kaggle/kaggle.json')

if not os.path.isfile(KAGGLE_JSON_DRIVE):
    print("⚠️ kaggle.json not on Drive. Upload it now (file picker will appear)")
    print("   File location on your Mac: ~/.kaggle/kaggle.json")
    from google.colab import files
    uploaded = files.upload()
    src_name = list(uploaded.keys())[0]
    import shutil
    shutil.move(src_name, KAGGLE_JSON_DRIVE)
    print(f"✅ Saved to Drive (won't ask again next session)")

os.makedirs(os.path.dirname(KAGGLE_JSON_LOCAL), exist_ok=True)
subprocess.run(['cp', KAGGLE_JSON_DRIVE, KAGGLE_JSON_LOCAL])
os.chmod(KAGGLE_JSON_LOCAL, 0o600)
subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)

# Verify Kaggle CLI works
r = subprocess.run(['kaggle', 'datasets', 'list', '-s', 'audio-fp-indian-music', '--user', 'aryanbanwala97'],
                   capture_output=True, text=True)
print(f"Kaggle CLI test: {'✅ OK' if 'aryanbanwala97' in r.stdout else '❌ FAILED'}")
print(r.stdout[:200])

# %% [markdown]
# ## CELL 3 — Install deps + verify imports

# %%
import importlib
deps = {'kapre': 'kapre==0.3.5', 'faiss': 'faiss-cpu', 'tf_keras': 'tf_keras', 'librosa': 'librosa', 'pyyaml': 'pyyaml'}
for pkg, target in deps.items():
    try:
        importlib.import_module(pkg.replace('-', '_'))
        print(f"  ✅ {pkg} present")
    except ImportError:
        r = subprocess.run(['pip', 'install', '-q', target], capture_output=True, text=True)
        print(f"  {'✅' if r.returncode==0 else '❌'} {pkg} install: {r.returncode}")

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
print(f"TF: {tf.__version__}, GPUs: {gpus}")
print("✅ Deps installed + GPU visible to TF")

# %% [markdown]
# ## CELL 4 — Download Indian dataset (small, ~100 MB)

# %%
DATASETS_DIR = f'{DRIVE_BASE}/datasets'
INDIAN_PARENT = f'{DATASETS_DIR}/audio-fp-indian-music-btp'  # nested location (if present)
os.makedirs(DATASETS_DIR, exist_ok=True)

# Check if dataset already there (either nested or top-level)
import glob
def find_indian_root(base):
    """Find directory that contains the song folders (each with original.mp3)."""
    if not os.path.isdir(base):
        return None
    # Direct children that have original.mp3 → base IS the root
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'original.mp3')):
            return base
    # One level deeper
    for d in os.listdir(base):
        p = os.path.join(base, d)
        if os.path.isdir(p):
            for d2 in os.listdir(p):
                p2 = os.path.join(p, d2)
                if os.path.isdir(p2) and os.path.isfile(os.path.join(p2, 'original.mp3')):
                    return p
    return None

INDIAN_DIR = find_indian_root(DATASETS_DIR)
if INDIAN_DIR is None:
    print("Downloading Indian dataset from Kaggle...")
    subprocess.run(['kaggle', 'datasets', 'download',
                    '-d', 'aryanbanwala97/audio-fp-indian-music-btp',
                    '-p', DATASETS_DIR, '--unzip'], check=True)
    INDIAN_DIR = find_indian_root(DATASETS_DIR)

assert INDIAN_DIR, f"Could not locate songs after download under {DATASETS_DIR}"
print(f"✅ Indian: {INDIAN_DIR}")

song_dirs = sorted([d for d in os.listdir(INDIAN_DIR)
                    if os.path.isdir(os.path.join(INDIAN_DIR, d))
                    and os.path.isfile(os.path.join(INDIAN_DIR, d, 'original.mp3'))])
n_songs = len(song_dirs)
n_clips = sum(len(glob.glob(os.path.join(INDIAN_DIR, sd, 'remix', 'clips_*', 'clip_*.mp3'))) for sd in song_dirs)
print(f"  {n_songs} songs, {n_clips} clip files")
assert n_songs == 21, f"Expected 21 songs, got {n_songs}"
assert n_clips >= 250, f"Expected ≥250 clips, got {n_clips}"

# %% [markdown]
# ## CELL 5 — Clone NeuralFP + verify imports

# %%
WORK = '/content/NeuralFP'
if os.path.isdir(WORK):
    subprocess.run(['rm', '-rf', WORK])
subprocess.run(['git', 'clone', '-q', 'https://github.com/mimbres/neural-audio-fp', WORK], check=True)
sys.path.insert(0, WORK)
%cd {WORK}

# Patch deprecated TF API
trainer_path = f'{WORK}/model/trainer.py'
with open(trainer_path) as f:
    src = f.read()
if 'tf.keras.experimental.CosineDecay' in src:
    src = src.replace('tf.keras.experimental.CosineDecay', 'tf.keras.optimizers.schedules.CosineDecay')
    with open(trainer_path, 'w') as f:
        f.write(src)

import yaml
with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)

from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter
print("✅ NeuralFP imports OK")

# %% [markdown]
# ## CELL 6 — Build random-init model + run quick inference

# %%
import numpy as np
import librosa
import faiss

EMB_SZ = int(cfg['MODEL']['EMB_SZ'])
FS = int(cfg['MODEL']['FS'])
DUR = float(cfg['MODEL']['DUR'])
SEG_LEN = int(FS * DUR)
HOP_LEN = int(FS * 0.5)

m_pre = get_melspec_layer(cfg, trainable=False)
m_fp  = get_fingerprinter(cfg, trainable=False)
dummy = np.zeros((1, 1, SEG_LEN), dtype=np.float32)
_ = m_fp(m_pre(dummy, training=False), training=False)
print(f"✅ Random-init model: {len(m_fp.variables)} vars")

# Inventory
songs = {}
for sd in song_dirs:
    sp = os.path.join(INDIAN_DIR, sd)
    original = os.path.join(sp, 'original.mp3')
    if not os.path.isfile(original): continue
    remix_full = sorted(glob.glob(os.path.join(sp, 'remix', 'remix_*.mp3')))
    clips = sorted(glob.glob(os.path.join(sp, 'remix', 'clips_*', 'clip_*.mp3')))
    songs[sd] = {'original': original, 'remix_full': remix_full[0] if remix_full else None, 'clips': clips}

def load_and_slice(path):
    audio, _ = librosa.load(path, sr=FS, mono=True)
    if len(audio) < SEG_LEN:
        audio = np.pad(audio, (0, SEG_LEN - len(audio)))
    n_segs = max(1, 1 + (len(audio) - SEG_LEN) // HOP_LEN)
    segments = np.zeros((n_segs, 1, SEG_LEN), dtype=np.float32)
    for i in range(n_segs):
        segments[i, 0, :] = audio[i*HOP_LEN:i*HOP_LEN+SEG_LEN]
    return segments

def embed_batch(segments, batch_size=128):
    out = []
    for i in range(0, len(segments), batch_size):
        spec = m_pre(segments[i:i+batch_size], training=False)
        emb = m_fp(spec, training=False)
        out.append(emb.numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

# Build DB + run all queries
print("\n=== Building DB + running 273 queries (random-init) ===")
t0 = time.time()
db_vectors, db_song_ids = [], []
song_id_to_name = {}
for sid, sn in enumerate(sorted(songs.keys())):
    song_id_to_name[sid] = sn
    embs = embed_batch(load_and_slice(songs[sn]['original']))
    db_vectors.append(embs)
    db_song_ids.extend([sid] * len(embs))
db_matrix = np.concatenate(db_vectors, axis=0)
db_song_ids_arr = np.array(db_song_ids, dtype=np.int32)
index = faiss.IndexFlatIP(EMB_SZ)
index.add(db_matrix)
print(f"  DB built ({db_matrix.shape}) in {time.time()-t0:.1f}s")

t0 = time.time()
total = 0; passed = 0
for sn in sorted(songs.keys()):
    s = songs[sn]
    queries = ([s['remix_full']] if s['remix_full'] else []) + s['clips']
    for qpath in queries:
        try:
            segs = load_and_slice(qpath)
            embs = embed_batch(segs)
            D, I = index.search(embs, 5)
            cand_scores = {}
            for row_d, row_i in zip(D, I):
                for d, idx in zip(row_d, row_i):
                    sid = int(db_song_ids_arr[idx])
                    cand_scores[sid] = cand_scores.get(sid, 0.0) + float(d)
            best = max(cand_scores, key=cand_scores.get)
            if song_id_to_name[best] == sn:
                passed += 1
        except Exception as e:
            pass
        total += 1

print(f"\n=== SMOKE TEST RESULT ===")
print(f"  GPU:      {gpu_line[0].strip() if gpu_line else 'unknown'}")
print(f"  Songs:    {len(songs)}")
print(f"  Queries:  {total}")
print(f"  Passed:   {passed}/{total} = {100*passed/total:.1f}%")
print(f"  Inference time: {time.time()-t0:.1f}s")
print(f"  Random baseline: 4.7%")
print(f"  Expected (random-init NeuralFP on Kaggle): ~27%")
print(f"")
if 20 <= 100*passed/total <= 35:
    print(f"✅ ✅ ✅  SMOKE TEST PASSED — pipeline works! Now run full_pipeline.ipynb")
else:
    print(f"⚠️ accuracy out of expected range — check setup")
