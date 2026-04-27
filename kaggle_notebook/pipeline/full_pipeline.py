# %% [markdown]
# # NeuralFP Full Pipeline — BTP-II
#
# **Goal:** Run end-to-end NeuralFP inference on our 21-song Indian music dataset.
# Validates pipeline (audio loading, embedding, FAISS, aggregation) BEFORE committing
# to long training runs.
#
# **Mode toggle (set in Cell 0):**
#  - `RUN_TRAINING = False` → 0-epoch run (random-init model, ~7-10 min)
#  - `RUN_TRAINING = True`  → 1-epoch first then inference (~40-50 min)
#
# **Output:** `/kaggle/working/neuralfp_dataset_results.json` (same schema as Dejavu/Olaf/Panako)

# %% [markdown]
# ## CELL 0 — Configuration

# %%
RUN_TRAINING = True    # multi-epoch run
TRAIN_EPOCHS = 10      # number of training epochs if RUN_TRAINING=True
WANDB_ENABLED = True
WANDB_PROJECT = "btp-neuralfp"
RUN_NAME = f"pipeline-{TRAIN_EPOCHS}ep" if RUN_TRAINING else "pipeline-0ep"
HOP_SECONDS = 0.5      # segment hop (50% overlap)
TOPK = 5               # FAISS top-K per query segment

print(f"=== CONFIG ===")
print(f"  RUN_TRAINING: {RUN_TRAINING}")
print(f"  TRAIN_EPOCHS: {TRAIN_EPOCHS}")
print(f"  WANDB_ENABLED: {WANDB_ENABLED}")
print(f"  RUN_NAME: {RUN_NAME}")

# %% [markdown]
# ## CELL 1 — Environment diagnostic + dependency install

# %%
import sys, os, subprocess, importlib

print("Python:", sys.version.split()[0])
print("Platform:", sys.platform)

# Install missing deps
deps_to_install = {
    'kapre': 'kapre==0.3.5',
    'faiss': 'faiss-cpu',
    'tf_keras': 'tf_keras',
    'wandb': 'wandb',
}
for pkg, target in deps_to_install.items():
    try:
        importlib.import_module(pkg)
        print(f"  ✅ {pkg} already present")
    except ImportError:
        r = subprocess.run(['pip', 'install', '-q', target], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  ✅ {pkg} installed")
        else:
            print(f"  ❌ {pkg} install failed: {r.stderr[-200:]}")

import tensorflow as tf
# Allow GPU memory growth (avoid pre-allocating full GPU memory)
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print(f"  ⚠️ memory_growth on {g}: {e}")
print(f"TF: {tf.__version__}, GPUs: {gpus}")

# %% [markdown]
# ## CELL 2 — Clone NeuralFP + apply TF compat patches

# %%
%cd /kaggle/working
subprocess.run(['rm', '-rf', 'NeuralFP'])
r = subprocess.run(
    ['git', 'clone', '-q', 'https://github.com/mimbres/neural-audio-fp', 'NeuralFP'],
    capture_output=True, text=True
)
print("Clone:", "✅" if r.returncode == 0 else f"❌ {r.stderr}")

%cd /kaggle/working/NeuralFP
sys.path.insert(0, '/kaggle/working/NeuralFP')

# TF 2.18+ compat: patch tf.keras.experimental.CosineDecay
patches = [
    ('model/trainer.py',
     'tf.keras.experimental.CosineDecay',
     'tf.keras.optimizers.schedules.CosineDecay'),
]
for path, old, new in patches:
    with open(path) as f:
        src = f.read()
    if old in src:
        src = src.replace(old, new)
        with open(path, 'w') as f:
            f.write(src)
        print(f"  ✅ patched {path}")

# Use legacy Keras for OrderedEnqueuer + experimental.* APIs
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# %% [markdown]
# ## CELL 3 — Verify datasets + locate originals/queries

# %%
# Mimbres dataset (FMA training data)
NAFP = '/kaggle/input/datasets/mimbres/neural-audio-fingerprint/neural-audio-fp-dataset'
assert os.path.isdir(NAFP), f"Mimbres dataset not at {NAFP}"
print(f"✅ Mimbres at {NAFP}")
print(f"   contents: {os.listdir(NAFP)}")

# Our Indian music dataset
OUR_BASE = '/kaggle/input/datasets/aryanbanwala97/audio-fp-indian-music-btp'
assert os.path.isdir(OUR_BASE), f"Our dataset not at {OUR_BASE}"
print(f"✅ Ours at {OUR_BASE}")
song_dirs = sorted([d for d in os.listdir(OUR_BASE)
                    if os.path.isdir(os.path.join(OUR_BASE, d))])
print(f"   {len(song_dirs)} song folders: {song_dirs[:5]}...")

# Build inventory: for each song, find original.mp3 + remix files + clip files
import glob
songs = {}
for sd in song_dirs:
    sp = os.path.join(OUR_BASE, sd)
    original = os.path.join(sp, 'original.mp3')
    if not os.path.isfile(original):
        continue
    remix_full = sorted(glob.glob(os.path.join(sp, 'remix', 'remix_*.mp3')))
    clips = sorted(glob.glob(os.path.join(sp, 'remix', 'clips_*', 'clip_*.mp3')))
    songs[sd] = {
        'original': original,
        'remix_full': remix_full[0] if remix_full else None,
        'clips': clips,
    }

n_songs = len(songs)
n_remix = sum(1 for s in songs.values() if s['remix_full'])
n_clips = sum(len(s['clips']) for s in songs.values())
print(f"\n📊 Inventory: {n_songs} songs | {n_remix} full-remix queries | {n_clips} clip queries")
print(f"   Total queries: {n_remix + n_clips}")
assert n_songs >= 20, "expected ~21 songs"

# %% [markdown]
# ## CELL 4 — Wandb login (read key from Kaggle Secrets or hardcoded)

# %%
import wandb

# Wandb key — read from Kaggle Secrets (NEVER hardcode).
# To enable: Kaggle UI → Add-ons → Secrets → Add label `WANDB_API_KEY` with your key.
WANDB_API_KEY = None
try:
    from kaggle_secrets import UserSecretsClient
    WANDB_API_KEY = UserSecretsClient().get_secret("WANDB_API_KEY")
    print("✅ Got WANDB_API_KEY from Kaggle Secrets")
except Exception as e:
    print(f"⚠️ Kaggle Secrets unavailable ({type(e).__name__}). Skipping wandb.")
    print("   To enable: Kaggle UI → Add-ons → Secrets → label=WANDB_API_KEY")

wandb_run = None
if WANDB_ENABLED and WANDB_API_KEY:
    try:
        wandb.login(key=WANDB_API_KEY)
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=RUN_NAME,
            config={
                'run_training': RUN_TRAINING,
                'train_epochs': TRAIN_EPOCHS,
                'n_songs': n_songs,
                'n_queries': n_remix + n_clips,
                'hop_seconds': HOP_SECONDS,
                'topk': TOPK,
            }
        )
        print(f"✅ wandb run: {wandb_run.url}")
    except Exception as e:
        print(f"⚠️ wandb init failed: {e}")
        WANDB_ENABLED = False
else:
    WANDB_ENABLED = False
    print("⚠️ wandb disabled (no API key found)")

# %% [markdown]
# ## CELL 5 — Build NeuralFP model (random init for 0-epoch, or train then load)

# %%
import yaml
import numpy as np

with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)

# Override paths so trainer doesn't crash on missing dirs (only matters if RUN_TRAINING=True)
cfg['DIR']['SOURCE_ROOT_DIR'] = f'{NAFP}/music/'
cfg['DIR']['BG_ROOT_DIR']     = f'{NAFP}/aug/bg/'
cfg['DIR']['IR_ROOT_DIR']     = f'{NAFP}/aug/ir/'
cfg['DIR']['SPEECH_ROOT_DIR'] = f'{NAFP}/aug/speech/common_voice_8k/en/'
cfg['DIR']['OUTPUT_ROOT_DIR'] = '/kaggle/working/logs/emb/'
cfg['DIR']['LOG_ROOT_DIR']    = '/kaggle/working/logs/'
cfg['TRAIN']['MAX_EPOCH']     = TRAIN_EPOCHS if RUN_TRAINING else 1
cfg['TRAIN']['MINI_TEST_IN_TRAIN'] = False

with open('config/pipeline.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

# Either train (writes checkpoint) or just build random-init model
checkpoint_dir = '/kaggle/working/logs/checkpoint/pipeline'
if RUN_TRAINING:
    print(f"--- Training {TRAIN_EPOCHS} epoch(s) ---")
    train_env = os.environ.copy()
    train_env['TF_USE_LEGACY_KERAS'] = '1'
    import time
    t0 = time.time()
    train_proc = subprocess.run(
        ['python', 'run.py', 'train', 'pipeline', '-c', 'pipeline',
         f'--max_epoch={TRAIN_EPOCHS}'],
        capture_output=False, text=True, timeout=39600, env=train_env  # 11 hr cap (Kaggle session limit is 12 hr)
    )
    print(f"Training elapsed: {time.time()-t0:.0f}s, exit={train_proc.returncode}")
    if train_proc.returncode != 0:
        print("⚠️ training failed — falling back to random-init model")
        RUN_TRAINING = False

# Build model directly (works for both random-init and post-training inference)
from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter

m_pre = get_melspec_layer(cfg, trainable=False)
m_fp  = get_fingerprinter(cfg, trainable=False)

# If we trained, restore checkpoint into m_fp
if RUN_TRAINING:
    try:
        ckpt = tf.train.Checkpoint(model=m_fp)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        if latest:
            ckpt.restore(latest).expect_partial()
            print(f"✅ Restored checkpoint: {latest}")
        else:
            print(f"⚠️ no checkpoint at {checkpoint_dir} — using random init")
    except Exception as e:
        print(f"⚠️ checkpoint restore failed: {e}")

print(f"✅ Model ready (trained={RUN_TRAINING})")
EMB_SZ = int(cfg['MODEL']['EMB_SZ'])
print(f"   Embedding size: {EMB_SZ}")
print(f"   Sample rate: {cfg['MODEL']['FS']} Hz")
print(f"   Segment duration: {cfg['MODEL']['DUR']} s")
print(f"   STFT win/hop: {cfg['MODEL']['STFT_WIN']}/{cfg['MODEL']['STFT_HOP']}")
print(f"   Mel bins: {cfg['MODEL']['N_MELS']}")

# %% [markdown]
# ## CELL 6 — Inference helpers

# %%
import librosa

FS  = int(cfg['MODEL']['FS'])              # 8000
DUR = float(cfg['MODEL']['DUR'])           # 1.0
SEG_LEN = int(FS * DUR)                    # 8000
HOP_LEN = int(FS * HOP_SECONDS)            # 4000

def load_and_slice(path):
    """Load mp3 → resample to 8 kHz mono → slice into 1-sec segments with 50% overlap.
    Returns array shape (n_segments, 1, 8000) ready for m_pre.
    """
    audio, _ = librosa.load(path, sr=FS, mono=True)
    if len(audio) < SEG_LEN:
        # Pad short clips
        audio = np.pad(audio, (0, SEG_LEN - len(audio)))
    n_segs = max(1, 1 + (len(audio) - SEG_LEN) // HOP_LEN)
    segments = np.zeros((n_segs, 1, SEG_LEN), dtype=np.float32)
    for i in range(n_segs):
        start = i * HOP_LEN
        segments[i, 0, :] = audio[start:start + SEG_LEN]
    return segments

def embed_batch(segments, batch_size=128):
    """Run m_pre + m_fp in batches. Returns (n, EMB_SZ) L2-normalized embeddings."""
    out = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        spec = m_pre(batch, training=False)
        emb = m_fp(spec, training=False)
        out.append(emb.numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

# Smoke test the helpers — also serves as model warm-up (first call builds weights)
print("Smoke testing inference helpers + model warm-up...")
try:
    test_segs = load_and_slice(songs[song_dirs[0]]['original'])
    print(f"  load_and_slice: {test_segs.shape}, dtype={test_segs.dtype}")
    # Test m_pre alone
    spec = m_pre(test_segs[:4], training=False)
    print(f"  m_pre output: shape={spec.shape}, dtype={spec.dtype}")
    # Test m_fp
    test_emb = m_fp(spec, training=False).numpy()
    print(f"  m_fp output: shape={test_emb.shape}, dtype={test_emb.dtype}")
    print(f"  L2 norms (should be ~1.0): {np.linalg.norm(test_emb, axis=1)[:4]}")
    assert test_emb.shape[1] == EMB_SZ, f"Embedding dim mismatch: {test_emb.shape[1]} vs {EMB_SZ}"
    print(f"  ✅ helpers + model verified")
except Exception as e:
    print(f"  ❌ Smoke test failed: {type(e).__name__}: {e}")
    raise

# %% [markdown]
# ## CELL 7 — Build database from 21 originals

# %%
import faiss
import time

print("Building DB from 21 originals...")
t0 = time.time()
db_vectors = []
db_song_ids = []          # which song each vector belongs to
song_id_to_name = {}
failed_originals = []

for song_id, song_name in enumerate(sorted(songs.keys())):
    song_id_to_name[song_id] = song_name
    try:
        segs = load_and_slice(songs[song_name]['original'])
        embs = embed_batch(segs)
        db_vectors.append(embs)
        db_song_ids.extend([song_id] * len(embs))
        if (song_id + 1) % 5 == 0 or song_id + 1 == len(songs):
            print(f"  [{song_id+1}/{len(songs)}] {song_name}: {len(embs)} segments")
    except Exception as e:
        print(f"  ❌ {song_name} failed: {type(e).__name__}: {e}")
        failed_originals.append(song_name)

if not db_vectors:
    raise RuntimeError("All originals failed to load — pipeline cannot continue")

db_matrix = np.concatenate(db_vectors, axis=0)
db_song_ids = np.array(db_song_ids, dtype=np.int32)
print(f"\n  DB shape: {db_matrix.shape}, build time: {time.time()-t0:.1f}s")
print(f"  Failed originals: {len(failed_originals)} ({failed_originals if failed_originals else 'none'})")

# Build FAISS index — Inner Product (works as cosine similarity for L2-normed vectors)
index = faiss.IndexFlatIP(EMB_SZ)
index.add(db_matrix)
print(f"  FAISS index ntotal: {index.ntotal}")

if WANDB_ENABLED:
    wandb.log({'db_n_segments': index.ntotal, 'db_build_time_sec': time.time()-t0})

# %% [markdown]
# ## CELL 8 — Run inference on all queries

# %%
from collections import Counter
from datetime import datetime

def predict_song(query_path):
    """Embed query, search FAISS top-K per segment, majority-vote across segments."""
    segs = load_and_slice(query_path)
    embs = embed_batch(segs)
    D, I = index.search(embs, TOPK)
    # Each row of I has TOPK indices; collect song_ids
    candidate_ids = []
    candidate_scores = {}
    for row_d, row_i in zip(D, I):
        for d, idx in zip(row_d, row_i):
            sid = int(db_song_ids[idx])
            candidate_ids.append(sid)
            candidate_scores[sid] = candidate_scores.get(sid, 0.0) + float(d)
    # Aggregate: pick song with highest summed similarity
    best_sid = max(candidate_scores, key=candidate_scores.get)
    confidence = candidate_scores[best_sid] / sum(candidate_scores.values())
    # Top-2 for diagnostics
    top2 = sorted(candidate_scores.items(), key=lambda x: -x[1])[:2]
    return song_id_to_name[best_sid], confidence, [song_id_to_name[s] for s, _ in top2]

print("Running inference on all queries...")
records_full = []   # full remix queries
records_clip = []   # clip queries
t_start = time.time()
total_queries = sum(1 for s in songs.values() if s['remix_full']) + sum(len(s['clips']) for s in songs.values())
done = 0

for song_name in sorted(songs.keys()):
    s = songs[song_name]
    # Full remix
    if s['remix_full']:
        t0 = time.time()
        try:
            pred, conf, top2 = predict_song(s['remix_full'])
            status = 'PASS' if pred == song_name else 'FAIL'
            err = None
        except Exception as e:
            pred, conf, top2, status, err = None, 0.0, [], 'ERROR', str(e)
        records_full.append({
            'file': os.path.relpath(s['remix_full'], OUR_BASE),
            'expected': song_name,
            'matched': pred,
            'confidence': round(float(conf), 4),
            'top2': top2,
            'time': round(time.time()-t0, 3),
            'status': status,
            **({'error': err} if err else {}),
        })
        done += 1
    # Clips
    for clip_path in s['clips']:
        t0 = time.time()
        try:
            pred, conf, top2 = predict_song(clip_path)
            status = 'PASS' if pred == song_name else 'FAIL'
            err = None
        except Exception as e:
            pred, conf, top2, status, err = None, 0.0, [], 'ERROR', str(e)
        records_clip.append({
            'file': os.path.relpath(clip_path, OUR_BASE),
            'expected': song_name,
            'matched': pred,
            'confidence': round(float(conf), 4),
            'top2': top2,
            'time': round(time.time()-t0, 3),
            'status': status,
            **({'error': err} if err else {}),
        })
        done += 1
    if done % 20 == 0 or done == total_queries:
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total_queries - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total_queries}] elapsed={elapsed:.0f}s rate={rate:.1f}q/s eta={eta:.0f}s")

infer_time = time.time() - t_start
print(f"\n✅ Inference done: {done} queries in {infer_time:.1f}s ({done/infer_time:.1f} q/s)")

# %% [markdown]
# ## CELL 9 — Aggregate, save JSON, log to wandb

# %%
def summarize(records, group_name):
    total = len(records)
    p = sum(1 for r in records if r['status'] == 'PASS')
    f = sum(1 for r in records if r['status'] == 'FAIL')
    nm = sum(1 for r in records if r['status'] == 'NO_MATCH')
    er = sum(1 for r in records if r['status'] == 'ERROR')
    avg_t = round(sum(r['time'] for r in records) / total, 3) if total else 0.0
    return {
        'total': total, 'pass': p, 'fail': f, 'no_match': nm, 'error': er,
        'pass_rate': round(p / total, 4) if total else 0.0,
        'avg_time_sec': avg_t,
    }

summary = {
    'test_full': summarize(records_full, 'full'),
    'test_clips': summarize(records_clip, 'clips'),
}
all_records = records_full + records_clip
summary['overall'] = summarize(all_records, 'all')

# Per-song breakdown
per_song = {}
for sn in sorted(songs.keys()):
    rs = [r for r in all_records if r['expected'] == sn]
    p = sum(1 for r in rs if r['status'] == 'PASS')
    per_song[sn] = {'total': len(rs), 'pass': p,
                    'pass_rate': round(p / len(rs), 4) if rs else 0.0}

result = {
    'library': 'neuralfp',
    'data_dir': 'dataset',
    'timestamp': datetime.now().isoformat(),
    'mode': 'training' if RUN_TRAINING else 'random_init',
    'training_epochs': TRAIN_EPOCHS if RUN_TRAINING else 0,
    'config': {
        'fs': FS, 'segment_dur_sec': DUR, 'hop_sec': HOP_SECONDS,
        'embedding_size': int(EMB_SZ), 'topk': TOPK,
        'db_n_segments': int(index.ntotal),
    },
    'db_verification': {
        'songs_in_dataset': n_songs,
        'expected_uploads': n_songs,
        'uploaded': n_songs,
        'all_uploaded': True,
    },
    'summary': summary,
    'per_song': per_song,
    'tests': {'test_full': records_full, 'test_clips': records_clip},
}

OUT_PATH = '/kaggle/working/neuralfp_dataset_results.json'
import json
with open(OUT_PATH, 'w') as f:
    json.dump(result, f, indent=2)
print(f"✅ Saved: {OUT_PATH}")
print(f"\n=== FINAL ===")
print(f"  Mode:          {'1+ epoch trained' if RUN_TRAINING else 'random init (0-epoch)'}")
print(f"  Overall:       {summary['overall']['pass']}/{summary['overall']['total']} = {summary['overall']['pass_rate']*100:.1f}%")
print(f"  Full remix:    {summary['test_full']['pass']}/{summary['test_full']['total']}")
print(f"  Clips:         {summary['test_clips']['pass']}/{summary['test_clips']['total']}")
print(f"\nFor reference:")
print(f"  Dejavu:        84/273 = 30.8%")
print(f"  Olaf:          20/273 = 7.3%")
print(f"  Panako:         9/273 = 3.3%")

if WANDB_ENABLED:
    # Flat scalar metrics (wandb handles these cleanly)
    wandb_metrics = {
        'overall_accuracy': summary['overall']['pass_rate'],
        'full_remix_accuracy': summary['test_full']['pass_rate'],
        'clip_accuracy': summary['test_clips']['pass_rate'],
        'inference_time_total_sec': infer_time,
        'inference_time_per_query_sec': infer_time / max(1, total_queries),
        'n_failed_originals': len(failed_originals),
    }
    # Per-song accuracy as separate metrics (flatten)
    for sn, ps in per_song.items():
        wandb_metrics[f'song_acc/{sn}'] = ps['pass_rate']
    wandb.log(wandb_metrics)

    # Parse training tensorboard events → log to wandb as time-series
    if RUN_TRAINING:
        try:
            tfev_glob = '/kaggle/working/logs/fit/pipeline/train/events.out.tfevents*'
            import glob as _glob
            tfev_files = _glob.glob(tfev_glob)
            if tfev_files:
                print(f"\n--- Streaming {len(tfev_files)} tfevents file(s) → wandb ---")
                n_logged = 0
                for ef in tfev_files:
                    for ev in tf.compat.v1.train.summary_iterator(ef):
                        for v in ev.summary.value:
                            if v.HasField('simple_value'):
                                wandb.log({f'train/{v.tag}': float(v.simple_value)},
                                          step=int(ev.step))
                                n_logged += 1
                print(f"  ✅ pushed {n_logged} training scalar events to wandb")
            else:
                print(f"⚠️ no tfevents files found at {tfev_glob}")
        except Exception as e:
            print(f"⚠️ tfevents → wandb failed: {type(e).__name__}: {e}")
    # Save full results JSON as wandb artifact
    try:
        artifact = wandb.Artifact(f'results-{RUN_NAME}', type='results')
        artifact.add_file(OUT_PATH)
        wandb_run.log_artifact(artifact)
    except Exception as e:
        print(f"⚠️ wandb artifact upload failed: {e}")
    wandb.finish()
    print(f"\n✅ wandb logs uploaded")
