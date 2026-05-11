# %% [markdown]
# # NeuralFP Full Pipeline — Colab version
#
# **Goal:** Train NeuralFP on FMA mini-dataset + run inference on 21-song Indian dataset.
#
# **Setup (one-time per session):**
#  1. Runtime → Change runtime type → **A100 GPU** (or T4 if A100 unavailable)
#  2. Mount Drive (Cell 1)
#  3. Upload `kaggle.json` (Cell 2 will prompt) — already on Drive after first session
#  4. Toggle `RUN_TRAINING` + `TRAIN_EPOCHS` in Cell 0
#  5. Runtime → Run all
#
# **Output:** checkpoints to `/content/drive/MyDrive/btp/checkpoints/<run_name>/`
#              results JSON to `/content/drive/MyDrive/btp/results/<run_name>.json`

# %% [markdown]
# ## CELL 0 — Configuration

# %%
RUN_TRAINING = True       # True = train then infer; False = random-init inference only
TRAIN_EPOCHS = 3          # ← always train from scratch this many epochs (resume disabled — see Cell 8 comment)
HOP_SECONDS  = 0.5
TOPK         = 5
WANDB_ENABLED = False     # set True after wandb login (Cell 4)

# === Phase 1 alterations (set toggles independently — most are orthogonal) ===
# Alteration 1: test-time pitch averaging (inference only, no retraining).
# Result so far: hurts accuracy (27.1% → 7.0%) — confirms NeuralFP is pitch-fragile.
USE_PITCH_TTA      = False
# Alteration 2: dilated time convolution in CNN (2x temporal receptive field for pitch ornaments).
# Architectural — REQUIRES retraining (REUSE_CKPT_FROM must be None).
USE_DILATED_CNN    = False
# (future: USE_CQT, USE_INDIAN_AUG)

# Reuse a previously-trained checkpoint instead of re-training. Set to e.g. 'colab-pipeline-1ep' to load
# baseline weights from Drive and run inference-only. AUTO-DISABLED when:
#  (a) an architectural alteration is on (baseline weights wouldn't fit the new arch), OR
#  (b) the reuse target's epoch count doesn't match TRAIN_EPOCHS (would mislabel weights).
REUSE_CKPT_FROM = 'colab-pipeline-1ep'
if USE_DILATED_CNN:
    REUSE_CKPT_FROM = None  # arch change → must retrain
if REUSE_CKPT_FROM and f'-{TRAIN_EPOCHS}ep' not in f'-{REUSE_CKPT_FROM}':
    # e.g. REUSE_CKPT_FROM='colab-pipeline-1ep' but TRAIN_EPOCHS=3 → don't reuse
    REUSE_CKPT_FROM = None

# Build run name reflecting active alterations
_alt_tag = ''
if USE_PITCH_TTA:   _alt_tag += '-pitchTTA'
if USE_DILATED_CNN: _alt_tag += '-dilated'
RUN_NAME = (f"colab-pipeline-{TRAIN_EPOCHS}ep" if RUN_TRAINING else "colab-pipeline-0ep") + _alt_tag

print(f"=== CONFIG ===")
print(f"  RUN_TRAINING: {RUN_TRAINING}")
print(f"  TRAIN_EPOCHS: {TRAIN_EPOCHS}")
print(f"  RUN_NAME: {RUN_NAME}")

# %% [markdown]
# ## CELL 1 — Mount Drive + verify GPU

# %%
import os, time
from google.colab import drive

# Resilient Drive mount: retry up to 3 times with force_remount.
# Drive auth occasionally fails on long sessions due to credential expiry — retry usually fixes.
_mounted = False
for _attempt in range(3):
    try:
        drive.mount('/content/drive', force_remount=True)
        if os.path.isdir('/content/drive/MyDrive'):
            print(f"✅ Drive mounted (attempt {_attempt+1})")
            _mounted = True
            break
    except Exception as e:
        print(f"⚠️ Drive mount attempt {_attempt+1} failed: {e}")
        time.sleep(3)
if not _mounted:
    raise RuntimeError("Drive mount failed 3 times. Try: Runtime → Disconnect and delete runtime → reconnect")

import subprocess
nvidia = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
print(nvidia.split('\n')[7] if len(nvidia.split('\n')) > 7 else nvidia[:500])

DRIVE_BASE = '/content/drive/MyDrive/btp'
os.makedirs(f'{DRIVE_BASE}/checkpoints', exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/results', exist_ok=True)
os.makedirs(f'{DRIVE_BASE}/secrets', exist_ok=True)
print(f"✅ Drive mounted, btp/ ready at {DRIVE_BASE}")

# %% [markdown]
# ## CELL 2 — Kaggle CLI auth (file picker first time, auto after)

# %%
KAGGLE_JSON_DRIVE = f'{DRIVE_BASE}/secrets/kaggle.json'
KAGGLE_JSON_LOCAL = os.path.expanduser('~/.kaggle/kaggle.json')

if not os.path.isfile(KAGGLE_JSON_DRIVE):
    print("⚠️ kaggle.json not on Drive. Click 'Choose Files' button below and select kaggle.json from your Mac")
    print("    (download from kaggle.com → Settings → API → Create New API Token)")
    from google.colab import files
    uploaded = files.upload()  # browser file picker
    src_name = list(uploaded.keys())[0]
    os.makedirs(os.path.dirname(KAGGLE_JSON_DRIVE), exist_ok=True)
    import shutil
    shutil.move(src_name, KAGGLE_JSON_DRIVE)
    print(f"✅ Saved to Drive (won't ask again next session)")

os.makedirs(os.path.dirname(KAGGLE_JSON_LOCAL), exist_ok=True)
subprocess.run(['cp', KAGGLE_JSON_DRIVE, KAGGLE_JSON_LOCAL])
os.chmod(KAGGLE_JSON_LOCAL, 0o600)
subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
print("✅ Kaggle CLI configured")

# %% [markdown]
# ## CELL 3 — Install Python deps (TF, kapre, faiss, etc)

# %%
import sys, importlib

deps = {
    'kapre': 'kapre==0.3.5',
    'faiss': 'faiss-cpu',
    'tf_keras': 'tf_keras',
    'librosa': 'librosa',
    'wandb': 'wandb',
    'pyyaml': 'pyyaml',
}
for pkg, target in deps.items():
    try:
        importlib.import_module(pkg.replace('-', '_'))
        print(f"  ✅ {pkg} present")
    except ImportError:
        r = subprocess.run(['pip', 'install', '-q', target], capture_output=True, text=True)
        print(f"  {'✅' if r.returncode==0 else '❌'} {pkg}: {r.returncode}")

os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try: tf.config.experimental.set_memory_growth(g, True)
    except Exception: pass
print(f"TF: {tf.__version__}, GPUs: {gpus}")

# %% [markdown]
# ## CELL 4 — Download datasets to Drive (one-time) + sync to /content (fast SSD)
#
# Drive = persistent across sessions but SLOW network I/O.
# /content = fast local SSD but wiped on session end.
# Strategy: download to Drive once, then mirror to /content for hot reads.

# %%
import time
DRIVE_DATASETS = f'{DRIVE_BASE}/datasets'
LOCAL_DATASETS = '/content/datasets'
os.makedirs(DRIVE_DATASETS, exist_ok=True)
os.makedirs(LOCAL_DATASETS, exist_ok=True)

def list_indian_song_dirs(base):
    """Return list of absolute paths to song dirs (each containing original.mp3).
    Handles both flat (songs at base) and nested (songs at base/<x>/<song>) layouts.
    Critical: returns ONLY song dirs, never the parent — so rsync can't accidentally
    pick up neighbor datasets (e.g. FMA) sharing the same parent.
    """
    if not os.path.isdir(base): return []
    flat = []
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'original.mp3')):
            flat.append(p)
    if flat: return flat
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if not os.path.isdir(p): continue
        nested = []
        for d2 in sorted(os.listdir(p)):
            p2 = os.path.join(p, d2)
            if os.path.isdir(p2) and os.path.isfile(os.path.join(p2, 'original.mp3')):
                nested.append(p2)
        if nested: return nested
    return []

# (a) Indian dataset — small (~100 MB), download to Drive if missing
indian_song_dirs = list_indian_song_dirs(DRIVE_DATASETS)
if not indian_song_dirs:
    # Always download into a dedicated subdir so it can never collide with FMA at parent level
    INDIAN_DOWNLOAD_DIR = f'{DRIVE_DATASETS}/audio-fp-indian-music-btp'
    os.makedirs(INDIAN_DOWNLOAD_DIR, exist_ok=True)
    print("Downloading Indian dataset from Kaggle to Drive...")
    subprocess.run(['kaggle', 'datasets', 'download',
                    '-d', 'aryanbanwala97/audio-fp-indian-music-btp',
                    '-p', INDIAN_DOWNLOAD_DIR, '--unzip'], check=True)
    indian_song_dirs = list_indian_song_dirs(INDIAN_DOWNLOAD_DIR)
assert indian_song_dirs, f"Indian dataset not found under {DRIVE_DATASETS}"
print(f"  Indian: {len(indian_song_dirs)} song dirs detected on Drive")

# (b) FMA mini — 11 GB. We DO NOT cache on Drive: Drive→/content rsync of FMA's 10K small
#     files runs at ~1-2 MB/s (would take ~90 min). Kaggle download to /content is ~50 MB/s,
#     so 11 GB direct from Kaggle = ~5-8 min per session. Net win even though it re-downloads.
#     (Optional future opt: tar FMA on Drive once, copy single tar file fast, untar locally.)
print(f"✅ Drive datasets ready (Indian only — FMA goes Kaggle→/content directly for speed)")

print("\n=== Syncing datasets to /content (local SSD) for fast I/O ===")
INDIAN_DIR = f'{LOCAL_DATASETS}/indian'
NAFP = f'{LOCAL_DATASETS}/neural-audio-fp-dataset'

import re as _re_sync

def rsync_with_progress(src, dst, label, min_size_gb=0):
    """rsync with live %% + ETA every ~5s. Validates dst by min size to detect partial sync."""
    if os.path.isdir(dst):
        # Validate cache: count files OR check size > expected min
        try:
            actual_gb = float(subprocess.run(['du', '-sb', dst], capture_output=True, text=True)
                              .stdout.split()[0]) / 1e9
        except Exception:
            actual_gb = 0
        if actual_gb >= min_size_gb:
            print(f"  ✅ {label} already cached at {dst} ({actual_gb:.2f} GB)")
            return
        else:
            print(f"  ⚠️ {label} at {dst} is incomplete ({actual_gb:.2f} GB < {min_size_gb} GB) — re-syncing")
            import shutil
            shutil.rmtree(dst, ignore_errors=True)
    t_start = time.time()
    print(f"  ⏳ Copying {label}...")
    proc = subprocess.Popen(
        ['rsync', '-a', '--info=progress2', '--no-i-r', '--exclude=._*', f'{src}/', f'{dst}/'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_print = 0
    last_pct = -1
    for line in iter(proc.stdout.readline, ''):
        # rsync progress line: "  4,317,892,608  39%   45.32MB/s    0:02:18 (xfr#234, to-chk=8/12)"
        m = _re_sync.search(r'(\d{1,3})%\s+(\S+)\s+(\d+:\d+:\d+)', line)
        if m:
            pct = int(m.group(1))
            speed = m.group(2)
            eta = m.group(3)
            now = time.time()
            # print every 5s OR every 5% jump
            if (now - last_print > 5) or (pct - last_pct >= 5):
                elapsed_m = (now - t_start)/60
                print(f"  [{elapsed_m:.1f}m] {label}: {pct:3d}% — {speed} — ETA {eta}")
                last_print = now
                last_pct = pct
    proc.wait()
    print(f"  ✅ {label} → {dst} in {(time.time()-t_start)/60:.1f} min")

def rsync_indian_per_song(song_dirs, dst, min_size_gb=0.05, max_size_gb=0.5):
    """Copy each Indian song dir into dst — avoids accidentally pulling FMA when songs
    are extracted to the same parent as FMA on Drive."""
    if os.path.isdir(dst):
        try:
            actual_gb = float(subprocess.run(['du', '-sb', dst], capture_output=True, text=True)
                              .stdout.split()[0]) / 1e9
        except Exception:
            actual_gb = 0
        import shutil
        if actual_gb > max_size_gb:
            # Probably contaminated with FMA from earlier buggy run
            print(f"  ⚠️ Indian at {dst} is {actual_gb:.2f} GB — too large (FMA contamination?), wiping")
            shutil.rmtree(dst, ignore_errors=True)
        elif actual_gb >= min_size_gb:
            print(f"  ✅ Indian (~100 MB) already cached at {dst} ({actual_gb*1000:.0f} MB)")
            return
        else:
            print(f"  ⚠️ Indian at {dst} incomplete ({actual_gb*1000:.0f} MB) — re-syncing")
            shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    t_start = time.time()
    n = len(song_dirs)
    print(f"  ⏳ Copying {n} Indian song dirs (~100 MB)...")
    for i, sd in enumerate(song_dirs):
        name = os.path.basename(sd)
        target = os.path.join(dst, name)
        subprocess.run(['rsync', '-a', '--exclude=._*', f'{sd}/', f'{target}/'], check=True)
        elapsed_m = (time.time()-t_start)/60
        print(f"  [{elapsed_m:.1f}m] Indian: {i+1}/{n} ({100*(i+1)/n:.0f}%) — {name}")
    print(f"  ✅ Indian → {dst} in {(time.time()-t_start)/60:.1f} min")

def fma_local_ready(path, min_size_gb=10):
    """Check if FMA already extracted at /content with full data."""
    if not os.path.isdir(f'{path}/music'): return False
    try:
        gb = float(subprocess.run(['du', '-sb', path], capture_output=True, text=True)
                   .stdout.split()[0]) / 1e9
    except Exception:
        gb = 0
    return gb >= min_size_gb

def kaggle_download_fma(dst_parent, label="FMA (~11 GB)"):
    """Download FMA from Kaggle directly to /content. Streams kaggle CLI progress.
    Runs ~5-8 min on Colab vs ~90 min if going through Drive. Re-runs every fresh session
    (Drive caching is too slow due to many small files)."""
    import shutil
    target = f'{dst_parent}/neural-audio-fp-dataset'
    if fma_local_ready(target):
        print(f"  ✅ {label} already at {target} (this session) — skipping download")
        return
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(dst_parent, exist_ok=True)
    print(f"  ⏳ Downloading {label} from Kaggle → {dst_parent} (one-time per session)...")
    t_dl = time.time()
    proc = subprocess.Popen(
        ['kaggle', 'datasets', 'download', '-d', 'mimbres/neural-audio-fingerprint',
         '-p', dst_parent, '--unzip'],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_print = 0
    last_line = ''
    for raw in iter(proc.stdout.readline, ''):
        # kaggle CLI uses \r-overwritten progress lines; take the last segment
        line = raw.replace('\r', '\n').strip().split('\n')[-1]
        if not line: continue
        last_line = line
        now = time.time()
        if now - last_print > 5:
            print(f"  [{(now-t_dl)/60:.1f}m] {line[:140]}")
            last_print = now
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle download failed (exit={proc.returncode}): {last_line}")
    print(f"  ✅ {label} downloaded + unzipped in {(time.time()-t_dl)/60:.1f} min")

t0 = time.time()
rsync_indian_per_song(indian_song_dirs, INDIAN_DIR, min_size_gb=0.05)
kaggle_download_fma(LOCAL_DATASETS)
print(f"\n  ✅ Total dataset prep done in {(time.time()-t0)/60:.1f} min")

OUR_BASE = INDIAN_DIR
assert os.path.isdir(NAFP), f"FMA at {NAFP} missing"
assert os.path.isdir(f'{NAFP}/music'), f"FMA music at {NAFP}/music missing"
assert os.path.isdir(OUR_BASE), f"Indian at {OUR_BASE} missing"
print(f"✅ Datasets ready on local SSD")

# %% [markdown]
# ## CELL 5 — Clone NeuralFP + apply TF compat patch

# %%
import shutil
WORK = '/content/NeuralFP'
# Ensure cwd is sane (in case prior interrupted run left cwd in deleted dir)
os.chdir('/content')
shutil.rmtree(WORK, ignore_errors=True)  # force-remove any leftover
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
    print("  ✅ patched trainer.py")

# Alteration 2: dilated time convolution.
# TF disallows strides>1 AND dilation>1 in same Conv2D. NeuralFP's ConvLayer has 2 conv2ds
# whose strides come from `strides[0]` and `strides[1]` (local __init__ arg). For each, we
# replace `dilation_rate=(1, 1)` with `(1, 2) if max(strides[N]) == 1 else (1, 1)` so dilation
# applies only on layers WITHOUT stride downsampling. Regex-based for whitespace robustness.
if USE_DILATED_CNN:
    import re as _re_patch
    nnfp_path = f'{WORK}/model/fp/nnfp.py'
    with open(nnfp_path) as f:
        src = f.read()
    new_src, n_changed = _re_patch.subn(
        r"strides=strides\[(\d)\],(\s*\n\s*)padding='SAME',(\s*\n\s*)dilation_rate=\(1, 1\)",
        r"strides=strides[\1],\2padding='SAME',\3dilation_rate=(1, 2) if max(strides[\1]) == 1 else (1, 1)",
        src
    )
    if n_changed == 0:
        print(f"  ⚠️ nnfp.py dilation patch did NOT match (regex). Skipping — file may have changed.")
    else:
        with open(nnfp_path, 'w') as f:
            f.write(new_src)
        print(f"  ✅ patched nnfp.py — {n_changed} conv2d layer(s) now use stride-aware dilation_rate")

print(f"✅ NeuralFP at {WORK}")

# %% [markdown]
# ## CELL 6 — Build inventory of Indian songs

# %%
import glob

song_dirs = sorted([d for d in os.listdir(OUR_BASE) if os.path.isdir(os.path.join(OUR_BASE, d))])
songs = {}
for sd in song_dirs:
    sp = os.path.join(OUR_BASE, sd)
    original = os.path.join(sp, 'original.mp3')
    if not os.path.isfile(original):
        continue
    remix_full = sorted(glob.glob(os.path.join(sp, 'remix', 'remix_*.mp3')))
    clips = sorted(glob.glob(os.path.join(sp, 'remix', 'clips_*', 'clip_*.mp3')))
    songs[sd] = {'original': original, 'remix_full': remix_full[0] if remix_full else None, 'clips': clips}

n_songs = len(songs)
n_remix = sum(1 for s in songs.values() if s['remix_full'])
n_clips = sum(len(s['clips']) for s in songs.values())
total_queries = n_remix + n_clips
print(f"✅ {n_songs} songs, {n_remix} full-remix, {n_clips} clips, total: {total_queries}")

# %% [markdown]
# ## CELL 7 — Wandb (optional, skip if WANDB_ENABLED=False)

# %%
import time
wandb_run = None
if WANDB_ENABLED:
    import wandb
    # Read API key from Drive secrets
    KEY_PATH = f'{DRIVE_BASE}/secrets/wandb_key.txt'
    if os.path.isfile(KEY_PATH):
        with open(KEY_PATH) as f:
            wandb.login(key=f.read().strip())
        wandb_run = wandb.init(
            project='btp-neuralfp', name=RUN_NAME,
            config={'run_training': RUN_TRAINING, 'train_epochs': TRAIN_EPOCHS,
                    'n_songs': n_songs, 'n_queries': total_queries}
        )
        print(f"✅ wandb: {wandb_run.url}")
    else:
        print(f"⚠️ no wandb key at {KEY_PATH}")
else:
    print("⚠️ wandb disabled (WANDB_ENABLED=False)")

# %% [markdown]
# ## CELL 8 — Build NeuralFP config + train (if RUN_TRAINING)

# %%
import yaml
import numpy as np

# Ensure cwd points to NeuralFP repo (relative paths in run.py expect this)
os.chdir(WORK)
print(f"cwd: {os.getcwd()}")

# Re-arm RUN_TRAINING in case prior interrupted run flipped it
if TRAIN_EPOCHS > 0:
    RUN_TRAINING = True

with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)

# Override paths (MAX_EPOCH set later after resume detection — see below)
cfg['DIR']['SOURCE_ROOT_DIR'] = f'{NAFP}/music/'
cfg['DIR']['BG_ROOT_DIR']     = f'{NAFP}/aug/bg/'
cfg['DIR']['IR_ROOT_DIR']     = f'{NAFP}/aug/ir/'
cfg['DIR']['SPEECH_ROOT_DIR'] = f'{NAFP}/aug/speech/common_voice_8k/en/'
cfg['DIR']['OUTPUT_ROOT_DIR'] = '/content/logs/emb/'
cfg['DIR']['LOG_ROOT_DIR']    = '/content/logs/'
cfg['TRAIN']['MINI_TEST_IN_TRAIN'] = False

CKPT_DIR_LOCAL = '/content/logs/checkpoint/colab'

# Skip-training path: if REUSE_CKPT_FROM is set AND that Drive checkpoint dir exists, copy
# its files into CKPT_DIR_LOCAL and SKIP training entirely. Cell 9 will then restore from
# CKPT_DIR_LOCAL as usual. Saves ~12 min for inference-only alterations (USE_PITCH_TTA etc)
# whose training is identical to baseline.
import shutil as _shutil
SKIPPED_TRAINING = False
if RUN_TRAINING and REUSE_CKPT_FROM:
    _src_ckpt = f'{DRIVE_BASE}/checkpoints/{REUSE_CKPT_FROM}'
    if os.path.isdir(_src_ckpt) and any(f.endswith('.index') for f in os.listdir(_src_ckpt)):
        _shutil.rmtree(CKPT_DIR_LOCAL, ignore_errors=True)
        os.makedirs(CKPT_DIR_LOCAL, exist_ok=True)
        for f in os.listdir(_src_ckpt):
            subprocess.run(['cp', f'{_src_ckpt}/{f}', CKPT_DIR_LOCAL], check=True)
        SKIPPED_TRAINING = True
        print(f"⏩ REUSE_CKPT_FROM='{REUSE_CKPT_FROM}' → skipping training, restored {len(os.listdir(CKPT_DIR_LOCAL))} files from Drive")
    else:
        print(f"⚠️ REUSE_CKPT_FROM='{REUSE_CKPT_FROM}' but {_src_ckpt} missing/empty — falling back to fresh training")

# Always train from scratch when not skipping — DO NOT resume mid-training.
# Why: NeuralFP uses tf.keras.optimizers.schedules.CosineDecay sized to MAX_EPOCH steps.
# Resuming would inherit the optimizer's step counter (past decay_steps) → effective LR=0 →
# zero weight updates (verified: ckpt-1 vs ckpt-2 model kernels were bit-identical).
# So either skip training (REUSE_CKPT_FROM) OR train fresh from empty dir.
if RUN_TRAINING and not SKIPPED_TRAINING:
    _shutil.rmtree(CKPT_DIR_LOCAL, ignore_errors=True)
    os.makedirs(CKPT_DIR_LOCAL, exist_ok=True)

cfg['TRAIN']['MAX_EPOCH'] = TRAIN_EPOCHS if RUN_TRAINING else 1

with open('config/colab.yaml', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False)

if RUN_TRAINING and not SKIPPED_TRAINING:
    import re as _re
    print(f"--- Training {TRAIN_EPOCHS} epoch(s) from scratch ---")
    train_env = os.environ.copy()
    train_env['TF_USE_LEGACY_KERAS'] = '1'
    train_env['PYTHONUNBUFFERED'] = '1'

    log_path = f'{DRIVE_BASE}/results/training_{RUN_NAME}.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    t0 = time.time()
    last_print_t = 0  # progress every 30s
    cur_epoch = '?'

    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(
            ['python', '-u', 'run.py', 'train', 'colab', '-c', 'colab', f'--max_epoch={TRAIN_EPOCHS}'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=train_env
        )
        for raw in iter(proc.stdout.readline, ''):
            logf.write(raw); logf.flush()

            # Each line may have multiple \r-separated progress updates — take last
            line = raw.replace('\r', '\n').strip().split('\n')[-1]

            # Epoch boundary
            m_ep = _re.search(r'Epoch\s+(\d+)/(\d+)', line)
            if m_ep:
                cur_epoch = f"{m_ep.group(1)}/{m_ep.group(2)}"
                elapsed_m = (time.time()-t0)/60
                print(f"\n🟢 [{elapsed_m:.1f}m] Starting epoch {cur_epoch}")
                last_print_t = time.time()
                continue

            # Progress: matches "50/350" or "100/200" patterns from Keras progbar
            m_step = _re.search(r'(\d+)/(\d+)', line)
            if m_step and (time.time() - last_print_t > 30):
                done, total = int(m_step.group(1)), int(m_step.group(2))
                if total > 10 and done <= total:  # plausible progress numbers
                    pct = 100 * done / total
                    elapsed = time.time() - t0
                    eta_sec = elapsed * (total - done) / done if done > 0 else 0
                    eta_min = eta_sec / 60
                    elapsed_m = elapsed / 60
                    print(f"⏳ [{elapsed_m:.1f}m] epoch {cur_epoch}: {pct:.0f}% ({done}/{total}) — ETA {eta_min:.1f}m")
                    last_print_t = time.time()
                    continue

            # Important keywords always print
            if any(k in line.lower() for k in ['saved', 'restoring', 'error']):
                print(f"   {line[:200]}")

        proc.wait()
        rc = proc.returncode

    print(f"\n✅ Training done — {(time.time()-t0)/60:.1f} min total, exit={rc}")
    print(f"📋 Full log: {log_path}")
    if rc != 0:
        print("⚠️ training failed — falling back to random-init")
        RUN_TRAINING = False

# Persist checkpoints to Drive (only when we actually trained — skip if reusing from Drive,
# since that would just create a duplicate folder with identical weights).
if RUN_TRAINING and not SKIPPED_TRAINING:
    DRIVE_CKPT_DIR = f'{DRIVE_BASE}/checkpoints/{RUN_NAME}'
    os.makedirs(DRIVE_CKPT_DIR, exist_ok=True)
    if os.path.isdir(CKPT_DIR_LOCAL):
        for f in os.listdir(CKPT_DIR_LOCAL):
            subprocess.run(['cp', f'{CKPT_DIR_LOCAL}/{f}', DRIVE_CKPT_DIR])
        print(f"✅ Checkpoints copied to Drive: {DRIVE_CKPT_DIR}")
        print(f"   Files: {sorted(os.listdir(DRIVE_CKPT_DIR))[:6]}")

# %% [markdown]
# ## CELL 9 — Build model + restore latest checkpoint (if trained)

# %%
from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter

m_pre = get_melspec_layer(cfg, trainable=False)
m_fp  = get_fingerprinter(cfg, trainable=False)

EMB_SZ = int(cfg['MODEL']['EMB_SZ'])
FS = int(cfg['MODEL']['FS'])
DUR = float(cfg['MODEL']['DUR'])
SEG_LEN = int(FS * DUR)
HOP_LEN = int(FS * HOP_SECONDS)

# Warm-up to instantiate variables
dummy = np.zeros((1, 1, SEG_LEN), dtype=np.float32)
_ = m_fp(m_pre(dummy, training=False), training=False)
print(f"✅ Model built: {len(m_fp.variables)} vars, "
      f"{sum(int(np.prod(v.shape)) for v in m_fp.variables if v.shape.rank):,} params")

# Manual restore (more reliable than framework — see docs/RESULTS_INTERPRETATION.md)
def manual_restore(m_fp, ckpt_path):
    print(f"  → Manual restore from {ckpt_path}")
    reader = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    model_keys = {k: shape_map[k] for k in shape_map if k.startswith('model/')}
    print(f"    Found {len(model_keys)} model/* keys, iterating {len(m_fp.variables)} model vars")
    used = set()
    assigned, skipped = 0, 0
    for var in m_fp.variables:
        target_shape = tuple(var.shape.as_list())
        var_path_parts = [p.replace(':0','') for p in var.name.split('/') if p]
        candidates = [k for k, sh in model_keys.items()
                      if tuple(sh) == target_shape and k not in used]
        best_key, best_score = None, -1
        for k in candidates:
            score = sum(1 for p in var_path_parts if p in k)
            if score > best_score:
                best_score, best_key = score, k
        if best_key:
            var.assign(reader.get_tensor(best_key))
            used.add(best_key)
            assigned += 1
        else:
            skipped += 1
    print(f"    Manual assign: {assigned} ok, {skipped} skipped")

if RUN_TRAINING and os.path.isdir(CKPT_DIR_LOCAL):
    ckpts = sorted([f for f in os.listdir(CKPT_DIR_LOCAL) if f.startswith('ckpt-') and f.endswith('.index')])
    if ckpts:
        latest_idx = max(int(c.split('-')[1].split('.')[0]) for c in ckpts)
        manual_restore(m_fp, f'{CKPT_DIR_LOCAL}/ckpt-{latest_idx}')
    else:
        print(f"⚠️ no checkpoints at {CKPT_DIR_LOCAL}")
print(f"✅ Model ready (trained={RUN_TRAINING})")

# %% [markdown]
# ## CELL 10 — Audio loading + inference

# %%
import librosa
import faiss
from datetime import datetime
import json

def load_and_slice(path):
    audio, _ = librosa.load(path, sr=FS, mono=True)
    if len(audio) < SEG_LEN:
        audio = np.pad(audio, (0, SEG_LEN - len(audio)))
    n_segs = max(1, 1 + (len(audio) - SEG_LEN) // HOP_LEN)
    segments = np.zeros((n_segs, 1, SEG_LEN), dtype=np.float32)
    for i in range(n_segs):
        start = i * HOP_LEN
        segments[i, 0, :] = audio[start:start + SEG_LEN]
    return segments

def embed_batch(segments, batch_size=128):
    out = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        spec = m_pre(batch, training=False)
        emb = m_fp(spec, training=False)
        out.append(emb.numpy())
    return np.concatenate(out, axis=0).astype(np.float32)

def acc_pct(p, t):
    return f"{(p/t)*100:.1f}%" if t else "0.0%"

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
N_WORKERS = max(4, min(16, multiprocessing.cpu_count()))

def load_silent(path):
    """librosa load with errors swallowed (returns None on failure)."""
    try:
        return load_and_slice(path)
    except Exception:
        return None

def load_silent_with_pitch_tta(path, semitones=(-1, 0, 1)):
    """Load audio + create pitch-shifted variants for test-time augmentation.
    Returns list of segment arrays (one per pitch shift). Used when USE_PITCH_TTA=True.
    Each pitch shift is wrapped individually so a failure in one variant doesn't kill the whole query —
    we fall back to whichever variants succeeded. Returns None only if ALL variants failed."""
    try:
        audio, _ = librosa.load(path, sr=FS, mono=True)
    except Exception:
        return None
    if len(audio) < SEG_LEN:
        audio = np.pad(audio, (0, SEG_LEN - len(audio)))
    out = []
    for st in semitones:
        try:
            shifted = librosa.effects.pitch_shift(audio, sr=FS, n_steps=st) if st != 0 else audio
            n_segs = max(1, 1 + (len(shifted) - SEG_LEN) // HOP_LEN)
            segments = np.zeros((n_segs, 1, SEG_LEN), dtype=np.float32)
            for i in range(n_segs):
                start = i * HOP_LEN
                segments[i, 0, :] = shifted[start:start + SEG_LEN]
            out.append(segments)
        except Exception:
            continue  # skip this variant, keep others
    return out if out else None

# === Build DB: parallel audio load → sequential GPU embed ===
n_songs_total = len(songs)
print(f"\n=== Building DB from {n_songs_total} originals (parallel load × {N_WORKERS}) ===")
t0 = time.time()
all_song_names = sorted(songs.keys())
db_paths = [songs[sn]['original'] for sn in all_song_names]
last_log = 0
loaded_segs = [None] * n_songs_total
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(load_silent, p): i for i, p in enumerate(db_paths)}
    done_count = 0
    for fut in futures:
        pass  # just to keep ref
    for fut in list(futures):
        idx = futures[fut]
        loaded_segs[idx] = fut.result()
        done_count += 1
        now = time.time()
        if (now - last_log > 4) or (done_count == n_songs_total):
            print(f"  [{now-t0:.1f}s] Audio load: {done_count}/{n_songs_total} ({100*done_count/n_songs_total:.0f}%)")
            last_log = now
print(f"  ⚡ Audio loaded in {time.time()-t0:.1f}s, embedding on GPU...")

t_emb = time.time()
db_vectors, db_song_ids = [], []
song_id_to_name = {}
for sid, (sn, segs) in enumerate(zip(all_song_names, loaded_segs)):
    song_id_to_name[sid] = sn
    if segs is None:
        continue
    embs = embed_batch(segs)
    db_vectors.append(embs)
    db_song_ids.extend([sid] * len(embs))
db_matrix = np.concatenate(db_vectors, axis=0)
db_song_ids_arr = np.array(db_song_ids, dtype=np.int32)
index = faiss.IndexFlatIP(EMB_SZ)
index.add(db_matrix)
print(f"  ✅ DB ({db_matrix.shape}) built in {time.time()-t0:.1f}s "
      f"(load {t_emb-t0:.1f}s + embed {time.time()-t_emb:.1f}s)")

# === Build query plan ===
queries = []  # list of (path, expected_sn, kind) where kind ∈ {'full','clip'}
for sn in sorted(songs.keys()):
    s = songs[sn]
    if s['remix_full']:
        queries.append((s['remix_full'], sn, 'full'))
    for cp in s['clips']:
        queries.append((cp, sn, 'clip'))
total_queries = len(queries)

# === Run queries: parallel audio load → sequential GPU embed + FAISS search ===
# When USE_PITCH_TTA=True, each query is loaded as a list of 3 pitch-shifted variants;
# the search aggregates similarity scores across all variants → more robust to pitch drift.
loader = load_silent_with_pitch_tta if USE_PITCH_TTA else load_silent
mode_label = " + pitch TTA (-1, 0, +1 semitone)" if USE_PITCH_TTA else ""
print(f"\n=== Running {total_queries} queries (parallel load × {N_WORKERS}){mode_label} ===")
t0 = time.time()
last_log = 0
loaded_q = [None] * total_queries
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(loader, q[0]): i for i, q in enumerate(queries)}
    done_count = 0
    for fut in list(futures):
        idx = futures[fut]
        loaded_q[idx] = fut.result()
        done_count += 1
        now = time.time()
        if (now - last_log > 4) or (done_count == total_queries):
            eta = (now - t0) * (total_queries - done_count) / done_count if done_count else 0
            print(f"  [{now-t0:.1f}s] Audio load: {done_count}/{total_queries} ({100*done_count/total_queries:.0f}%) — ETA {eta:.0f}s")
            last_log = now
print(f"  ⚡ Audio loaded in {time.time()-t0:.1f}s, embedding+searching on GPU...")

t_search = time.time()
records_full, records_clip = [], []
last_log = 0
for qi, ((qpath, sn, kind), payload) in enumerate(zip(queries, loaded_q)):
    if payload is None:
        rec = {'expected': sn, 'matched': None, 'status': 'ERROR'}
    else:
        try:
            # If pitch-TTA: payload is list of segment arrays (one per pitch shift). Aggregate.
            seg_lists = payload if USE_PITCH_TTA else [payload]
            cand_scores = {}
            for segs in seg_lists:
                embs = embed_batch(segs)
                D, I = index.search(embs, TOPK)
                for row_d, row_i in zip(D, I):
                    for d, idx in zip(row_d, row_i):
                        sid = int(db_song_ids_arr[idx])
                        cand_scores[sid] = cand_scores.get(sid, 0.0) + float(d)
            best = max(cand_scores, key=cand_scores.get)
            pred = song_id_to_name[best]
            rec = {'expected': sn, 'matched': pred, 'status': 'PASS' if pred == sn else 'FAIL'}
        except Exception as e:
            rec = {'expected': sn, 'matched': None, 'status': 'ERROR'}
    (records_full if kind == 'full' else records_clip).append(rec)
    now = time.time()
    if (now - last_log > 4) or (qi+1 == total_queries):
        done_q = qi + 1
        eta = (now - t_search) * (total_queries - done_q) / done_q if done_q else 0
        print(f"  [{now-t_search:.1f}s] Search: {done_q}/{total_queries} ({100*done_q/total_queries:.0f}%) — ETA {eta:.0f}s")
        last_log = now
print(f"  ✅ Inference done in {time.time()-t0:.1f}s "
      f"(load {t_search-t0:.1f}s + search {time.time()-t_search:.1f}s)")

# Summarize
def summarize(records):
    total = len(records)
    p = sum(1 for r in records if r['status']=='PASS')
    return {'total': total, 'pass': p, 'accuracy': acc_pct(p, total)}

s_full = summarize(records_full)
s_clip = summarize(records_clip)
all_records = records_full + records_clip
overall_pass = sum(1 for r in all_records if r['status']=='PASS')
s_over = {'total': len(all_records), 'pass': overall_pass,
          'accuracy': acc_pct(overall_pass, len(all_records))}

print(f"\n=== RESULTS ({RUN_NAME}) ===")
print(f"  Overall:    {s_over['accuracy']} ({s_over['pass']}/{s_over['total']})")
print(f"  Full-remix: {s_full['accuracy']} ({s_full['pass']}/{s_full['total']})")
print(f"  Clips:      {s_clip['accuracy']} ({s_clip['pass']}/{s_clip['total']})")

# Save to Drive
result = {
    'library': 'neuralfp', 'run_name': RUN_NAME,
    'timestamp': datetime.now().isoformat(),
    'training_epochs': TRAIN_EPOCHS if RUN_TRAINING else 0,
    'reused_ckpt_from': REUSE_CKPT_FROM if SKIPPED_TRAINING else None,
    'alterations': {'pitch_tta': USE_PITCH_TTA, 'dilated_cnn': USE_DILATED_CNN},
    'config': {'fs': FS, 'segment_dur_sec': DUR, 'hop_sec': HOP_SECONDS,
               'embedding_size': EMB_SZ, 'topk': TOPK},
    'summary': {'test_full': s_full, 'test_clips': s_clip, 'overall': s_over},
}
result_path = f'{DRIVE_BASE}/results/{RUN_NAME}.json'
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\n✅ Saved {result_path}")

if wandb_run:
    wandb.log({'overall_acc': overall_pass/len(all_records),
               'full_acc': s_full['pass']/s_full['total'],
               'clip_acc': s_clip['pass']/s_clip['total']})
    wandb.finish()

print("\n=== DONE ===")
