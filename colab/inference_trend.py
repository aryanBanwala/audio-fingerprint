# %% [markdown]
# # NeuralFP Trend Analysis — Colab version
#
# **Goal:** Inference across all 6 saved checkpoints (epochs 2/4/6/8/9/10) using
# manual restore (576/576 vars). Same logic as `kaggle_notebook/inference_trend.py` v4.
#
# **Setup:** Run after `full_pipeline.ipynb` has trained + saved checkpoints to
# `/content/drive/MyDrive/btp/checkpoints/<run_name>/`

# %% [markdown]
# ## CELL 0 — Configuration

# %%
RUN_NAME = "colab-pipeline-10ep"   # Must match the training run_name in full_pipeline.py
HOP_SECONDS = 0.5
TOPK = 5

# %% [markdown]
# ## CELL 1 — Mount Drive + setup

# %%
from google.colab import drive
drive.mount('/content/drive')

import os, sys, subprocess, time
DRIVE_BASE = '/content/drive/MyDrive/btp'
CKPT_DIR = f'{DRIVE_BASE}/checkpoints/{RUN_NAME}'
RESULTS_DIR = f'{DRIVE_BASE}/results/{RUN_NAME}_trend'
os.makedirs(RESULTS_DIR, exist_ok=True)

assert os.path.isdir(CKPT_DIR), f"Checkpoints not at {CKPT_DIR}. Run full_pipeline.ipynb first."
ckpt_files = sorted([f for f in os.listdir(CKPT_DIR) if f.startswith('ckpt-') and f.endswith('.index')])
ckpt_indices = sorted([int(f.split('-')[1].split('.')[0]) for f in ckpt_files])
print(f"✅ Found {len(ckpt_indices)} checkpoints: {ckpt_indices}")

# %% [markdown]
# ## CELL 2 — Kaggle auth (load from Drive)

# %%
KAGGLE_JSON_DRIVE = f'{DRIVE_BASE}/secrets/kaggle.json'
KAGGLE_JSON_LOCAL = os.path.expanduser('~/.kaggle/kaggle.json')
assert os.path.isfile(KAGGLE_JSON_DRIVE), f"Run full_pipeline.ipynb first to set up kaggle.json"
os.makedirs(os.path.dirname(KAGGLE_JSON_LOCAL), exist_ok=True)
subprocess.run(['cp', KAGGLE_JSON_DRIVE, KAGGLE_JSON_LOCAL])
os.chmod(KAGGLE_JSON_LOCAL, 0o600)
subprocess.run(['pip', 'install', '-q', 'kaggle'], check=True)
print("✅ Kaggle CLI ready")

# %% [markdown]
# ## CELL 3 — Install deps + clone NeuralFP

# %%
import importlib

deps = {'kapre': 'kapre==0.3.5', 'faiss': 'faiss-cpu', 'tf_keras': 'tf_keras', 'librosa': 'librosa'}
for pkg, target in deps.items():
    try:
        importlib.import_module(pkg)
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

WORK = '/content/NeuralFP'
if os.path.isdir(WORK):
    subprocess.run(['rm', '-rf', WORK])
subprocess.run(['git', 'clone', '-q', 'https://github.com/mimbres/neural-audio-fp', WORK], check=True)
sys.path.insert(0, WORK)
%cd {WORK}

trainer_path = f'{WORK}/model/trainer.py'
with open(trainer_path) as f:
    src = f.read()
if 'tf.keras.experimental.CosineDecay' in src:
    src = src.replace('tf.keras.experimental.CosineDecay', 'tf.keras.optimizers.schedules.CosineDecay')
    with open(trainer_path, 'w') as f:
        f.write(src)
    print("  ✅ patched trainer.py")
print(f"✅ NeuralFP at {WORK}")

# %% [markdown]
# ## CELL 4 — Download Indian dataset (if not cached)

# %%
DATASETS_DIR = f'{DRIVE_BASE}/datasets'
INDIAN_DIR = f'{DATASETS_DIR}/audio-fp-indian-music-btp'
if not os.path.isdir(INDIAN_DIR) or not os.listdir(INDIAN_DIR):
    print("Downloading Indian dataset from Kaggle...")
    os.makedirs(DATASETS_DIR, exist_ok=True)
    subprocess.run(['kaggle', 'datasets', 'download',
                    '-d', 'aryanbanwala97/audio-fp-indian-music-btp',
                    '-p', DATASETS_DIR, '--unzip'], check=True)
    candidates = [d for d in os.listdir(DATASETS_DIR) if 'indian' in d.lower() or 'btp' in d.lower()]
    if candidates and candidates[0] != 'audio-fp-indian-music-btp':
        os.rename(f'{DATASETS_DIR}/{candidates[0]}', INDIAN_DIR)
print(f"✅ Indian: {INDIAN_DIR}")

OUR_BASE = INDIAN_DIR
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
print(f"  {n_songs} songs, {n_remix} full-remix, {n_clips} clips, total: {total_queries}")

# %% [markdown]
# ## CELL 5 — Build model + cache audio (one-time)

# %%
import yaml
import numpy as np
import librosa

with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)

EMB_SZ = int(cfg['MODEL']['EMB_SZ'])
FS = int(cfg['MODEL']['FS'])
DUR = float(cfg['MODEL']['DUR'])
SEG_LEN = int(FS * DUR)
HOP_LEN = int(FS * HOP_SECONDS)

from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter

m_pre = get_melspec_layer(cfg, trainable=False)
m_fp  = get_fingerprinter(cfg, trainable=False)
dummy = np.zeros((1, 1, SEG_LEN), dtype=np.float32)
_ = m_fp(m_pre(dummy, training=False), training=False)
print(f"✅ Model built: {len(m_fp.variables)} vars, "
      f"{sum(int(np.prod(v.shape)) for v in m_fp.variables if v.shape.rank):,} params")

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

print("\n=== Caching all audio segments (one-time) ===")
t0 = time.time()
cached_audio = {}
for sn in sorted(songs.keys()):
    s = songs[sn]
    cached_audio[(sn, 'orig')] = load_and_slice(s['original'])
    if s['remix_full']:
        cached_audio[(sn, 'remix')] = load_and_slice(s['remix_full'])
    for i, cp in enumerate(s['clips']):
        cached_audio[(sn, f'clip_{i}', cp)] = load_and_slice(cp)
print(f"  ✅ Cached {len(cached_audio)} segments in {time.time()-t0:.1f}s")

# %% [markdown]
# ## CELL 6 — Manual restore + per-checkpoint inference

# %%
import faiss
from datetime import datetime
import json

def acc_pct(p, t):
    return f"{(p/t)*100:.1f}%" if t else "0.0%"

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

def evaluate_checkpoint(ckpt_idx):
    print(f"\n{'='*60}\nCheckpoint {ckpt_idx}\n{'='*60}")
    t_total = time.time()
    ckpt_path = f'{CKPT_DIR}/ckpt-{ckpt_idx}'
    manual_restore(m_fp, ckpt_path)

    def embed_batch(segments, batch_size=128):
        out = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            spec = m_pre(batch, training=False)
            emb = m_fp(spec, training=False)
            out.append(emb.numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    t0 = time.time()
    db_vectors, db_song_ids = [], []
    song_id_to_name = {}
    for sid, sn in enumerate(sorted(songs.keys())):
        song_id_to_name[sid] = sn
        embs = embed_batch(cached_audio[(sn, 'orig')])
        db_vectors.append(embs)
        db_song_ids.extend([sid] * len(embs))
    db_matrix = np.concatenate(db_vectors, axis=0)
    db_song_ids_arr = np.array(db_song_ids, dtype=np.int32)
    index = faiss.IndexFlatIP(EMB_SZ)
    index.add(db_matrix)
    print(f"  DB built ({db_matrix.shape}) in {time.time()-t0:.1f}s")
    print(f"  DB[0] sig: mean={db_matrix[0].mean():+.6f}, std={db_matrix[0].std():.6f}")

    t0 = time.time()
    records_full, records_clip = [], []
    for sn in sorted(songs.keys()):
        s = songs[sn]
        if s['remix_full']:
            try:
                segs = cached_audio[(sn, 'remix')]
                embs = embed_batch(segs)
                D, I = index.search(embs, TOPK)
                cand_scores = {}
                for row_d, row_i in zip(D, I):
                    for d, idx in zip(row_d, row_i):
                        sid = int(db_song_ids_arr[idx])
                        cand_scores[sid] = cand_scores.get(sid, 0.0) + float(d)
                best = max(cand_scores, key=cand_scores.get)
                pred = song_id_to_name[best]
                conf = cand_scores[best] / sum(cand_scores.values())
                top2 = [song_id_to_name[s_] for s_, _ in sorted(cand_scores.items(), key=lambda x:-x[1])[:2]]
                status_lbl = 'PASS' if pred == sn else 'FAIL'
            except Exception as e:
                pred, conf, top2, status_lbl = None, 0.0, [], 'ERROR'
            records_full.append({'file': os.path.relpath(s['remix_full'], OUR_BASE),
                                 'expected': sn, 'matched': pred,
                                 'confidence': round(float(conf), 4), 'top2': top2,
                                 'status': status_lbl})
        for i, cp in enumerate(s['clips']):
            try:
                segs = cached_audio[(sn, f'clip_{i}', cp)]
                embs = embed_batch(segs)
                D, I = index.search(embs, TOPK)
                cand_scores = {}
                for row_d, row_i in zip(D, I):
                    for d, idx in zip(row_d, row_i):
                        sid = int(db_song_ids_arr[idx])
                        cand_scores[sid] = cand_scores.get(sid, 0.0) + float(d)
                best = max(cand_scores, key=cand_scores.get)
                pred = song_id_to_name[best]
                conf = cand_scores[best] / sum(cand_scores.values())
                top2 = [song_id_to_name[s_] for s_, _ in sorted(cand_scores.items(), key=lambda x:-x[1])[:2]]
                status_lbl = 'PASS' if pred == sn else 'FAIL'
            except Exception as e:
                pred, conf, top2, status_lbl = None, 0.0, [], 'ERROR'
            records_clip.append({'file': os.path.relpath(cp, OUR_BASE),
                                 'expected': sn, 'matched': pred,
                                 'confidence': round(float(conf), 4), 'top2': top2,
                                 'status': status_lbl})
    print(f"  Inference done in {time.time()-t0:.1f}s")

    def summarize(records):
        total = len(records)
        p = sum(1 for r in records if r['status']=='PASS')
        return {'total': total, 'pass': p, 'fail': total-p, 'accuracy': acc_pct(p, total)}

    s_full = summarize(records_full)
    s_clip = summarize(records_clip)
    all_records = records_full + records_clip
    overall_pass = sum(1 for r in all_records if r['status']=='PASS')
    s_over = {'total': len(all_records), 'pass': overall_pass,
              'fail': len(all_records)-overall_pass,
              'accuracy': acc_pct(overall_pass, len(all_records))}

    per_song = {}
    for sn in sorted(songs.keys()):
        rs = [r for r in all_records if r['expected']==sn]
        per_song[sn] = {'total_pass': sum(1 for r in rs if r['status']=='PASS'),
                        'total_tests': len(rs)}

    result = {
        'library': 'neuralfp', 'training_epochs': ckpt_idx,
        'timestamp': datetime.now().isoformat(),
        'summary': {'test_full': s_full, 'test_clips': s_clip, 'overall': s_over},
        'per_song': per_song,
        'tests': {'test_full': records_full, 'test_clips': records_clip},
    }
    out_path = f'{RESULTS_DIR}/neuralfp_dataset_results_ckpt-{ckpt_idx}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✅ Saved {out_path}")
    print(f"  RESULT — overall: {s_over['accuracy']} | full: {s_full['accuracy']} | clips: {s_clip['accuracy']}")
    print(f"  Total: {time.time()-t_total:.1f}s")
    return result

# %% [markdown]
# ## CELL 7 — Run all checkpoints

# %%
print("="*60)
print(f"Running inference on {len(ckpt_indices)} checkpoints: {ckpt_indices}")
print("="*60)

trend = []
t_grand = time.time()
for idx in ckpt_indices:
    res = evaluate_checkpoint(idx)
    trend.append({
        'epoch': idx,
        'overall_pass': res['summary']['overall']['pass'],
        'overall_total': res['summary']['overall']['total'],
        'overall_acc_pct': float(res['summary']['overall']['accuracy'].rstrip('%')),
        'full_acc_pct': float(res['summary']['test_full']['accuracy'].rstrip('%')),
        'clips_acc_pct': float(res['summary']['test_clips']['accuracy'].rstrip('%')),
    })
print(f"\n{'='*60}\nALL DONE in {time.time()-t_grand:.0f}s\n{'='*60}")

# %% [markdown]
# ## CELL 8 — Trend CSV + plot

# %%
import csv

csv_path = f'{RESULTS_DIR}/trend_summary.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(trend[0].keys()))
    w.writeheader()
    w.writerows(trend)
print(f"✅ CSV saved: {csv_path}")

print("\n=== ACCURACY TREND ===")
print(f"{'Epoch':<8} {'Overall':<10} {'Full':<10} {'Clips':<10}")
print("-" * 40)
for t in trend:
    print(f"{t['epoch']:<8} {t['overall_acc_pct']:>5.1f}%   {t['full_acc_pct']:>5.1f}%   {t['clips_acc_pct']:>5.1f}%")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = [t['epoch'] for t in trend]
    ax.plot(epochs, [t['overall_acc_pct'] for t in trend], 'o-', label='Overall', linewidth=2)
    ax.plot(epochs, [t['full_acc_pct'] for t in trend], 's-', label='Full-remix', linewidth=2)
    ax.plot(epochs, [t['clips_acc_pct'] for t in trend], '^-', label='Clips', linewidth=2)
    ax.axhline(y=30.8, color='red', linestyle='--', alpha=0.5, label='Dejavu (30.8%)')
    ax.axhline(y=4.7, color='gray', linestyle=':', alpha=0.5, label='Random (4.7%)')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('NeuralFP on Indian Music — Accuracy vs Training Epoch')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'{RESULTS_DIR}/trend_plot.png'
    plt.savefig(plot_path, dpi=120)
    print(f"✅ Plot saved: {plot_path}")
except Exception as e:
    print(f"⚠️ Plot failed: {e}")

print("\n=== DONE ===")
