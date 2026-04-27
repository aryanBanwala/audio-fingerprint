# %% [markdown]
# # NeuralFP Trend Analysis — Inference across all checkpoints (v4)
#
# **v3 result:** Framework restore left 64 vars unmatched (partial restore). Fixed via
# manual fallback for ckpts 4-10 (got correct values), but ckpt-2 used framework path
# only → 24.9% might be off. v4: force manual_restore for ALL ckpts including ckpt-2.

# %% [markdown]
# ## CELL 1 — Setup: deps + clone NeuralFP

# %%
import sys, os, subprocess, importlib

print(f"Python: {sys.version.split()[0]}")
deps = {'kapre': 'kapre==0.3.5', 'faiss': 'faiss-cpu', 'tf_keras': 'tf_keras', 'librosa': 'librosa'}
for pkg, target in deps.items():
    try:
        importlib.import_module(pkg)
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

# %% [markdown]
# ## CELL 2 — Clone NeuralFP source

# %%
%cd /kaggle/working
subprocess.run(['rm', '-rf', 'NeuralFP'])
subprocess.run(['git', 'clone', '-q', 'https://github.com/mimbres/neural-audio-fp', 'NeuralFP'])
sys.path.insert(0, '/kaggle/working/NeuralFP')

trainer_path = '/kaggle/working/NeuralFP/model/trainer.py'
with open(trainer_path) as f:
    src = f.read()
if 'tf.keras.experimental.CosineDecay' in src:
    src = src.replace('tf.keras.experimental.CosineDecay', 'tf.keras.optimizers.schedules.CosineDecay')
    with open(trainer_path, 'w') as f:
        f.write(src)
    print("  ✅ patched trainer.py")
%cd /kaggle/working/NeuralFP
print("✅ NeuralFP source ready")

# %% [markdown]
# ## CELL 3 — Locate checkpoints + inspect structure

# %%
ckpt_root = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'ckpt-10.index' in files:
        ckpt_root = root
        break

if not ckpt_root:
    raise FileNotFoundError("Couldn't find checkpoint files at /kaggle/input.")

print(f"✅ Checkpoints found at: {ckpt_root}")
ckpt_files = sorted([f for f in os.listdir(ckpt_root) if f.startswith('ckpt-') and f.endswith('.index')])
ckpt_indices = sorted([int(f.split('-')[1].split('.')[0]) for f in ckpt_files])
print(f"  Indices: {ckpt_indices}")

# DIAGNOSTIC: dump variable names from one checkpoint to verify structure
import numpy as np
print(f"\n=== Inspecting ckpt-{ckpt_indices[0]} structure ===")
reader_first = tf.train.load_checkpoint(f'{ckpt_root}/ckpt-{ckpt_indices[0]}')
shape_map_first = reader_first.get_variable_to_shape_map()
print(f"  Total variables in checkpoint: {len(shape_map_first)}")
model_keys_first = sorted([k for k in shape_map_first if k.startswith('model/')])
print(f"  model/* keys: {len(model_keys_first)}")
for k in model_keys_first[:5]:
    print(f"    {k} -> {shape_map_first[k]}")

# DIAGNOSTIC: verify two checkpoints actually differ
print(f"\n=== Verifying ckpt-{ckpt_indices[0]} vs ckpt-{ckpt_indices[-1]} differ ===")
reader_last = tf.train.load_checkpoint(f'{ckpt_root}/ckpt-{ckpt_indices[-1]}')
sample_keys = [k for k in shape_map_first if k.startswith('model/') and shape_map_first[k]][:3]
ckpts_actually_differ = False
for k in sample_keys:
    v1 = reader_first.get_tensor(k)
    v2 = reader_last.get_tensor(k)
    diff = float(np.abs(v1 - v2).mean())
    print(f"  {k[:80]}... abs_diff={diff:.4e} {'✅' if diff > 1e-6 else '⚠️ identical'}")
    if diff > 1e-6:
        ckpts_actually_differ = True

if not ckpts_actually_differ:
    print("⚠️ Checkpoints are IDENTICAL on disk — cannot show training trend!")
else:
    print("✅ Checkpoints DO differ on disk — restore should produce different results")

# %% [markdown]
# ## CELL 4 — Verify Indian dataset

# %%
OUR_BASE = '/kaggle/input/datasets/aryanbanwala97/audio-fp-indian-music-btp'
assert os.path.isdir(OUR_BASE), f"Indian dataset not at {OUR_BASE}"
song_dirs = sorted([d for d in os.listdir(OUR_BASE) if os.path.isdir(os.path.join(OUR_BASE, d))])
print(f"✅ Indian dataset: {len(song_dirs)} songs")

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
total_queries = n_remix + n_clips
print(f"  {n_songs} songs, {n_remix} full-remix queries, {n_clips} clips, total: {total_queries}")

# %% [markdown]
# ## CELL 5 — Build model ONCE + cache audio

# %%
import yaml
import librosa
import time

with open('config/default.yaml') as f:
    cfg = yaml.safe_load(f)

EMB_SZ = int(cfg['MODEL']['EMB_SZ'])
FS = int(cfg['MODEL']['FS'])
DUR = float(cfg['MODEL']['DUR'])
HOP_SECONDS = 0.5
SEG_LEN = int(FS * DUR)
HOP_LEN = int(FS * HOP_SECONDS)
TOPK = 5

print(f"Config: EMB_SZ={EMB_SZ}, FS={FS}Hz, DUR={DUR}s, SEG_LEN={SEG_LEN}")

# Build model ONCE — NeuralFP's own inference pattern (model/generate.py)
print("\n=== Building model (one-time) ===")
from model.fp.melspec.melspectrogram import get_melspec_layer
from model.fp.nnfp import get_fingerprinter
m_pre = get_melspec_layer(cfg, trainable=False)
m_fp  = get_fingerprinter(cfg, trainable=False)

# Warm-up: run forward pass to instantiate weights
dummy = np.zeros((1, 1, SEG_LEN), dtype=np.float32)
_ = m_fp(m_pre(dummy, training=False), training=False)

n_total_vars = len(m_fp.variables)
n_trainable_vars = len(m_fp.trainable_variables)
n_params = sum(int(np.prod(v.shape)) for v in m_fp.variables if v.shape.rank)
print(f"  ✅ Model built: {n_total_vars} total vars ({n_trainable_vars} trainable, "
      f"rest non-trainable due to trainable=False), {n_params:,} parameters")
print(f"  First 3 variable names:")
for v in m_fp.variables[:3]:
    print(f"    {v.name}  shape={tuple(v.shape)}")

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

cache_size_mb = sum(v.nbytes for v in cached_audio.values()) / 1024 / 1024
print(f"  ✅ Cached {len(cached_audio)} audio segments in {time.time()-t0:.1f}s ({cache_size_mb:.1f} MB)")

# %% [markdown]
# ## CELL 6 — Restore helpers (framework first, manual fallback)

# %%
import faiss
from datetime import datetime
import json

def acc_pct(p, t):
    return f"{(p/t)*100:.1f}%" if t else "0.0%"

def weight_snapshot(m_fp, n=5):
    """Read first N variable values via .variables (NOT trainable_variables — empty here)."""
    out = []
    vars_list = [v for v in m_fp.variables if v.shape.rank and v.shape.num_elements()]
    for v in vars_list[:n]:
        arr = v.numpy()
        out.append({'name': v.name, 'shape': tuple(arr.shape),
                    'mean': float(arr.mean()), 'std': float(arr.std())})
    return out

def print_snapshot(label, snap):
    print(f"  {label}:")
    for s in snap:
        print(f"    {s['name']:<55s} shape={str(s['shape']):<20s} mean={s['mean']:+.6f} std={s['std']:.6f}")

def manual_restore(m_fp, ckpt_path):
    """Fallback: read variables from checkpoint file directly + assign by name+shape match."""
    print(f"  → Manual restore from {ckpt_path}")
    reader = tf.train.load_checkpoint(ckpt_path)
    shape_map = reader.get_variable_to_shape_map()
    model_keys = {k: shape_map[k] for k in shape_map if k.startswith('model/')}
    print(f"    Found {len(model_keys)} model/* keys in checkpoint")
    print(f"    Iterating {len(m_fp.variables)} model variables")

    used_keys = set()
    assigned, skipped = 0, 0
    for var in m_fp.variables:
        target_shape = tuple(var.shape.as_list())
        # var.name is like 'finger_printer/conv_layer_2/conv2d_1x3/bias:0'
        # ckpt key is like 'model/front_conv/.../conv2d_1x3/bias/.ATTRIBUTES/VARIABLE_VALUE'
        var_path_parts = [p.replace(':0','') for p in var.name.split('/') if p]

        candidates = [k for k, sh in model_keys.items()
                      if tuple(sh) == target_shape and k not in used_keys]

        # Score candidates by matching path fragments
        best_key = None
        best_score = -1
        for k in candidates:
            score = sum(1 for p in var_path_parts if p in k)
            if score > best_score:
                best_score = score
                best_key = k

        if best_key:
            val = reader.get_tensor(best_key)
            var.assign(val)
            used_keys.add(best_key)
            assigned += 1
        else:
            skipped += 1
    print(f"    Manual assign: {assigned} ok, {skipped} skipped, {len(model_keys)-len(used_keys)} ckpt keys unused")

def restore_checkpoint(m_fp, ckpt_path, prev_snap):
    """Forced manual restore for ALL checkpoints (skips framework path entirely).

    v3 lesson: framework `tf.train.Checkpoint.restore` left 64 vars unmatched even on
    first call → produced wrong accuracy numbers. Manual path reads checkpoint file
    directly via `tf.train.load_checkpoint` and assigns by shape+name-fragment match.
    """
    print(f"  → Restoring from {ckpt_path}")
    pre_snap = weight_snapshot(m_fp)
    print_snapshot("Pre-restore  weights (first 5)", pre_snap)

    # Use manual restore directly — skip framework path which is unreliable here
    manual_restore(m_fp, ckpt_path)

    post_snap = weight_snapshot(m_fp)
    print_snapshot("Post-restore weights (first 5)", post_snap)

    # Sanity check: weights must have changed
    changed = any(abs(p['mean'] - q['mean']) > 1e-9 or abs(p['std'] - q['std']) > 1e-9
                  for p, q in zip(pre_snap, post_snap))
    if not changed:
        print(f"  ⚠️ Manual restore did NOT change weights — possible bug")
    else:
        print(f"  ✅ Weights changed — manual restore confirmed")

    return post_snap

def evaluate_checkpoint(ckpt_idx, m_pre, m_fp, prev_snap):
    print(f"\n{'='*60}\nCheckpoint {ckpt_idx}\n{'='*60}")
    t_total = time.time()

    ckpt_path = f'{ckpt_root}/ckpt-{ckpt_idx}'
    post_snap = restore_checkpoint(m_fp, ckpt_path, prev_snap)

    def embed_batch(segments, batch_size=128):
        out = []
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i+batch_size]
            spec = m_pre(batch, training=False)
            emb = m_fp(spec, training=False)
            out.append(emb.numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    # Build DB
    t0 = time.time()
    db_vectors = []
    db_song_ids = []
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

    # CRITICAL diagnostic: DB embedding signature.
    # If two checkpoints produce identical embeddings, restore did NOT work.
    db_sig = {
        'first_vec_mean': float(db_matrix[0].mean()),
        'first_vec_std': float(db_matrix[0].std()),
        'all_vecs_mean': float(db_matrix.mean()),
        'first_vec_first5': db_matrix[0][:5].tolist(),
    }
    print(f"  DB signature: first_vec[mean={db_sig['first_vec_mean']:+.6f}, "
          f"std={db_sig['first_vec_std']:.6f}], all_mean={db_sig['all_vecs_mean']:+.6f}")
    print(f"  DB[0][:5] = {db_sig['first_vec_first5']}")

    # Inference
    t0 = time.time()
    records_full, records_clip = [], []
    for sn in sorted(songs.keys()):
        s = songs[sn]
        if s['remix_full']:
            tq = time.time()
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
                err = None
            except Exception as e:
                pred, conf, top2, status_lbl, err = None, 0.0, [], 'ERROR', str(e)
            records_full.append({'file': os.path.relpath(s['remix_full'], OUR_BASE),
                                 'expected': sn, 'matched': pred,
                                 'confidence': round(float(conf), 4), 'top2': top2,
                                 'time': round(time.time()-tq, 3), 'status': status_lbl,
                                 **({'error': err} if err else {})})
        for i, cp in enumerate(s['clips']):
            tq = time.time()
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
                err = None
            except Exception as e:
                pred, conf, top2, status_lbl, err = None, 0.0, [], 'ERROR', str(e)
            records_clip.append({'file': os.path.relpath(cp, OUR_BASE),
                                 'expected': sn, 'matched': pred,
                                 'confidence': round(float(conf), 4), 'top2': top2,
                                 'time': round(time.time()-tq, 3), 'status': status_lbl,
                                 **({'error': err} if err else {})})
    print(f"  Inference done in {time.time()-t0:.1f}s")

    def summarize(records):
        total = len(records)
        p = sum(1 for r in records if r['status'] == 'PASS')
        f = sum(1 for r in records if r['status'] == 'FAIL')
        nm = sum(1 for r in records if r['status'] == 'NO_MATCH')
        er = sum(1 for r in records if r['status'] == 'ERROR')
        avg_t = round(sum(r['time'] for r in records) / total, 3) if total else 0.0
        return {'total': total, 'pass': p, 'fail': f, 'no_match': nm, 'error': er,
                'accuracy': acc_pct(p, total), 'avg_query_time_sec': avg_t}

    s_full = summarize(records_full)
    s_clip = summarize(records_clip)
    all_records = records_full + records_clip
    overall_pass = sum(1 for r in all_records if r['status'] == 'PASS')
    s_over = {'total': len(all_records), 'pass': overall_pass,
              'fail': len(all_records) - overall_pass,
              'accuracy': acc_pct(overall_pass, len(all_records))}

    per_song = {}
    for sn in sorted(songs.keys()):
        rs = [r for r in all_records if r['expected'] == sn]
        per_song[sn] = {
            'full_pass': sum(1 for r in rs if r['status']=='PASS' and '/clips_' not in r['file']),
            'full_fail': sum(1 for r in rs if r['status']=='FAIL' and '/clips_' not in r['file']),
            'full_nomatch': 0, 'full_error': 0,
            'clip_pass': sum(1 for r in rs if r['status']=='PASS' and '/clips_' in r['file']),
            'clip_fail': sum(1 for r in rs if r['status']=='FAIL' and '/clips_' in r['file']),
            'clip_nomatch': 0, 'clip_error': 0,
            'total_tests': len(rs),
            'total_pass': sum(1 for r in rs if r['status']=='PASS'),
            'total_fail': sum(1 for r in rs if r['status']!='PASS'),
        }

    result = {
        'library': 'neuralfp',
        'data_dir': 'dataset',
        'timestamp': datetime.now().isoformat(),
        'mode': 'training',
        'training_epochs': ckpt_idx,
        'config': {'fs': FS, 'segment_dur_sec': DUR, 'hop_sec': HOP_SECONDS,
                   'embedding_size': EMB_SZ, 'topk': TOPK,
                   'db_n_segments': int(index.ntotal),
                   'db_signature': db_sig,
                   'weights_snapshot': post_snap},
        'db_verification': {'songs_in_dataset': n_songs, 'expected_uploads': n_songs,
                            'uploaded': n_songs, 'all_uploaded': True},
        'summary': {'test_full': s_full, 'test_clips': s_clip, 'overall': s_over},
        'per_song': per_song,
        'tests': {'test_full': records_full, 'test_clips': records_clip},
    }

    out_path = f'/kaggle/working/neuralfp_dataset_results_ckpt-{ckpt_idx}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  ✅ Saved {out_path}")
    print(f"  RESULT — overall: {s_over['accuracy']} | full: {s_full['accuracy']} | clips: {s_clip['accuracy']}")
    print(f"  Total checkpoint time: {time.time()-t_total:.1f}s")
    return result, post_snap

# %% [markdown]
# ## CELL 7 — Loop through all checkpoints

# %%
print("="*60)
print(f"Running inference on {len(ckpt_indices)} checkpoints: {ckpt_indices}")
print("="*60)

trend = []
prev_snap = None
t_grand = time.time()
for idx in ckpt_indices:
    res, prev_snap = evaluate_checkpoint(idx, m_pre, m_fp, prev_snap)
    trend.append({
        'epoch': idx,
        'overall_pass': res['summary']['overall']['pass'],
        'overall_total': res['summary']['overall']['total'],
        'overall_acc_pct': float(res['summary']['overall']['accuracy'].rstrip('%')),
        'full_acc_pct': float(res['summary']['test_full']['accuracy'].rstrip('%')),
        'clips_acc_pct': float(res['summary']['test_clips']['accuracy'].rstrip('%')),
        'db_first_mean': res['config']['db_signature']['first_vec_mean'],
        'db_first_std': res['config']['db_signature']['first_vec_std'],
    })
print(f"\n{'='*60}\nALL DONE in {time.time()-t_grand:.0f}s\n{'='*60}")

# Sanity check: db signatures must vary
print("\n=== SANITY: DB embedding signature across checkpoints ===")
print(f"{'Epoch':<8} {'db_mean':<14} {'db_std':<10} {'overall_acc':<10}")
for t in trend:
    print(f"{t['epoch']:<8} {t['db_first_mean']:+.8f}   {t['db_first_std']:.6f}   {t['overall_acc_pct']:.1f}%")
unique_means = len(set(round(t['db_first_mean'], 8) for t in trend))
if unique_means == 1:
    print(f"⚠️ All checkpoints produced IDENTICAL embeddings — restore still broken")
else:
    print(f"✅ {unique_means}/{len(trend)} unique embedding signatures — restore working")

# %% [markdown]
# ## CELL 8 — Generate trend summary CSV + plot

# %%
import csv

trend_full = [
    {'epoch': 0, 'overall_pass': 74, 'overall_total': 273, 'overall_acc_pct': 27.1,
     'full_acc_pct': 28.6, 'clips_acc_pct': 27.0, 'note': 'random init (v2)'},
    {'epoch': 1, 'overall_pass': 83, 'overall_total': 273, 'overall_acc_pct': 30.4,
     'full_acc_pct': 38.1, 'clips_acc_pct': 29.8, 'note': 'v3 reference'},
] + [{'epoch': t['epoch'], 'overall_pass': t['overall_pass'], 'overall_total': t['overall_total'],
      'overall_acc_pct': t['overall_acc_pct'], 'full_acc_pct': t['full_acc_pct'],
      'clips_acc_pct': t['clips_acc_pct'], 'note': 'v5 trend run'} for t in trend]

csv_path = '/kaggle/working/trend_summary.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(trend_full[0].keys()))
    w.writeheader()
    w.writerows(trend_full)
print(f"✅ CSV saved: {csv_path}")

print("\n=== ACCURACY TREND ===")
print(f"{'Epoch':<8} {'Overall':<10} {'Full':<10} {'Clips':<10} {'Note':<25}")
print("-" * 65)
for t in trend_full:
    print(f"{t['epoch']:<8} {t['overall_acc_pct']:>5.1f}%   {t['full_acc_pct']:>5.1f}%   {t['clips_acc_pct']:>5.1f}%   {t['note']}")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = [t['epoch'] for t in trend_full]
    ax.plot(epochs, [t['overall_acc_pct'] for t in trend_full], 'o-', label='Overall', linewidth=2)
    ax.plot(epochs, [t['full_acc_pct'] for t in trend_full], 's-', label='Full-remix', linewidth=2)
    ax.plot(epochs, [t['clips_acc_pct'] for t in trend_full], '^-', label='Clips', linewidth=2)
    ax.axhline(y=30.8, color='red', linestyle='--', alpha=0.5, label='Dejavu (30.8%)')
    ax.axhline(y=4.7, color='gray', linestyle=':', alpha=0.5, label='Random (4.7%)')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('NeuralFP on Indian Music — Accuracy vs Training Epoch')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = '/kaggle/working/trend_plot.png'
    plt.savefig(plot_path, dpi=120)
    print(f"✅ Plot saved: {plot_path}")
except Exception as e:
    print(f"⚠️ Plot failed: {e}")

print("\n=== DONE ===")
