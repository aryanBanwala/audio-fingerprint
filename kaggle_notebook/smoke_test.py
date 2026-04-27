# %% [markdown]
# # NeuralFP Smoke Test (v5) — BTP-II
#
# Defensive smoke test: diagnose Kaggle env, install deps, clone NeuralFP,
# verify both datasets, run 1-epoch training in Keras 2 legacy mode.
#
# **v5 changes vs v4:**
#  - Install `tf_keras` (Keras 2 standalone) and run training with `TF_USE_LEGACY_KERAS=1`
#  - This restores `OrderedEnqueuer`, `experimental.CosineDecay`, and other Keras 2 APIs
#    that Keras 3 (TF 2.16+ default) removed.
#
# **v4 changes vs v3 (kept):**
#  - Patch `tf.keras.experimental.CosineDecay` → `tf.keras.optimizers.schedules.*` (defense in depth)
#  - Direct-path lookup for our dataset
#  - Better stderr dump on training failure
#
# **Datasets attached:**
#  - `mimbres/neural-audio-fingerprint` (FMA training data)
#  - `aryanbanwala97/audio-fp-indian-music-btp` (our 21 Indian songs)

# %% [markdown]
# ## CELL 1 — Pure diagnostic (no installs)
# Just shows what Kaggle gives us. Zero risk.

# %%
import sys, os, platform, subprocess
print("=" * 60)
print("CELL 1 — Environment diagnostic")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Machine: {platform.machine()}")

# GPU info
print("\n--- GPU ---")
gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
if gpu_info.returncode == 0:
    for line in gpu_info.stdout.split('\n')[:15]:
        print(line)
else:
    print("nvidia-smi unavailable — GPU may not be enabled")

# Pre-installed packages relevant to us
print("\n--- Pre-installed packages (relevant) ---")
pip_out = subprocess.run(['pip', 'list', '--format=columns'], capture_output=True, text=True).stdout
for line in pip_out.split('\n'):
    for kw in ['tensorflow', 'torch', 'kapre', 'faiss', 'librosa', 'numpy', 'kaggle', 'jupyter']:
        if kw.lower() in line.lower() and 'WARNING' not in line:
            print(f"  {line}")
            break

# Check both attached datasets
print("\n--- /kaggle/input/ contents ---")
input_dir = '/kaggle/input'
if os.path.isdir(input_dir):
    for d in os.listdir(input_dir):
        full = f'{input_dir}/{d}'
        if os.path.isdir(full):
            try:
                contents = os.listdir(full)[:5]
                print(f"  📁 {d}/ -> {contents}{'...' if len(os.listdir(full)) > 5 else ''}")
            except Exception as e:
                print(f"  📁 {d}/ -> ERROR: {e}")
else:
    print("  ⚠️ /kaggle/input does not exist")

print("\n✅ CELL 1 done — diagnostic complete")

# %% [markdown]
# ## CELL 2 — Try imports of NeuralFP's required deps
# Don't fail hard. Catalog what works and what's broken.

# %%
print("=" * 60)
print("CELL 2 — Dependency imports (defensive)")
print("=" * 60)

import importlib
results = {}

deps_to_check = [
    'tensorflow',
    'numpy',
    'librosa',
    'kapre',
    'faiss',
    'wavio',
    'click',
    'yaml',
    'matplotlib',
]

for pkg in deps_to_check:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, '__version__', 'no_version_attr')
        results[pkg] = ('OK', v)
        print(f"  ✅ {pkg}: {v}")
    except ImportError as e:
        results[pkg] = ('MISSING', str(e)[:80])
        print(f"  ❌ {pkg}: MISSING")
    except Exception as e:
        results[pkg] = ('ERROR', f"{type(e).__name__}: {str(e)[:80]}")
        print(f"  ⚠️ {pkg}: {type(e).__name__}: {str(e)[:80]}")

# GPU check via TF if available
if results.get('tensorflow', ('FAIL', ''))[0] == 'OK':
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n  TF GPUs: {len(gpus)}")
    for g in gpus:
        print(f"    {g}")

# Try installing missing pieces (defensive — don't crash if install fails)
missing = [p for p, (s, _) in results.items() if s == 'MISSING']
if missing:
    print(f"\n--- Attempting to install missing: {missing} ---")
    install_map = {
        'kapre': 'kapre==0.3.5',
        'wavio': 'wavio',
        'click': 'click',
        'yaml': 'pyyaml',
        'faiss': 'faiss-cpu',  # use CPU faiss — more reliable than GPU
    }
    for pkg in missing:
        target = install_map.get(pkg, pkg)
        print(f"\n  Installing {target}...")
        r = subprocess.run(['pip', 'install', '-q', target], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"    ✅ {target} installed")
            try:
                importlib.invalidate_caches()
                m = importlib.import_module(pkg)
                results[pkg] = ('OK_AFTER_INSTALL', getattr(m, '__version__', '?'))
            except Exception as e:
                results[pkg] = ('INSTALL_BUT_BROKEN', str(e)[:80])
                print(f"    ⚠️ Installed but import still fails: {e}")
        else:
            print(f"    ❌ Install failed: {r.stderr[-200:]}")
            results[pkg] = ('INSTALL_FAILED', r.stderr[-100:])

print("\n--- Final dependency status ---")
for p, (s, v) in results.items():
    icon = '✅' if 'OK' in s else '❌' if 'MISSING' in s or 'FAIL' in s else '⚠️'
    print(f"  {icon} {p}: {s} ({v})")

# --- v5: Install tf_keras (Keras 2 legacy) for NeuralFP compatibility ---
# TF 2.16+ ships Keras 3 by default; NeuralFP needs Keras 2 APIs (OrderedEnqueuer, experimental.CosineDecay).
# tf_keras + TF_USE_LEGACY_KERAS=1 makes tf.keras alias to Keras 2.
print("\n--- Installing tf_keras (Keras 2 legacy) for NeuralFP compat ---")
r = subprocess.run(['pip', 'install', '-q', 'tf_keras'], capture_output=True, text=True)
if r.returncode == 0:
    print("  ✅ tf_keras installed")
    try:
        import tf_keras
        print(f"     tf_keras version: {tf_keras.__version__}")
        results['tf_keras'] = ('OK_AFTER_INSTALL', tf_keras.__version__)
    except Exception as e:
        print(f"  ⚠️ tf_keras imports broken: {e}")
        results['tf_keras'] = ('INSTALL_BUT_BROKEN', str(e)[:80])
else:
    print(f"  ❌ tf_keras install failed: {r.stderr[-200:]}")
    results['tf_keras'] = ('INSTALL_FAILED', r.stderr[-100:])

print("\n✅ CELL 2 done")

# %% [markdown]
# ## CELL 3 — Clone NeuralFP and try importing its modules
# Most likely break point. We catalog what fails.

# %%
print("=" * 60)
print("CELL 3 — NeuralFP module imports")
print("=" * 60)

%cd /kaggle/working
subprocess.run(['rm', '-rf', 'NeuralFP'])
clone_result = subprocess.run(
    ['git', 'clone', '-q', 'https://github.com/mimbres/neural-audio-fp', 'NeuralFP'],
    capture_output=True, text=True
)
if clone_result.returncode != 0:
    print(f"❌ Clone failed: {clone_result.stderr}")
else:
    print("✅ NeuralFP repo cloned")
    sys.path.insert(0, '/kaggle/working/NeuralFP')

%cd /kaggle/working/NeuralFP

# --- TF 2.18 compatibility patches ---
# NeuralFP was written for TF 2.4-2.10 era. Patch deprecated APIs.
print("\n--- Applying TF 2.18 compat patches ---")
patches = [
    # tf.keras.experimental.CosineDecay → tf.keras.optimizers.schedules.CosineDecay
    ('model/trainer.py',
     'tf.keras.experimental.CosineDecay',
     'tf.keras.optimizers.schedules.CosineDecay'),
]
for path, old, new in patches:
    try:
        with open(path) as f:
            src = f.read()
        if old in src:
            src = src.replace(old, new)
            with open(path, 'w') as f:
                f.write(src)
            print(f"  ✅ patched {path}: {old} → {new}")
        else:
            print(f"  ⚠️ {path}: pattern not found (already patched?)")
    except Exception as e:
        print(f"  ❌ {path} patch failed: {e}")

# Try importing key NeuralFP modules
modules_to_test = [
    'model.fp.nnfp',
    'model.dataset',
    'model.utils.config_gpu_memory_lim',
    'model.trainer',
]

nafp_results = {}
for mod in modules_to_test:
    try:
        m = importlib.import_module(mod)
        nafp_results[mod] = 'OK'
        print(f"  ✅ {mod}")
    except Exception as e:
        nafp_results[mod] = f"{type(e).__name__}: {str(e)[:200]}"
        print(f"  ❌ {mod}")
        print(f"     Error: {type(e).__name__}: {str(e)[:300]}")

# Check if eval module works (no TF needed for eval_faiss.py per README)
try:
    sys.path.insert(0, '/kaggle/working/NeuralFP/eval')
    import eval_faiss
    print(f"  ✅ eval.eval_faiss")
except Exception as e:
    print(f"  ⚠️ eval.eval_faiss: {type(e).__name__}: {str(e)[:200]}")

print("\n✅ CELL 3 done")

# %% [markdown]
# ## CELL 4 — Datasets accessible?
# Just check paths, don't extract our tar yet.

# %%
print("=" * 60)
print("CELL 4 — Datasets accessibility")
print("=" * 60)

# Smart path detection — Kaggle nests datasets under /kaggle/input/datasets/<user>/<slug>/
def find_dataset(keyword, marker_files=None):
    """Walk /kaggle/input/ to find a folder matching keyword, optionally with marker file/folder inside."""
    marker_files = marker_files or []
    candidates = []
    for root, dirs, files in os.walk('/kaggle/input'):
        # Avoid walking too deep
        if root.count(os.sep) > 6:
            dirs[:] = []
            continue
        rl = root.lower()
        if keyword.lower() in rl:
            # If marker files specified, check at least one is present
            if not marker_files or any(os.path.exists(f'{root}/{m}') for m in marker_files):
                candidates.append(root)
                # Don't recurse further into matched folder
                dirs[:] = []
    return candidates

# Mimbres dataset (look for 'music' + 'aug' subfolders as markers)
mimbres_candidates = find_dataset('mimbres', ['music', 'aug'])
if not mimbres_candidates:
    mimbres_candidates = find_dataset('neural-audio-fingerprint', ['music', 'aug'])
if not mimbres_candidates:
    mimbres_candidates = find_dataset('neural-audio', ['music'])

mimbres_ok = False
NAFP = None
if mimbres_candidates:
    NAFP = mimbres_candidates[0]
    contents = os.listdir(NAFP)
    print(f"✅ Mimbres dataset found at {NAFP}")
    print(f"   Contents: {contents}")
    if 'music' in contents and 'aug' in contents:
        mimbres_ok = True
        try:
            n_music = len(os.listdir(f'{NAFP}/music'))
            n_aug = len(os.listdir(f'{NAFP}/aug'))
            print(f"   music/ has {n_music} items, aug/ has {n_aug} items")
        except Exception as e:
            print(f"   (couldn't list deeper: {e})")
else:
    print("❌ Mimbres dataset NOT found")
    print(f"   /kaggle/input/ tree (top 2 levels):")
    for root, dirs, files in os.walk('/kaggle/input'):
        depth = root.replace('/kaggle/input', '').count(os.sep)
        if depth <= 2:
            print(f"     {root}: dirs={dirs[:5]} files={files[:3]}")

# Our dataset — Kaggle may auto-extract tar.gz, so try direct path first then walk
our_ok = False
OUR_BASE = None
OUR_TAR = None

# Try known/likely paths first
direct_paths = [
    '/kaggle/input/audio-fp-indian-music-btp',
    '/kaggle/input/datasets/aryanbanwala97/audio-fp-indian-music-btp',
    '/kaggle/input/aryanbanwala97/audio-fp-indian-music-btp',
]
for p in direct_paths:
    if os.path.isdir(p):
        OUR_BASE = p
        print(f"\n✅ Our dataset found via direct path: {OUR_BASE}")
        break

# Fallback: walk looking for any folder containing 'audio-fp' or 'aryanbanwala'
if not OUR_BASE:
    for root, dirs, files in os.walk('/kaggle/input'):
        depth = root.count(os.sep)
        if depth > 8:
            dirs[:] = []
            continue
        rl = root.lower()
        if ('audio-fp' in rl or 'aryanbanwala' in rl) and 'mimbres' not in rl:
            # Skip the bare /datasets/aryanbanwala97 wrapper, want the actual dataset folder
            try:
                entries = os.listdir(root)
            except Exception:
                continue
            if entries:
                OUR_BASE = root
                print(f"\n✅ Our dataset found via walk: {OUR_BASE}")
                break

if OUR_BASE:
    contents = sorted(os.listdir(OUR_BASE))
    print(f"   Top-level contents ({len(contents)} items): {contents[:10]}")
    # Look for tar archive
    if 'dataset_audio.tar.gz' in contents:
        OUR_TAR = f'{OUR_BASE}/dataset_audio.tar.gz'
        size_mb = os.path.getsize(OUR_TAR) / 1024 / 1024
        print(f"   ✅ tar archive: {size_mb:.1f} MB")
        our_ok = True
    else:
        # Check if Kaggle auto-extracted — look for mp3 files anywhere within
        mp3_count = 0
        for r, _, fs in os.walk(OUR_BASE):
            mp3_count += sum(1 for f in fs if f.endswith('.mp3'))
            if mp3_count > 5:
                break
        if mp3_count > 0:
            print(f"   ✅ Auto-extracted: {mp3_count}+ mp3 files found")
            our_ok = True
        else:
            print(f"   ⚠️ Folder exists but no tar.gz or mp3s — needs investigation")
else:
    print(f"\n❌ Our dataset NOT found")
    # Diagnostic: show what IS available
    print("   /kaggle/input/ structure:")
    for root, dirs, files in os.walk('/kaggle/input'):
        depth = root.replace('/kaggle/input', '').count(os.sep)
        if depth <= 3:
            print(f"     {root}: dirs={dirs[:6]} files={files[:3]}")

print(f"\n--- Status ---")
print(f"  Mimbres dataset OK: {mimbres_ok}")
print(f"  Our dataset OK: {our_ok}")
print("\n✅ CELL 4 done")

# %% [markdown]
# ## CELL 5 — Try minimal training (1 epoch only)
# Only proceed if CELL 2 + 3 + 4 are healthy.

# %%
print("=" * 60)
print("CELL 5 — Minimal training (1 epoch)")
print("=" * 60)

# Decision: should we even try training?
key_imports_ok = (results.get('tensorflow', ('FAIL',))[0] in ('OK', 'OK_AFTER_INSTALL') and
                   results.get('kapre', ('FAIL',))[0] in ('OK', 'OK_AFTER_INSTALL'))
nafp_imports_ok = nafp_results.get('model.trainer', '') == 'OK'

print(f"Pre-conditions:")
print(f"  TF + kapre OK: {key_imports_ok}")
print(f"  NeuralFP modules OK: {nafp_imports_ok}")
print(f"  Mimbres dataset OK: {mimbres_ok}")
print(f"  Our dataset OK:     {our_ok}")

if not (key_imports_ok and nafp_imports_ok and mimbres_ok):
    print("\n⚠️ SKIPPING training — critical pre-conditions not met")
    print("   Above output shows exactly what's broken.")
else:
    print("\n--- Pre-conditions met. Attempting 1-epoch training. ---")
    import yaml
    try:
        with open('config/default.yaml') as f:
            cfg = yaml.safe_load(f)
        cfg['DIR']['SOURCE_ROOT_DIR'] = f'{NAFP}/music/'
        cfg['DIR']['BG_ROOT_DIR'] = f'{NAFP}/aug/bg/'
        cfg['DIR']['IR_ROOT_DIR'] = f'{NAFP}/aug/ir/'
        cfg['DIR']['SPEECH_ROOT_DIR'] = f'{NAFP}/aug/speech/common_voice_8k/en/'
        cfg['DIR']['OUTPUT_ROOT_DIR'] = '/kaggle/working/logs/emb/'
        cfg['DIR']['LOG_ROOT_DIR'] = '/kaggle/working/logs/'
        cfg['TRAIN']['MAX_EPOCH'] = 1
        cfg['TRAIN']['MINI_TEST_IN_TRAIN'] = False
        with open('config/smoke.yaml', 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
        print("Config saved. Starting 1-epoch training (~5 min)...")

        import time
        t0 = time.time()
        # v5: Force Keras 2 mode for NeuralFP (OrderedEnqueuer, experimental.* APIs)
        train_env = os.environ.copy()
        train_env['TF_USE_LEGACY_KERAS'] = '1'
        train_proc = subprocess.run(
            ['python', 'run.py', 'train', 'smoke', '-c', 'smoke', '--max_epoch=1'],
            capture_output=True, text=True, timeout=900, env=train_env
        )
        elapsed = time.time() - t0

        stdout_lines = train_proc.stdout.split('\n')
        print(f"\n--- Training stdout ({len(stdout_lines)} lines, last 50) ---")
        for line in stdout_lines[-50:]:
            print(line)
        if train_proc.returncode != 0:
            stderr_lines = train_proc.stderr.split('\n')
            print(f"\n--- Training stderr ({len(stderr_lines)} lines, last 80) ---")
            for line in stderr_lines[-80:]:
                print(line)
            print(f"\n❌ Training failed (exit {train_proc.returncode}, {elapsed:.0f}s)")
        else:
            print(f"\n✅ Training succeeded ({elapsed:.0f}s)")
            ckpt_dir = '/kaggle/working/logs/checkpoint/smoke'
            if os.path.isdir(ckpt_dir):
                print(f"   Checkpoints: {os.listdir(ckpt_dir)}")
    except Exception as e:
        print(f"\n❌ Training setup failed: {type(e).__name__}: {e}")

print("\n✅ CELL 5 done")

# %% [markdown]
# ## CELL 6 — Final summary

# %%
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("\nDeps:")
for p, (s, v) in results.items():
    icon = '✅' if 'OK' in s else '❌'
    print(f"  {icon} {p}: {s}")

print("\nNeuralFP modules:")
for m, r in nafp_results.items():
    icon = '✅' if r == 'OK' else '❌'
    print(f"  {icon} {m}: {r[:80]}")

print(f"\nDatasets: mimbres={mimbres_ok}, ours={our_ok}")

# Save summary JSON
import json
from datetime import datetime
summary = {
    "timestamp": datetime.now().isoformat(),
    "python": sys.version.split()[0],
    "deps_status": {p: s for p, (s, _) in results.items()},
    "nafp_modules_status": nafp_results,
    "mimbres_dataset_ok": mimbres_ok,
    "our_dataset_ok": our_ok,
}
with open('/kaggle/working/smoke_test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nWritten: /kaggle/working/smoke_test_summary.json")

print("\n=== DECISION GUIDE ===")
if all('OK' in s for p, (s, _) in results.items() if p in ('tensorflow', 'kapre', 'librosa')) and \
   all(r == 'OK' for r in nafp_results.values()):
    print("✅ ALL GREEN — proceed with full 30-epoch training")
else:
    print("❌ Issues found — review above. Likely pivot to PyTorch port or MERT.")
