# HANDOFF — Post-Compact Continuation Notes

**Date:** 2026-04-26 (Sunday afternoon)
**Critical deadline:** Monday morning — show progress to professor

---

## 🎯 Where we are RIGHT NOW (just before compact)

### Smoke test v3 just completed on Kaggle (64.8s success)

✅ **Compatibility verified** (huge win):
- Python 3.12 + TF 2.15+ on Kaggle works with NeuralFP code
- kapre 0.3.5, faiss-cpu install successfully
- All 4 NeuralFP modules import OK: `model.fp.nnfp`, `model.dataset`, `model.utils.config_gpu_memory_lim`, `model.trainer`

✅ **Mimbres dataset found** (after path fix):
- Path: `/kaggle/input/datasets/mimbres/neural-audio-fingerprint/neural-audio-fp-dataset`
- Has `music/` and `aug/` subfolders as needed

❌ **Our dataset still NOT found**:
- Expected at: `/kaggle/input/datasets/aryanbanwala97/audio-fp-indian-music-btp/...`
- The `find_dataset()` walk didn't pick it up — likely needs deeper search or different keyword

⚠️ **Cell 5 (1-epoch training) was SKIPPED** because preconditions not met (our dataset missing)

### Kaggle CLI status was lagging — UI showed COMPLETE but CLI kept showing RUNNING. Don't trust CLI status alone, also poll output files.

---

## 🚀 Immediate next step

1. **Debug our dataset path** — likely the walk function didn't go deep enough or the path has unexpected nesting
2. Inspect `/tmp/smoke_v3/neuralfp-btp-smoke-test.log` — search for "aryanbanwala" or "audio-fp" lines to see exact path
3. Push smoke test v4 with fixed path detection
4. Once datasets ok + 1-epoch training succeeds → push full training notebook (30 epochs, ~3-5 hrs on P100)

---

## 📋 Project context (don't lose this)

**BTP-II at DTU CSE, AY 2025-26**
**Title:** Audio Fingerprinting for Indian Music
**Synopsis:** Existing fingerprinters fail on Indian music due to ornaments (Meend/Murki/Alaap/Gamak/Khatka). Synopsis specifically points at NeuralFP (Mimbres = author's GitHub handle, NOT a dataset).

**Already done (Phase 0 baseline):**
- Dejavu: 30.8% (84/273) on `dataset/`
- Olaf: 7.3% (20/273)
- Panako: 3.3% (9/273)
- Results in `scripts/results/<lib>_dataset_results.json`

**Today's goal:** Get NeuralFP 4th data point for 4-way comparison demo on Monday.

**Future (Phase 1 next month):** Modifications — STFT→CQT, Indian augmentation, dilated CNN.

---

## 🗂️ Key files & paths (verify these still exist)

```
/Users/aryanbanwala/Desktop/git/audio-fingerprint/
├── .venv/bin/kaggle                          # kaggle CLI (use this, not system pip)
├── BTP_PLAN.md                               # symlink to ~/.claude/plans/ab-hume-...
├── NeuralFP/                                 # cloned mimbres repo
├── Panako/, olaf/, dejavu/                   # classical libs (already benchmarked)
├── dataset/                                  # 21 Hindi songs, 273 queries
│   └── dataset-metadata.json                 # for Kaggle upload
├── scripts/                                  # uniform pipeline (3 libs done)
│   ├── common.py                             # discover_songs, write_results
│   ├── compare.py                            # 3-way table (extend to 4-way)
│   ├── dejavu_test.{sh,py}, olaf_test.*, panako_test.*
│   └── results/<lib>_dataset_results.json    # baseline accuracies
├── kaggle_notebook/
│   ├── smoke_test.py                         # source (defensive v3)
│   ├── smoke_test.ipynb                      # Kaggle-ready notebook
│   ├── kernel-metadata.json                  # Kaggle config
│   └── notebook.py                           # full training (skeleton, complete it after smoke passes)
└── journal/
    ├── 2026-04-26.md                         # append-only progress (don't alter past entries)
    └── HANDOFF.md                            # this file
```

---

## 🔑 Kaggle setup (already working)

- **Username:** `aryanbanwala97`
- **API key:** `~/.kaggle/kaggle.json` (chmod 600 done)
- **Smoke test kernel:** https://www.kaggle.com/code/aryanbanwala97/neuralfp-btp-smoke-test
- **Kernel ID:** `aryanbanwala97/neuralfp-btp-smoke-test`
- **Dataset URL:** https://www.kaggle.com/datasets/aryanbanwala97/audio-fp-indian-music-btp
- **Dataset has v2:** `dataset_audio.tar.gz` (176 MB, 21 songs + 273 queries)

### Reusable Kaggle CLI commands

```bash
# Status
.venv/bin/kaggle kernels status aryanbanwala97/neuralfp-btp-smoke-test

# Download outputs
rm -rf /tmp/smoke_vN && .venv/bin/kaggle kernels output aryanbanwala97/neuralfp-btp-smoke-test -p /tmp/smoke_vN/

# Push new version (after editing smoke_test.py)
cd kaggle_notebook && rm -f smoke_test.ipynb && \
  ../.venv/bin/jupytext --to ipynb smoke_test.py && \
  python3 -c "
import json
with open('smoke_test.ipynb') as f: nb = json.load(f)
nb.setdefault('metadata', {})
nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nb['metadata']['language_info'] = {'name': 'python', 'version': '3.10'}
with open('smoke_test.ipynb', 'w') as f: json.dump(nb, f, indent=1)
" && ../.venv/bin/kaggle kernels push -p .
```

**Important:** kernelspec MUST be manually injected after jupytext conversion (jupytext alone doesn't add it — papermill will fail with "No kernel name found").

---

## ⚠️ User's workflow preferences (CRITICAL — follow strictly)

These are ALREADY in `~/.claude/projects/-Users-aryanbanwala-Desktop-git-audio-fingerprint/memory/`:
- `feedback_workflow.md` — confirm before commands, narrate actions, append journal, narrate cloud actions
- `feedback_external_deps.md` — ask before cloning new repos / installing new packages / cloud uploads
- `user_role.md` — Hinglish, terse direct answers, DTU CSE student
- `project_btp.md` — full BTP context

**Hard rules:**
1. Confirm before any command (even local ones)
2. Tell user what each action does + risk + expected outcome
3. Append to `journal/2026-04-26.md` for major events (don't edit past entries)
4. Narrate Kaggle/cloud actions so user can verify on dashboard
5. Hinglish replies, terse, no over-explanation

---

## 🧠 Compatibility findings (important — don't re-research)

These are PROVEN working on Kaggle:
- ✅ Python 3.12 + TensorFlow 2.18 (Kaggle current default)
- ✅ kapre 0.3.5 (auto-installs and imports fine)
- ✅ faiss-cpu (use this, NOT faiss-gpu — faiss-gpu 1.6.5 is broken on Python 3.10+)
- ✅ NeuralFP source code imports cleanly
- ⚠️ Don't try to install TF 2.5 / older — Kaggle's preinstalled TF 2.18 works

**Earlier concerns (RESOLVED):**
- ~~TF version mismatch~~ — works fine
- ~~Python 3.12 incompatibility~~ — works fine
- ~~kapre + new TF~~ — works fine
- Only real issue: dataset path detection (Kaggle nests under `/kaggle/input/datasets/`)

---

## 🛠️ Smoke test v3 result (full)

```json
{
  "timestamp": "2026-04-26T11:21:53.136028",
  "python": "3.12.12",
  "deps_status": {
    "tensorflow": "OK", "numpy": "OK", "librosa": "OK",
    "kapre": "OK_AFTER_INSTALL", "faiss": "OK_AFTER_INSTALL",
    "wavio": "OK", "click": "OK", "yaml": "OK", "matplotlib": "OK"
  },
  "nafp_modules_status": {
    "model.fp.nnfp": "OK", "model.dataset": "OK",
    "model.utils.config_gpu_memory_lim": "OK", "model.trainer": "OK"
  },
  "mimbres_dataset_ok": true,
  "our_dataset_ok": false
}
```

---

## 📅 Time budget

- Total: 10-15 hours till Monday morning demo
- Used so far: ~2-3 hours (setup, classical baselines confirmed, NeuralFP smoke test v1-v3)
- Remaining: 7-12 hours
- Critical path: get full NeuralFP training done in this window

**Realistic worst case fallback:** if NeuralFP keeps failing, pivot to MERT (foundation model, 1-2 hrs total, plug-and-play). User initially questioned MERT (correctly — limited modification potential), so try NeuralFP first.

---

## 🚦 Decision tree for next session

```
If our dataset path issue resolves → 1-epoch training works → push full 30-epoch training (3-5 hrs)
If our dataset issue persists → debug path one more time → if still fails, manually upload via different method
If training fails on Kaggle → try PyTorch port (stdio2016/pfann) — needs user permission to clone
If everything fails → MERT pivot (last resort, document why we deviated from synopsis)
```

---

## 📝 What can be SKIPPED in compaction

These are noise — don't preserve detail:
- All "RUNNING / RUNNING / RUNNING" heartbeat exchanges
- Detailed back-and-forth about Kaggle signup steps (already done)
- Full research synthesis (already in `~/.claude/plans/ab-hume-ye-krne-soft-melody.md`)
- v1 attempt details (kernelspec error, fixed and superseded)
- Detailed compatibility research (now empirically verified — see `Compatibility findings` above)

## 📌 What MUST be preserved

- Project context (BTP, synopsis, professor.md decode)
- User preferences (workflow, communication style)
- Current state (v3 just done, our dataset path issue)
- Kaggle setup (URL, kernel ID, auth)
- File paths in repo
- Time deadline (Monday)
- Decision tree for next steps
