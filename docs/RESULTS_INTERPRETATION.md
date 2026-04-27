# Results Interpretation Guide

How to read the numbers in `scripts/results/`. Includes the bug saga that informs which numbers to trust.

## The 4-way comparison (top-level)

Each `<lib>_dataset_results.json` is the canonical run on the full 21-song / 273-query dataset:

| Library | Overall | Full-remix | Clips |
|---|---|---|---|
| Dejavu | 30.8% | 33.3% | 30.6% |
| Olaf | 7.3% | 23.8% | 6.0% |
| Panako | 3.3% | 4.8% | 3.2% |
| **NeuralFP (ckpt-10, manual restore)** | **30.8%** | 28.6% | **31.0%** |

NeuralFP-final ties Dejavu overall, beats it on clip queries (31.0% vs 30.6%).

## The trend (per checkpoint)

`scripts/results/trend/` contains 6 checkpoint snapshots from a single 10-epoch FMA training run. The accuracy trajectory is **U-shaped**:

```
Epoch  Overall  Full   Clips
0      27.1%    28.6%  27.0%   ← random init (no checkpoint, separate run)
1      30.4%    38.1%  29.8%   ← peak full-remix (separate run, in-memory inference)
2      26.0%    23.8%  26.2%   ← early dip
4      20.5%    23.8%  20.2%   ← valley
6      28.6%    33.3%  28.2%   ← recovery
8      30.0%    33.3%  29.8%
9      30.0%    28.6%  30.2%
10     30.8%    28.6%  31.0%   ← final, ties Dejavu
```

## What each row in the JSON means

```json
{
  "tests": {
    "test_full":  [{"file": "Ehsaas/remix/remix_001.mp3",
                    "expected": "Ehsaas",
                    "matched":  "Ehsaas",
                    "confidence": 0.42,
                    "top2": ["Ehsaas", "Mi_Amor"],
                    "status": "PASS"}, ...],
    "test_clips": [...]
  },
  "per_song": {
    "Ehsaas": {"full_pass": 1, "clip_pass": 12, "total_pass": 13,
               "total_tests": 13, ...}
  },
  "summary": { "overall": {"accuracy": "30.8%", ...}, ... }
}
```

- `expected` = ground-truth song
- `matched`  = top-1 prediction
- `status`   = `PASS` if matched==expected, else `FAIL` (or `NO_MATCH`/`ERROR`)
- `top2`     = top-2 candidates for ambiguity analysis

## How NeuralFP's checkpoints were verified — the bug saga

We hit three bugs while writing the trend kernel. Read this before trusting any NeuralFP number.

### Bug 1 — Reused Checkpoint object (TF #25081)

**Symptom:** all 6 checkpoints produced identical 24.5% accuracy.
**Cause:** `tf.train.Checkpoint(model=m_fp)` was created once outside the loop. Subsequent `.restore()` calls silently no-op'd.
**Fix:** create a **fresh `tf.train.Checkpoint` object per iteration**.

### Bug 2 — `clear_session()` breaks restore (TF #27937)

**Symptom:** `RuntimeError: Manual restore ALSO failed for ckpt-2`.
**Cause:** between iterations we called `tf.keras.backend.clear_session()` and rebuilt the model. clear_session severs the Keras backend's variable tracking; subsequent restores can't find variables.
**Fix:** **build model ONCE outside the loop** (NeuralFP's own `model/generate.py` pattern). No `clear_session`.

### Bug 3 — Framework `restore()` leaves vars unmatched

**Symptom:** even with bugs 1+2 fixed, `assert_existing_objects_matched()` reported "Found 64 Python objects unmatched" for every checkpoint. Accuracy was wrong.
**Cause:** for our specific NeuralFP model graph, `tf.train.Checkpoint.restore` doesn't bind ~64/576 variables to their checkpoint values. They stay at random init.
**Fix:** **manual restore** — read the checkpoint file directly via `tf.train.load_checkpoint(path)`, walk variables, assign by shape + name-fragment scoring. v4 result: `Manual assign: 576 ok, 0 skipped, 0 ckpt keys unused` for every ckpt.

### Verifying the fix worked

The trend kernel records the first DB embedding vector signature for each checkpoint:

```
ckpt 2:  db_first_mean=-0.011733
ckpt 4:  db_first_mean=+0.005633
ckpt 6:  db_first_mean=+0.006026
ckpt 8:  db_first_mean=+0.005678
ckpt 9:  db_first_mean=+0.005348
ckpt 10: db_first_mean=+0.005204
```

**Sanity check:** "✅ 6/6 unique embedding signatures — restore working".

### Numbers to discard from earlier work

| Source | Number | Why wrong |
|---|---|---|
| `full_pipeline.py` v5 stdout | NeuralFP 10-epoch = 24.2% | Bug 3 (partial restore — 64 vars unmatched) |
| v1 trend kernel | All 6 ckpts = 24.5% | Bug 1 (Checkpoint reuse) |
| v3 trend kernel ckpt-2 | 24.9% | Manual fallback didn't trigger for first iter |

The 30.8% number for ckpt-10 in `scripts/results/trend/neuralfp_dataset_results_ckpt-10.json` is the correct one.

## Numbers in `LATEST_RESULTS.md` and `MONDAY_DEMO.md` reference these files

If you change a number in either of those documents, update the source JSON it came from too. Currently:
- All `Dejavu/Olaf/Panako` numbers → `scripts/results/<lib>_dataset_results.json`
- All `NeuralFP epoch N` numbers → `scripts/results/trend/neuralfp_dataset_results_ckpt-N.json`
- Random-init (epoch 0) and 1-epoch numbers → from earlier in-memory inference runs (not in this folder; logged in `journal/2026-04-26.md`)

## Reproducing

```bash
# 4-way classical baseline (local Docker)
bash scripts/dejavu_test.sh
bash scripts/olaf_test.sh
bash scripts/panako_test.sh
python3 scripts/compare.py

# NeuralFP trend (push to Kaggle)
cd kaggle_notebook/trend
kaggle kernels push -p .
# wait ~12 min, then:
kaggle kernels output aryanbanwala97/neuralfp-btp-trend-analysis -p /tmp/out
```
