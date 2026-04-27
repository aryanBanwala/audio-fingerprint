# BTP-II — Latest Results (One-pager)

**Updated:** 2026-04-27 (Monday)
**Goal:** Audio Fingerprinting for Indian Music

---

## 🎯 Headline numbers (4-way comparison on 21 Hindi songs / 273 queries)

```
                Dejavu  Olaf  Panako  NeuralFP-final-10ep
Overall:        30.8%   7.3%   3.3%    30.8%       ← TIES Dejavu
Full-remix:     33.3%   23.8%  4.8%    28.6%
Clip queries:   30.6%   6.0%   3.2%    31.0%       ← beats Dejavu
```

**Random baseline = 4.7% (1/21)**

---

## 📈 Verified epoch-by-epoch trend (manual checkpoint restore — airtight)

```
Epoch  Overall  Full   Clips   Note
0      27.1%    28.6%  27.0%   random init (5.7× random)
1      30.4%    38.1%  29.8%   peak full-remix (beats Dejavu's 33.3%)
2      26.0%    23.8%  26.2%   ← early dip
4      20.5%    23.8%  20.2%   ← valley
6      28.6%    33.3%  28.2%   recovery
8      30.0%    33.3%  29.8%
9      30.0%    28.6%  30.2%
10     30.8%    28.6%  31.0%   ← overall peak, TIES Dejavu
```

See `scripts/results/trend/trend_plot.png` for U-shape visualisation.

---

## 🔬 3 key findings

### 1. Synopsis claim verified
Olaf and Panako catastrophically fail (<10%) on Indian music. Dejavu best classical at 30.8% but still misses 70% of queries.

### 2. Random-init NeuralFP already 27.1%
0-epoch (untrained) NeuralFP scores 5.7× above random. Mel-spectrogram preprocessing + CNN inductive bias → architecture is well-suited.

### 3. Non-monotonic training trajectory (paper-worthy)
NeuralFP on FMA (Western) shows **U-shape**: 1ep peak (30.4% overall, 38.1% full-remix) → 4ep valley (20.5%) → 10ep recovery (30.8%).
- **Mid-training instability** — 2-4 epoch dip suggests temporary overfitting to Western patterns
- Final epoch ties Dejavu but doesn't surpass it — **10× compute for parity**, not improvement
- 1-epoch full-remix (38.1%) beats Dejavu (33.3%) — **shorter training has different sweet spot per query type**
- **Motivates Phase 1 Indian-specific modifications** to push past Dejavu ceiling

---

## 📈 Phase 1 plan — 3 modifications (May 2026)

| Mod | Change | Expected gain |
|---|---|---|
| **CQT** replace STFT | Semitone-aligned features | +10-15% |
| **Indian augmentation** (Meend, Murki, Gamak) | Train on synthetic ornaments | +15-20% |
| **Dilated CNN** | Wider receptive field for pitch contours | +5-10% |
| **Combined target** | All 3 mods | **65-75%** overall |

Paper venue: ICASSP 2026 / ISMIR 2026 (both A* tier)

---

## 🎬 Live demo commands

```bash
# Show 4-way comparison (using ckpt-10 as NeuralFP)
python3 scripts/compare.py

# View trend plot
open scripts/results/trend/trend_plot.png

# View detailed per-checkpoint results
ls scripts/results/trend/*.json

# Read journal
cat journal/2026-04-26.md
```

---

*Status: Phase 0 done. Phase 1 starts post-demo.*
