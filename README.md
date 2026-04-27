# Audio Fingerprinting for Indian Music

**B.Tech Project-II, DTU CSE, AY 2025-26**

Evaluating audio fingerprinting methods (classical Shazam-style + neural CNN-based) on Indian music. Demonstrates that existing systems fail on Indian ornaments (Meend, Murki, Gamak), and proposes Indian-specific modifications.

## Where to start

| Question | File |
|---|---|
| What's this project about? | [BTP_PLAN.md](BTP_PLAN.md) |
| Latest demo numbers (1-pager) | [LATEST_RESULTS.md](LATEST_RESULTS.md) |
| Full demo doc with slides outline | [MONDAY_DEMO.md](MONDAY_DEMO.md) |
| What's in each folder? | [docs/REPO_STRUCTURE.md](docs/REPO_STRUCTURE.md) |
| How to read the result numbers | [docs/RESULTS_INTERPRETATION.md](docs/RESULTS_INTERPRETATION.md) |
| Day-by-day progress log | [journal/2026-04-26.md](journal/2026-04-26.md) |

## Headline result

```
                  Dejavu  Olaf  Panako  NeuralFP-final-10ep
Overall:          30.8%   7.3%   3.3%    30.8%       ← ties Dejavu
Full-remix:       33.3%   23.8%  4.8%    28.6%
Clip queries:     30.6%   6.0%   3.2%    31.0%       ← beats Dejavu

Random baseline = 4.7% (1/21)
```

NeuralFP at 1-epoch peaks **38.1% on full-remix** (beats Dejavu's 33.3%). U-shape trend — see [scripts/results/trend/trend_plot.png](scripts/results/trend/trend_plot.png).

## Status

- ✅ **Phase 0** (April 2026) — baseline 4-way comparison, NeuralFP 10-epoch trend on Indian music
- 🟡 **Phase 1** (May 2026) — STFT→CQT, Indian augmentation (Meend/Murki/Gamak), Dilated CNN
- 🔵 **Phase 2** (June 2026) — Saraga external eval, expand custom dataset
- 📝 **Phase 3** (July-Aug 2026) — paper draft, ICASSP 2026 / ISMIR 2026 target

## Quick reproduction

```bash
# Classical baselines (local, Docker for Dejavu)
bash scripts/dejavu_test.sh    # ~5 min
bash scripts/olaf_test.sh      # ~30 sec
bash scripts/panako_test.sh    # ~2 min
python3 scripts/compare.py     # 4-way table

# NeuralFP trend (Kaggle GPU)
cd kaggle_notebook/trend
kaggle kernels push -p .
# wait ~12 min, then download
kaggle kernels output aryanbanwala97/neuralfp-btp-trend-analysis -p /tmp/out
```
