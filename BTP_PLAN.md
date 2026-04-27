# BTP-II: Audio Fingerprinting for Indian Music — Study Notes & Plan

> **How to read this file:** 8 chapters. Read one chapter, sleep on it, come back. Don't try to absorb everything in one sitting. End mein decision points hain — un par tum decide karoge.

---

## 📑 Table of Contents

1. [Context — kyu, kya, kahan tak](#1-context)
2. [Aaj tak ka kaam — baseline benchmark](#2-baseline-done)
3. [Synopsis ki secret decode](#3-synopsis-decode)
4. [4 candidate models — kya kya available hai](#4-candidate-models)
5. [Deep dive — NeuralFP (recommended)](#5-neuralfp-deep-dive)
6. [4 possible modifications samajh ke](#6-modifications)
7. [Proposed plan + timeline](#7-proposed-plan)
8. [Decisions to lock](#8-decisions)

---

## 1. Context

**Project:** BTP-II at DTU CSE, AY 2025-26
**Title (per synopsis):** Audio Fingerprinting for Indian Music
**Synopsis claim:** Existing fingerprinting systems work for Western pop/rock but fail on Indian music (Meend, Murki, Alaap, Gamak, Khatka).
**Professor's instruction (from `professor.md`):**
> A* conferences, NeurIPS, ICMR, latest paper check, esa model jisme modification hojae, run karo implement karo, ek conference paper likho.

**Translation:** Top-tier conference target. Recent SoTA paper find. Modify it for Indian music. Implement + evaluate. Write conference paper.

**End goal:** A B.Tech-feasible (4-8 weeks) project with:
- Clear baseline (proves problem exists)
- Reproduced SoTA (proves you can run real research)
- 1-3 amendments (your contribution)
- Clean comparison story (panel explainable)
- Optional: paper draft for ICASSP/ISMIR/NCMR

---

## 2. Baseline (DONE — aaj ka kaam)

We benchmarked 3 classical fingerprinting libraries against `dataset/` (21 Hindi/Bollywood songs, 273 test queries):

| Library | Algorithm type | Overall accuracy | Notes |
|---|---|---:|---|
| **Dejavu** | Shazam-style (peaks + hashing) | **30.8%** | Best classical baseline |
| **Olaf** | Same algorithm, C lightweight | 7.3% | Returns NO_MATCH on uncertainty |
| **Panako** | Speed/pitch-shift robust (Gabor) | 3.3% | Worst — even with pitch-shift handling |

This **proves synopsis claim** — Western fingerprinting fails on Indian music. Per-song breakdown: songs with heavy ornamentation (Ehsaas, Ishqa_Ve, Makhna, Meherbaan, Rangi_Saari, Tu_Hi_Haqeeqat) score **0/12** across all 3 libraries. Songs that are more Western-structured (California_Love, Namo_Namo, Mi_Amor) pass through fine.

**Files:** `scripts/results/dejavu_dataset_results.json`, `olaf_dataset_results.json`, `panako_dataset_results.json`. Run `python3 scripts/compare.py --data-dir dataset` to see comparison.

---

## 3. Synopsis decode — the BIG insight

Synopsis says: **"Dataset: Mimbres audio fingerprinting dataset, Custom Dataset"**

🚨 **"Mimbres" is NOT a dataset name.** It's the GitHub username of the author of a key research paper:

- GitHub repo: `github.com/mimbres/neural-audio-fp`
- Paper: **"Neural Audio Fingerprint for High-Specific Audio Retrieval Based on Contrastive Learning"** (Chang et al., **ICASSP 2021**)

So the synopsis literally points at this paper as the base. Professor told you "use NeuralFP, modify it." Project is exactly aligned.

---

## 4. Candidate Models — what's out there

I ran research on top conferences (NeurIPS, ICASSP, ICMR, ISMIR, ICLR, 2021-2025). Here are the realistic candidates:

### Quick comparison

| Model | Year/Venue | GitHub stars | Difficulty | Code+weights ready? | Recommendation |
|---|---|---:|---|---|---|
| **NeuralFP (mimbres)** | ICASSP 2021 | 206 | ⭐⭐ Easy | ✅ Both ready | 🏆 **Pick this** |
| **GraFPrint** | ICASSP 2025 | 38 | ⭐⭐⭐⭐ Hard (GNN concepts) | ⚠️ Code yes, less mature | Skip — too complex for BTP |
| **ByteCover3** | ICASSP 2023 | ~50 | ⭐⭐⭐ Medium | ✅ But task is cover-song, not fingerprint | Skip — adaptation overhead |
| **MERT/CLAP foundation** | ICLR 2024 / various | 300+ | ⭐ Easiest | ✅ Just embeddings | Skip — novelty too low for paper |

### What each one is, in 1 line

- **NeuralFP:** Shazam ko "neural" bana diya — peaks instead of hand-coded ho gaya, contrastive learning se sikhaya gaya
- **GraFPrint:** Audio ko graph banata hai (peaks=nodes), Graph Neural Network use karta hai — too abstract for our level
- **ByteCover3:** Cover song detection — "Tum hi ho original vs Arijit live cover" find karta hai. Task slightly different
- **MERT/CLAP:** Pre-trained huge models, sirf embeddings nikaal ke similarity check karte hain — paper ke liye boring

### Why NeuralFP wins

1. ✅ Synopsis already points at it (Mimbres = author handle)
2. ✅ Code + pretrained weights + mini training dataset (11.2GB) all available, MIT license
3. ✅ Simple architecture — CNN encoder (something you already understand from CSE coursework)
4. ✅ Published at top venue (ICASSP 2021, A* tier)
5. ✅ Has been **never evaluated on Indian music** — first such evaluation = publishable contribution
6. ✅ Single GPU enough (RTX 3090 or college lab GPU). 5-10 hours training on mini-dataset
7. ✅ 3-4 clean modification points for Indian music

---

## 5. NeuralFP — Deep Dive

### What it does (in plain Hinglish)

1. Audio ko **spectrogram** mein convert karta hai (heat-map: x-axis = time, y-axis = frequency, color = volume)
2. **CNN** (Convolutional Neural Network) us spectrogram ko ek **128-dimension vector** mein squeeze karta hai
3. Yeh vector us song ka "DNA fingerprint" hai
4. Match karne ke liye: query song ka vector nikaalo, database mein closest vectors search karo (cosine similarity)
5. Sabse close vector wala song = answer

### Architecture diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT                                                       │
│ Audio clip (1 second, 16kHz mono)                           │
│         │                                                   │
│         ▼                                                   │
│ STFT spectrogram (256 freq bins × 32 time frames)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ CNN ENCODER (the "brain")                                    │
│  Conv2D → BatchNorm → ReLU                                   │
│  Conv2D → BatchNorm → ReLU                                   │
│  Conv2D → BatchNorm → ReLU                                   │
│  Conv2D → BatchNorm → ReLU                                   │
│  GlobalAvgPool → FC (128) → L2-normalize                     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                    128-D vector
                  (the "fingerprint")
                           │
                           ▼
                  store in FAISS index
```

### Training (the "contrastive" part)

Naya word: **contrastive learning**. Saral analogy:

> Imagine bachhe ko twin brothers identify karna sikhaana hai.
> - Twin A ke 2 photos → "yeh same hain" (positive pair) → their vectors should be CLOSE
> - Twin A vs Twin B → "yeh alag hain" (negative pair) → vectors should be FAR

Same logic applied to audio:
- Same song ka clean version + noisy version → "same" → close vectors
- Same song clean + DIFFERENT song → "alag" → far vectors

Loss function (NT-Xent loss): close pairs ke liye reward, far pairs ke liye penalty.

After training: model has learned "what makes 2 clips THE SAME song, regardless of noise/distortion."

### Why this fails on Indian music (problem statement)

NeuralFP was trained on FMA dataset (Western pop/rock). Its contrastive training augmentations include: noise, compression, EQ changes — but NOT pitch ornamentations.

Result: when an Indian singer adds Meend (smooth pitch slide), the spectrogram changes shape, and NeuralFP treats it as a "different" song. Hence low accuracy.

### Repo state (verified)

- URL: https://github.com/mimbres/neural-audio-fp
- License: MIT
- Stars: 206 (active)
- Pretrained weights: included
- Training script: included (TensorFlow v1, may need conversion to PyTorch)
- Mini training dataset: 11.2 GB FMA subset, ready to download
- Documented inference + training pipeline

---

## 6. The 4 Modifications — pick which to do

Each modification is a "what we changed and why" story for the panel. More mods = stronger paper, but more time.

### Modification 1: STFT → CQT (input feature change) ⭐ EASIEST

**Plain English:** Bas input ka data type change karna hai.

**STFT vs CQT analogy:**

> Imagine measuring music with two different rulers:
> - **STFT** = ruler in centimeters (linear). Pitch shift karne pe peaks scatter ho jaate hain.
> - **CQT** = ruler in piano keys (logarithmic, semitone-based). Pitch shift karne pe peaks neat-neatly translate hote hain.

**Why for Indian music:**
- Meend = pitch slide karta hai 2-3 semitones over 0.5-1 sec
- STFT mein: peaks "smear" ho jaate hain, fingerprint break
- CQT mein: peaks ek **straight diagonal line** banate hain — CNN easily seekh leta hai

**Code change:** Literally 1 line. `librosa.cqt()` instead of `librosa.stft()` in data loading.

**Time:** 1-2 days
**Risk:** Low
**Expected gain:** +10-15% accuracy on Indian music
**Panel explainability:** ⭐⭐⭐⭐⭐ "Indian music semitone-based hai, CQT semitone-based feature deta hai. STFT linear hai, mismatch."

---

### Modification 2: Synthetic Indian Augmentation ⭐⭐ MEDIUM

**Plain English:** Training data mein artificially Meend/Murki/Gamak generate karke add karo.

**Bachhe ko kutta sikhane wala analogy:**

> Bachhe ko sirf ek angle se ek Labrador dikhao → wo sirf Labrador pehchanega.
> Different angles, breeds, lighting dikhao → general "kutta" concept seekhta hai.

**Yahan kya:**
- Original song: `song.mp3`
- Synthetic Meend version: programmatically pitch-bend 2 semitones over 500ms at random points
- Synthetic Murki version: rapid pitch oscillation 10Hz at random spots
- Synthetic Gamak version: vibrato-like wobble

Training mein:
- "song.mp3" + "song_with_synthetic_meend.mp3" → contrastive_loss says "these are SAME"
- Model is forced to learn ornament-invariant features

**Implementation:**
```python
# pseudocode using librosa + pyrubberband
def add_meend(audio, semitones=2, duration_ms=500):
    # gradually pitch-shift over duration_ms then return to original
    return pyrubberband.pitch_shift(audio, sr, n_steps=semitones, ...)

def augment_for_indian(audio):
    aug = random.choice([meend, murki, gamak, vibrato])
    return aug(audio)
```

**Time:** 1-1.5 weeks (writing augmentation library + retraining)
**Risk:** Medium (training takes ~10 hrs)
**Expected gain:** +15-20% accuracy
**Panel explainability:** ⭐⭐⭐⭐ "We can't change Indian singers, so we make our model used to ornaments by simulating them in training data."

---

### Modification 3: Dilated Convolutions (architecture change) ⭐⭐ MEDIUM

**Plain English:** CNN ke kernel ko "spread out" karna, taaki wo wider time-context dekh sake.

**Magnifying glass vs binoculars analogy:**

> - **Normal CNN kernel** = magnifying glass: chhota area dikhta hai. 3-5 frames see hote hain.
> - **Dilated CNN kernel** = binoculars: same effort, much wider view. Same 5 elements, but frames 1, 4, 7, 10, 13 (skipping in between).

**Why for Indian music:**
- Meend takes 200-500ms = 5-10 spectrogram frames
- Normal kernel sees 3-5 frames → misses end-to-end Meend pattern
- Dilated kernel sees 10-15 frame span → captures whole Meend trajectory

**Code change:** PyTorch CNN definition mein `nn.Conv2d(..., dilation=2)` or `dilation=4`.

**Time:** 2-3 days code change + retrain (~10 hrs)
**Risk:** Medium (might give marginal improvement only)
**Expected gain:** +5-10% accuracy
**Panel explainability:** ⭐⭐⭐ "Same parameters, wider receptive field for long-range pitch contours."

---

### Modification 4: Auxiliary Ornament Loss ⭐⭐⭐⭐ HARD — SKIP

**Plain English:** Model ko 2 kaam ek saath sikhao — fingerprint generate karo + ornament classify karo.

**Why skip:**
- Need ornament-labeled dataset (ROD has only 4 hours, very small)
- Complex multi-task balancing
- Risk of harming fingerprinting performance for marginal gain
- Hard to explain to panel cleanly

**Verdict:** Don't do for BTP. Save for PhD if anyone continues.

---

### Recommended combo: **Mod 1 + Mod 2** (CQT + Augmentation)

| Combo | Time | Accuracy boost | Story strength |
|---|---|---:|---:|
| Mod 1 only | 1-2 days | +10-15% | ⭐⭐⭐ |
| **Mod 1 + 2** ⭐ | 2 weeks | **+25-35%** | ⭐⭐⭐⭐⭐ |
| Mod 1 + 2 + 3 | 3 weeks | +30-40% | ⭐⭐⭐⭐⭐ |
| All 4 | 5+ weeks | +30-45% (risky) | ⭐⭐⭐ |

---

## 7. Proposed Plan (Timeline)

Assumes Mod 1 + 2 (CQT + Indian augmentation), which is the recommended path.

### Week 1: Setup + Reproduce baseline
- [ ] Clone `github.com/mimbres/neural-audio-fp`
- [ ] Set up environment (PyTorch / TensorFlow whichever the repo uses)
- [ ] Download mini-dataset (FMA, 11.2 GB) + pretrained weights
- [ ] Run pretrained NeuralFP inference on `dataset/` (21 songs, 273 queries)
- [ ] Get baseline accuracy number — expected ~55-65% (vs Dejavu's 30.8%)
- [ ] Write up: "NeuralFP unmodified on Indian music = X%"

### Week 2: Reproduce training (proves you can train)
- [ ] Train NeuralFP from scratch on FMA mini-dataset
- [ ] Match (within 2%) the paper's reported FMA accuracy
- [ ] Verify training loop end-to-end works on your GPU

### Week 3: Modification 1 — CQT input
- [ ] Replace STFT with CQT in data loading pipeline
- [ ] Adjust input dimensions for CNN (CQT bins ≠ STFT bins)
- [ ] Retrain on FMA mini
- [ ] Evaluate on `dataset/` — expected +10-15% gain

### Week 4-5: Modification 2 — Indian augmentation
- [ ] Implement synthetic Meend, Murki, Gamak augmentations using `librosa` + `pyrubberband`
- [ ] Add to training augmentation pipeline
- [ ] Retrain (this time may take longer — augmentation slows things down)
- [ ] Evaluate on `dataset/` — expected +15-20% additional gain

### Week 6: Comprehensive evaluation
- [ ] Run all 5 systems on `dataset/`:
   1. Dejavu (DONE: 30.8%)
   2. Olaf (DONE: 7.3%)
   3. Panako (DONE: 3.3%)
   4. NeuralFP unmodified
   5. NeuralFP + CQT + Augmentation (your method)
- [ ] Optional: Also run on Saraga (Hindustani) test split for stronger paper
- [ ] Build comparison tables, plots (precision/recall curves, per-song breakdown)

### Week 7: Paper draft
- [ ] Title: "Audio Fingerprinting for Indian Music: Adapting Neural Methods for Ornamentation"
- [ ] Sections: Intro, Related Work, Method, Experiments, Results, Discussion, Conclusion
- [ ] Target venue: ICASSP 2026 (Sept deadline) / ISMIR 2026 (Apr) / NCMR (Indian conf)

### Week 8: Buffer + presentation prep
- [ ] Slides for BTP presentation
- [ ] Final report
- [ ] Demo if possible

---

## 8. Decisions to Lock

Before I write the formal plan, you need to answer:

### Decision A: Which base model?
- [ ] **NeuralFP (recommended)** — synopsis aligned, lowest risk
- [ ] GraFPrint (cutting-edge, complex)
- [ ] ByteCover3 (cover-song, needs adaptation)
- [ ] MERT/CLAP (low novelty)

### Decision B: How many modifications?
- [ ] 1 mod (CQT only) — safe, modest paper
- [ ] **2 mods (CQT + Augmentation)** — recommended sweet spot
- [ ] 3 mods (+ Dilated CNN) — strongest paper, more time
- [ ] 4 mods (+ Aux Loss) — risky, skip

### Decision C: Dataset for paper?
- [ ] Just our existing `dataset/` (21 songs)
- [ ] **Add Saraga (Hindustani classical, 60 hrs) for stronger paper** — recommended
- [ ] Add ROD (Raga Ornamentation Detection, 4 hrs labeled) for ornament-specific eval

### Decision D: Conference target?
- [ ] **ICASSP 2026** (deadline ~Sept 2026, A* venue, signal processing focus)
- [ ] ISMIR 2026 (deadline ~Apr 2026, A venue, music IR focus — perfect topic match)
- [ ] NCMR (Indian conf, easier acceptance, less prestige)
- [ ] ICMR 2026 (multimedia retrieval)

### Decision E: Custom Indian dataset expansion?
Synopsis mentions "Custom Dataset" — abhi 21 songs hain. Paper ke liye need:
- [ ] Stay with 21 + use Saraga for additional eval
- [ ] Expand custom dataset to 50-100 songs (1-2 days extra work)
- [ ] Crowdsource via friends recording covers (slow, 1-2 weeks)

---

## 📚 Key references / links to bookmark

### Code repos (the most important)
- **NeuralFP:** https://github.com/mimbres/neural-audio-fp
- **CQTNet (cover song with CQT):** https://github.com/yzspku/CQTNet
- **Panako (already in our repo):** https://github.com/JorenSix/Panako
- **Olaf (already in our repo):** https://github.com/JorenSix/Olaf

### Papers (priority read order)
1. **Chang et al. 2021** — NeuralFP base paper (must read): https://arxiv.org/abs/2010.11910
2. **Wang 2003** — Original Shazam paper (foundational): "An Industrial-Strength Audio Search Algorithm"
3. **2025 ROD paper** — ornament detection in Indian music: https://arxiv.org/html/2505.04419v1
4. **Indian Song Retrieval (Springer 2015)** — proves SHAZAM fails on Indian songs: https://link.springer.com/chapter/10.1007/978-81-322-2464-8_6

### Datasets
- **Saraga (Hindustani):** https://mtg.github.io/saraga/
- **Saraga (Carnatic):** same link, separate split
- **CompMusic Raga datasets:** https://compmusic.upf.edu/datasets
- **ROD (ornament-labeled):** linked from arxiv.org/abs/2505.04419
- **FMA (training data, included in NeuralFP repo):** https://github.com/mdeff/fma

---

## 🔄 Plan status

This file is a STUDY DOC. Not the final plan yet. After you read all 8 chapters and lock decisions A-E, I'll rewrite this as the formal implementation plan.

**Next step:** Read at your pace. Bata kya stuck ho raha hai — har section mein deep dive ho sakta hai.
