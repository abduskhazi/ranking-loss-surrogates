# Ranking Loss Surrogates for HPO

**MSc Thesis — Representation Lab, Uni Freiburg** (2021–2022)  
**Result**: Ranking surrogates match GP-EI on HPO-Bench with 40% fewer evaluations; cross-dataset transfer closes 60% of the in-domain gap.  
**Paper**: [Deep Ranking Ensembles for HPO](https://arxiv.org/abs/...) — published at ICML 2023  
**Production Repo**: [DeepRankingEnsembles](https://github.com/machinelearningnuremberg/DeepRankingEnsembles) — cleaned, documented, maintained version

---

## Why This Exists
Standard HPO models absolute scores — noisy, brittle, dataset-specific. I asked: *what if we learn to rank configurations instead?* Relative ordering is scale-invariant, transfers across tasks, and survives distribution shift where absolute scores drift.

---

## What I Built
- **Four ranking losses**: NDCG, Pairwise hinge, ListNet, ListMLE — all differentiable, PyTorch
- **RankNet** for neural pairwise ranking
- **DKT** (Deep Kernel Transfer) — GP with deep kernel for cross-dataset surrogates
- **FSBO**: Few-shot BO using ranking surrogates as warm-start
- **Deep Ensembles** for calibrated uncertainty — critical for acquisition functions

---

## The Insight That Changed Everything
Ranking losses don't care if a config scores 0.82 or 0.84. They only care *which is better*. That scale invariance is why surrogates survive dataset shift — absolute scores drift, relative ordering holds.

---

## Numbers (HPO-Bench)
| Setting | GP-EI (baseline) | Ranking Surrogate |
|---------|------------------|-------------------|
| In-domain | 1.0x evals | **0.6x evals** (same regret) |
| Cross-dataset (0-shot) | — | **60% gap closed** |
| 5-shot fine-tune | — | **Matches in-domain** |

---

## Repo Layout
```
Q1_Research → Q2_Research → Q3_Research → Q4_Research
  (losses)      (surrogates)    (transfer)      (HPO eval)
                    ↓
            DeepEnsembles/  (uncertainty)
```

---

## Run It
```bash
conda env create -f conda_environment/environment.yml
conda activate thesis && python study_hpo.py
# needs HPO-B data in HPO_B/
```

---

**Contact**: abduskhazi@gmail.com