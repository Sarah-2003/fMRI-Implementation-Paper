# fMRI + ML/AI Implementation Paper

Implementation paper exploring novel machine learning approaches for fMRI-based neuropsychiatric disorder diagnosis.

## Three Novel Ideas Under Consideration

| # | Idea | Key Innovation | Datasets | Timeline | GPU? |
|---|------|---------------|----------|----------|------|
| 1 | **BrainDevScore** | Normative FC deviation maps + conformal prediction | HCP + ABIDE + REST-MDD + COBRE | 3-4 weeks | No |
| 2 | **FC-ContrastNet** | Self-supervised contrastive learning on brain graphs | HCP + SRPBS + ADHD-200 | 4-5 weeks | Yes (T4) |
| 3 | **BrainAge-Dx** | Regional brain functional age gap profiles | HCP + ABIDE + UCLA CNP | 2-3 weeks | No |

See [`research/ideas/THREE_NOVEL_IDEAS.md`](research/ideas/THREE_NOVEL_IDEAS.md) for full details.

## Repository Structure

```
.
├── README.md
├── research/
│   ├── ideas/                    # Novel idea proposals and analysis
│   │   └── THREE_NOVEL_IDEAS.md  # Detailed comparison of 3 ideas
│   ├── datasets/                 # Dataset documentation and access guides
│   │   └── DATASET_GUIDE.md      # All datasets with access instructions
│   └── literature/               # Key papers and references
│       └── KEY_PAPERS.md         # 26 key papers organized by topic
├── code/
│   ├── preprocessing/            # FC matrix extraction, atlas handling
│   ├── models/                   # Model architectures
│   ├── evaluation/               # Metrics, cross-validation, conformal prediction
│   └── utils/                    # Data loading, visualization helpers
├── paper/
│   ├── figures/                  # Publication-quality figures
│   └── tables/                   # Results tables
└── results/                      # Experiment outputs, logs, saved models
```

## Datasets Used

All publicly available:
- **HCP-YA** (1,206 healthy) — normative baseline
- **ABIDE I+II** (2,156 ASD) — autism classification
- **REST-meta-MDD** (2,428 MDD) — depression classification
- **SRPBS** (2,414, 7 disorders) — multi-disorder classification
- **ADHD-200** (973 ADHD) — ADHD classification
- **UCLA CNP** (272, SZ/BD/ADHD) — multi-disorder validation
- **COBRE** (146 SZ) — schizophrenia validation

## Requirements

```
python >= 3.9
nilearn >= 0.10
scikit-learn >= 1.3
pytorch >= 2.0 (for Idea 2 only)
torch-geometric >= 2.4 (for Idea 2 only)
matplotlib >= 3.7
seaborn >= 0.12
mapie >= 0.8 (conformal prediction)
```

## Status

- [x] Literature review completed
- [x] Research gap analysis
- [x] 3 novel ideas proposed
- [x] Dataset identification and access guide
- [ ] Idea selection (pending)
- [ ] Implementation
- [ ] Experiments
- [ ] Paper writing
