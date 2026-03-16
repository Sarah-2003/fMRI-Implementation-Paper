# BrainAge-Dx

**Regional Functional Connectivity Age-Gap Profiles as Transdiagnostic Fingerprints for Neuropsychiatric Diagnosis**

## One-Line Summary

> Train a brain age predictor on healthy subjects, then show that the **pattern** of which brain networks age faster/slower uniquely identifies ASD, schizophrenia, bipolar disorder, and ADHD — using nothing but ridge regression.

## Why This Matters

- **Brain age gap** is established — but gives ONE number (crude)
- **Different disorders affect different networks** — the regional *pattern* is the real biomarker
- **2025 evidence** shows simpler models are MORE sensitive to clinical deviations than deep learning
- **Result:** A 7-dimensional "brain aging fingerprint" per patient — visualized as an intuitive radar chart

## The Killer Figure

```
        DMN
       / | \
     FP  |  VIS
    /    |    \
   DA ---+--- SM
    \    |    /
     VA  |  LIM
       \ | /
      [center]

  ASD: DMN delayed, others normal
  SZ:  Salience + Executive accelerated
  BD:  Limbic + DMN accelerated
  ADHD: Frontoparietal delayed
```

Each disorder has a distinct radar chart shape = diagnostic fingerprint.

## Datasets (~2,590 subjects)

| Dataset | N | Disorder | Role |
|---------|---|----------|------|
| **HCP-YA** | 1,206 | Healthy | Train brain age model |
| **ABIDE I** | 1,112 | ASD (539) + HC (573) | Test — developmental |
| **UCLA CNP** | 272 | SZ (58) + BD (49) + ADHD (45) + HC (138) | Test — multi-disorder |

## Method

```
fMRI → FC Matrix → Yeo 7-Network Features (28 dim) → Ridge Regression → Brain Age Gap → Regional Profile → Diagnosis
```

1. Extract network-level FC features (7 within-network + 21 between-network = 28)
2. Train ridge regression on HCP healthy subjects (age prediction)
3. Compute global + 7 regional brain age gaps for patients
4. Use regional gap profiles for cross-disorder classification
5. Correlate with symptom severity

**No GPU needed. Runs on a laptop in minutes.**

## Repository Structure

```
.
├── README.md
├── IMPLEMENTATION_PLAN.md         # Detailed step-by-step plan
├── research/
│   ├── ideas/                     # All 3 ideas considered
│   ├── datasets/                  # Dataset access guides
│   └── literature/                # Key references
├── code/
│   ├── preprocessing/             # FC extraction, feature computation
│   ├── models/                    # Ridge regression, classification
│   ├── evaluation/                # CV, statistics, metrics
│   └── utils/                     # Data loading, visualization
├── paper/
│   ├── figures/                   # Radar charts, brain maps, confusion matrices
│   └── tables/                    # Results tables
└── results/                       # Experiment outputs
```

## Requirements

```
python >= 3.9
nilearn >= 0.10
scikit-learn >= 1.3
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.11
statsmodels >= 0.14
```

## Timeline

- **Week 1:** Data download + FC feature extraction
- **Week 2:** Train models + run experiments + statistics
- **Week 3:** Figures + paper writing

## Status

- [x] Literature review completed
- [x] Research gap analysis
- [x] 3 novel ideas proposed and evaluated
- [x] **Idea selected: BrainAge-Dx**
- [x] Detailed implementation plan
- [x] Dataset identification and access guide
- [ ] Data download and preprocessing
- [ ] Model training and experiments
- [ ] Figures and visualizations
- [ ] Paper draft

## Target Journals

- **NeuroImage** (IF 4.7)
- **Human Brain Mapping** (IF 3.5)
- **NeuroImage: Clinical** (IF 3.4)
- Stretch: **Nature Communications** (IF 14.7) if results are strong
