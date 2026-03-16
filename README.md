# fMRI + ML/AI Implementation Papers

Repository for fMRI-based neuropsychiatric disorder diagnosis implementation papers.

## Papers

### Paper 1: BrainAge-Dx
**Regional Functional Connectivity Age-Gap Profiles as Transdiagnostic Fingerprints for Neuropsychiatric Diagnosis**

> Train a brain age predictor on healthy subjects, then show that the **pattern** of which brain networks age faster/slower uniquely identifies ASD, schizophrenia, bipolar disorder, and ADHD — using nothing but ridge regression.

- **Datasets:** HCP (1,206) + ABIDE (1,112) + UCLA CNP (272) = ~2,590 subjects
- **Method:** Ridge regression on Yeo 7-network FC features → regional brain age gaps → diagnostic fingerprints
- **Timeline:** 2-3 weeks | No GPU needed
- **Status:** Code pipeline complete, 13 supporting papers collected
- **Folder:** [`paper-1-brainage-dx/`](paper-1-brainage-dx/)

### Paper 2: TBD
*Slot reserved for second implementation paper.*

---

## Repository Structure

```
.
├── README.md
├── .gitignore
├── paper-1-brainage-dx/           # Paper 1: BrainAge-Dx
│   ├── IMPLEMENTATION_PLAN.md
│   ├── code/
│   │   ├── config.py              # All parameters in one place
│   │   ├── run_pipeline.py        # Main entry point
│   │   ├── requirements.txt
│   │   ├── preprocessing/
│   │   │   ├── fetch_data.py      # Dataset download
│   │   │   └── extract_features.py # FC → 28 network features
│   │   ├── models/
│   │   │   └── brain_age.py       # Global + 7 regional Ridge models
│   │   ├── evaluation/
│   │   │   └── statistics.py      # Group comparisons, classification, ANOVA
│   │   └── utils/
│   │       └── visualizations.py  # Radar charts, brain maps, figures
│   ├── supporting-papers/         # 13 PDFs + 2 manual download guide
│   ├── research/                  # Ideas, dataset guides, key references
│   ├── paper/                     # Manuscript figures and tables
│   └── results/                   # Experiment outputs
└── paper-2-tbd/                   # Reserved for Paper 2
```
