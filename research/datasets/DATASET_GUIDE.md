# Dataset Guide for fMRI Implementation Paper

## Overview

All three ideas use combinations of these **5 core datasets**. All are publicly available and well-recognized.

---

## Primary Datasets

### 1. HCP-YA (Human Connectome Project — Young Adult)
**Role in all 3 ideas:** Healthy control baseline / normative training

| Field | Detail |
|-------|--------|
| **URL** | https://www.humanconnectome.org/study/hcp-young-adult |
| **Subjects** | 1,206 (1,113 with imaging), ages 22-35 |
| **Disorders** | Healthy controls only |
| **fMRI Type** | Resting-state (4 runs, ~1hr) + task-based (7 tasks) |
| **Preprocessing** | HCP minimal preprocessing pipeline |
| **Format** | NIfTI, CIFTI (BIDS-compatible) |
| **Access** | Free — register at ConnectomeDB, accept Data Use Terms |
| **Citations** | 5,000+ (Van Essen et al., NeuroImage 2013) |
| **Python Access** | `nilearn.datasets.fetch_development_fmri` (partial) |
| **Download Size** | ~2TB full; ~50GB for preprocessed FC matrices |
| **Why This Dataset** | Gold standard for healthy brain connectivity. Unmatched data quality. Essential for normative modeling and pretraining. |

**Quick Start:**
```python
# For preprocessed connectomes, download from ConnectomeDB
# Or use nilearn for parcellated time series
from nilearn import datasets
atlas = datasets.fetch_atlas_aal()  # 116 ROIs
```

---

### 2. ABIDE I+II (Autism Brain Imaging Data Exchange)
**Role:** Primary test dataset for ASD

| Field | Detail |
|-------|--------|
| **URL** | http://fcon_1000.projects.nitrc.org/indi/abide/ |
| **Preprocessed** | http://preprocessed-connectomes-project.org/abide/ |
| **Subjects** | ABIDE I: 1,112 (539 ASD, 573 controls), ABIDE II: 1,044 — **Total: 2,156** |
| **Sites** | 17+ international sites |
| **Disorders** | Autism Spectrum Disorder (ASD) |
| **fMRI Type** | Resting-state |
| **Preprocessing** | 4 pipelines available: C-PAC, CCS, DPARSF, NIAK |
| **Derivatives** | ALFF, fALFF, ReHo, VMHC, degree centrality, ROI time series (AAL, CC200, CC400, Dosenbach160, Harvard-Oxford) |
| **Access** | Free — NITRC account for raw; preprocessed on Amazon S3 (no restrictions) |
| **Citations** | 2,400+ (Di Martino et al., Molecular Psychiatry 2014) |
| **Python Access** | `nilearn.datasets.fetch_abide_pcp()` — **one line of code!** |

**Quick Start:**
```python
from nilearn import datasets
# Download preprocessed ABIDE data with one command
abide = datasets.fetch_abide_pcp(
    derivatives=['rois_aal'],  # AAL atlas ROI time series
    pipeline='cpac',
    quality_checked=True
)
# abide.rois_aal → list of ROI time series files
# abide.phenotypic → DataFrame with diagnosis, age, sex, site
```

**Why This Dataset:**
- THE benchmark for fMRI + deep learning
- Preprocessed data available via Python API — zero preprocessing effort
- Multi-site design tests generalization
- Large enough for deep learning

---

### 3. REST-meta-MDD (DIRECT Consortium)
**Role:** Primary test dataset for Major Depressive Disorder

| Field | Detail |
|-------|--------|
| **URL** | http://rfmri.org/REST-meta-MDD |
| **Subjects** | 2,428 (1,300 MDD, 1,128 controls) from 25 sites |
| **Disorders** | Major Depressive Disorder (MDD) |
| **fMRI Type** | Resting-state |
| **Preprocessing** | DPARSF standardized protocol |
| **Phenotypic Data** | Age, sex, episode status, medication, illness duration, HAMD-17 scores |
| **Access** | Free — de-identified derivatives on Science Data Bank (SciDB) |
| **Citations** | High (Yan et al., 2019) |

**Why This Dataset:**
- Largest public MDD-specific fMRI dataset worldwide
- Depression is the #1 disability cause globally — massive clinical relevance
- 25 sites → multi-site generalization
- Rich clinical phenotyping (severity scores, medication status)

**Access Instructions:**
1. Visit http://rfmri.org/REST-meta-MDD
2. Download the de-identified R-fMRI indices (free since Jan 2020)
3. Data includes preprocessed connectivity features

---

### 4. SRPBS Multi-Disorder Dataset (Japan)
**Role:** Multi-disorder classification (7 conditions)

| Field | Detail |
|-------|--------|
| **URL** | https://bicr-resource.atr.jp/srpbsopen/ |
| **Subjects** | 2,414 (993 patients + 1,421 controls) from 14 sites |
| **Disorders** | ASD, MDD, Bipolar, Schizophrenia, OCD, Chronic Pain, Stroke |
| **fMRI Type** | Resting-state + structural MRI |
| **Special Feature** | "Traveling subject" data: 9 subjects scanned at ALL 14 sites (12 scanners) |
| **Access** | Free — Synapse (ID: syn22317076) |
| **Citations** | Tanaka et al., Scientific Data 2021 |

**Why This Dataset:**
- **ONLY** public dataset covering 7 neuropsychiatric disorders
- Traveling subject design is invaluable for site-effect harmonization research
- Perfect for cross-disorder / transdiagnostic classification
- Unified imaging protocol across sites

**Access Instructions:**
1. Register at Synapse (https://www.synapse.org/)
2. Request access to syn22317076
3. Download connectivity matrices and phenotypic data

---

### 5. ADHD-200
**Role:** External validation for ADHD

| Field | Detail |
|-------|--------|
| **URL** | http://fcon_1000.projects.nitrc.org/indi/adhd200/ |
| **Preprocessed** | http://preprocessed-connectomes-project.org/adhd200/ |
| **Subjects** | 973 (491 controls, 285 ADHD with subtypes), ages 7-21 |
| **Subtypes** | ADHD-Combined, ADHD-Inattentive, ADHD-Hyperactive |
| **Sites** | 8 sites |
| **fMRI Type** | Resting-state |
| **Preprocessing** | 3 pipelines: Athena/AFNI+FSL, Burner/SPM8, NIAK |
| **Access** | Free — NITRC |
| **Python Access** | `nilearn.datasets.fetch_adhd()` (7 subjects for quick testing) |
| **Citations** | Highly cited (Milham et al., 2012) |

**Why This Dataset:**
- Established ADHD benchmark with competition results for comparison
- Multiple preprocessing pipelines available
- Pediatric population (ages 7-21) — different from adult datasets

---

### 6. UCLA CNP (Consortium for Neuropsychiatric Phenomics)
**Role:** Multi-disorder validation (SZ, BD, ADHD)

| Field | Detail |
|-------|--------|
| **URL** | https://openneuro.org/datasets/ds000030 |
| **Subjects** | 272 (138 controls, 58 SZ, 49 BD, 45 ADHD) |
| **Disorders** | Schizophrenia, Bipolar Disorder, ADHD |
| **fMRI Type** | Resting-state + 5 task-based paradigms |
| **Format** | BIDS on OpenNeuro |
| **Access** | Free, immediate — no application needed |
| **Citations** | Poldrack et al., Scientific Data 2016 |

**Why This Dataset:**
- 3 disorders in one dataset — perfect for multi-class experiments
- BIDS format → compatible with fMRIPrep and all standard pipelines
- OpenNeuro = instant download, no gatekeeping
- Task-based AND resting-state fMRI available

---

### 7. COBRE (Center for Biomedical Research Excellence)
**Role:** Schizophrenia-specific validation

| Field | Detail |
|-------|--------|
| **URL** | http://fcon_1000.projects.nitrc.org/indi/retro/cobre.html |
| **Subjects** | 146 (72 schizophrenia, 74 controls) |
| **fMRI Type** | Resting-state (150 volumes, TR=2s, 5 min) |
| **Access** | Free — NITRC, SchizConnect, figshare |
| **Preprocessing** | NIAK pipeline available on figshare |

---

## Dataset Mapping Per Idea

### Idea 1: BrainDevScore
| Dataset | Role | N |
|---------|------|---|
| HCP-YA | Train normative model | 1,206 |
| ABIDE I+II | Test (ASD) | 2,156 |
| REST-meta-MDD | Test (MDD) | 2,428 |
| COBRE | Test (SZ) | 146 |
| **Total** | | **~5,936** |

### Idea 2: FC-ContrastNet
| Dataset | Role | N |
|---------|------|---|
| HCP-YA | Pretraining (unlabeled) | 1,206 |
| SRPBS | Fine-tuning + multi-class | 2,414 |
| ADHD-200 | External validation | 973 |
| **Total** | | **~4,593** |

### Idea 3: BrainAge-Dx
| Dataset | Role | N |
|---------|------|---|
| HCP-YA | Train brain age model | 1,206 |
| ABIDE I | Test (ASD) | 1,112 |
| UCLA CNP | Test (SZ, BD, ADHD) | 272 |
| **Total** | | **~2,590** |

---

## Download Priority Order

If choosing **Idea 1** or **Idea 3** (recommended):
1. **ABIDE preprocessed** — `nilearn.datasets.fetch_abide_pcp()` (instant, ~2GB)
2. **HCP-YA** — Register at ConnectomeDB (~1 day approval, ~50GB preprocessed)
3. **COBRE** — figshare preprocessed (~500MB)
4. **REST-meta-MDD** — SciDB download (~5GB)

If choosing **Idea 2**:
1. **SRPBS** — Synapse registration + download (~1-2 days)
2. **HCP-YA** — ConnectomeDB
3. **ADHD-200** — preprocessed-connectomes-project.org

---

## Preprocessing Notes

For all ideas, we need **functional connectivity matrices**:

```python
# Standard FC extraction pipeline (works for any dataset)
from nilearn import connectome, datasets
from nilearn.maskers import NiftiLabelsMasker
import numpy as np

# 1. Choose atlas
atlas = datasets.fetch_atlas_aal()  # 116 ROIs
# or: atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200)

# 2. Extract ROI time series
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0  # adjust per dataset
)
time_series = masker.fit_transform(fmri_file)

# 3. Compute FC matrix
correlation_measure = connectome.ConnectivityMeasure(kind='correlation')
fc_matrix = correlation_measure.fit_transform([time_series])[0]
# fc_matrix shape: (116, 116) — symmetric correlation matrix
```

**Note:** For ABIDE and ADHD-200, preprocessed ROI time series are already available — skip steps 1-2.
