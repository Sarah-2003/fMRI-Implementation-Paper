# BrainAge-Dx: Regional Brain Functional Age Gap as a Universal Diagnostic Biomarker

## Implementation Plan

---

## Paper Title (Working)
**"BrainAge-Dx: Regional Functional Connectivity Age-Gap Profiles as Transdiagnostic Fingerprints for Neuropsychiatric Diagnosis"**

## One-Line Pitch
> Train a brain age predictor on healthy subjects, then show that the **pattern** of which brain networks age faster or slower uniquely identifies ASD, schizophrenia, bipolar disorder, and ADHD — using nothing but ridge regression.

---

## The Story Arc (Why Reviewers Will Love This)

1. **Opening hook:** Brain age prediction is well-established, but gives ONE number. That's like measuring "overall body health" with a single metric — useful but crude.
2. **The insight:** Different disorders affect different brain networks differently. ASD may show *delayed* maturation in social networks. Schizophrenia may show *accelerated* aging in executive networks. The **pattern** is the biomarker, not the number.
3. **The twist:** A 2025 PLOS Biology paper proved that simpler brain age models are MORE sensitive to clinical deviations than complex deep learning. We leverage this.
4. **The result:** A 7-dimensional "brain aging fingerprint" that:
   - Differentiates disorders from each other (not just patient vs healthy)
   - Correlates with symptom severity
   - Is visualizable as an intuitive radar chart
   - Needs zero GPU and runs in minutes

---

## Datasets

| Dataset | N | Role | Access Method |
|---------|---|------|---------------|
| **HCP-YA** | 1,206 | Train age predictor on healthy brains | ConnectomeDB registration |
| **ABIDE I** | 1,112 (539 ASD, 573 HC) | Test — developmental disorder | `nilearn.datasets.fetch_abide_pcp()` |
| **UCLA CNP** | 272 (138 HC, 58 SZ, 49 BD, 45 ADHD) | Test — multi-disorder psychiatric | OpenNeuro ds000030 |

**Total: ~2,590 subjects across 4 disorders + healthy controls**

---

## Method — Step by Step

### Step 1: Feature Extraction — Network-Level Functional Connectivity

**Atlas:** Yeo 7-Network parcellation (mapped to ROIs via Schaefer 200-ROI atlas)

The 7 canonical brain networks:
1. **Visual** (VIS) — primary and secondary visual cortex
2. **Somatomotor** (SM) — motor and somatosensory cortex
3. **Dorsal Attention** (DA) — top-down attention, intraparietal sulcus
4. **Ventral Attention / Salience** (VA) — stimulus-driven attention, anterior insula, ACC
5. **Limbic** (LIM) — orbitofrontal, temporal pole — emotion
6. **Frontoparietal / Executive** (FP) — lateral PFC, posterior parietal — cognitive control
7. **Default Mode** (DMN) — mPFC, PCC, angular gyrus — self-referential, mind-wandering

**Features per subject (28 total):**
- 7 **within-network** FC values: mean correlation within each network
- 21 **between-network** FC values: mean correlation between each pair of networks (7 choose 2 = 21)
- Total: **28 features** per subject

```python
# Pseudocode
from nilearn import datasets, connectome
from nilearn.maskers import NiftiLabelsMasker
import numpy as np

# Use Schaefer 200 parcellation with Yeo 7-network labels
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)

# For each subject:
#   1. Extract 200 ROI time series
#   2. Compute 200x200 FC matrix
#   3. Average FC within each of 7 networks → 7 within-network features
#   4. Average FC between each network pair → 21 between-network features
#   5. Result: 28-dimensional feature vector
```

### Step 2: Train Brain Age Model on HCP Healthy Subjects

**Model:** Ridge Regression (deliberately simple — validated by Dafflon et al. 2025)

```
Input: 28 FC features + sex (binary)
Output: Predicted age (continuous)
Training: HCP-YA subjects (N=1,206, ages 22-35)
Validation: 5-fold CV within HCP to get prediction error estimates
```

**Why Ridge and not Deep Learning?**
- Dafflon et al. (2025) proved: simpler models have HIGHER sensitivity to clinical deviations
- Ridge regression is interpretable — you can see which features contribute
- Robust to small samples and overfitting
- Reproducible — no random seed sensitivity

**Also train 7 regional models:**
```
Global model:  All 28 features → age
DMN model:     4 features (within-DMN + DMN-to-other) → age
FP model:      4 features (within-FP + FP-to-other) → age
VA model:      4 features (within-VA + VA-to-other) → age
... (one per network)
```

Each regional model captures how well that specific network predicts age → regional brain age.

### Step 3: Compute Brain Age Gap for Patients

For each patient in ABIDE + UCLA CNP:

```
Global Brain Age Gap = Predicted_Age(global model) - Actual_Age
Regional Brain Age Gap[network] = Predicted_Age(network model) - Actual_Age
```

**Result per subject:** 1 global + 7 regional brain age gaps = **8-dimensional brain aging profile**

**Key correction:** Apply bias correction (de Lange & Cole, 2020) to remove the known age-dependent bias in brain age prediction:
```python
# Bias correction: regress out the effect of age on the gap
from sklearn.linear_model import LinearRegression
bias_model = LinearRegression().fit(age_train.reshape(-1,1), gap_train)
corrected_gap = gap - bias_model.predict(age.reshape(-1,1))
```

### Step 4: Statistical Analysis

**4a. Group-level comparisons:**
- For each disorder (ASD, SZ, BD, ADHD) vs. healthy controls:
  - t-tests on global brain age gap
  - t-tests on each of 7 regional gaps
  - FDR correction for multiple comparisons
  - Effect sizes (Cohen's d)

**4b. Disorder differentiation (the key novelty):**
- ANOVA across disorders on regional gaps
- Post-hoc pairwise comparisons: does ASD have a different aging pattern than SZ?
- Multinomial classification using the 7 regional gaps as features
  - Leave-one-out or 10-fold CV
  - Compare: global gap alone vs. regional gap profile

**4c. Severity correlation:**
- ABIDE: correlate regional gaps with ADOS scores (autism severity)
- UCLA CNP: correlate with symptom scales where available

**4d. The killer experiment — can regional patterns DIFFERENTIATE disorders?**
```
Experiment: 4-class classification (ASD vs SZ vs BD vs ADHD)
Features: 7 regional brain age gaps
Model: Logistic Regression (multi-class)
Evaluation: Stratified 10-fold CV, report accuracy, F1, confusion matrix
Baseline: Same classification using only global brain age gap (1 feature)
Hypothesis: Regional profile (7 features) >> global gap (1 feature)
```

### Step 5: Visualizations (The Paper-Selling Figures)

**Figure 1 — The Pipeline Overview**
```
[fMRI] → [FC Matrix] → [7 Network Features] → [Age Prediction] → [Gap Profiles] → [Diagnosis]
```

**Figure 2 — THE RADAR CHART (the figure everyone will remember)**
```
Spider/radar chart with 7 axes (one per network)
4 colored polygons overlaid: ASD (blue), SZ (red), BD (orange), ADHD (green)
Each shows the mean regional brain age gap pattern
→ ASD: DMN delayed, others normal
→ SZ: Salience + Executive accelerated
→ BD: Limbic + DMN accelerated
→ ADHD: Frontoparietal delayed, Attention disrupted
```

**Figure 3 — Brain Surface Maps**
- 7 brain renderings showing which regions are "older" or "younger" per disorder
- Using nilearn's `plot_surf_stat_map` with Yeo network parcellation

**Figure 4 — Individual-Level Scatter**
- Each patient plotted in regional-gap space
- Color by disorder → shows clustering and overlap
- t-SNE or PCA of the 7D regional gap profiles

**Figure 5 — Confusion Matrix**
- 4-class classification results
- Side-by-side: global gap vs. regional profile

**Figure 6 — Severity Correlation**
- Scatter plots: regional gap (y) vs symptom severity (x) for each disorder

### Step 6: Paper Structure

```
Abstract (250 words)
1. Introduction
   1.1 Brain age as a biomarker — established but limited (one number)
   1.2 Regional heterogeneity — different disorders affect different networks
   1.3 The simplicity argument — Dafflon 2025 (simpler = more sensitive)
   1.4 Our contribution: regional FC-based brain age profiles for transdiagnostic diagnosis

2. Methods
   2.1 Datasets (HCP, ABIDE, UCLA CNP)
   2.2 Preprocessing and FC extraction
   2.3 Network-level feature computation (Yeo 7)
   2.4 Brain age model (Ridge Regression)
   2.5 Regional brain age gap computation
   2.6 Statistical analysis
   2.7 Classification experiments

3. Results
   3.1 Brain age prediction accuracy (HCP validation)
   3.2 Global brain age gap across disorders
   3.3 Regional brain age gap profiles (THE KEY RESULT)
   3.4 Disorder differentiation using regional profiles
   3.5 Severity correlations
   3.6 Individual-level visualization

4. Discussion
   4.1 Regional aging fingerprints — what the patterns mean neuroscientifically
   4.2 Clinical utility — from single number to personalized profile
   4.3 Simplicity as strength
   4.4 Limitations (age range, cross-sectional, etc.)
   4.5 Future directions (longitudinal, treatment monitoring, more disorders)

5. Conclusion
```

---

## Timeline

### Week 1: Data + Features
| Day | Task |
|-----|------|
| 1-2 | Download ABIDE via nilearn, UCLA CNP from OpenNeuro, register for HCP |
| 3 | Extract ROI time series (Schaefer 200, Yeo 7) for ABIDE |
| 4 | Extract ROI time series for UCLA CNP |
| 5 | Compute FC matrices → 28 network-level features for all subjects |

### Week 2: Models + Experiments
| Day | Task |
|-----|------|
| 1 | Train global + 7 regional Ridge Regression models on HCP |
| 2 | Compute brain age gaps for ABIDE + UCLA CNP |
| 3 | Statistical analysis: group comparisons, effect sizes |
| 4 | Classification experiments: global vs. regional profiles |
| 5 | Severity correlations, ablation studies |

### Week 3: Figures + Paper
| Day | Task |
|-----|------|
| 1 | Generate radar charts, brain surface maps |
| 2 | Generate scatter plots, confusion matrices |
| 3-5 | Write paper (Methods → Results → Introduction → Discussion) |

---

## Tools & Dependencies

```
# Core
python >= 3.9
nilearn >= 0.10.4      # FC extraction, brain plots, ABIDE download
scikit-learn >= 1.3     # Ridge regression, classification, CV
numpy >= 1.24
pandas >= 2.0

# Visualization
matplotlib >= 3.7
seaborn >= 0.12

# Statistics
scipy >= 1.11           # t-tests, ANOVA
statsmodels >= 0.14     # FDR correction, regression diagnostics

# Optional
plotly >= 5.15          # Interactive radar charts
nilearn                 # Brain surface plotting

# NO GPU REQUIRED
# Google Colab free tier or any laptop with 8GB RAM sufficient
```

---

## Expected Results (Hypotheses)

Based on existing literature, we predict:

| Disorder | Expected Regional Pattern | Rationale |
|----------|--------------------------|-----------|
| **ASD** | DMN delayed (negative gap), Salience disrupted | DMN is the "social brain" — underconnected in autism |
| **Schizophrenia** | Executive + Salience accelerated (positive gap) | Known accelerated brain aging in SZ (Koutsouleris et al.) |
| **Bipolar** | Limbic accelerated, DMN disrupted | Emotion regulation networks affected |
| **ADHD** | Frontoparietal delayed, Attention disrupted | Executive control networks immature |

If these patterns are DISTINCT, we can differentiate disorders — that's the paper's main claim.

---

## Novelty Checklist

- [x] Regional (not just global) brain age on functional connectivity
- [x] Cross-disorder comparison (4 disorders) in a single framework
- [x] Ridge regression validated as MORE sensitive than deep learning (Dafflon 2025)
- [x] 7-network aging fingerprint as a diagnostic tool
- [x] Multi-class classification using regional age gaps
- [x] Severity correlation with clinical scores
- [x] ~2,590 subjects across 3 datasets
- [x] Fully reproducible (no random seeds, no GPU, simple code)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| HCP age range narrow (22-35) | Use ABIDE controls (wider age range) for augmented training; report this as limitation |
| Small UCLA CNP sample per disorder | Use ABIDE healthy controls as additional test set; report CIs |
| Regional models may be noisy | Aggregate: use both individual network models AND global model features |
| Disorders may not differentiate | Even null result is publishable ("regional brain age does NOT differentiate disorders — implications for precision psychiatry") |
| Site effects in ABIDE | Use ComBat harmonization as preprocessing step; also test without it |
