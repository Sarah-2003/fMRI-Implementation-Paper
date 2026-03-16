"""
BrainAge-Dx: Configuration
All paths, parameters, and constants in one place.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
TABLES_DIR = PROJECT_ROOT / "paper" / "tables"

# Create dirs on import
for d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Atlas ──────────────────────────────────────────────────────────────
ATLAS_NAME = "schaefer_200_yeo7"
N_ROIS = 200
N_NETWORKS = 7

# Yeo 7-network labels (order matches Schaefer atlas)
NETWORK_NAMES = [
    "Visual",
    "Somatomotor",
    "Dorsal Attention",
    "Ventral Attention",
    "Limbic",
    "Frontoparietal",
    "Default Mode",
]

NETWORK_ABBREVS = ["VIS", "SM", "DA", "VA", "LIM", "FP", "DMN"]

NETWORK_COLORS = {
    "VIS": "#781286",
    "SM": "#4682B4",
    "DA": "#00760E",
    "VA": "#C43AFA",
    "LIM": "#DCF8A4",
    "FP": "#E69422",
    "DMN": "#CD3E4E",
}

# ── fMRI Preprocessing ────────────────────────────────────────────────
TR = 2.0  # Repetition time (seconds) — adjusted per dataset
LOW_PASS = 0.1  # Hz
HIGH_PASS = 0.01  # Hz
STANDARDIZE = True
DETREND = True

# ── Brain Age Model ───────────────────────────────────────────────────
RIDGE_ALPHA = 1.0  # Regularization (tuned via CV)
N_FOLDS = 10  # Cross-validation folds
RANDOM_STATE = 42

# ── Datasets ──────────────────────────────────────────────────────────
DATASETS = {
    "hcp": {
        "name": "Human Connectome Project (Young Adult)",
        "role": "train",
        "n_subjects": 1206,
        "age_range": (22, 35),
        "disorders": [],
    },
    "abide": {
        "name": "ABIDE I",
        "role": "test",
        "n_subjects": 1112,
        "age_range": (6, 64),
        "disorders": ["ASD"],
        "pipeline": "cpac",
        "quality_checked": True,
    },
    "ucla": {
        "name": "UCLA CNP (OpenNeuro ds000030)",
        "role": "test",
        "n_subjects": 272,
        "age_range": (21, 50),
        "disorders": ["SZ", "BD", "ADHD"],
    },
}

# ── Feature Config ────────────────────────────────────────────────────
# 7 within-network + 21 between-network = 28 FC features
N_WITHIN_FEATURES = N_NETWORKS  # 7
N_BETWEEN_FEATURES = N_NETWORKS * (N_NETWORKS - 1) // 2  # 21
N_TOTAL_FEATURES = N_WITHIN_FEATURES + N_BETWEEN_FEATURES  # 28
# +1 for sex = 29 features for age prediction

# ── Statistical Thresholds ────────────────────────────────────────────
ALPHA = 0.05
FDR_METHOD = "fdr_bh"  # Benjamini-Hochberg
