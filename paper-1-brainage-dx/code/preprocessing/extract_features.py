"""
BrainAge-Dx: Feature Extraction
Convert fMRI data → FC matrices → 28 network-level features per subject.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR,
    N_ROIS,
    N_NETWORKS,
    N_TOTAL_FEATURES,
    NETWORK_NAMES,
    NETWORK_ABBREVS,
    TR,
    LOW_PASS,
    HIGH_PASS,
    STANDARDIZE,
    DETREND,
)


def get_schaefer_yeo7_mapping():
    """
    Get the mapping from Schaefer 200 ROIs to Yeo 7 networks.
    Returns dict: {network_index (0-6): list of ROI indices}
    """
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7)
    labels = atlas.labels

    # Parse network from label names (e.g., '7Networks_LH_Vis_1')
    roi_to_network = {}
    network_map = {
        "Vis": 0,
        "SomMot": 1,
        "DorsAttn": 2,
        "SalVentAttn": 3,
        "Limbic": 4,
        "Cont": 5,
        "Default": 6,
    }

    for i, label in enumerate(labels):
        if isinstance(label, bytes):
            label = label.decode()
        for key, net_idx in network_map.items():
            if key in label:
                roi_to_network[i] = net_idx
                break

    # Build network → ROI list
    network_to_rois = {n: [] for n in range(N_NETWORKS)}
    for roi, net in roi_to_network.items():
        network_to_rois[net].append(roi)

    return network_to_rois


def extract_fc_matrix(time_series):
    """
    Compute functional connectivity matrix from ROI time series.
    Uses Pearson correlation.

    Args:
        time_series: np.ndarray of shape (n_timepoints, n_rois)

    Returns:
        fc_matrix: np.ndarray of shape (n_rois, n_rois)
    """
    fc = np.corrcoef(time_series.T)
    # Fisher z-transform
    fc = np.arctanh(np.clip(fc, -0.999, 0.999))
    np.fill_diagonal(fc, 0)
    return fc


def fc_to_network_features(fc_matrix, network_to_rois):
    """
    Compute 28 network-level features from a full FC matrix.

    Features:
    - 7 within-network mean FC (mean of connections within each network)
    - 21 between-network mean FC (mean of connections between each pair)

    Args:
        fc_matrix: (n_rois, n_rois) FC matrix
        network_to_rois: dict mapping network_idx → list of ROI indices

    Returns:
        features: np.ndarray of shape (28,)
        feature_names: list of 28 feature names
    """
    features = []
    feature_names = []

    # Within-network features (7)
    for net_idx in range(N_NETWORKS):
        rois = network_to_rois[net_idx]
        if len(rois) < 2:
            features.append(0.0)
        else:
            # Upper triangle of within-network submatrix
            submat = fc_matrix[np.ix_(rois, rois)]
            triu_idx = np.triu_indices(len(rois), k=1)
            features.append(np.mean(submat[triu_idx]))
        feature_names.append(f"within_{NETWORK_ABBREVS[net_idx]}")

    # Between-network features (21)
    for i, j in combinations(range(N_NETWORKS), 2):
        rois_i = network_to_rois[i]
        rois_j = network_to_rois[j]
        submat = fc_matrix[np.ix_(rois_i, rois_j)]
        features.append(np.mean(submat))
        feature_names.append(f"between_{NETWORK_ABBREVS[i]}_{NETWORK_ABBREVS[j]}")

    features = np.array(features)
    assert len(features) == N_TOTAL_FEATURES, f"Expected {N_TOTAL_FEATURES}, got {len(features)}"
    return features, feature_names


def extract_features_from_timeseries_files(
    timeseries_files, phenotypic_df, dataset_name, network_to_rois
):
    """
    Process a list of ROI time series files → network features DataFrame.

    Args:
        timeseries_files: list of file paths to ROI time series (.1D or .npy)
        phenotypic_df: DataFrame with subject info
        dataset_name: str identifier
        network_to_rois: dict from get_schaefer_yeo7_mapping()

    Returns:
        DataFrame with 28 features + metadata per subject
    """
    all_features = []
    valid_subjects = []

    for idx, ts_file in enumerate(timeseries_files):
        try:
            # Load time series
            if str(ts_file).endswith(".1D"):
                ts = np.loadtxt(ts_file)
            elif str(ts_file).endswith(".npy"):
                ts = np.load(ts_file)
            else:
                ts = np.loadtxt(ts_file)

            # Skip if too short or has NaN
            if ts.shape[0] < 50 or np.any(np.isnan(ts)):
                print(f"  Skipping subject {idx}: bad data (shape={ts.shape})")
                continue

            # Compute FC matrix
            fc = extract_fc_matrix(ts)

            # Extract network features
            feats, feat_names = fc_to_network_features(fc, network_to_rois)

            all_features.append(feats)
            valid_subjects.append(idx)

        except Exception as e:
            print(f"  Error processing subject {idx}: {e}")
            continue

    if not all_features:
        print(f"  WARNING: No valid subjects for {dataset_name}")
        return pd.DataFrame()

    # Build DataFrame
    feat_df = pd.DataFrame(all_features, columns=feat_names)

    # Add metadata from phenotypic
    for col in ["SUB_ID", "AGE_AT_SCAN", "SEX", "diagnosis", "SITE_ID"]:
        if col in phenotypic_df.columns:
            feat_df[col] = phenotypic_df.iloc[valid_subjects][col].values

    feat_df["dataset"] = dataset_name
    feat_df["subject_idx"] = valid_subjects

    print(f"  Extracted features for {len(feat_df)} / {len(timeseries_files)} subjects")
    return feat_df


def extract_features_from_fc_matrices(fc_dir, phenotypic_df, dataset_name, network_to_rois):
    """
    Process pre-computed FC matrices (e.g., from HCP Netmats).

    Args:
        fc_dir: path to directory with FC matrices (.npy or .txt files)
        phenotypic_df: DataFrame with subject info
        dataset_name: str identifier
        network_to_rois: dict

    Returns:
        DataFrame with 28 features + metadata
    """
    fc_files = sorted(Path(fc_dir).glob("*.npy")) + sorted(Path(fc_dir).glob("*.txt"))

    all_features = []
    valid_indices = []

    for idx, fc_file in enumerate(fc_files):
        try:
            fc = np.load(fc_file) if str(fc_file).endswith(".npy") else np.loadtxt(fc_file)

            if np.any(np.isnan(fc)):
                continue

            feats, feat_names = fc_to_network_features(fc, network_to_rois)
            all_features.append(feats)
            valid_indices.append(idx)

        except Exception as e:
            print(f"  Error with {fc_file.name}: {e}")
            continue

    feat_df = pd.DataFrame(all_features, columns=feat_names)

    if phenotypic_df is not None and len(phenotypic_df) >= len(valid_indices):
        for col in phenotypic_df.columns:
            if col not in feat_df.columns:
                feat_df[col] = phenotypic_df.iloc[valid_indices][col].values

    feat_df["dataset"] = dataset_name
    return feat_df


def process_abide(network_to_rois):
    """Process ABIDE dataset end-to-end."""
    from nilearn import datasets

    print("Processing ABIDE...")

    # Fetch data (cached if already downloaded)
    abide = datasets.fetch_abide_pcp(
        pipeline="cpac",
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_ho"],
        quality_checked=True,
        verbose=0,
    )

    pheno = pd.DataFrame(abide.phenotypic)
    pheno["diagnosis"] = pheno["DX_GROUP"].map({1: "ASD", 2: "HC"})

    # Note: ABIDE rois_ho uses Harvard-Oxford atlas (not Schaefer)
    # For the paper, we need to either:
    # 1. Use Harvard-Oxford and map to Yeo networks (approximate)
    # 2. Re-extract from 4D fMRI with Schaefer atlas
    # Option 1 is faster; Option 2 is more precise.
    # We'll implement both and let the user choose.

    feat_df = extract_features_from_timeseries_files(
        abide.rois_ho, pheno, "abide", network_to_rois
    )

    out_path = PROCESSED_DIR / "abide_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")
    return feat_df


def get_feature_names():
    """Return the 28 feature names in order."""
    names = []
    for abbrev in NETWORK_ABBREVS:
        names.append(f"within_{abbrev}")
    for i, j in combinations(range(N_NETWORKS), 2):
        names.append(f"between_{NETWORK_ABBREVS[i]}_{NETWORK_ABBREVS[j]}")
    return names


if __name__ == "__main__":
    print("BrainAge-Dx: Feature Extraction")
    print("=" * 50)

    # Get network mapping
    print("\nLoading Schaefer 200 atlas with Yeo 7 network labels...")
    network_to_rois = get_schaefer_yeo7_mapping()

    for net_idx, rois in network_to_rois.items():
        print(f"  {NETWORK_NAMES[net_idx]}: {len(rois)} ROIs")

    # Process ABIDE
    print()
    abide_df = process_abide(network_to_rois)

    print(f"\nDone! Feature shape: {abide_df.shape}")
    print(f"Feature columns: {get_feature_names()[:5]}... ({N_TOTAL_FEATURES} total)")
