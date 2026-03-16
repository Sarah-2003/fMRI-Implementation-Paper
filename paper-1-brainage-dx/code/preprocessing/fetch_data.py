"""
BrainAge-Dx: Data Fetching
Download ABIDE and UCLA CNP datasets.
HCP requires manual registration at ConnectomeDB.
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR


def fetch_abide(pipeline="cpac", quality_checked=True):
    """
    Download ABIDE I preprocessed data via nilearn.
    Returns phenotypic DataFrame and list of ROI time series file paths.
    """
    from nilearn import datasets

    print("Fetching ABIDE I preprocessed data...")
    print("  Pipeline:", pipeline)
    print("  Quality checked:", quality_checked)

    abide = datasets.fetch_abide_pcp(
        data_dir=str(RAW_DIR),
        pipeline=pipeline,
        band_pass_filtering=True,
        global_signal_regression=False,
        derivatives=["rois_ho"],  # Harvard-Oxford atlas ROI time series
        quality_checked=quality_checked,
        verbose=1,
    )

    # Extract phenotypic info
    pheno = pd.DataFrame(abide.phenotypic)
    pheno["dataset"] = "abide"
    pheno["diagnosis"] = pheno["DX_GROUP"].map({1: "ASD", 2: "HC"})

    print(f"  Downloaded {len(pheno)} subjects")
    print(f"  ASD: {(pheno['diagnosis'] == 'ASD').sum()}")
    print(f"  HC: {(pheno['diagnosis'] == 'HC').sum()}")

    # Save phenotypic data
    out_path = PROCESSED_DIR / "abide_phenotypic.csv"
    pheno.to_csv(out_path, index=False)
    print(f"  Saved phenotypic data to {out_path}")

    return pheno, abide.rois_ho


def fetch_ucla_cnp():
    """
    Download UCLA CNP (ds000030) from OpenNeuro.
    Requires openneuro-py: pip install openneuro-py

    Alternative: manual download from https://openneuro.org/datasets/ds000030
    """
    ucla_dir = RAW_DIR / "ucla_cnp"
    ucla_dir.mkdir(parents=True, exist_ok=True)

    print("UCLA CNP dataset:")
    print("  Manual download required from: https://openneuro.org/datasets/ds000030")
    print(f"  Place BIDS data in: {ucla_dir}")
    print()
    print("  Quick download via CLI:")
    print("    pip install openneuro-py")
    print(f'    openneuro download --dataset ds000030 --target-dir "{ucla_dir}"')
    print()
    print("  Subjects: 272 (138 HC, 58 SZ, 49 BD, 45 ADHD)")

    # Create phenotypic template if participants.tsv exists
    tsv_path = ucla_dir / "participants.tsv"
    if tsv_path.exists():
        pheno = pd.read_csv(tsv_path, sep="\t")
        pheno["dataset"] = "ucla"
        out_path = PROCESSED_DIR / "ucla_phenotypic.csv"
        pheno.to_csv(out_path, index=False)
        print(f"  Saved phenotypic data to {out_path}")
        return pheno
    else:
        print("  (participants.tsv not yet available — download data first)")
        return None


def fetch_hcp_info():
    """
    HCP requires manual registration. Print instructions.
    """
    hcp_dir = RAW_DIR / "hcp"
    hcp_dir.mkdir(parents=True, exist_ok=True)

    print("HCP Young Adult dataset:")
    print("  1. Register at https://db.humanconnectome.org/")
    print("  2. Accept Data Use Terms")
    print("  3. Download resting-state fMRI (preprocessed)")
    print(f"  4. Place data in: {hcp_dir}")
    print()
    print("  Recommended: Download the 'Resting State fMRI FIX-Denoised' package")
    print("  Subjects: 1,206 healthy, ages 22-35")
    print()
    print("  Alternative: Use HCP 1200 Parcellation+Timeseries+Netmats data")
    print("  (pre-extracted time series — much smaller download ~2GB)")
    print("  URL: https://db.humanconnectome.org/data/projects/HCP_1200")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download datasets for BrainAge-Dx")
    parser.add_argument(
        "--dataset",
        choices=["abide", "ucla", "hcp", "all"],
        default="all",
        help="Which dataset to fetch",
    )
    args = parser.parse_args()

    if args.dataset in ("abide", "all"):
        fetch_abide()
        print()

    if args.dataset in ("ucla", "all"):
        fetch_ucla_cnp()
        print()

    if args.dataset in ("hcp", "all"):
        fetch_hcp_info()
