"""
BrainAge-Dx: Main Pipeline
Run the complete experiment end-to-end.

Usage:
    python run_pipeline.py --step all       # Full pipeline
    python run_pipeline.py --step fetch     # Download data only
    python run_pipeline.py --step extract   # Extract features only
    python run_pipeline.py --step train     # Train models only
    python run_pipeline.py --step evaluate  # Run evaluation only
    python run_pipeline.py --step figures   # Generate figures only
    python run_pipeline.py --step test      # Test with synthetic data
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

sys.path.append(str(Path(__file__).parent))
from config import (
    N_NETWORKS,
    N_FOLDS,
    RANDOM_STATE,
    NETWORK_NAMES,
    NETWORK_ABBREVS,
    PROCESSED_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)


def step_fetch():
    """Step 1: Download datasets."""
    from preprocessing.fetch_data import fetch_abide, fetch_ucla_cnp, fetch_hcp_info

    print("\n" + "=" * 60)
    print("STEP 1: FETCH DATA")
    print("=" * 60)

    fetch_abide()
    print()
    fetch_ucla_cnp()
    print()
    fetch_hcp_info()


def step_extract():
    """Step 2: Extract network-level FC features."""
    from preprocessing.extract_features import (
        get_schaefer_yeo7_mapping,
        process_abide,
    )

    print("\n" + "=" * 60)
    print("STEP 2: EXTRACT FEATURES")
    print("=" * 60)

    network_to_rois = get_schaefer_yeo7_mapping()
    abide_df = process_abide(network_to_rois)

    print(f"\nFeature extraction complete.")
    print(f"ABIDE: {len(abide_df)} subjects")


def step_train():
    """Step 3: Train brain age models."""
    from models.brain_age import BrainAgeModel

    print("\n" + "=" * 60)
    print("STEP 3: TRAIN BRAIN AGE MODELS")
    print("=" * 60)

    # Load HCP features
    hcp_path = PROCESSED_DIR / "hcp_features.csv"
    if not hcp_path.exists():
        print(f"ERROR: HCP features not found at {hcp_path}")
        print("Please extract HCP features first (see preprocessing/extract_features.py)")
        return None

    hcp_df = pd.read_csv(hcp_path)
    print(f"Loaded HCP data: {len(hcp_df)} subjects")

    model = BrainAgeModel()
    model.train(hcp_df)
    model.save()

    return model


def step_evaluate(model=None):
    """Step 4: Predict brain age gaps and run statistical analysis."""
    from models.brain_age import BrainAgeModel
    from evaluation.statistics import (
        run_all_group_comparisons,
        compare_global_vs_regional,
        anova_across_disorders,
    )

    print("\n" + "=" * 60)
    print("STEP 4: EVALUATE")
    print("=" * 60)

    # Load model
    if model is None:
        model = BrainAgeModel()
        model.load()

    # Load test datasets
    all_results = []

    for dataset in ["abide", "ucla"]:
        feat_path = PROCESSED_DIR / f"{dataset}_features.csv"
        if feat_path.exists():
            df = pd.read_csv(feat_path)
            print(f"\nPredicting for {dataset}: {len(df)} subjects")
            results = model.predict(df)
            all_results.append(results)
        else:
            print(f"WARNING: {feat_path} not found, skipping")

    if not all_results:
        print("No test data available!")
        return

    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(RESULTS_DIR / "all_predictions.csv", index=False)
    print(f"\nAll predictions saved: {len(results_df)} subjects")

    # Statistical analysis
    print("\n--- Group Comparisons ---")
    comparison_df = run_all_group_comparisons(results_df)

    print("\n--- ANOVA ---")
    anova_df = anova_across_disorders(results_df)

    print("\n--- Classification ---")
    global_res, regional_res, combined_res = compare_global_vs_regional(results_df)

    return results_df, comparison_df, {
        "Global": global_res,
        "Regional": regional_res,
        "Combined": combined_res,
    }


def step_figures(results_df=None, comparison_df=None, classification_results=None):
    """Step 5: Generate all publication figures."""
    from utils.visualizations import generate_all_figures

    print("\n" + "=" * 60)
    print("STEP 5: GENERATE FIGURES")
    print("=" * 60)

    if results_df is None:
        pred_path = RESULTS_DIR / "all_predictions.csv"
        if pred_path.exists():
            results_df = pd.read_csv(pred_path)
        else:
            print("No predictions found. Run evaluation first.")
            return

    if comparison_df is None:
        comp_path = RESULTS_DIR / "group_comparisons.csv"
        if comp_path.exists():
            comparison_df = pd.read_csv(comp_path)

    generate_all_figures(results_df, comparison_df, classification_results or {})


def step_test():
    """Run full pipeline with synthetic data to verify everything works."""
    from models.brain_age import BrainAgeModel
    from evaluation.statistics import (
        run_all_group_comparisons,
        compare_global_vs_regional,
        anova_across_disorders,
    )
    from utils.visualizations import generate_all_figures

    print("\n" + "=" * 60)
    print("TEST RUN WITH SYNTHETIC DATA")
    print("=" * 60)

    np.random.seed(RANDOM_STATE)

    # Feature names
    feat_names = []
    for abbrev in NETWORK_ABBREVS:
        feat_names.append(f"within_{abbrev}")
    for i, j in combinations(range(N_NETWORKS), 2):
        feat_names.append(f"between_{NETWORK_ABBREVS[i]}_{NETWORK_ABBREVS[j]}")

    # ── Synthetic HCP (training) ──
    n_train = 800
    train_data = {}
    train_data["AGE_AT_SCAN"] = np.random.uniform(22, 35, n_train)
    train_data["SEX"] = np.random.choice([0, 1], n_train)

    for name in feat_names:
        # Features correlated with age
        train_data[name] = (
            0.3 * np.random.randn(n_train)
            + 0.02 * train_data["AGE_AT_SCAN"]
            + np.random.uniform(-0.5, 0.5)
        )

    train_df = pd.DataFrame(train_data)
    print(f"Synthetic HCP: {n_train} subjects")

    # ── Synthetic test data ──
    disorders = {"HC": 200, "ASD": 150, "SZ": 80, "BD": 60, "ADHD": 60}

    # Define disorder-specific brain age patterns
    disorder_patterns = {
        "HC": {abbrev: 0.0 for abbrev in NETWORK_ABBREVS},
        "ASD": {"VIS": -0.1, "SM": 0.0, "DA": 0.0, "VA": -0.2, "LIM": 0.1, "FP": -0.4, "DMN": -0.8},
        "SZ": {"VIS": 0.1, "SM": 0.2, "DA": 0.3, "VA": 0.9, "LIM": 0.4, "FP": 0.7, "DMN": 0.3},
        "BD": {"VIS": 0.0, "SM": 0.1, "DA": -0.1, "VA": 0.3, "LIM": 0.8, "FP": 0.2, "DMN": 0.5},
        "ADHD": {"VIS": 0.1, "SM": -0.1, "DA": -0.3, "VA": 0.2, "LIM": 0.0, "FP": -0.6, "DMN": -0.2},
    }

    test_dfs = []
    for diagnosis, n in disorders.items():
        d = {}
        d["AGE_AT_SCAN"] = np.random.uniform(15, 50, n)
        d["SEX"] = np.random.choice([0, 1], n)
        d["diagnosis"] = diagnosis

        pattern = disorder_patterns[diagnosis]
        for name in feat_names:
            base = 0.3 * np.random.randn(n) + 0.02 * d["AGE_AT_SCAN"]
            # Add disorder-specific deviation to within-network features
            for abbrev in NETWORK_ABBREVS:
                if name == f"within_{abbrev}":
                    base += pattern[abbrev] * np.random.uniform(0.8, 1.2, n)
            d[name] = base

        test_dfs.append(pd.DataFrame(d))

    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"Synthetic test: {len(test_df)} subjects")
    for diag, n in disorders.items():
        print(f"  {diag}: {n}")

    # ── Train ──
    print("\n--- Training ---")
    model = BrainAgeModel()
    model.train(train_df)

    # ── Predict ──
    print("\n--- Predicting ---")
    results_df = model.predict(test_df)
    results_df.to_csv(RESULTS_DIR / "all_predictions.csv", index=False)

    # ── Statistics ──
    print("\n--- Group Comparisons ---")
    comparison_df = run_all_group_comparisons(results_df)

    print("\n--- ANOVA ---")
    anova_df = anova_across_disorders(results_df)
    print(anova_df.to_string())

    print("\n--- Classification ---")
    global_res, regional_res, combined_res = compare_global_vs_regional(results_df)

    # ── Figures ──
    print("\n--- Generating Figures ---")
    classification_results = {
        "Global": global_res,
        "Regional": regional_res,
        "Combined": combined_res,
    }
    generate_all_figures(results_df, comparison_df, classification_results)

    print("\n" + "=" * 60)
    print("TEST RUN COMPLETE!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BrainAge-Dx: Main Pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "fetch", "extract", "train", "evaluate", "figures", "test"],
        default="test",
        help="Which step to run (default: test with synthetic data)",
    )
    args = parser.parse_args()

    if args.step == "fetch":
        step_fetch()
    elif args.step == "extract":
        step_extract()
    elif args.step == "train":
        step_train()
    elif args.step == "evaluate":
        step_evaluate()
    elif args.step == "figures":
        step_figures()
    elif args.step == "test":
        step_test()
    elif args.step == "all":
        step_fetch()
        step_extract()
        model = step_train()
        if model:
            results = step_evaluate(model)
            if results:
                results_df, comparison_df, class_results = results
                step_figures(results_df, comparison_df, class_results)


if __name__ == "__main__":
    main()
