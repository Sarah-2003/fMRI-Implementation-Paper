"""
BrainAge-Dx: Statistical Analysis
Group comparisons, effect sizes, classification, severity correlations.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    N_NETWORKS,
    N_FOLDS,
    RANDOM_STATE,
    ALPHA,
    FDR_METHOD,
    NETWORK_NAMES,
    NETWORK_ABBREVS,
    RESULTS_DIR,
)


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def group_comparison(results_df, disorder, control_label="HC"):
    """
    Compare brain age gaps between a disorder group and healthy controls.

    Args:
        results_df: DataFrame with gap columns and 'diagnosis'
        disorder: str (e.g., 'ASD', 'SZ', 'BD', 'ADHD')
        control_label: str

    Returns:
        DataFrame with t-stat, p-value, Cohen's d, mean gap for each network + global
    """
    dis_mask = results_df["diagnosis"] == disorder
    hc_mask = results_df["diagnosis"] == control_label

    gap_cols = ["gap_global"] + [f"gap_{abbrev}" for abbrev in NETWORK_ABBREVS]
    labels = ["Global"] + NETWORK_NAMES

    rows = []
    p_values = []

    for col, label in zip(gap_cols, labels):
        dis_gaps = results_df.loc[dis_mask, col].dropna().values
        hc_gaps = results_df.loc[hc_mask, col].dropna().values

        if len(dis_gaps) < 5 or len(hc_gaps) < 5:
            rows.append({
                "network": label,
                "mean_gap_disorder": np.nan,
                "mean_gap_hc": np.nan,
                "t_stat": np.nan,
                "p_value": np.nan,
                "cohens_d": np.nan,
            })
            p_values.append(1.0)
            continue

        t_stat, p_val = stats.ttest_ind(dis_gaps, hc_gaps)
        d = cohens_d(dis_gaps, hc_gaps)

        rows.append({
            "network": label,
            "mean_gap_disorder": np.mean(dis_gaps),
            "mean_gap_hc": np.mean(hc_gaps),
            "std_gap_disorder": np.std(dis_gaps),
            "std_gap_hc": np.std(hc_gaps),
            "n_disorder": len(dis_gaps),
            "n_hc": len(hc_gaps),
            "t_stat": t_stat,
            "p_value": p_val,
            "cohens_d": d,
        })
        p_values.append(p_val)

    comp_df = pd.DataFrame(rows)

    # FDR correction
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method=FDR_METHOD)
    comp_df["p_corrected"] = p_corrected
    comp_df["significant"] = rejected
    comp_df["disorder"] = disorder

    return comp_df


def run_all_group_comparisons(results_df, disorders=None):
    """
    Run group comparisons for all disorders vs HC.

    Returns combined DataFrame.
    """
    if disorders is None:
        disorders = [d for d in results_df["diagnosis"].unique() if d != "HC"]

    all_comparisons = []
    for disorder in disorders:
        print(f"\n{disorder} vs HC:")
        comp = group_comparison(results_df, disorder)
        all_comparisons.append(comp)

        # Print summary
        for _, row in comp.iterrows():
            sig = "*" if row["significant"] else ""
            print(
                f"  {row['network']:20s}: d={row['cohens_d']:+.3f}, "
                f"p={row['p_corrected']:.4f} {sig}"
            )

    combined = pd.concat(all_comparisons, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "group_comparisons.csv", index=False)
    return combined


def classify_disorders(results_df, use_regional=True, use_global=False):
    """
    Multi-class classification using brain age gap profiles.

    Args:
        results_df: DataFrame with gap columns and 'diagnosis'
        use_regional: bool, use 7 regional gaps
        use_global: bool, use global gap

    Returns:
        dict with accuracy, F1, confusion matrix, classification report
    """
    # Exclude HC for disorder-vs-disorder classification
    disorder_mask = results_df["diagnosis"] != "HC"
    df = results_df[disorder_mask].copy()

    if len(df) < 20:
        print("Too few samples for classification")
        return None

    # Select features
    feat_cols = []
    if use_regional:
        feat_cols += [f"gap_{abbrev}" for abbrev in NETWORK_ABBREVS]
    if use_global:
        feat_cols += ["gap_global"]

    # Drop NaN rows
    df = df.dropna(subset=feat_cols)

    X = df[feat_cols].values
    y = df["diagnosis"].values

    print(f"\nClassification: {' + '.join(np.unique(y))}")
    print(f"  Features: {feat_cols}")
    print(f"  Samples: {len(y)}")
    for label in np.unique(y):
        print(f"    {label}: {(y == label).sum()}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=min(N_FOLDS, min(np.bincount(pd.factorize(y)[0]))),
                         shuffle=True, random_state=RANDOM_STATE)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    y_pred = cross_val_predict(clf, X_scaled, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")
    cm = confusion_matrix(y, y_pred, labels=np.unique(y))
    report = classification_report(y, y_pred)

    print(f"\n  Accuracy:    {acc:.3f}")
    print(f"  F1 (macro):  {f1_macro:.3f}")
    print(f"  F1 (weighted): {f1_weighted:.3f}")
    print(f"\n{report}")

    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "labels": np.unique(y),
        "report": report,
        "feature_type": "regional" if use_regional else "global",
    }

    return results


def compare_global_vs_regional(results_df):
    """
    Compare classification performance: global gap only vs. regional profile.
    This is the KEY experiment.
    """
    print("\n" + "=" * 60)
    print("KEY EXPERIMENT: Global Gap vs Regional Profile")
    print("=" * 60)

    print("\n--- Global Gap Only (1 feature) ---")
    global_results = classify_disorders(results_df, use_regional=False, use_global=True)

    print("\n--- Regional Profile (7 features) ---")
    regional_results = classify_disorders(results_df, use_regional=True, use_global=False)

    print("\n--- Combined (8 features) ---")
    combined_results = classify_disorders(results_df, use_regional=True, use_global=True)

    # Save comparison
    comparison = {
        "model": ["Global (1 feat)", "Regional (7 feat)", "Combined (8 feat)"],
        "accuracy": [
            global_results["accuracy"] if global_results else None,
            regional_results["accuracy"] if regional_results else None,
            combined_results["accuracy"] if combined_results else None,
        ],
        "f1_macro": [
            global_results["f1_macro"] if global_results else None,
            regional_results["f1_macro"] if regional_results else None,
            combined_results["f1_macro"] if combined_results else None,
        ],
    }
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(RESULTS_DIR / "global_vs_regional.csv", index=False)
    print(f"\nComparison saved to {RESULTS_DIR / 'global_vs_regional.csv'}")

    return global_results, regional_results, combined_results


def severity_correlation(results_df, severity_col, gap_cols=None):
    """
    Correlate brain age gaps with clinical severity scores.

    Args:
        results_df: DataFrame
        severity_col: column name with severity scores
        gap_cols: list of gap columns to test

    Returns:
        DataFrame with Pearson r, p-value per network
    """
    if gap_cols is None:
        gap_cols = ["gap_global"] + [f"gap_{abbrev}" for abbrev in NETWORK_ABBREVS]

    # Only subjects with severity scores
    df = results_df.dropna(subset=[severity_col])

    if len(df) < 10:
        print(f"Too few subjects with {severity_col} scores")
        return None

    rows = []
    p_values = []

    for col in gap_cols:
        gaps = df[col].values
        severity = df[severity_col].values

        r, p = stats.pearsonr(gaps, severity)
        rows.append({
            "network": col.replace("gap_", ""),
            "pearson_r": r,
            "p_value": p,
            "n": len(df),
        })
        p_values.append(p)

    corr_df = pd.DataFrame(rows)
    rejected, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method=FDR_METHOD)
    corr_df["p_corrected"] = p_corrected
    corr_df["significant"] = rejected

    return corr_df


def anova_across_disorders(results_df):
    """
    One-way ANOVA for each regional gap across all disorder groups.
    Tests whether disorders have significantly different regional patterns.
    """
    gap_cols = [f"gap_{abbrev}" for abbrev in NETWORK_ABBREVS]
    disorders = [d for d in results_df["diagnosis"].unique() if d != "HC"]

    rows = []
    p_values = []

    for col, name in zip(gap_cols, NETWORK_NAMES):
        groups = [
            results_df.loc[results_df["diagnosis"] == d, col].dropna().values
            for d in disorders
        ]
        groups = [g for g in groups if len(g) >= 3]

        if len(groups) < 2:
            continue

        f_stat, p_val = stats.f_oneway(*groups)
        rows.append({
            "network": name,
            "f_stat": f_stat,
            "p_value": p_val,
            "n_groups": len(groups),
        })
        p_values.append(p_val)

    anova_df = pd.DataFrame(rows)
    if p_values:
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=ALPHA, method=FDR_METHOD)
        anova_df["p_corrected"] = p_corrected
        anova_df["significant"] = rejected

    anova_df.to_csv(RESULTS_DIR / "anova_results.csv", index=False)
    return anova_df


if __name__ == "__main__":
    # Test with synthetic data
    print("BrainAge-Dx: Statistical Analysis — Test Run")
    print("=" * 50)

    np.random.seed(RANDOM_STATE)
    n = 300

    data = {
        "diagnosis": np.random.choice(["HC", "ASD", "SZ", "BD", "ADHD"], n),
        "gap_global": np.random.randn(n),
    }

    for abbrev in NETWORK_ABBREVS:
        data[f"gap_{abbrev}"] = np.random.randn(n)

    # Add disorder-specific patterns
    df = pd.DataFrame(data)
    df.loc[df["diagnosis"] == "ASD", "gap_DMN"] -= 0.5
    df.loc[df["diagnosis"] == "SZ", "gap_VA"] += 0.7
    df.loc[df["diagnosis"] == "SZ", "gap_FP"] += 0.5
    df.loc[df["diagnosis"] == "BD", "gap_LIM"] += 0.6

    # Group comparisons
    run_all_group_comparisons(df)

    # ANOVA
    print("\n\nANOVA across disorders:")
    anova_df = anova_across_disorders(df)
    print(anova_df)

    # Classification
    compare_global_vs_regional(df)

    print("\nTest run complete!")
