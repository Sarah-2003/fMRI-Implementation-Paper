"""
BrainAge-Dx: Visualizations
Radar charts, brain maps, confusion matrices, scatter plots.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from itertools import combinations

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    N_NETWORKS,
    NETWORK_NAMES,
    NETWORK_ABBREVS,
    NETWORK_COLORS,
    FIGURES_DIR,
)

# Consistent style
plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

DISORDER_COLORS = {
    "ASD": "#2196F3",
    "SZ": "#F44336",
    "BD": "#FF9800",
    "ADHD": "#4CAF50",
    "HC": "#9E9E9E",
}


def plot_radar_chart(mean_gaps_per_disorder, save_path=None, title=None):
    """
    THE KILLER FIGURE — Radar chart of regional brain age gaps per disorder.

    Args:
        mean_gaps_per_disorder: dict of {disorder: np.array of 7 mean gaps}
        save_path: path to save figure
        title: optional title
    """
    angles = np.linspace(0, 2 * np.pi, N_NETWORKS, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for disorder, gaps in mean_gaps_per_disorder.items():
        values = gaps.tolist()
        values += values[:1]
        color = DISORDER_COLORS.get(disorder, "#333333")

        ax.plot(angles, values, "o-", linewidth=2.5, label=disorder, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(NETWORK_ABBREVS, fontsize=14, fontweight="bold")

    # Zero line
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    # Y-axis
    ax.set_ylabel("Brain Age Gap (years)", labelpad=30, fontsize=12)

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    else:
        ax.set_title(
            "Regional Brain Age Gap Profiles\nper Neuropsychiatric Disorder",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

    if save_path:
        fig.savefig(save_path)
        print(f"Radar chart saved to {save_path}")
    plt.close(fig)
    return fig


def plot_group_comparison_bars(comparison_df, disorder, save_path=None):
    """
    Bar plot of mean brain age gap per network for a specific disorder vs HC.
    Bars colored by significance.

    Args:
        comparison_df: output from group_comparison()
        disorder: str
    """
    df = comparison_df[comparison_df["disorder"] == disorder].copy()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    colors = ["#F44336" if sig else "#BDBDBD" for sig in df["significant"]]

    bars = ax.bar(x, df["mean_gap_disorder"], color=colors, edgecolor="black", linewidth=0.5)

    # Error bars
    ax.errorbar(
        x, df["mean_gap_disorder"], yerr=df["std_gap_disorder"],
        fmt="none", ecolor="black", capsize=4
    )

    # Zero line
    ax.axhline(y=0, color="black", linewidth=1, linestyle="-")

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(df["network"], rotation=45, ha="right")
    ax.set_ylabel("Brain Age Gap (years)")
    ax.set_title(f"Regional Brain Age Gaps — {disorder} vs HC")

    # Legend
    sig_patch = mpatches.Patch(color="#F44336", label="Significant (FDR < 0.05)")
    ns_patch = mpatches.Patch(color="#BDBDBD", label="Not significant")
    ax.legend(handles=[sig_patch, ns_patch])

    # Effect sizes as text
    for i, (_, row) in enumerate(df.iterrows()):
        if row["significant"]:
            ax.text(
                i, row["mean_gap_disorder"] + row["std_gap_disorder"] + 0.1,
                f"d={row['cohens_d']:.2f}",
                ha="center", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Bar plot saved to {save_path}")
    plt.close(fig)
    return fig


def plot_confusion_matrix(cm, labels, title="", save_path=None):
    """
    Publication-quality confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        square=True, linewidths=1, linecolor="white",
        cbar_kws={"label": "Proportion"},
        ax=ax,
    )

    # Add raw counts as secondary annotation
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.75,
                f"(n={cm[i, j]})",
                ha="center", va="center",
                fontsize=8, color="gray",
            )

    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
    ax.set_title(title or "Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)
    return fig


def plot_individual_scatter(results_df, save_path=None):
    """
    Scatter plot of individual subjects in brain age gap space.
    Uses t-SNE or first 2 PCA components of 7 regional gaps.
    """
    from sklearn.decomposition import PCA

    gap_cols = [f"gap_{abbrev}" for abbrev in NETWORK_ABBREVS]
    df = results_df.dropna(subset=gap_cols).copy()

    X = df[gap_cols].values
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    for disorder in df["diagnosis"].unique():
        mask = df["diagnosis"] == disorder
        color = DISORDER_COLORS.get(disorder, "#333333")
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=color, label=disorder, alpha=0.6, s=30, edgecolors="white", linewidth=0.5,
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Individual Brain Age Gap Profiles\n(PCA of 7 Regional Gaps)")
    ax.legend(title="Diagnosis", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Scatter plot saved to {save_path}")
    plt.close(fig)
    return fig


def plot_severity_correlation(results_df, severity_col, gap_col, disorder, save_path=None):
    """
    Scatter plot with regression line: gap vs severity.
    """
    df = results_df[results_df["diagnosis"] == disorder].dropna(subset=[severity_col, gap_col])

    if len(df) < 10:
        print(f"Too few samples for {disorder} severity correlation")
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(df[severity_col], df[gap_col], alpha=0.6, color=DISORDER_COLORS.get(disorder))

    # Regression line
    from scipy import stats
    slope, intercept, r, p, se = stats.linregress(df[severity_col], df[gap_col])
    x_line = np.linspace(df[severity_col].min(), df[severity_col].max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=2)

    network_name = gap_col.replace("gap_", "")
    ax.set_xlabel(severity_col, fontsize=12)
    ax.set_ylabel(f"Brain Age Gap — {network_name} (years)", fontsize=12)
    ax.set_title(f"{disorder}: Severity vs Brain Age Gap (r={r:.3f}, p={p:.4f})")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig


def plot_pipeline_overview(save_path=None):
    """
    Generate the pipeline overview figure (Figure 1).
    """
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis("off")

    steps = [
        ("fMRI\nData", 0.5),
        ("FC\nMatrix", 2.0),
        ("7-Network\nFeatures", 3.5),
        ("Ridge\nRegression", 5.0),
        ("Brain Age\nGap", 6.5),
        ("Regional\nProfile", 8.0),
        ("Diagnosis", 9.5),
    ]

    for i, (label, x) in enumerate(steps):
        color = "#E3F2FD" if i < 3 else "#FFF3E0" if i < 5 else "#E8F5E9"
        rect = mpatches.FancyBboxPatch(
            (x - 0.5, 0.5), 0.95, 1.0,
            boxstyle="round,pad=0.1", facecolor=color, edgecolor="black", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, 1.0, label, ha="center", va="center", fontsize=11, fontweight="bold")

        if i < len(steps) - 1:
            ax.annotate(
                "", xy=(steps[i + 1][1] - 0.55, 1.0), xytext=(x + 0.5, 1.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=2),
            )

    ax.set_title(
        "BrainAge-Dx Pipeline", fontsize=14, fontweight="bold", y=1.05
    )

    if save_path:
        fig.savefig(save_path)
        print(f"Pipeline figure saved to {save_path}")
    plt.close(fig)
    return fig


def generate_all_figures(results_df, comparison_df, classification_results):
    """
    Generate all publication figures and save to paper/figures/.
    """
    print("\nGenerating all figures...")
    print("=" * 50)

    # Figure 1: Pipeline overview
    plot_pipeline_overview(FIGURES_DIR / "fig1_pipeline.png")

    # Figure 2: THE RADAR CHART
    disorders = [d for d in results_df["diagnosis"].unique() if d != "HC"]
    mean_gaps = {}
    for disorder in disorders:
        mask = results_df["diagnosis"] == disorder
        gaps = [results_df.loc[mask, f"gap_{abbrev}"].mean() for abbrev in NETWORK_ABBREVS]
        mean_gaps[disorder] = np.array(gaps)

    plot_radar_chart(mean_gaps, FIGURES_DIR / "fig2_radar_chart.png")

    # Figure 3: Bar plots per disorder
    for disorder in disorders:
        plot_group_comparison_bars(
            comparison_df, disorder,
            FIGURES_DIR / f"fig3_bars_{disorder.lower()}.png",
        )

    # Figure 4: Confusion matrices
    if classification_results:
        for name, res in classification_results.items():
            if res and "confusion_matrix" in res:
                plot_confusion_matrix(
                    res["confusion_matrix"], res["labels"],
                    title=f"Classification — {name}",
                    save_path=FIGURES_DIR / f"fig4_cm_{name.lower().replace(' ', '_')}.png",
                )

    # Figure 5: Individual scatter
    plot_individual_scatter(results_df, FIGURES_DIR / "fig5_individual_scatter.png")

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    # Test with synthetic data
    print("BrainAge-Dx: Visualizations — Test Run")
    print("=" * 50)

    np.random.seed(42)

    # Synthetic radar data
    mean_gaps = {
        "ASD": np.array([-0.2, 0.1, 0.0, -0.1, 0.05, -0.3, -0.8]),
        "SZ": np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.7, 0.3]),
        "BD": np.array([0.0, 0.1, -0.1, 0.3, 0.8, 0.2, 0.5]),
        "ADHD": np.array([0.1, -0.1, -0.3, 0.2, 0.0, -0.5, -0.2]),
    }

    plot_radar_chart(mean_gaps, FIGURES_DIR / "test_radar.png",
                     title="Test Radar Chart")
    plot_pipeline_overview(FIGURES_DIR / "test_pipeline.png")

    print("\nTest run complete!")
