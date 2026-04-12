"""
Clustering visualizations — loads model output from data/processed/ and saves
four figures to data/visualizations/.

Run AFTER train_clustering.py.

Usage (from project root):
    python src/visualizations_clustering.py
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/raw/train.csv"
MODEL_PATH  = "data/processed/clustering_model.pkl"
LABELS_PATH = "data/processed/cluster_labels.npy"
OUT_DIR     = "data/visualizations"
RANDOM_STATE = 42
UMAP_SAMPLE  = 4000


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_inputs(data_path, model_path, labels_path):
    """Given file paths, return df, binary feature matrix, cluster labels, and saved model dict."""
    df = pd.read_csv(data_path)
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    labels = np.load(labels_path)
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    X_bin = df[binary_cols].values
    return df, X_bin, labels, saved


def save_fig(fig, name):
    """Given a figure and filename stem, save PNG to data/visualizations/."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_silhouette(sil_scores):
    """Given a list of silhouette scores for k=2..6, save silhouette sweep plot."""
    k_range = list(range(2, 2 + len(sil_scores)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, sil_scores, "o-", color="#1f77b4", lw=2.2, ms=8)
    ax.axvline(2, color="gray", linestyle="--", alpha=0.6, label="Selected k=2")
    for k, s in zip(k_range, sil_scores):
        ax.annotate(f"{s:.3f}", (k, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs. k\n(Binary Adverse Event Features)", fontsize=13)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "cluster_silhouette")


def plot_umap(X_bin, labels, y, le):
    """Given feature matrix, cluster labels, and severity labels, save UMAP side-by-side plot."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed — skipping UMAP. pip install umap-learn")
        return

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_bin), min(UMAP_SAMPLE, len(X_bin)), replace=False)
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE,
                        n_neighbors=20, min_dist=0.15)
    X_2d = reducer.fit_transform(X_bin[idx])
    labs_s = labels[idx]
    y_s = y[idx]

    ari = saved["ari"]
    sev_labels = le.classes_
    cluster_colors = ["#1f77b4", "#ff7f0e"]
    sev_colors = {"Major": "#d62728", "Moderate": "#1f77b4", "Minor": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ci in [0, 1]:
        mask = labs_s == ci
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=cluster_colors[ci], s=4, alpha=0.35, label=f"Cluster {ci + 1}")
    axes[0].set_title("UMAP — Cluster Assignment (k=2)", fontsize=12)
    axes[0].legend(markerscale=4, fontsize=10)
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")
    axes[0].spines[["top", "right"]].set_visible(False)

    for si, sname in enumerate(sev_labels):
        mask = y_s == si
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=sev_colors[sname], s=4, alpha=0.35, label=sname)
    axes[1].set_title("UMAP — True Severity Label", fontsize=12)
    axes[1].legend(markerscale=4, fontsize=10)
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Adverse Event Space: Clusters vs. Severity (ARI = {ari:.3f})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    save_fig(fig, "cluster_umap")


def plot_ae_profile(df, labels):
    """Given the DataFrame and cluster labels, save adverse event prevalence heatmap."""
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    short_names = [c.replace("Target_Binary_", "").replace("_", " ") for c in binary_cols]

    df2 = df.copy()
    df2["Cluster"] = labels
    agg = df2.groupby("Cluster")[binary_cols].mean()
    agg.columns = short_names

    top20 = (agg.loc[0] - agg.loc[1]).abs().nlargest(20).index
    agg.index = ["Cluster 1", "Cluster 2"]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(
        agg[top20], ax=ax, cmap="Blues", annot=True, fmt=".2f",
        linewidths=0.4, annot_kws={"size": 8},
        cbar_kws={"label": "Mean prevalence"},
    )
    ax.set_title("Top 20 Differentiating Adverse Events by Cluster", fontsize=13, pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8.5)
    plt.tight_layout()
    save_fig(fig, "cluster_ae_profile")


def plot_severity_composition(df, labels):
    """Given the DataFrame and cluster labels, save severity proportion bar chart per cluster."""
    df2 = df.copy()
    df2["Cluster"] = labels
    comp = df2.groupby(["Cluster", "Severity"]).size().unstack(fill_value=0)
    comp_pct = comp.div(comp.sum(axis=1), axis=0) * 100
    comp_pct.index = ["Cluster 1", "Cluster 2"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    comp_pct[["Major", "Moderate", "Minor"]].plot(
        kind="bar", ax=ax,
        color=["#d62728", "#1f77b4", "#2ca02c"],
        edgecolor="white", width=0.5,
    )
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Proportion (%)", fontsize=11)
    ax.set_title(
        "Severity Composition Within Each Cluster\n"
        "(Near-identical proportions confirm AE clusters are severity-independent)",
        fontsize=11,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
    ax.legend(title="Severity", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    for p in ax.patches:
        h = p.get_height()
        if h > 3:
            ax.text(p.get_x() + p.get_width() / 2, h + 0.5,
                    f"{h:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    save_fig(fig, "cluster_severity_composition")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading inputs ===")
    df, X_bin, labels, saved = load_inputs(DATA_PATH, MODEL_PATH, LABELS_PATH)
    le = saved["le"]
    sil_scores = saved["sil_scores"]
    y = le.transform(df["Severity"])
    print(f"  {len(df):,} rows  |  {X_bin.shape[1]} features  |  k={len(np.unique(labels))} clusters")

    print("\n=== Generating visualizations ===")
    plot_silhouette(sil_scores)
    plot_umap(X_bin, labels, y, le)
    plot_ae_profile(df, labels)
    plot_severity_composition(df, labels)

    print(f"\nDone. All figures in {OUT_DIR}/\n")


if __name__ == "__main__":
    main()