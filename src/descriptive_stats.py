"""
descriptive_stats.py
Generates descriptive statistics tables and visuals for the methods section.

Usage (from project root):
    python3 src/descriptive_stats.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/raw/train.csv"
VIS_DIR   = Path("data/visualizations")
TAB_DIR   = Path("data/tables")
VIS_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_fig(fig, name):
    """Given a figure and stem, save PNG to VIS_DIR."""
    path = VIS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def save_table(df, name):
    """Given a dataframe and stem, save CSV to TAB_DIR."""
    path = TAB_DIR / f"{name}.csv"
    df.to_csv(path)
    print(f"  saved → {path}")


# ── Load ──────────────────────────────────────────────────────────────────────

def load(data_path):
    """Given a CSV path, return the full dataframe."""
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows  |  {df.shape[1]} columns")
    return df


# ── Tables ────────────────────────────────────────────────────────────────────

def table_class_distribution(df):
    """Given the dataframe, save severity class distribution table."""
    counts = df["Severity"].value_counts()
    pct    = (df["Severity"].value_counts(normalize=True) * 100).round(1)
    out = pd.DataFrame({"Count": counts, "Percent (%)": pct})
    out.index.name = "Severity"
    out["Imbalance Ratio"] = (counts / counts.min()).round(1)
    print("\nClass Distribution:")
    print(out.to_string())
    save_table(out, "class_distribution")
    return out


def table_ae_burden(df):
    """Given the dataframe, save adverse event burden summary table."""
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    row_sums    = df[binary_cols].sum(axis=1)
    out = pd.DataFrame({
        "Statistic": ["Mean AEs per pair", "Median AEs per pair",
                      "Min AEs", "Max AEs",
                      "Pairs with 0 AEs (%)", "Pairs with 40+ AEs (%)"],
        "Value": [
            round(row_sums.mean(), 2),
            round(row_sums.median(), 2),
            int(row_sums.min()),
            int(row_sums.max()),
            round((row_sums == 0).mean() * 100, 1),
            round((row_sums >= 40).mean() * 100, 1),
        ]
    }).set_index("Statistic")
    print("\nAdverse Event Burden:")
    print(out.to_string())
    save_table(out, "ae_burden")
    return out


def table_prr_stats(df):
    """Given the dataframe, save PRR distribution summary table."""
    prr_cols   = [c for c in df.columns if c.startswith("Target_PRR_")]
    prr_vals   = df[prr_cols]
    nonzero    = prr_vals[prr_vals > 0].stack()
    out = pd.DataFrame({
        "Statistic": [
            "Total PRR entries",
            "Non-zero PRR entries (%)",
            "Mean PRR (non-zero)",
            "Median PRR (non-zero)",
            "Max PRR",
            "PRR cols",
        ],
        "Value": [
            prr_vals.size,
            round((prr_vals > 0).mean().mean() * 100, 1),
            round(nonzero.mean(), 3),
            round(nonzero.median(), 3),
            round(prr_vals.values.max(), 3),
            len(prr_cols),
        ]
    }).set_index("Statistic")
    print("\nPRR Statistics:")
    print(out.to_string())
    save_table(out, "prr_stats")
    return out


def table_missingness(df):
    """Given the dataframe, save top missing value rates table."""
    miss = df.isnull().mean().mul(100).round(1)
    miss = miss[miss > 0].sort_values(ascending=False)
    out  = miss.rename("Missing (%)").to_frame()
    print("\nMissingness (columns with any missing):")
    print(out.head(10).to_string())
    save_table(out, "missingness")
    return out


# ── Visuals ───────────────────────────────────────────────────────────────────

def plot_class_distribution(df):
    """Given the dataframe, save severity class distribution bar chart."""
    counts = df["Severity"].value_counts().reindex(["Major", "Moderate", "Minor"])
    pct    = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values,
                  color=["#d62728", "#1f77b4", "#2ca02c"], edgecolor="white", width=0.5)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"{p:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Severity Class", fontsize=12)
    ax.set_title("Severity Class Distribution (Training Data)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "desc_class_distribution")


def plot_ae_burden(df):
    """Given the dataframe, save adverse event burden histogram."""
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    row_sums    = df[binary_cols].sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(row_sums, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(row_sums.mean(), color="#C44E52", lw=2,
               linestyle="--", label=f"Mean = {row_sums.mean():.1f}")
    ax.axvline(row_sums.median(), color="#dd8452", lw=2,
               linestyle=":", label=f"Median = {row_sums.median():.0f}")
    ax.set_xlabel("Number of Adverse Events per Drug Pair", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Adverse Event Burden per Drug Pair",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "desc_ae_burden")


def plot_ae_prevalence(df):
    """Given the dataframe, save per-adverse-event prevalence bar chart."""
    binary_cols  = [c for c in df.columns if c.startswith("Target_Binary_")]
    short_names  = [c.replace("Target_Binary_", "").replace("_", " ") for c in binary_cols]
    prevalence   = df[binary_cols].mean().mul(100)
    prevalence.index = short_names
    prevalence   = prevalence.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 12))
    prevalence.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Prevalence (%)", fontsize=12)
    ax.set_title("Prevalence of Each Adverse Event\n(% of Drug Pairs)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "desc_ae_prevalence")


def plot_prr_distribution(df):
    """Given the dataframe, save PRR value distribution for non-zero entries."""
    prr_cols = [c for c in df.columns if c.startswith("Target_PRR_")]
    nonzero  = df[prr_cols][df[prr_cols] > 0].stack()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(nonzero.clip(upper=nonzero.quantile(0.99)),
            bins=40, color="#55A868", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Proportional Reporting Ratio (PRR)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Non-Zero PRR Values (99th Percentile Clipped)",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "desc_prr_distribution")


def plot_missingness(df):
    """Given the dataframe, save missingness bar chart for columns with missing values."""
    miss = df.isnull().mean().mul(100)
    miss = miss[miss > 0].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, len(miss) * 0.4)))
    miss.plot(kind="barh", ax=ax, color="#C44E52", edgecolor="white")
    ax.set_xlabel("Missing (%)", fontsize=12)
    ax.set_title("Feature Missingness Rate", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "desc_missingness")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading data ===")
    df = load(DATA_PATH)

    print("\n=== Generating tables ===")
    table_class_distribution(df)
    table_ae_burden(df)
    table_prr_stats(df)
    table_missingness(df)

    print("\n=== Generating visuals ===")
    plot_class_distribution(df)
    plot_ae_burden(df)
    plot_ae_prevalence(df)
    plot_prr_distribution(df)
    plot_missingness(df)

    print(f"\nDone.\n  Tables  → {TAB_DIR}/\n  Visuals → {VIS_DIR}/\n")


if __name__ == "__main__":
    main()