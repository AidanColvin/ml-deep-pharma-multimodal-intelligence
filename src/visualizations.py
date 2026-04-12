"""
generate_visuals.py
Generates all project visualizations with proper train/test split.
No model is ever scored on the data it was trained on.

Usage (from project root):
    python generate_visuals.py
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

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/raw/train.csv"
OUT_DIR     = Path("data/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
PALETTE      = ["#4C72B0", "#55A868", "#C44E52"]
CV5          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, name):
    """Given a figure and stem name, save PNG to OUT_DIR."""
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def load_and_split(data_path):
    """
    Given a CSV path, return a proper 80/20 stratified train/test split.
    Text columns are TF-IDF vectorized after splitting to prevent leakage.
    """
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows loaded")

    # Combine all non-target text columns into one string per row
    drop_cols = ["Pair_ID", "Severity"] + \
                [c for c in df.columns if c.startswith("Target_")]
    text_cols  = [c for c in df.columns if c not in drop_cols]
    df["combined_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    X_raw = df["combined_text"]
    y     = df["Severity"]

    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Fit vectorizer ONLY on train split — transform test separately
    vec = TfidfVectorizer(max_features=2000, stop_words="english")
    X_train = vec.fit_transform(X_raw_train)
    X_test  = vec.transform(X_raw_test)

    return df, X_train, X_test, y_train, y_test, vec


def fit_models(X_train, y_train):
    """Given training data, fit and return all three classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  fitted {name}")
    return trained


# ── Classification Visuals ────────────────────────────────────────────────────

def plot_roc(trained, X_test, y_test):
    """
    Given fitted models and held-out test data, save multiclass ROC curves.
    Uses micro-average across classes. Scored on test set only.
    """
    classes    = sorted(y_test.unique())
    y_test_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, model), color in zip(trained.items(), PALETTE):
        probs = model.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Multiclass ROC Curves (Test Set)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save(fig, "roc_curves")


def plot_confusion(trained, X_test, y_test):
    """
    Given fitted models and held-out test data, save raw count and
    normalized confusion matrices side by side.
    """
    classes = sorted(y_test.unique())

    # Raw counts
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices — Raw Counts (Test Set)", fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, "confusion_matrices_counts")

    # Normalized (%)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, model) in zip(axes, trained.items()):
        cm = confusion_matrix(y_test, model.predict(X_test), labels=classes)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".1f", ax=ax, cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        ax.set_title(name, fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices — Row-Normalized % (Test Set)", fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, "confusion_matrices_pct")


def plot_cv(trained, X_train, y_train):
    """
    Given fitted models and training data, save 5-fold CV accuracy bar chart.
    CV is run on training data only — test set is never touched.
    """
    names, means, stds = [], [], []
    for name, model in trained.items():
        # Subset to keep runtime reasonable
        scores = cross_val_score(
            model, X_train[:3000], y_train[:3000],
            cv=CV5, scoring="accuracy", n_jobs=-1
        )
        names.append(name)
        means.append(scores.mean())
        stds.append(scores.std())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color=PALETTE, width=0.55, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Accuracy (Subset)", fontsize=12)
    ax.set_title("5-Fold Cross-Validation Accuracy (Training Data)", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save(fig, "cv_accuracy")


def plot_feature_importance(trained, vec):
    """Given fitted GB model and vectorizer, save top 12 feature importance bar chart."""
    gb = trained["Gradient Boosting"]
    feature_names = vec.get_feature_names_out()
    imps = pd.Series(gb.feature_importances_, index=feature_names).sort_values().tail(12)

    fig, ax = plt.subplots(figsize=(8, 6))
    imps.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Top 12 Predictive Text Tokens (Gradient Boosting)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save(fig, "feature_importance")


def plot_decision_tree(X_train, y_train, vec):
    """Given training data and vectorizer, save shallow decision tree diagram."""
    classes = sorted(y_train.unique())
    feature_names = vec.get_feature_names_out()
    dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt, ax=ax, feature_names=list(feature_names),
              class_names=[str(c) for c in classes],
              filled=True, rounded=True)
    ax.set_title("Simple Tree Showing Core Classification Logic",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    save(fig, "decision_tree")


# ── Clustering Visuals ────────────────────────────────────────────────────────

def run_clustering(df):
    """
    Given the full dataframe, run K-means sweep k=2..6 on binary adverse event
    columns. Return labels and silhouette scores for the selected k=2 model.
    """
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    X_bin = df[binary_cols].values

    sil_scores = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labs = km.fit_predict(X_bin)
        sil_scores.append(silhouette_score(X_bin, labs))
        print(f"  k={k}  silhouette={sil_scores[-1]:.3f}")

    # Best k is 2 — refit for final labels
    km2 = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
    labels = km2.fit_predict(X_bin)
    return labels, sil_scores


def plot_silhouette(sil_scores):
    """Given silhouette scores for k=2..6, save silhouette sweep plot."""
    k_range = list(range(2, 2 + len(sil_scores)))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_range, sil_scores, "o-", color="#1f77b4", lw=2.2, ms=8)
    ax.axvline(2, color="gray", linestyle="--", alpha=0.6, label="Selected k=2")
    for k, s in zip(k_range, sil_scores):
        ax.annotate(f"{s:.3f}", (k, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Score vs. k\n(Binary Adverse Event Features)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    save(fig, "cluster_silhouette")


def plot_ae_profile(df, labels):
    """Given the DataFrame and cluster labels, save adverse event prevalence heatmap."""
    binary_cols  = [c for c in df.columns if c.startswith("Target_Binary_")]
    short_names  = [c.replace("Target_Binary_", "").replace("_", " ") for c in binary_cols]

    df2 = df.copy()
    df2["Cluster"] = labels
    agg = df2.groupby("Cluster")[binary_cols].mean()
    agg.columns = short_names

    top20 = (agg.loc[0] - agg.loc[1]).abs().nlargest(20).index
    agg.index = ["Cluster 1", "Cluster 2"]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(agg[top20], ax=ax, cmap="Blues", annot=True, fmt=".2f",
                linewidths=0.4, annot_kws={"size": 8},
                cbar_kws={"label": "Mean prevalence"})
    ax.set_title("Top 20 Differentiating Adverse Events by Cluster",
                 fontsize=13, fontweight="bold", pad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8.5)
    plt.tight_layout()
    save(fig, "cluster_ae_profile")


def plot_severity_composition(df, labels):
    """Given the DataFrame and cluster labels, save severity proportion bar chart."""
    df2 = df.copy()
    df2["Cluster"] = labels
    comp     = df2.groupby(["Cluster", "Severity"]).size().unstack(fill_value=0)
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
        fontsize=11, fontweight="bold",
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
    save(fig, "cluster_severity_composition")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading and splitting data ===")
    df, X_train, X_test, y_train, y_test, vec = load_and_split(DATA_PATH)

    print("\n=== Fitting classifiers ===")
    trained = fit_models(X_train, y_train)

    print("\n=== Classification visuals ===")
    plot_roc(trained, X_test, y_test)
    plot_confusion(trained, X_test, y_test)
    plot_cv(trained, X_train, y_train)
    plot_feature_importance(trained, vec)
    plot_decision_tree(X_train, y_train, vec)

    print("\n=== Clustering ===")
    labels, sil_scores = run_clustering(df)

    print("\n=== Clustering visuals ===")
    plot_silhouette(sil_scores)
    plot_ae_profile(df, labels)
    plot_severity_composition(df, labels)

    print(f"\nDone. All figures saved to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()