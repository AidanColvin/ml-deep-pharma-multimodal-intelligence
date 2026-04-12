"""
train_multitask.py
Trains binary side effect classifiers and PRR regressors.
Saves results tables to data/tables/ and visualizations to data/visualizations/.

Usage (from project root):
    python3 src/train_multitask.py
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

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score, roc_auc_score,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/raw/train.csv"
TAB_DIR   = Path("data/tables")
VIS_DIR   = Path("data/visualizations")
TAB_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# ── Load and preprocess ───────────────────────────────────────────────────────

def load_and_split(data_path):
    """Given CSV path, return train/test splits for text, binary targets, and PRR targets."""
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} rows loaded")

    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    prr_cols    = [c for c in df.columns if c.startswith("Target_PRR_")]
    drop_cols   = ["Pair_ID", "Severity"] + binary_cols + prr_cols
    text_cols   = [c for c in df.columns if c not in drop_cols]

    df["combined_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)

    X_raw = df["combined_text"]
    y_bin = df[binary_cols]
    y_prr = df[prr_cols]

    (X_tr, X_te,
     yb_tr, yb_te,
     yp_tr, yp_te) = train_test_split(
        X_raw, y_bin, y_prr,
        test_size=0.2, random_state=RANDOM_STATE
    )

    vec    = TfidfVectorizer(max_features=2000, stop_words="english")
    X_train = vec.fit_transform(X_tr)
    X_test  = vec.transform(X_te)

    short_bin = [c.replace("Target_Binary_", "").replace("_", " ") for c in binary_cols]
    short_prr = [c.replace("Target_PRR_",    "").replace("_", " ") for c in prr_cols]

    return X_train, X_test, yb_tr, yb_te, yp_tr, yp_te, short_bin, short_prr


# ── Binary side effect prediction ─────────────────────────────────────────────

def train_binary(X_train, X_test, yb_train, yb_test, short_names):
    """
    Given train/test splits, fit multi-label classifier and return per-label metrics.
    Uses Logistic Regression wrapped in MultiOutputClassifier.
    """
    print("\n=== Binary Side Effect Prediction ===")
    clf = MultiOutputClassifier(
        LogisticRegression(max_iter=500, C=1.0, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    clf.fit(X_train, yb_train)
    preds = clf.predict(X_test)
    probs = np.array([e.predict_proba(X_test)[:, 1] for e in clf.estimators_]).T

    # Per-label metrics
    rows = []
    for i, name in enumerate(short_names):
        f1  = f1_score(yb_test.iloc[:, i], preds[:, i], zero_division=0)
        try:
            auc = roc_auc_score(yb_test.iloc[:, i], probs[:, i])
        except Exception:
            auc = float("nan")
        rows.append({"Adverse Event": name, "F1": round(f1, 3), "AUC": round(auc, 3)})

    results = pd.DataFrame(rows).set_index("Adverse Event")

    # Global micro F1
    micro_f1 = f1_score(yb_test.values, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(yb_test.values, preds, average="macro", zero_division=0)
    print(f"  Micro F1: {micro_f1:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Mean per-label AUC: {results['AUC'].mean():.4f}")

    # Summary table
    summary = pd.DataFrame({
        "Metric": ["Micro F1", "Macro F1", "Mean per-label AUC"],
        "Value":  [round(micro_f1, 4), round(macro_f1, 4), round(results["AUC"].mean(), 4)]
    }).set_index("Metric")
    summary.to_csv(TAB_DIR / "binary_prediction_summary.csv")
    results.to_csv(TAB_DIR / "binary_prediction_per_label.csv")
    print(f"  saved → data/tables/binary_prediction_summary.csv")
    print(f"  saved → data/tables/binary_prediction_per_label.csv")

    return results, micro_f1, macro_f1


# ── PRR regression ────────────────────────────────────────────────────────────

def train_prr(X_train, X_test, yp_train, yp_test, short_names):
    """
    Given train/test splits, fit multi-output Ridge regressor and return per-label RMSE.
    Only evaluates RMSE on rows where ground truth PRR > 0 (masked evaluation).
    """
    print("\n=== PRR Regression ===")
    reg = MultiOutputRegressor(
        Ridge(alpha=1.0),
        n_jobs=-1
    )
    reg.fit(X_train, yp_train)
    preds = reg.predict(X_test)
    preds = np.clip(preds, 0, None)  # PRR cannot be negative

    rows = []
    for i, name in enumerate(short_names):
        true = yp_test.iloc[:, i].values
        pred = preds[:, i]
        mask = true > 0
        if mask.sum() == 0:
            rmse = float("nan")
            inv_rmse = float("nan")
        else:
            rmse = np.sqrt(mean_squared_error(true[mask], pred[mask]))
            inv_rmse = round(1 / (1 + rmse), 4)
        rows.append({
            "Adverse Event": name,
            "RMSE (masked)": round(rmse, 3),
            "Inverse RMSE":  inv_rmse
        })

    results = pd.DataFrame(rows).set_index("Adverse Event")

    mean_inv_rmse = results["Inverse RMSE"].mean()
    mean_rmse     = results["RMSE (masked)"].mean()
    print(f"  Mean masked RMSE: {mean_rmse:.4f}")
    print(f"  Mean inverse RMSE (score): {mean_inv_rmse:.4f}")

    summary = pd.DataFrame({
        "Metric": ["Mean Masked RMSE", "Mean Inverse RMSE (score component)"],
        "Value":  [round(mean_rmse, 4), round(mean_inv_rmse, 4)]
    }).set_index("Metric")
    summary.to_csv(TAB_DIR / "prr_regression_summary.csv")
    results.to_csv(TAB_DIR / "prr_regression_per_label.csv")
    print(f"  saved → data/tables/prr_regression_summary.csv")
    print(f"  saved → data/tables/prr_regression_per_label.csv")

    return results, mean_inv_rmse


# ── Visuals ───────────────────────────────────────────────────────────────────

def plot_binary_f1(results):
    """Given per-label binary results, save F1 bar chart sorted by score."""
    sorted_r = results.sort_values("F1", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(sorted_r.index, sorted_r["F1"], color="#4C72B0", edgecolor="white")
    ax.axvline(sorted_r["F1"].mean(), color="#C44E52", lw=2,
               linestyle="--", label=f"Mean F1 = {sorted_r['F1'].mean():.3f}")
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Label F1 Score — Binary Side Effect Prediction",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = VIS_DIR / "binary_f1_per_label.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def plot_prr_inv_rmse(results):
    """Given per-label PRR results, save inverse RMSE bar chart sorted by score."""
    sorted_r = results.sort_values("Inverse RMSE", ascending=True).dropna()

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(sorted_r.index, sorted_r["Inverse RMSE"], color="#55A868", edgecolor="white")
    ax.axvline(sorted_r["Inverse RMSE"].mean(), color="#C44E52", lw=2,
               linestyle="--", label=f"Mean = {sorted_r['Inverse RMSE'].mean():.3f}")
    ax.set_xlabel("Inverse RMSE (higher = better)", fontsize=12)
    ax.set_title("Per-Label Inverse RMSE — PRR Regression (Masked)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = VIS_DIR / "prr_inv_rmse_per_label.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading data ===")
    (X_train, X_test,
     yb_train, yb_test,
     yp_train, yp_test,
     short_bin, short_prr) = load_and_split(DATA_PATH)

    bin_results, micro_f1, macro_f1 = train_binary(
        X_train, X_test, yb_train, yb_test, short_bin)

    prr_results, mean_inv_rmse = train_prr(
        X_train, X_test, yp_train, yp_test, short_prr)

    print("\n=== Generating visuals ===")
    plot_binary_f1(bin_results)
    plot_prr_inv_rmse(prr_results)

    print("\n=== Composite score estimate ===")
    # Approximate Hardcore Clinical Score using your severity AUC as proxy for Macro F1
    severity_macro_f1 = 0.5666  # from existing results
    score = (0.40 * severity_macro_f1) + (0.30 * micro_f1) + (0.30 * mean_inv_rmse)
    print(f"  Severity Macro F1:     {severity_macro_f1:.4f} (weight 0.40)")
    print(f"  Side Effect Micro F1:  {micro_f1:.4f} (weight 0.30)")
    print(f"  PRR Mean Inv RMSE:     {mean_inv_rmse:.4f} (weight 0.30)")
    print(f"  Estimated composite:   {score:.4f}")

    pd.DataFrame({
        "Component": ["Severity Macro F1 (×0.40)",
                      "Side Effect Micro F1 (×0.30)",
                      "PRR Inv RMSE (×0.30)",
                      "Estimated Composite Score"],
        "Value": [round(severity_macro_f1, 4),
                  round(micro_f1, 4),
                  round(mean_inv_rmse, 4),
                  round(score, 4)]
    }).set_index("Component").to_csv(TAB_DIR / "composite_score_estimate.csv")
    print(f"  saved → data/tables/composite_score_estimate.csv")

    print(f"\nDone. Tables → {TAB_DIR}/  Visuals → {VIS_DIR}/\n")


if __name__ == "__main__":
    main()