"""
Clustering model — binary adverse event features.
Runs KMeans on the 50 Target_Binary columns, saves labels to data/processed/.

Usage (from project root):
    python src/train_clustering.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = "data/raw/train.csv"
OUT_DIR     = "data/processed"
MODEL_OUT   = os.path.join(OUT_DIR, "clustering_model.pkl")
LABELS_OUT  = os.path.join(OUT_DIR, "cluster_labels.npy")
K           = 2
RANDOM_STATE = 42
SAMPLE_SIZE  = 3000  # used for silhouette scoring only


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_binary_features(path):
    """Given a CSV path, return X (binary AE matrix), y (encoded severity), and the encoder."""
    df = pd.read_csv(path)
    binary_cols = [c for c in df.columns if c.startswith("Target_Binary_")]
    X = df[binary_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df["Severity"])
    return X, y, le, df


def sweep_silhouette(X, k_range, random_state):
    """Given feature matrix X and a range of k values, return list of silhouette scores."""
    scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labs = km.fit_predict(X)
        score = silhouette_score(X, labs, sample_size=SAMPLE_SIZE)
        scores.append(score)
        print(f"  k={k}  silhouette={score:.4f}")
    return scores


def fit_kmeans(X, k, random_state):
    """Given feature matrix X and k, return fitted KMeans model and cluster labels."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def evaluate(y_true, labels, X):
    """Given true labels and cluster labels, print ARI and silhouette."""
    ari = adjusted_rand_score(y_true, labels)
    sil = silhouette_score(X, labels, sample_size=SAMPLE_SIZE)
    print(f"\n  Final k={K} — silhouette={sil:.4f}  ARI={ari:.4f}")
    print(f"  Interpretation: clusters are {'well-separated' if sil > 0.5 else 'weakly separated'}, "
          f"{'align with severity' if ari > 0.1 else 'do not align with severity'}")
    return ari, sil


def save_results(model, labels, le, sil_scores, ari, sil):
    """Given model and outputs, save to data/processed/."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"model": model, "le": le,
                     "sil_scores": sil_scores, "ari": ari, "sil": sil}, f)
    np.save(LABELS_OUT, labels)
    print(f"\n  model  → {MODEL_OUT}")
    print(f"  labels → {LABELS_OUT}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Loading data ===")
    X, y, le, df = load_binary_features(DATA_PATH)
    print(f"  {X.shape[0]:,} rows  |  {X.shape[1]} binary AE features")
    print(f"  Severity classes: {list(le.classes_)}")

    print("\n=== Silhouette sweep k=2..6 ===")
    sil_scores = sweep_silhouette(X, range(2, 7), RANDOM_STATE)

    best_k = int(np.argmax(sil_scores)) + 2
    print(f"\n  Best k by silhouette: k={best_k}  (using k={K} per research design)")

    print(f"\n=== Fitting KMeans k={K} ===")
    model, labels = fit_kmeans(X, K, RANDOM_STATE)

    print("\n=== Evaluation ===")
    ari, sil = evaluate(y, labels, X)

    print("\n=== Saving ===")
    save_results(model, labels, le, sil_scores, ari, sil)

    print("\nDone.\n")


if __name__ == "__main__":
    main()