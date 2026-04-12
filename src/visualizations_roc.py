import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

FIGURES_DIR = Path("data/visualizations")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def run_zoomed_roc():
    df = pd.read_csv('data/raw/train.csv')
    text_cols = [c for c in df.columns if not c.startswith('Target_') and c not in ['Pair_ID', 'Severity']]
    X_text = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
    
    # Fix: Convert Severity to numeric
    y_numeric = pd.to_numeric(df['Severity'], errors='coerce').fillna(0)
    y_bin = (y_numeric > 0).astype(int)

    vec = TfidfVectorizer(max_features=1000)
    X = vec.fit_transform(X_text)
    
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000), '#4C72B0', '--'),
        'Random Forest': (RandomForestClassifier(n_estimators=50), '#55A868', '-.')
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for name, (model, color, ls) in models.items():
        model.fit(X, y_bin)
        fpr, tpr, _ = roc_curve(y_bin, model.predict_proba(X)[:, 1])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, linestyle=ls, lw=2, label=f"{name} (AUC={roc_auc:.4f})")
        axes[1].plot(fpr, tpr, color=color, linestyle=ls, lw=3, label=f"{name}")

    axes[0].plot([0,1],[0,1],'k--', alpha=0.4)
    axes[0].set_title('Full ROC Range', fontweight='bold')
    axes[0].legend()

    axes[1].set_xlim([0, 0.25]); axes[1].set_ylim([0.7, 1.01])
    axes[1].set_title('Zoomed (FPR 0–0.25)', fontweight='bold')
    axes[1].legend()
    
    plt.savefig(FIGURES_DIR / 'roc_analysis_zoomed.png', dpi=150)
    plt.close()
    print(f"  ✔ Zoomed ROC saved to {FIGURES_DIR}/")

if __name__ == "__main__":
    run_zoomed_roc()
