import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

OUT_DIR = Path("data/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_all_viz():
    df = pd.read_csv('data/raw/train.csv')
    y_raw, uniques = pd.factorize(df['Severity'])
    y_bin = (y_raw > 0).astype(int)
    text_cols = [c for c in df.columns if not c.startswith('Target_') and c not in ['Pair_ID', 'Severity']]
    X_text = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
    vec = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vec.fit_transform(X_text).toarray()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'Gradient Boosting': HistGradientBoostingClassifier(max_iter=50)
    }
    
    trained = {}
    for name, model in models.items():
        model.fit(X, y_raw)
        trained[name] = (model.predict_proba(X)[:, 1:].sum(axis=1), model.predict(X), model)

    plt.figure(figsize=(7, 6))
    for name, (probs, preds, model) in trained.items():
        fpr, tpr, _ = roc_curve(y_bin, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr,tpr):.4f})")
    plt.plot([0,1],[0,1], 'k--', alpha=0.5)
    plt.legend(); plt.savefig(OUT_DIR / 'roc_curves.png'); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, (probs, preds, model)) in zip(axes, trained.items()):
        sns.heatmap(confusion_matrix(y_raw, preds), annot=True, fmt='d', ax=ax, cmap='Blues', xticklabels=uniques, yticklabels=uniques)
        ax.set_title(name)
    plt.tight_layout(); plt.savefig(OUT_DIR / 'confusion_matrices.png'); plt.close()

    rf_model = trained['Random Forest'][2]
    pd.Series(rf_model.feature_importances_, index=vec.get_feature_names_out()).sort_values().tail(12).plot(kind='barh')
    plt.title('Top Predictive Features'); plt.tight_layout(); plt.savefig(OUT_DIR / 'feature_importance.png'); plt.close()
    print(f"✔ 4 Visuals saved to {OUT_DIR}/")

if __name__ == "__main__": run_all_viz()
