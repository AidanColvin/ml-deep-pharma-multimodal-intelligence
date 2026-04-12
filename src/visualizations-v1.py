import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize

# --- Path Setup ---
FIGURES_DIR = Path("data/visualizations")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PALETTE = ['#4C72B0', '#55A868', '#C44E52']

print("Loading raw data...")
# Load directly from your raw folder
df = pd.read_csv('data/raw/train.csv')

# --- On-the-Fly Preprocessing ---
print("Processing text and generating TF-IDF features...")
# Grab all text columns, ignore the target label
text_cols = df.select_dtypes(include=['object']).columns.drop('Severity', errors='ignore')

# Combine all text fields into one giant string per row, replacing NaN with empty strings
df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

# Split into train/test (80/20) so we can evaluate it
X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    df['combined_text'], df['Severity'], test_size=0.2, random_state=42, stratify=df['Severity']
)

# Run TF-IDF (limiting to 2000 features to keep your Mac running fast)
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train = vectorizer.fit_transform(X_raw_train)
X_test = vectorizer.transform(X_raw_test)
feature_names = vectorizer.get_feature_names_out()

# Identify the exact class labels found in your data (e.g., Major, Moderate, Minor)
CLASSES = sorted(list(y_train.unique()))
y_test_bin = label_binarize(y_test, classes=CLASSES)

# --- Define Models ---
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
}

print("Running models (this will take a minute or two)...")
trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    trained[name] = (model, probs, preds)
    print(f"  ✔ {name} ready")

# --- Fig 1: Feature Importance ---
print("Saving Feature Importance...")
gb = trained['Gradient Boosting'][0]
imps = pd.Series(gb.feature_importances_, index=feature_names).sort_values().tail(12)
plt.figure(figsize=(8, 6))
imps.plot(kind='barh', color='#4C72B0', edgecolor='white')
plt.title('Top 12 Predictive Text Tokens (Gradient Boosting)', fontweight='bold')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'feature_importance_final.png')
plt.close()

# --- Fig 2: Decision Tree logic ---
print("Saving Tree Visual...")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=list(feature_names), class_names=[str(c) for c in CLASSES], filled=True, rounded=True)
plt.title("Simple Tree showing core classification logic", fontsize=15, fontweight='bold')
plt.savefig(FIGURES_DIR / 'tree_visual_final.png', dpi=300)
plt.close()

# --- Fig 3: ROC Curves ---
print("Saving ROC Curves...")
plt.figure(figsize=(8, 6))
for (name, (model, probs, preds)), color in zip(trained.items(), PALETTE[:len(models)]):
    # Micro-average ROC for multi-class comparison
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig(FIGURES_DIR / 'roc_curves_final.png')
plt.close()

# --- Fig 4: 5-Fold Cross Validation ---
print("Saving CV Results...")
cv_means, cv_stds = [], []
for name, (model, probs, preds) in trained.items():
    # Doing CV on a smaller subset so you aren't waiting forever
    scores = cross_val_score(model, X_train[:2000], y_train[:2000], cv=CV5, scoring='accuracy', n_jobs=-1)
    cv_means.append(scores.mean())
    cv_stds.append(scores.std())

plt.figure(figsize=(8, 5))
x = np.arange(len(models))
plt.bar(x, cv_means, yerr=cv_stds, capsize=5, color=PALETTE[:len(models)], width=0.6)
plt.xticks(x, list(models.keys()))
plt.ylabel('Accuracy (Subset)')
plt.title('5-Fold Cross-Validation Accuracy', fontweight='bold')
plt.savefig(FIGURES_DIR / 'cv_comparison_final.png')
plt.close()

# --- Fig 5: Confusion Matrices ---
print("Saving Confusion Matrices...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, (model, probs, preds)) in zip(axes, trained.items()):
    cm = confusion_matrix(y_test, preds, labels=CLASSES)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', ax=ax, cmap='Blues', cbar=False,
                xticklabels=CLASSES, yticklabels=CLASSES)
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrices_final.png')
plt.close()

print(f"\nAll visuals saved to {FIGURES_DIR}")