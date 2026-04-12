import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize

TAB_DIR = Path("data/tables")
TAB_DIR.mkdir(exist_ok=True)

df = pd.read_csv("data/raw/train.csv")
drop_cols = ["Pair_ID","Severity"] + [c for c in df.columns if c.startswith("Target_")]
text_cols = [c for c in df.columns if c not in drop_cols]
df["combined_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    df["combined_text"], df["Severity"], test_size=0.2, random_state=42, stratify=df["Severity"])
vec = TfidfVectorizer(max_features=2000, stop_words="english")
X_train = vec.fit_transform(X_raw_train)
X_test  = vec.transform(X_raw_test)
classes = sorted(y_test.unique())
y_test_bin = label_binarize(y_test, classes=classes)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
}
trained = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained[name] = model
    print(f"  fitted {name}")

# ROC AUC table
auc_rows = []
for name, model in trained.items():
    probs = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
    auc_rows.append({"Model": name, "AUC": round(auc(fpr, tpr), 4)})
pd.DataFrame(auc_rows).set_index("Model").to_csv(TAB_DIR / "roc_auc.csv")
print("  saved → data/tables/roc_auc.csv")

# CV accuracy table
CV5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for name, model in trained.items():
    scores = cross_val_score(model, X_train[:3000], y_train[:3000], cv=CV5, scoring="accuracy", n_jobs=-1)
    cv_rows.append({"Model": name, "CV_Mean": round(scores.mean(),4), "CV_Std": round(scores.std(),4)})
pd.DataFrame(cv_rows).set_index("Model").to_csv(TAB_DIR / "cv_accuracy.csv")
print("  saved → data/tables/cv_accuracy.csv")

# Confusion matrices
for name, model in trained.items():
    cm = confusion_matrix(y_test, model.predict(X_test), labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        TAB_DIR / f"confusion_{name.lower().replace(' ','_')}.csv")
    print(f"  saved → data/tables/confusion_{name.lower().replace(' ','_')}.csv")

# Model performance summary
rows = []
for name, model in trained.items():
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
    from sklearn.metrics import f1_score, accuracy_score
    rows.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Macro_F1": round(f1_score(y_test, preds, average="macro"), 4),
        "AUC": round(auc(fpr, tpr), 4)
    })
pd.DataFrame(rows).set_index("Model").to_csv(TAB_DIR / "model_performance_summary.csv")
print("  saved → data/tables/model_performance_summary.csv")
print("\nDone.")
