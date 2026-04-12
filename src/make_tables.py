import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Make the tables folder
TABLES_DIR = Path("data/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("Loading raw data...")
df = pd.read_csv('data/raw/train.csv')

# 2. Make the Missing Data Table
print("Making missing data table...")
# Count blank spots in text columns
text_cols = df.select_dtypes(include=['object']).columns.drop('Severity', errors='ignore')
missing_counts = df[text_cols].isnull().sum()
missing_df = pd.DataFrame({
    'Feature': missing_counts.index,
    'Missing Values': missing_counts.values
})
# Sort so the worst ones are at the top
missing_df = missing_df.sort_values(by='Missing Values', ascending=False)
missing_df.to_csv(TABLES_DIR / 'missing_data.csv', index=False)

# 3. Setup text and models
print("Cleaning text and setting up models...")
df['combined_text'] = df[text_cols].fillna('').agg(' '.join, axis=1)

X_raw_train, X_raw_test, y_train, y_test = train_test_split(
    df['combined_text'], df['Severity'], test_size=0.2, random_state=42, stratify=df['Severity']
)

# Keep it fast with 2000 words
vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train = vectorizer.fit_transform(X_raw_train)
X_test = vectorizer.transform(X_raw_test)
feature_names = vectorizer.get_feature_names_out()
classes = sorted(list(y_train.unique()))

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
}

# 4. Make the Performance Table
print("Running models to get scores...")
results = []
trained_gb = None

for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)
    
    # Save the GB model for the feature table later
    if name == 'Gradient Boosting':
        trained_gb = model

    # Get overall AUC
    auc_score = roc_auc_score(y_test, probs, multi_class='ovr')
    
    # Get recall for each class
    recalls = recall_score(y_test, preds, average=None, labels=classes)
    
    # Put it in a row
    row = {
        'Model': name,
        'AUC': round(auc_score, 4)
    }
    # Add a column for each class recall
    for i, class_name in enumerate(classes):
        row[f'{class_name} Recall'] = round(recalls[i], 3)
        
    results.append(row)

perf_df = pd.DataFrame(results)
perf_df.to_csv(TABLES_DIR / 'model_performance.csv', index=False)

# 5. Make the Top Features Table
print("Making top features table...")
imps = trained_gb.feature_importances_
# Sort and get the top 12
top_indices = np.argsort(imps)[-12:][::-1]

feature_rows = []
for rank, idx in enumerate(top_indices, 1):
    feature_rows.append({
        'Rank': rank,
        'Feature Token': feature_names[idx],
        'Importance': round(imps[idx], 4)
    })

features_df = pd.DataFrame(feature_rows)
features_df.to_csv(TABLES_DIR / 'top_features.csv', index=False)

print(f"\nAll tables saved to {TABLES_DIR}")