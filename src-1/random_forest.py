import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def load_data():
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    binary_cols = [c for c in train.columns if c.startswith('Target_Binary_')]
    prr_cols = [c for c in train.columns if c.startswith('Target_PRR_')]
    
    def get_text(df):
        cols = [c for c in df.columns if not c.startswith('Target_') and c not in ['Pair_ID', 'Severity']]
        return df[cols].fillna('').astype(str).agg(' '.join, axis=1)

    return train, test, get_text(train), get_text(test), binary_cols, prr_cols

def main():
    train_df, test_df, X_all, X_test_final, binary_cols, prr_cols = load_data()
    X_train, X_val, y_sev_train, y_sev_val, y_bin_train, y_bin_val = train_test_split(
        X_all, train_df['Severity'], train_df[binary_cols], test_size=0.2, random_state=42
    )

    rf_params = {'n_estimators': 50, 'n_jobs': -1, 'random_state': 42}
    sev_model = Pipeline([('tfidf', TfidfVectorizer(max_features=5000)), ('rf', RandomForestClassifier(**rf_params))])
    bin_model = Pipeline([('tfidf', TfidfVectorizer(max_features=5000)), ('rf', MultiOutputClassifier(RandomForestClassifier(**rf_params)))])
    prr_model = Pipeline([('tfidf', TfidfVectorizer(max_features=5000)), ('rf', MultiOutputRegressor(RandomForestRegressor(**rf_params)))])

    sev_model.fit(X_train, y_sev_train)
    bin_model.fit(X_train, y_bin_train)

    sev_acc = accuracy_score(y_sev_val, sev_model.predict(X_val))
    bin_acc = (bin_model.predict(X_val) == y_bin_val.values).mean()
    print(f"RESULT_RF_SEV: {sev_acc:.4f}")
    print(f"RESULT_RF_BIN: {bin_acc:.4f}")

    sev_model.fit(X_all, train_df['Severity'])
    bin_model.fit(X_all, train_df[binary_cols])
    prr_model.fit(X_all, train_df[prr_cols])

    os.makedirs('data/processed', exist_ok=True)
    sub = pd.DataFrame({'Pair_ID': test_df['Pair_ID'], 'Severity': sev_model.predict(X_test_final)})
    df_bin = pd.DataFrame(bin_model.predict(X_test_final), columns=binary_cols)
    df_prr = pd.DataFrame(prr_model.predict(X_test_final), columns=prr_cols)
    sub = pd.concat([sub, df_bin, df_prr], axis=1)
    sub.to_csv('data/processed/submission_rf.csv', index=False)

if __name__ == "__main__":
    main()