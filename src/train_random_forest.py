import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
try:
    from src.train_helper import load_and_vectorize
except ImportError:
    from train_helper import load_and_vectorize

def train():
    train_df, test_df, X_train, X_test, bin_cols, prr_cols = load_and_vectorize()
    pbar = tqdm(total=3, desc="Training RF")
    sev_m = RandomForestClassifier(n_estimators=100, n_jobs=-1).fit(X_train, train_df['Severity'])
    pbar.update(1)
    bin_m = MultiOutputClassifier(RandomForestClassifier(n_estimators=10, n_jobs=-1), n_jobs=-1).fit(X_train, train_df[bin_cols])
    pbar.update(1)
    prr_m = MultiOutputRegressor(Ridge()).fit(X_train, train_df[prr_cols])
    pbar.update(1)
    pbar.close()
    sub = pd.DataFrame({'Pair_ID': test_df['Pair_ID'], 'Severity': sev_m.predict(X_test)})
    sub = pd.concat([sub, pd.DataFrame(bin_m.predict(X_test), columns=bin_cols)], axis=1)
    sub = pd.concat([sub, pd.DataFrame(prr_m.predict(X_test), columns=prr_cols)], axis=1)
    sub.to_csv('data/processed/submission_rf.csv', index=False)
    print("✔ RF Complete.")
if __name__ == "__main__": train()
