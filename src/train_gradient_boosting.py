import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
try:
    from src.train_helper import load_and_vectorize
except ImportError:
    from train_helper import load_and_vectorize

def train():
    train_df, test_df, X_train, X_test, bin_cols, prr_cols = load_and_vectorize()
    
    print("Converting to Dense for HistGB...")
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    pbar = tqdm(total=3, desc="Training Turbo Gradient Boosting")
    
    sev_m = HistGradientBoostingClassifier(max_iter=30).fit(X_train_dense, train_df['Severity'])
    pbar.update(1)
    
    bin_m = MultiOutputClassifier(
        HistGradientBoostingClassifier(max_iter=10), n_jobs=-1
    ).fit(X_train_dense, train_df[bin_cols])
    pbar.update(1)
    
    prr_m = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=10), n_jobs=-1
    ).fit(X_train_dense, train_df[prr_cols])
    pbar.update(1)
    pbar.close()
    
    sub = pd.DataFrame({'Pair_ID': test_df['Pair_ID'], 'Severity': sev_m.predict(X_test_dense)})
    sub = pd.concat([sub, pd.DataFrame(bin_m.predict(X_test_dense), columns=bin_cols)], axis=1)
    sub = pd.concat([sub, pd.DataFrame(prr_m.predict(X_test_dense), columns=prr_cols)], axis=1)
    
    sub.to_csv('data/processed/submission_gb.csv', index=False)
    print("✔ Parallel GB Complete.")

if __name__ == "__main__": train()
