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
    
    # HistGradientBoosting requires dense input but handles it much faster internally
    # We convert to dense once here to avoid repeating it 100 times
    print("Converting features for HistGB...")
    X_train_dense = X_train.toarray()
    X_test_dense = X_test.toarray()

    pbar = tqdm(total=3, desc="Training Fast Gradient Boosting")
    
    # 1. Severity
    sev_m = HistGradientBoostingClassifier(max_iter=50).fit(X_train_dense, train_df['Severity'])
    pbar.update(1)
    
    # 2. Binary Side Effects (Reduced iterations for speed)
    bin_m = MultiOutputClassifier(
        HistGradientBoostingClassifier(max_iter=20)
    ).fit(X_train_dense, train_df[bin_cols])
    pbar.update(1)
    
    # 3. PRR Risk
    prr_m = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=20)
    ).fit(X_train_dense, train_df[prr_cols])
    pbar.update(1)
    pbar.close()
    
    print("Generating predictions...")
    sub = pd.DataFrame({'Pair_ID': test_df['Pair_ID'], 'Severity': sev_m.predict(X_test_dense)})
    sub = pd.concat([sub, pd.DataFrame(bin_m.predict(X_test_dense), columns=bin_cols)], axis=1)
    sub = pd.concat([sub, pd.DataFrame(prr_m.predict(X_test_dense), columns=prr_cols)], axis=1)
    
    sub.to_csv('data/processed/submission_gb.csv', index=False)
    print("✔ GB Training Complete and Saved.")

if __name__ == "__main__": train()
