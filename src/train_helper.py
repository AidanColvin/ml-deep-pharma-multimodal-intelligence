import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_vectorize():
    print("Loading data...")
    train = pd.read_csv('data/raw/train.csv')
    test = pd.read_csv('data/raw/test.csv')
    
    bin_cols = [c for c in train.columns if c.startswith('Target_Binary_')]
    prr_cols = [c for c in train.columns if c.startswith('Target_PRR_')]
    
    def get_text(df):
        cols = [c for c in df.columns if not c.startswith('Target_') and c not in ['Pair_ID', 'Severity']]
        return df[cols].fillna('').astype(str).agg(' '.join, axis=1)

    print("Vectorizing medical text (Cleaning stop words)...")
    X_train_text = get_text(train)
    X_test_text = get_text(test)
    
    vec = TfidfVectorizer(
        max_features=2000, 
        stop_words='english', 
        ngram_range=(1, 2)
    )
    
    X_train = vec.fit_transform(X_train_text)
    X_test = vec.transform(X_test_text)
    
    return train, test, X_train, X_test, bin_cols, prr_cols
