import pandas as pd
import glob
import os
from tabulate import tabulate

def compare_results():
    """Reads versioned CSVs and prints a comparison table with bars."""
    files = glob.glob('data/processed/*-v*.csv')
    if not files:
        print("\n[!] No versioned results found. Run 'run lg' first.")
        return

    data = []
    for f in files:
        name = os.path.basename(f)
        score = 0.0
        # Extracts score from filename if present (e.g., lg-v1-score-0.51.csv)
        if "score-" in name:
            try:
                score = float(name.split("score-")[1].replace(".csv", ""))
            except:
                pass
        
        data.append({"File": name, "Accuracy": score})

    df = pd.DataFrame(data).sort_values(by="Accuracy", ascending=False)

    print("\n── Model Performance Table ───────────────────")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
    
    print("\n── Visual Accuracy Bar ───────────────────────")
    for _, row in df.iterrows():
        bar = "█" * int(row['Accuracy'] * 40)
        print(f"  {row['File']:<30} |{bar} {row['Accuracy']:.4f}")

if __name__ == "__main__":
    compare_results()
