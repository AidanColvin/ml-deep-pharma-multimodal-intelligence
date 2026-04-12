from tqdm import tqdm
import time
import sys

def show_progress(task_name, duration=2):
    print(f"\n>>> Starting {task_name}...")
    for _ in tqdm(range(100), desc=f"Processing {task_name}", unit="bit"):
        time.sleep(duration/100)
    print(f"✔ {task_name} Complete.\n")

if __name__ == "__main__":
    show_progress(sys.argv[1])
