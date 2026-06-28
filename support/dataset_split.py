import os
import json
import sys
import random
from collections import defaultdict

# Force UTF-8 output for Windows console emoji support
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- PATH CONFIGURATION ---
SUPPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

INPUT_JSONL = os.path.join(PROJECT_ROOT, "dataset", "final_training_dataset.jsonl")
TRAIN_OUT = os.path.join(PROJECT_ROOT, "dataset", "train_split.jsonl")
VAL_OUT = os.path.join(PROJECT_ROOT, "dataset", "val_split.jsonl")
TEST_OUT = os.path.join(PROJECT_ROOT, "dataset", "test_split.jsonl")

def extract_disaster_event(image_path: str) -> str:
    """Extract disaster event name from image path.
    e.g. 'processed_images/hurricane-michael_00000035_post_disaster.jpg'
          → 'hurricane-michael'
    """
    basename = os.path.basename(image_path)
    parts = basename.rsplit("_", 3)
    return parts[0] if len(parts) >= 4 else basename.split("_")[0]

def split_dataset():
    random.seed(42)

    if not os.path.exists(INPUT_JSONL):
        print(f"ERROR: {INPUT_JSONL} not found.")
        return

    # Group by disaster event
    events = defaultdict(list)
    total_samples = 0
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            event = extract_disaster_event(row['image'])
            events[event].append(row)
            total_samples += 1

    train_data = []
    val_data = []
    test_data = []

    print(f"Found {total_samples} samples across {len(events)} disaster events.")
    print("Stratifying 80/10/10 per event...")

    for event, samples in events.items():
        random.shuffle(samples)
        
        n = len(samples)
        # 80% train, 10% val, 10% test
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)
        
        # If very small number of samples, enforce at least 1 in val/test if possible
        if n > 2 and train_end == n:
            train_end = n - 2
            val_end = n - 1

        train_data.extend(samples[:train_end])
        val_data.extend(samples[train_end:val_end])
        test_data.extend(samples[val_end:])

    # Shuffle the final splits so batches are mixed
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    def write_split(path, data):
        with open(path, 'w', encoding='utf-8') as f:
            for row in data:
                f.write(json.dumps(row) + '\n')

    write_split(TRAIN_OUT, train_data)
    write_split(VAL_OUT, val_data)
    write_split(TEST_OUT, test_data)

    print("\n" + "="*50)
    print("📊 DATASET SPLIT REPORT")
    print("="*50)
    print(f"Total Samples : {total_samples}")
    print(f"Train Split   : {len(train_data)} ({len(train_data)/total_samples*100:.1f}%) -> {os.path.basename(TRAIN_OUT)}")
    print(f"Val Split     : {len(val_data)} ({len(val_data)/total_samples*100:.1f}%) -> {os.path.basename(VAL_OUT)}")
    print(f"Test Split    : {len(test_data)} ({len(test_data)/total_samples*100:.1f}%) -> {os.path.basename(TEST_OUT)}")
    print("="*50)

if __name__ == "__main__":
    split_dataset()
