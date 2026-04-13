"""
download_dataset.py — Downloads AG News from HuggingFace and saves train/val/test splits.

Run:
    python ml_pipeline/data/download_dataset.py
"""

import os
import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# OUTPUT_DIR = Path("ml_pipeline/data/raw")
PROJECT_ROOT = Path(__file__).parents[2].resolve()
OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "data" / "raw"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}


def download_and_split():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("⬇️  Downloading AG News from HuggingFace (≈ 30 MB) …")
    dataset = load_dataset("ag_news", split="train+test")

    # Convert to list of dicts
    all_records = []
    for item in tqdm(dataset, desc="Processing"):
        all_records.append(
            {
                "text": item["text"],
                "label": item["label"],
                "category": LABEL_MAP[item["label"]],
            }
        )

    # Shuffle reproducibly
    random.seed(SEED)
    random.shuffle(all_records)

    # Split
    total = len(all_records)
    n_train = int(total * SPLITS["train"])
    n_val = int(total * SPLITS["val"])

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train : n_train + n_val],
        "test": all_records[n_train + n_val :],
    }

    # Save
    for name, records in splits.items():
        out_path = OUTPUT_DIR / f"{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(records):,} records → {out_path}")

    # Class distribution
    print("\n📊 Class distribution (train set):")
    from collections import Counter
    counts = Counter(r["category"] for r in splits["train"])
    for cat, n in sorted(counts.items()):
        pct = n / len(splits["train"]) * 100
        print(f"   {cat:<12} {n:>7,}  ({pct:.1f}%)")

    print(f"\n🎉 Dataset ready!  Total: {total:,} articles")


if __name__ == "__main__":
    download_and_split()
