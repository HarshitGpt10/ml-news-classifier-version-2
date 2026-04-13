"""
create_custom_dataset.py
────────────────────────
Interactive CLI tool that lets you:
  1. Add individual news articles with labels
  2. Import from a CSV  (columns: text, category)
  3. Bulk-label articles using keyword matching
  4. Export a combined dataset for training

Usage:
    python ml_pipeline/data/create_custom_dataset.py
    python ml_pipeline/data/create_custom_dataset.py --import my_news.csv
    python ml_pipeline/data/create_custom_dataset.py --export
"""

import json
import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parents[2]))
from ml_pipeline.data.categories import (
    CATEGORIES, CATEGORY_BY_SHORT, LABEL_MAP_INV, NUM_CLASSES
)
PROJECT_ROOT = Path(__file__).parents[2].resolve()
CUSTOM_DATASET_PATH = PROJECT_ROOT / "ml_pipeline" / "data" / "custom_dataset.json"
# CUSTOM_DATASET_PATH = Path("ml_pipeline/data/custom_dataset.json")
CUSTOM_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Core helpers ───────────────────────────────────────────────────────────────

def load_custom() -> list[dict]:
    if CUSTOM_DATASET_PATH.exists():
        with open(CUSTOM_DATASET_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_custom(records: list[dict]):
    with open(CUSTOM_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved {len(records):,} records → {CUSTOM_DATASET_PATH}")


def _show_categories():
    print("\nAvailable categories:")
    for c in CATEGORIES:
        print(f"  {c.id:2d}. {c.icon} {c.name}")
    print()


def _make_record(text: str, label: int, source: str = "custom") -> dict:
    return {
        "text":     text.strip(),
        "label":    label,
        "category": CATEGORIES[label].name,
        "source":   source,
        "added_at": datetime.now().isoformat(),
    }


# ── Add single article ─────────────────────────────────────────────────────────

def add_article_interactive(records: list[dict]):
    print("\n─── Add a new article ───────────────────────────────")
    text = input("Paste the news text (or headline):\n> ").strip()
    if not text:
        print("Empty text — skipped.")
        return records

    _show_categories()
    while True:
        raw = input("Enter category number: ").strip()
        if raw.isdigit() and int(raw) in range(NUM_CLASSES):
            label = int(raw)
            break
        print(f"  ⚠️  Enter a number between 0 and {NUM_CLASSES - 1}")

    records.append(_make_record(text, label))
    print(f"✅ Added as '{CATEGORIES[label].name}'")
    return records


# ── Import from CSV ───────────────────────────────────────────────────────────

def import_csv(filepath: str, records: list[dict]) -> list[dict]:
    """
    CSV must have columns: text, category
    category can be:  numeric id (0-11), short slug, or full name
    """
    added, skipped = 0, 0
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            text = (row.get("text") or row.get("Text") or "").strip()
            cat  = (row.get("category") or row.get("Category") or "").strip()
            if not text or not cat:
                skipped += 1
                continue

            # Resolve label
            label = None
            if cat.isdigit():
                idx = int(cat)
                if 0 <= idx < NUM_CLASSES:
                    label = idx
            elif cat in LABEL_MAP_INV:
                label = LABEL_MAP_INV[cat]
            elif cat.lower() in CATEGORY_BY_SHORT:
                label = CATEGORY_BY_SHORT[cat.lower()].id
            else:
                # Try partial match
                for c in CATEGORIES:
                    if cat.lower() in c.name.lower():
                        label = c.id
                        break

            if label is None:
                print(f"  ⚠️  Row {i}: unknown category '{cat}' — skipped")
                skipped += 1
                continue

            records.append(_make_record(text, label, source=f"csv:{Path(filepath).name}"))
            added += 1

    print(f"✅ Imported {added:,} articles  ({skipped} skipped) from {filepath}")
    return records


# ── Auto-label via keywords ───────────────────────────────────────────────────

def auto_label_texts(texts: list[str]) -> list[dict]:
    """
    Naive keyword-based labelling for bootstrap. 
    Returns records; unmatched texts are labelled with label=-1 (review needed).
    """
    records = []
    for text in texts:
        lower = text.lower()
        best_cat, best_score = None, 0
        for c in CATEGORIES:
            score = sum(1 for kw in c.keywords if kw in lower)
            if score > best_score:
                best_score, best_cat = score, c
        if best_cat and best_score > 0:
            records.append(_make_record(text, best_cat.id, source="auto-label"))
        else:
            records.append({
                "text":     text,
                "label":    -1,          # needs manual review
                "category": "UNLABELLED",
                "source":   "auto-label",
                "added_at": datetime.now().isoformat(),
            })
    return records


# ── Export combined dataset ────────────────────────────────────────────────────

def export_combined(output_path: str = "ml_pipeline/data/raw/custom_train.json"):
    records = load_custom()
    labelled = [r for r in records if r.get("label", -1) >= 0]
    unlabelled = len(records) - len(labelled)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(labelled, f, ensure_ascii=False, indent=2)

    print(f"\n📦 Export summary:")
    print(f"   Total records   : {len(records):,}")
    print(f"   Labelled        : {len(labelled):,}")
    print(f"   Unlabelled      : {unlabelled:,}  (excluded from export)")
    print(f"   Output file     : {out}")

    # Per-category breakdown
    from collections import Counter
    counts = Counter(r["category"] for r in labelled)
    print("\n   Category breakdown:")
    for c in CATEGORIES:
        n = counts.get(c.name, 0)
        bar = "▓" * min(n // 5, 40)
        print(f"   {c.icon} {c.name:<26} {n:>5}  {bar}")

    return labelled


# ── Stats ─────────────────────────────────────────────────────────────────────

def show_stats(records: list[dict]):
    from collections import Counter
    print(f"\n📊 Custom dataset — {len(records):,} total records")
    counts = Counter(r.get("category", "UNKNOWN") for r in records)
    for c in CATEGORIES:
        n = counts.get(c.name, 0)
        bar = "█" * min(n // 2, 30)
        print(f"  {c.icon} {c.name:<26}  {n:>5}  {bar}")
    unlabelled = counts.get("UNLABELLED", 0)
    if unlabelled:
        print(f"  ❓ {'UNLABELLED':<26}  {unlabelled:>5}  (need review)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Custom dataset builder for ML News Classifier")
    parser.add_argument("--import",   dest="import_csv",  metavar="CSV",  help="Import labelled CSV")
    parser.add_argument("--export",   action="store_true", help="Export combined dataset")
    parser.add_argument("--stats",    action="store_true", help="Show dataset statistics")
    parser.add_argument("--add",      action="store_true", help="Add article interactively")
    args = parser.parse_args()

    records = load_custom()

    if args.import_csv:
        records = import_csv(args.import_csv, records)
        save_custom(records)

    elif args.export:
        export_combined()

    elif args.stats:
        show_stats(records)

    elif args.add:
        records = add_article_interactive(records)
        save_custom(records)

    else:
        # Interactive menu
        while True:
            print("\n" + "═" * 50)
            print("  ML News Classifier — Custom Dataset Builder")
            print("═" * 50)
            print("  1. Add article manually")
            print("  2. Import from CSV")
            print("  3. Show statistics")
            print("  4. Export for training")
            print("  5. Exit")
            choice = input("\nChoice: ").strip()

            if choice == "1":
                records = add_article_interactive(records)
                save_custom(records)
            elif choice == "2":
                path = input("CSV file path: ").strip()
                if os.path.exists(path):
                    records = import_csv(path, records)
                    save_custom(records)
                else:
                    print(f"File not found: {path}")
            elif choice == "3":
                show_stats(records)
            elif choice == "4":
                export_combined()
            elif choice == "5":
                break
            else:
                print("Invalid choice")


if __name__ == "__main__":
    main()
