"""
train_baseline.py — TF-IDF + Logistic Regression baseline model.
Expected accuracy: ~87–88 % on AG News test set.

Run:
    python ml_pipeline/training/train_baseline.py
"""

import json
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parents[2]))
from ml_pipeline.data.preprocessor import TextPreprocessor, LABEL_MAP
from ml_pipeline.data.categories import CATEGORIES

# ── Paths ─────────────────────────────────────────────────────────────────────
# DATA_DIR   = Path("ml_pipeline/data/raw")
# OUTPUT_DIR = Path("ml_pipeline/models/baseline")
PROJECT_ROOT = Path(__file__).parents[2].resolve()
OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "models" / "baseline"
DATA_DIR   = PROJECT_ROOT /"ml_pipeline"/"data"/"raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CATEGORIES = ["World", "Sports", "Business", "Technology"]
NUM_CLASSES = len(CATEGORIES)          # ← CHANGED to 12
CATEGORIES_NAMES = [c.name for c in CATEGORIES]   # ← NEW

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_split(name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    texts  = [r["text"]  for r in records]
    labels = [r["label"] for r in records]
    return texts, labels


def preprocess(texts: list[str]) -> list[str]:
    prep = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    print(f"  Cleaning {len(texts):,} texts …")
    return prep.clean_batch(texts)


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CATEGORIES_NAMES,
        yticklabels=CATEGORIES_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Baseline (TF-IDF + LR) — Confusion Matrix (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("  BASELINE MODEL: TF-IDF + Logistic Regression")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data …")
    X_train_raw, y_train = load_split("train")
    X_val_raw,   y_val   = load_split("val")
    X_test_raw,  y_test  = load_split("test")
    print(f"  Train {len(X_train_raw):,}  |  Val {len(X_val_raw):,}  |  Test {len(X_test_raw):,}")

    # 2. Preprocess
    print("\n[2/5] Preprocessing …")
    X_train = preprocess(X_train_raw)
    X_val   = preprocess(X_val_raw)
    X_test  = preprocess(X_test_raw)

    # 3. Build pipeline
    print("\n[3/5] Building TF-IDF + LR pipeline …")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=5.0,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    # 4. Train
    print("\n[4/5] Training …")
    start = datetime.now()
    pipeline.fit(X_train, y_train)
    elapsed = (datetime.now() - start).seconds

    # 5. Evaluate
    print("\n[5/5] Evaluating …")
    val_preds  = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)

    val_acc  = accuracy_score(y_val,  val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1  = f1_score(y_test, test_preds, average="weighted")

    print(f"\n{'─'*40}")
    print(f"  Val  Accuracy : {val_acc*100:.2f}%")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test F1       : {test_f1*100:.2f}%")
    print(f"  Train time    : {elapsed}s")
    print(f"{'─'*40}")

    print("\n📋 Per-category report (test):")
    # print(classification_report(y_test, test_preds, target_names=CATEGORIES))
    print(classification_report(y_test, test_preds, target_names=CATEGORIES_NAMES))

    # Top features per class
    print("\n🔍 Top 10 TF-IDF features per category:")
    vectorizer = pipeline.named_steps["tfidf"]
    clf        = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    for i, cat in enumerate(CATEGORIES_NAMES):
        top_idx = np.argsort(clf.coef_[i])[-10:][::-1]
        top_words = ", ".join(feature_names[top_idx])
        print(f"  {cat:<12} → {top_words}")

    # Save model
    model_path = OUTPUT_DIR / "tfidf_lr_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Model saved → {model_path}")

    # Save metrics
    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "val_accuracy":  round(val_acc,  4),
        "test_accuracy": round(test_acc, 4),
        "test_f1":       round(test_f1,  4),
        "train_time_s":  elapsed,
        "trained_at":    datetime.now().isoformat(),
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📝 Metrics saved → {metrics_path}")

    # Confusion matrix
    plot_confusion_matrix(y_test, test_preds, OUTPUT_DIR / "confusion_matrix.png")

    print("\n✅ Baseline training complete!")
    return pipeline, metrics


if __name__ == "__main__":
    train()
