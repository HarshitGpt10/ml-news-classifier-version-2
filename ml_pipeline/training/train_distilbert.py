"""
train_distilbert.py — Fine-tune DistilBERT for news classification.
Expected accuracy: ~94–95 % on AG News test set.

⚠️  Requires ~4 GB VRAM (GPU) or runs slowly on CPU (~1 hr per epoch).
    Reduce BATCH_SIZE to 8 if you get out-of-memory errors.

Run:
    python ml_pipeline/training/train_distilbert.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report
import sys
sys.path.append(str(Path(__file__).parents[2]))

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 16   # reduce to 8 if OOM
NUM_EPOCHS  = 3
LR          = 2e-5
NUM_CLASSES = 8

PROJECT_ROOT = Path(__file__).parents[2].resolve()
OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "models" / "distilbert"
DATA_DIR   = PROJECT_ROOT /"ml_pipeline"/"data"/"raw"
# DATA_DIR   = Path("ml_pipeline/data/raw")
# OUTPUT_DIR = Path("ml_pipeline/models/distilbert")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "World & International",
    "Politics & Governance",
    "Business & Finance",
    "Technology",
    "Sports",
    "Health & Medicine",
    "Entertainment & Culture",
    "Lifestyle & Society"
]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, records, tokenizer, max_len=MAX_LEN):
        self.records   = records
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r   = self.records[idx]
        enc = self.tokenizer(
            r["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(r["label"], dtype=torch.long),
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            logits = model(ids, attention_mask=mask).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds  += preds
            all_labels += batch["label"].tolist()
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / len(all_labels)
    return acc, all_preds, all_labels


def train():
    print("=" * 60)
    print("  DISTILBERT: Fine-tuning for news classification")
    print("=" * 60)

    print(f"\n[1/5] Loading tokenizer ({MODEL_NAME}) …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print("\n[2/5] Loading data …")
    train_records = load_split("train")
    val_records   = load_split("val")
    test_records  = load_split("test")

    train_ds = NewsDataset(train_records, tokenizer)
    val_ds   = NewsDataset(val_records,   tokenizer)
    test_ds  = NewsDataset(test_records,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\n[3/5] Loading model ({MODEL_NAME}) …")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    print("\n[4/5] Fine-tuning …")
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out  = model(ids, attention_mask=mask, labels=lbls)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(lbls)
            correct    += (out.logits.argmax(1) == lbls).sum().item()
            total      += len(lbls)

        train_acc = correct / total
        val_acc, _, _ = evaluate(model, val_loader)

        tag = " ← best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  "
              f"train={train_acc*100:.2f}%  val={val_acc*100:.2f}%{tag}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(OUTPUT_DIR / "best_model")
            tokenizer.save_pretrained(OUTPUT_DIR / "best_model")

    print("\n[5/5] Evaluating best model …")
    best_model = DistilBertForSequenceClassification.from_pretrained(
        OUTPUT_DIR / "best_model"
    ).to(DEVICE)
    test_acc, preds, labels = evaluate(best_model, test_loader)

    print(f"\n  Test Accuracy: {test_acc*100:.2f}%")
    print("\n📋 Per-category report:")
    print(classification_report(labels, preds, target_names=CATEGORIES))

    metrics = {
        "model": "DistilBERT fine-tuned",
        "test_accuracy": round(test_acc, 4),
        "trained_at": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ DistilBERT fine-tuning complete!  Model → {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
