"""
train_lstm.py — Bidirectional LSTM model for news classification.
Expected accuracy: ~91–92 % on AG News test set.

Run:
    python ml_pipeline/training/train_lstm.py
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
from collections import Counter
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parents[2]))
from ml_pipeline.data.preprocessor import TextPreprocessor

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE   = 50_000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.4
MAX_SEQ_LEN  = 128
BATCH_SIZE   = 64
NUM_EPOCHS   = 10
LR           = 1e-3
PATIENCE     = 3        # early stopping
NUM_CLASSES  = 8

# DATA_DIR   = Path("ml_pipeline/data/raw")
# OUTPUT_DIR = Path("ml_pipeline/models/lstm")
PROJECT_ROOT = Path(__file__).parents[2].resolve()
OUTPUT_DIR = PROJECT_ROOT / "ml_pipeline" / "models" / "lstm"
DATA_DIR   = PROJECT_ROOT /"ml_pipeline"/"data"/"raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── Vocabulary ────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_size=VOCAB_SIZE):
        self.max_size = max_size
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build(self, texts: list[str]):
        counter = Counter(w for text in texts for w in text.split())
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"  Vocabulary size: {len(self.word2idx):,}")

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> list[int]:
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        # Pad / truncate
        ids += [0] * (max_len - len(ids))
        return ids[:max_len]


# ── Dataset ───────────────────────────────────────────────────────────────────

class NewsDataset(Dataset):
    def __init__(self, records: list[dict], vocab: Vocabulary):
        self.vocab = vocab
        self.X = [vocab.encode(r["clean"]) for r in records]
        self.y = [r["label"] for r in records]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout   = nn.Dropout(dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))                     # (B, T, E)
        out, _ = self.lstm(emb)                                   # (B, T, 2H)
        attn_w = torch.softmax(self.attention(out), dim=1)        # (B, T, 1)
        context = (attn_w * out).sum(dim=1)                       # (B, 2H)
        return self.fc(self.dropout(context))                     # (B, C)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train(train)
    total_loss, total_correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        if train:
            optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss    += loss.item() * len(y_batch)
        total_correct += (logits.argmax(1) == y_batch).sum().item()
        total         += len(y_batch)
    return total_loss / total, total_correct / total


def load_and_clean(name: str):
    path = DATA_DIR / f"{name}.json"
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    prep = TextPreprocessor()
    for r in records:
        r["clean"] = prep.clean(r["text"])
    return records


def train():
    print("=" * 60)
    print("  LSTM MODEL: Bidirectional LSTM + Attention")
    print("=" * 60)

    print("\n[1/5] Loading & preprocessing data …")
    train_records = load_and_clean("train")
    val_records   = load_and_clean("val")
    test_records  = load_and_clean("test")

    print("\n[2/5] Building vocabulary …")
    vocab = Vocabulary()
    vocab.build([r["clean"] for r in train_records])

    print("\n[3/5] Creating data loaders …")
    train_ds = NewsDataset(train_records, vocab)
    val_ds   = NewsDataset(val_records,   vocab)
    test_ds  = NewsDataset(test_records,  vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("\n[4/5] Training …")
    model = BiLSTMClassifier(
        vocab_size=len(vocab.word2idx),
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_acc = 0.0
    no_improve   = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, train=False)
        scheduler.step(val_loss)

        tag = " ← best" if val_acc > best_val_acc else ""
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS}  "
              f"train_acc={train_acc*100:.2f}%  val_acc={val_acc*100:.2f}%{tag}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "lstm_best.pt")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test
    print("\n[5/5] Evaluating best model on test set …")
    model.load_state_dict(torch.load(OUTPUT_DIR / "lstm_best.pt", map_location=DEVICE))
    _, test_acc = run_epoch(model, test_loader, criterion, train=False)
    print(f"\n  Test Accuracy: {test_acc*100:.2f}%")

    # Save vocab
    import pickle
    with open(OUTPUT_DIR / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    # Save metrics
    metrics = {
        "model": "BiLSTM + Attention",
        "test_accuracy": round(test_acc, 4),
        "trained_at": datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ LSTM training complete!  Model → {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
