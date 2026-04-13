"""
classifier.py - Auto downloads models from Google Drive if not present
"""

import re
import json
import sys
import joblib
import pickle
import torch
import gdown
import shutil
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parents[2]))

from ml_pipeline.data.categories import CATEGORIES, NUM_CLASSES
from ml_pipeline.data.preprocessor import TextPreprocessor
from ml_pipeline.training.train_lstm import BiLSTMClassifier

# =========================
# Google Drive Config
# =========================
DRIVE_FOLDER_ID = "1ztji2Gs6RP8lnYUQoiR7NAF89qdGx178"   # From your link
MODELS_DIR = Path(__file__).parents[2] / "ml_pipeline" / "models"

def download_models_from_drive():
    """Download models from Google Drive if missing."""
    if MODELS_DIR.exists() and list(MODELS_DIR.glob("**/*.joblib")):
        print("✅ Models already present locally")
        return True

    print("📥 Models not found. Downloading from Google Drive... (this may take a few minutes)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = MODELS_DIR / "models.zip"

    try:
        # Download the entire folder as zip
        gdown.download_folder(id=DRIVE_FOLDER_ID, output=str(MODELS_DIR), quiet=False, remaining_ok=True)

        print("✅ Models downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download models: {e}")
        return False

# ── Vocabulary class ─────────────────────
class Vocabulary:
    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def encode(self, text: str, max_len=128) -> list[int]:
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (max_len - len(ids))
        return ids[:max_len]


# ── Ensemble weights ─────────────────────
WEIGHTS = {"baseline": 0.15, "lstm": 0.30, "distilbert": 0.55}


class NewsClassifier:
    def __init__(self):
        # Download models if needed
        download_models_from_drive()

        self.prep = TextPreprocessor()
        self._baseline   = self._load_baseline()
        self._lstm       = self._load_lstm()
        self._distilbert = self._load_distilbert()

        loaded = [k for k, v in {
            "baseline": self._baseline,
            "lstm": self._lstm,
            "distilbert": self._distilbert,
        }.items() if v is not None]

        print(f"✅ Classifier ready — models loaded: {loaded}")

    # ... (keep all your predict, _keyword_classify, _baseline_probs, _ensemble_probs, _lstm_probs, _distilbert_probs methods exactly as they are)

    # ── Model loaders (unchanged except paths) ─────────────────────
    def _load_baseline(self):
        p = MODELS_DIR / "baseline/tfidf_lr_pipeline.joblib"
        if p.exists():
            try:
                return joblib.load(p)
            except Exception as e:
                print(f"⚠️ Failed to load baseline: {e}")
        print("⚠️ Baseline model not found")
        return None

    def _load_lstm(self):
        pt = MODELS_DIR / "lstm/lstm_best.pt"
        vfile = MODELS_DIR / "lstm/vocab.pkl"

        if not pt.exists() or not vfile.exists():
            print("⚠️ LSTM not found")
            return None

        try:
            from ml_pipeline.data.Vocabulary import Vocabulary
            import sys
            sys.modules['__main__'].Vocabulary = Vocabulary

            with open(vfile, "rb") as f:
                vocab = pickle.load(f)

            model = BiLSTMClassifier(
                vocab_size=len(vocab.word2idx),
                embed_dim=128,
                hidden_dim=256,
                num_layers=2,
                num_classes=8,
                dropout=0.4,
            )
            model.load_state_dict(torch.load(pt, map_location="cpu"))
            model.eval()
            print("✅ LSTM loaded successfully")
            return model, vocab
        except Exception as e:
            print(f"⚠️ LSTM load failed: {e}")
            return None

    def _load_distilbert(self):
        p = MODELS_DIR / "distilbert/best_model"
        if not p.exists():
            print("⚠️ DistilBERT not found")
            return None
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            tok = DistilBertTokenizerFast.from_pretrained(p)
            model = DistilBertForSequenceClassification.from_pretrained(p)
            print("✅ DistilBERT loaded successfully")
            return model, tok
        except Exception as e:
            print(f"⚠️ DistilBERT load failed: {e}")
            return None


# Singleton
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = NewsClassifier()
    return _classifier


if __name__ == "__main__":
    clf = get_classifier()
