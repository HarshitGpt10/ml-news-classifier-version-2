"""
classifier.py
─────────────
8-category news classifier (minimal changes for your trained models).
Supports short and long text with smart routing.
"""

import re
import json
import sys
import joblib
import pickle
import torch
from ml_pipeline.training.train_lstm import BiLSTMClassifier
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter

sys.path.append(str(Path(__file__).parents[2]))
from ml_pipeline.data.categories import CATEGORIES, NUM_CLASSES
from ml_pipeline.data.preprocessor import TextPreprocessor

# ── Vocabulary class (required to load saved LSTM model) ─────────────────────
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


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parents[2].resolve()
MODEL_DIR = PROJECT_ROOT / "ml_pipeline" / "models"

# ── Ensemble weights ──────────────────────────────────────────────────────────
WEIGHTS = {"baseline": 0.15, "lstm": 0.30, "distilbert": 0.55}


class NewsClassifier:

    def __init__(self):
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

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, text: str) -> dict:
        text = text.strip()
        if not text:
            raise ValueError("Empty text")

        word_count = len(text.split())
        probs, method = self._ensemble_probs(text), "ensemble"

        if word_count < 4:
            probs, method = self._keyword_classify(text), "keyword"
        # elif word_count < 50 or (self._lstm is None and self._distilbert is None):
        #     probs, method = self._baseline_probs(text), "baseline"
        # else:
        #     probs, method = self._ensemble_probs(text), "ensemble"

        pred_idx = int(np.argmax(probs))
        cat = CATEGORIES[pred_idx]

        return {
            "category":    cat.name,
            "label":       pred_idx,
            "icon":        cat.icon,
            "color":       cat.color,
            "confidence":  float(round(probs[pred_idx], 4)),
            "word_count":  word_count,
            "method":      method,
            "probabilities": {
                c.name: float(round(probs[c.id], 4)) for c in CATEGORIES
            },
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]

    # ── Internal methods ──────────────────────────────────────────────────────
    def _keyword_classify(self, text: str) -> np.ndarray:
        lower = text.lower()
        scores = np.zeros(NUM_CLASSES)
        for c in CATEGORIES:
            for kw in c.keywords:
                if kw in lower:
                    scores[c.id] += 1
        if scores.sum() == 0:
            return self._baseline_probs(text) if self._baseline else np.ones(NUM_CLASSES)/NUM_CLASSES
        return scores / scores.sum()

    def _baseline_probs(self, text: str) -> np.ndarray:
        clean = self.prep.clean(text)
        raw = self._baseline.predict_proba([clean])[0]
        probs = np.zeros(NUM_CLASSES)
        from ml_pipeline.data.categories import AG_NEWS_REMAP
        for ag_label, our_label in AG_NEWS_REMAP.items():
            if ag_label < len(raw):
                probs[our_label] += raw[ag_label]
        return probs / max(probs.sum(), 1e-9)

    def _ensemble_probs(self, text: str) -> np.ndarray:
        parts, w_sum = [], 0.0
        if self._baseline:
            p = self._baseline_probs(text)
            parts.append(p * WEIGHTS["baseline"])
            w_sum += WEIGHTS["baseline"]
        if self._lstm:
            p = self._lstm_probs(text)
            parts.append(p * WEIGHTS["lstm"])
            w_sum += WEIGHTS["lstm"]
        if self._distilbert:
            p = self._distilbert_probs(text)
            parts.append(p * WEIGHTS["distilbert"])
            w_sum += WEIGHTS["distilbert"]
        if not parts:
            return self._keyword_classify(text)
        combined = np.sum(parts, axis=0) / w_sum
        return combined / combined.sum()

    def _lstm_probs(self, text: str) -> np.ndarray:
        if self._lstm is None:
            return self._baseline_probs(text) if self._baseline else np.ones(NUM_CLASSES) / NUM_CLASSES

        model, vocab = self._lstm
        clean = self.prep.clean(text)
        ids = torch.tensor([vocab.encode(clean)], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            logits = model(ids)
        p = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
        return p

    def _distilbert_probs(self, text: str) -> np.ndarray:
        model, tokenizer = self._distilbert
        enc = tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        model.eval()
        with torch.no_grad():
            logits = model(**enc).logits
        import torch.nn.functional as F
        p = F.softmax(logits, dim=-1).numpy()[0]
        return p

    # ── Model loaders ─────────────────────────────────────────────────────────
    def _load_baseline(self):
        p = MODEL_DIR / "baseline/tfidf_lr_pipeline.joblib"
        if p.exists():
            return joblib.load(p)
        print("⚠️  Baseline model not found")
        return None

    def _load_lstm(self):
        pt = MODEL_DIR / "lstm/lstm_best.pt"
        vfile = MODEL_DIR / "lstm/vocab.pkl"

        if not pt.exists() or not vfile.exists():
            print("⚠️ LSTM not found (run train_lstm.py)")
            return None

        try:
            # === FIX FOR VOCABULARY PICKLE ERROR ===
            # Register Vocabulary class so pickle can find it
            from ml_pipeline.data.Vocabulary import Vocabulary
            import sys
            sys.modules['__main__'].Vocabulary = Vocabulary
            if 'ml_pipeline.training.train_lstm' in sys.modules:
                sys.modules['ml_pipeline.training.train_lstm'].Vocabulary = Vocabulary
            # =======================================

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
        p = MODEL_DIR / "distilbert/best_model"
        if not p.exists():
            print("⚠️  DistilBERT not found (run train_distilbert.py)")
            return None
        try:
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            tok = DistilBertTokenizerFast.from_pretrained(p)
            model = DistilBertForSequenceClassification.from_pretrained(p)
            print("✅ DistilBERT loaded successfully")
            return model, tok
        except Exception as e:
            print(f"⚠️  DistilBERT load failed: {e}")
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = NewsClassifier()
    return _classifier


# ── Quick test when running directly ─────────────────────────────────────────
if __name__ == "__main__":
    clf = get_classifier()
    tests = [
        "IPL",
        "India wins World Cup",
        "PM Modi inaugurates new metro line in Delhi today",
        "Apple launches new iPhone with AI chip — stock rises 5%",
        "Scientists discover new exoplanet in habitable zone",
        "Trisha at 42: What’s behind the romance rumours, and why is she trending again?",
        "Supreme Court acquits accused in 2012 Delhi gang rape case",
        "Raghav Chadha releases new video: Let my work do the talking Amid an internal rift within the Aam Aadmi Party (AAP), Rajya Sabha MP Raghav Chadha has responded"
    ]
    for t in tests:
        r = clf.predict(t)
        print(f"\n[{len(t.split()):2d} words] {t[:60]:<60}")
        print(f"  → {r['icon']} {r['category']} ({r['confidence']*100:.1f}%) via {r['method']}")