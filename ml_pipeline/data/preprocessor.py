"""
preprocessor.py — Shared text cleaning pipeline used by ALL model streams.
Every model (TF-IDF, LSTM, DistilBERT) passes text through this first.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)

# AG News category mapping
LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Technology"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


class TextPreprocessor:
    """
    Cleans and normalises raw news text for ML pipelines.

    Usage:
        prep = TextPreprocessor()
        clean = prep.clean("Apple Inc. reported $3.4B in Q2 earnings!")
    """

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    # ── public API ────────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Full pipeline: lowercase → strip HTML → normalise → tokenise → filter → join."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = self._strip_html(text)
        text = self._remove_urls(text)
        text = self._normalise_numbers(text)
        text = self._remove_punctuation(text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]
        if self.lemmatize:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if len(t) > 1]
        return " ".join(tokens)

    def clean_batch(self, texts: list[str]) -> list[str]:
        return [self.clean(t) for t in texts]

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", text)

    @staticmethod
    def _normalise_numbers(text: str) -> str:
        text = re.sub(r"\$[\d,]+\.?\d*[bmk]?", " MONEY ", text)
        text = re.sub(r"\d+%", " PERCENT ", text)
        text = re.sub(r"\b\d{4}\b", " YEAR ", text)
        text = re.sub(r"\b\d+\b", " NUM ", text)
        return text

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
