class Vocabulary:
    """Vocabulary class used by LSTM model."""

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

    def __len__(self):
        return len(self.word2idx)