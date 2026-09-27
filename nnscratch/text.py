"""Text preprocessing and a TF-IDF vectorizer built on raw counts (output is a scipy sparse matrix)."""
import re
from collections import Counter
from typing import List, Sequence, Tuple

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy import sparse

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def tokenize(text: str) -> List[str]:
    """Lowercase, strip URLs, punctuation and digits, drop stopwords, Porter-stem the rest."""
    text = text.lower()
    text = re.sub(r"https?\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]|\d+|_", " ", text)
    return [STEMMER.stem(w) for w in text.split() if w not in STOPWORDS]


def load_tsv(path) -> Tuple[List[str], List[str]]:
    """Read `label<TAB>text` lines."""
    labels, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[1].strip():
                labels.append(parts[0])
                texts.append(parts[1])
    return labels, texts


def split(n: int, fractions: Sequence[float], seed: int) -> List[np.ndarray]:
    """Shuffle indices 0..n-1 and cut them into consecutive chunks of the given fractions."""
    idx = np.random.default_rng(seed).permutation(n)
    cuts = np.cumsum([int(f * n) for f in fractions[:-1]])
    return np.split(idx, cuts)


class TfIdf:
    """tf-idf(t, d) = (count of t in d / length of d) * log(N / df(t)).

    Fit on training documents only; words never seen in training are ignored at transform time.
    """

    def __init__(self, max_features: int = None):
        self.max_features = max_features

    def fit(self, docs: List[List[str]]) -> "TfIdf":
        term_counts, doc_freq = Counter(), Counter()
        for tokens in docs:
            term_counts.update(tokens)
            doc_freq.update(set(tokens))
        vocab = [w for w, _ in term_counts.most_common(self.max_features)]
        self.index = {w: i for i, w in enumerate(vocab)}
        self.idf = np.array([np.log(len(docs) / doc_freq[w]) for w in vocab])
        return self

    def transform(self, docs: List[List[str]]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for r, tokens in enumerate(docs):
            counts = Counter(t for t in tokens if t in self.index)
            for word, c in counts.items():
                rows.append(r)
                cols.append(self.index[word])
                vals.append(c / len(tokens))
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.index)))
        return tf.multiply(self.idf).tocsr()

    def __len__(self):
        return len(self.index)
