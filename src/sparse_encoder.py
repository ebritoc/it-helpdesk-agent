"""BM25 Sparse Vector Encoder for Hybrid Search"""
import re
import math
import pickle
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter, defaultdict


class BM25SparseEncoder:
    """Lightweight BM25 sparse vector encoder for keyword-based search

    Generates sparse vectors using BM25 algorithm:
    - Term Frequency (TF) with saturation
    - Inverse Document Frequency (IDF) weighting
    - Document length normalization
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """Initialize BM25 encoder

        Args:
            k1: Term saturation parameter (default: 1.2)
                Controls how quickly term frequency saturates
            b: Length normalization parameter (default: 0.75)
                Controls impact of document length (0=no normalization, 1=full)
        """
        self.k1 = k1
        self.b = b

        # Vocabulary mapping: word -> index
        self.vocabulary: Dict[str, int] = {}

        # IDF scores: word -> IDF value
        self.idf_scores: Dict[str, float] = {}

        # Corpus statistics
        self.doc_count = 0
        self.avg_doc_len = 0.0

        # Fitted flag
        self.is_fitted = False

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words

        Args:
            text: Input text

        Returns:
            List of lowercase tokens
        """
        # Lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def fit(self, documents: List[str]):
        """Build vocabulary and compute IDF from document corpus

        Args:
            documents: List of text documents
        """
        if not documents:
            raise ValueError("Cannot fit on empty document list")

        # Tokenize all documents
        tokenized_docs = [self.tokenize(doc) for doc in documents]

        # Compute document frequencies (how many docs contain each word)
        doc_frequencies = defaultdict(int)
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_frequencies[token] += 1

        # Build vocabulary
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(doc_frequencies.keys()))}

        # Compute IDF scores using BM25 IDF formula
        # IDF(qi) = ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        # where N = total docs, n(qi) = docs containing term qi
        N = len(documents)
        self.doc_count = N

        for word, df in doc_frequencies.items():
            # BM25 IDF with smoothing
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_scores[word] = max(0.0, idf)  # Ensure non-negative

        # Compute average document length
        doc_lengths = [len(tokens) for tokens in tokenized_docs]
        self.avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0

        self.is_fitted = True

        print(f"[BM25] Fitted on {N} documents")
        print(f"[BM25] Vocabulary size: {len(self.vocabulary)}")
        print(f"[BM25] Average document length: {self.avg_doc_len:.1f} tokens")

    def encode(self, text: str) -> Dict[str, Any]:
        """Generate sparse vector for text using BM25 scoring

        Args:
            text: Input text to encode

        Returns:
            Dictionary with 'indices' and 'values' lists for sparse vector
            Format: {"indices": [int, ...], "values": [float, ...]}
        """
        if not self.is_fitted:
            raise RuntimeError("Encoder must be fitted before encoding. Call fit() first.")

        # Tokenize text
        tokens = self.tokenize(text)
        doc_len = len(tokens)

        # Compute term frequencies
        term_freqs = Counter(tokens)

        # Compute BM25 scores for each term
        indices = []
        values = []

        for word, tf in term_freqs.items():
            # Skip words not in vocabulary
            if word not in self.vocabulary:
                continue

            # Get word index and IDF
            word_idx = self.vocabulary[word]
            idf = self.idf_scores.get(word, 0.0)

            # BM25 formula:
            # score = IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))

            bm25_score = idf * (numerator / denominator)

            # Only include non-zero scores
            if bm25_score > 0:
                indices.append(word_idx)
                values.append(bm25_score)

        return {
            "indices": indices,
            "values": values
        }

    def save(self, path: Path):
        """Save encoder state to disk

        Args:
            path: File path to save encoder state
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted encoder")

        state = {
            'k1': self.k1,
            'b': self.b,
            'vocabulary': self.vocabulary,
            'idf_scores': self.idf_scores,
            'doc_count': self.doc_count,
            'avg_doc_len': self.avg_doc_len,
            'is_fitted': self.is_fitted
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)

        print(f"[BM25] Encoder saved to {path}")

    def load(self, path: Path) -> bool:
        """Load encoder state from disk

        Args:
            path: File path to load encoder state from

        Returns:
            True if loaded successfully, False otherwise
        """
        if not path.exists():
            print(f"[BM25] Encoder file not found: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)

            self.k1 = state['k1']
            self.b = state['b']
            self.vocabulary = state['vocabulary']
            self.idf_scores = state['idf_scores']
            self.doc_count = state['doc_count']
            self.avg_doc_len = state['avg_doc_len']
            self.is_fitted = state['is_fitted']

            print(f"[BM25] Encoder loaded from {path}")
            print(f"[BM25] Vocabulary size: {len(self.vocabulary)}")
            return True

        except Exception as e:
            print(f"[BM25] Error loading encoder: {e}")
            return False
