"""Enhanced retrieval module with dense, sparse, and hybrid retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import importlib
import importlib.util

from .candidates import Passage


@dataclass
class RetrievalResult:
    """Results from retrieval."""
    passages: List[Passage]
    scores: List[float]
    metadata: Dict[str, Any] = None


class DenseRetriever:
    """Dense retrieval using sentence transformers."""

    def __init__(
        self,
        encoder_model: str = "facebook/contriever",
        corpus_path: Optional[str] = None,
        index_path: Optional[str] = None,
        device: str = "cuda:0",
        top_k: int = 100
    ):
        self.device = device
        self.top_k = top_k

        # Load encoder lazily to avoid heavyweight imports when unused
        if "contriever" in encoder_model.lower():
            transformers_spec = importlib.util.find_spec("transformers")
            if transformers_spec is None:
                raise ModuleNotFoundError(
                    "transformers is required for dense retrieval. "
                    "Install it or use --retrieval_method sparse"
                )

            transformers = importlib.import_module("transformers")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(encoder_model)  # nosec B615
            self.encoder = transformers.AutoModel.from_pretrained(encoder_model).to(device)  # nosec B615
            self.use_contriever = True
        else:
            st_spec = importlib.util.find_spec("sentence_transformers")
            if st_spec is None:
                raise ModuleNotFoundError(
                    "sentence-transformers is required for dense retrieval. "
                    "Install it or use --retrieval_method sparse"
                )

            sentence_transformers = importlib.import_module("sentence_transformers")
            self.encoder = sentence_transformers.SentenceTransformer(encoder_model, device=device)
            self.use_contriever = False

        # Load corpus and index
        self.corpus = []
        self.corpus_embeddings = None

        if corpus_path:
            self.load_corpus(corpus_path)

        if index_path and Path(index_path).exists():
            self.load_index(index_path)

    def load_corpus(self, corpus_path: str) -> None:
        """Load document corpus."""
        path = Path(corpus_path)

        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.corpus = data
                elif isinstance(data, dict):
                    # Handle different corpus formats
                    if 'documents' in data:
                        self.corpus = data['documents']
                    elif 'passages' in data:
                        self.corpus = data['passages']
                    else:
                        # Assume dict of id -> text
                        self.corpus = list(data.values())
        elif path.suffix == '.jsonl':
            self.corpus = []
            with open(path) as f:
                for line in f:
                    doc = json.loads(line.strip())
                    if isinstance(doc, str):
                        self.corpus.append(doc)
                    elif isinstance(doc, dict):
                        # Extract text field
                        text = doc.get('text', doc.get('passage', doc.get('content', '')))
                        self.corpus.append(text)
        else:
            # Plain text, one document per line
            with open(path) as f:
                self.corpus = [line.strip() for line in f if line.strip()]

    def build_index(self, save_path: Optional[str] = None) -> None:
        """Build embedding index for corpus."""
        if not self.corpus:
            raise ValueError("No corpus loaded")

        print(f"Building index for {len(self.corpus)} documents...")
        embeddings = []
        batch_size = 32

        total_batches = (len(self.corpus) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(self.corpus), batch_size)):
            batch = self.corpus[i:i + batch_size]

            # Progress logging
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f"  Encoding batch {batch_idx + 1}/{total_batches} ({i}/{len(self.corpus)} docs)")

            if self.use_contriever:
                batch_embeddings = self._encode_contriever(batch)
            else:
                batch_embeddings = self.encoder.encode(
                    batch,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
                batch_embeddings = batch_embeddings.cpu().numpy()

            embeddings.append(batch_embeddings)

        self.corpus_embeddings = np.vstack(embeddings)
        print(f"Index built: {self.corpus_embeddings.shape}")

        if save_path:
            self.save_index(save_path)

    def _encode_contriever(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Contriever model."""
        torch = importlib.import_module("torch")
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = self._mean_pooling(outputs[0], inputs['attention_mask'])
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling for sentence embeddings."""
        torch = importlib.import_module("torch")
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def save_index(self, path: str) -> None:
        """Save an embedding index without executable object serialization."""
        if self.corpus_embeddings is None:
            raise ValueError("No embedding index has been built")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            np.savez_compressed(
                f,
                embeddings=np.asarray(self.corpus_embeddings),
                corpus=np.asarray(self.corpus, dtype=np.str_),
            )

    def load_index(self, path: str) -> None:
        """Load an embedding index created by :meth:`save_index`."""
        with np.load(path, allow_pickle=False) as data:
            self.corpus_embeddings = np.asarray(data['embeddings'])
            self.corpus = data['corpus'].astype(str).tolist()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve passages for a query."""
        # CRITICAL FIX: Handle empty corpus gracefully
        if not self.corpus:
            # No corpus loaded - return empty result
            return RetrievalResult(passages=[], scores=[])

        if self.corpus_embeddings is None:
            self.build_index()

        top_k = top_k or self.top_k

        # Encode query
        if self.use_contriever:
            query_embedding = self._encode_contriever([query])[0]
        else:
            query_embedding = self.encoder.encode(
                query,
                convert_to_tensor=True,
                show_progress_bar=False
            ).cpu().numpy()

        # Compute similarities
        similarities = np.dot(self.corpus_embeddings, query_embedding)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Create passages
        passages = []
        scores = []

        for idx in top_indices:
            text = self.corpus[idx]
            passage = Passage(
                pid=f"dense_{idx}",
                text=text,
                cost=len(text.split()) * 4,  # Rough token estimate
                metadata={'retriever': 'dense', 'corpus_idx': int(idx)}
            )
            passages.append(passage)
            scores.append(float(similarities[idx]))

        return RetrievalResult(passages=passages, scores=scores)


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods."""

    def __init__(
        self,
        encoder_model: str = "facebook/contriever",
        corpus_path: Optional[str] = None,
        device: str = "cuda:0",
        top_k: int = 100,
        alpha: float = 0.5  # Weight for dense retrieval
    ):
        self.top_k = top_k
        self.alpha = alpha

        # Initialize dense retriever
        self.dense_retriever = DenseRetriever(
            encoder_model=encoder_model,
            corpus_path=corpus_path,
            device=device,
            top_k=top_k * 2  # Get more candidates for fusion
        )

        # Initialize sparse retriever (BM25)
        self.sparse_retriever = BM25Retriever(corpus_path=corpus_path)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve using hybrid approach."""
        top_k = top_k or self.top_k

        # Get dense results
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)

        # Get sparse results
        sparse_results = self.sparse_retriever.retrieve(query, top_k=top_k * 2)

        # Combine scores with reciprocal rank fusion
        combined_scores = {}

        # Add dense scores
        for rank, (passage, score) in enumerate(zip(dense_results.passages, dense_results.scores)):
            pid = passage.pid
            combined_scores[pid] = self.alpha * (1.0 / (rank + 1))

        # Add sparse scores
        for rank, (passage, score) in enumerate(zip(sparse_results.passages, sparse_results.scores)):
            pid = passage.pid
            if pid in combined_scores:
                combined_scores[pid] += (1 - self.alpha) * (1.0 / (rank + 1))
            else:
                combined_scores[pid] = (1 - self.alpha) * (1.0 / (rank + 1))

        # Create passage lookup
        all_passages = {p.pid: p for p in dense_results.passages + sparse_results.passages}

        # Sort by combined score
        sorted_pids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)

        # Get top-k passages
        passages = []
        scores = []

        for pid in sorted_pids[:top_k]:
            passages.append(all_passages[pid])
            scores.append(combined_scores[pid])

        return RetrievalResult(
            passages=passages,
            scores=scores,
            metadata={'method': 'hybrid', 'alpha': self.alpha}
        )


class BM25Retriever:
    """Sparse retrieval using BM25."""

    def __init__(self, corpus_path: Optional[str] = None, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.tokenized_corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf_scores = {}
        self.N = 0

        if corpus_path:
            self.load_corpus(corpus_path)
            self.build_index()

    def load_corpus(self, corpus_path: str) -> None:
        """Load corpus for BM25."""
        path = Path(corpus_path)

        if path.suffix == '.json':
            with open(path) as f:
                data = json.load(f)
                if isinstance(data, list):
                    self.corpus = data
                elif isinstance(data, dict) and 'documents' in data:
                    self.corpus = data['documents']
        else:
            with open(path) as f:
                self.corpus = [line.strip() for line in f if line.strip()]

    def build_index(self) -> None:
        """Build BM25 index."""
        self.N = len(self.corpus)

        # Tokenize corpus
        for doc in self.corpus:
            tokens = doc.lower().split()
            self.tokenized_corpus.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        # Calculate IDF scores
        for token, df in self.doc_freqs.items():
            self.idf_scores[token] = np.log((self.N - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 100) -> RetrievalResult:
        """Retrieve using BM25."""
        query_tokens = query.lower().split()

        scores = []
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            score = self._compute_bm25_score(query_tokens, doc_tokens, self.doc_lengths[idx])
            scores.append((idx, score))

        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Create passages
        passages = []
        passage_scores = []

        for idx, score in scores[:top_k]:
            text = self.corpus[idx]
            passage = Passage(
                pid=f"bm25_{idx}",
                text=text,
                cost=len(text.split()) * 4,
                metadata={'retriever': 'bm25', 'corpus_idx': idx}
            )
            passages.append(passage)
            passage_scores.append(score)

        return RetrievalResult(passages=passages, scores=passage_scores)

    def _compute_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], doc_length: int) -> float:
        """Compute BM25 score for a document."""
        score = 0.0
        doc_token_counts = {}

        for token in doc_tokens:
            doc_token_counts[token] = doc_token_counts.get(token, 0) + 1

        for query_token in query_tokens:
            if query_token not in self.idf_scores:
                continue

            idf = self.idf_scores[query_token]
            tf = doc_token_counts.get(query_token, 0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)

            score += idf * (numerator / denominator)

        return score
