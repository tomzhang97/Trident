"""NLI/Cross-encoder scoring module for facet sufficiency testing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .facets import Facet
from .candidates import Passage
from .config import NLIConfig


@dataclass
class NLIScore:
    """NLI scoring result."""
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    final_score: float  # Sufficiency score


class NLIScorer:
    """Cross-encoder/NLI model for scoring passage-facet pairs."""

    def __init__(self, config: NLIConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)  # nosec B615
        self.model = AutoModelForSequenceClassification.from_pretrained(  # nosec B615
            config.model_name
        ).to(device)
        self.model.eval()

        # Cache with LRU eviction
        self.cache_size = config.cache_size
        self.cache: OrderedDict[Tuple[str, str], float] = OrderedDict()

        # Label mappings for common NLI models
        self.label_mapping = self._get_label_mapping(config.model_name)

    def _get_label_mapping(self, model_name: str) -> Dict[str, int]:
        """Get label mapping for different NLI models."""
        # Dynamically detect number of labels
        num_labels = self.model.config.num_labels
        if num_labels == 2:
            return {"ENTAILMENT": 1, "NEUTRAL": 0, "CONTRADICTION": 0}
        elif "deberta" in model_name.lower() or "mnli" in model_name.lower():
            return {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
        elif "roberta" in model_name.lower():
            return {"entailment": 2, "neutral": 1, "contradiction": 0}
        else:
            return {"entailment": 0, "neutral": 1, "contradiction": 2}

    def score(self, passage: Passage, facet: Facet) -> float:
        """Score passage-facet pair for sufficiency."""
        # Check cache
        cache_key = (passage.pid, facet.facet_id)
        if cache_key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        # Prepare input for NLI
        premise = passage.text
        hypothesis = facet.to_hypothesis()

        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1)

        # Extract scores
        entail_idx = self.label_mapping.get("ENTAILMENT", self.label_mapping.get("entailment", 0))
        neutral_idx = self.label_mapping.get("NEUTRAL", self.label_mapping.get("neutral", 1))
        contra_idx = self.label_mapping.get("CONTRADICTION", self.label_mapping.get("contradiction", 2))

        entailment_score = probs[entail_idx].item()
        neutral_score = probs[neutral_idx].item()
        contradiction_score = probs[contra_idx].item()

        # Compute sufficiency score
        # Higher entailment and lower contradiction indicates sufficiency
        sufficiency_score = entailment_score - 0.5 * contradiction_score

        # Update cache with LRU eviction
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest
        self.cache[cache_key] = sufficiency_score

        return sufficiency_score

    def batch_score(
        self,
        pairs: List[Tuple[Passage, Facet]]
    ) -> List[float]:
        """Batch scoring for efficiency."""
        scores = []

        # Process in batches
        for i in range(0, len(pairs), self.config.batch_size):
            batch = pairs[i:i + self.config.batch_size]

            # Check cache first
            batch_to_score = []
            cached_scores = {}

            for passage, facet in batch:
                cache_key = (passage.pid, facet.facet_id)
                if cache_key in self.cache:
                    cached_scores[cache_key] = self.cache[cache_key]
                else:
                    batch_to_score.append((passage, facet))

            # Score uncached pairs
            if batch_to_score:
                premises = [p.text for p, _ in batch_to_score]
                hypotheses = [f.to_hypothesis() for _, f in batch_to_score]

                # Batch tokenization
                inputs = self.tokenizer(
                    premises,
                    hypotheses,
                    truncation=True,
                    max_length=self.config.max_length,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)

                # Batch inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)

                # Extract scores
                entail_idx = self.label_mapping.get("ENTAILMENT", self.label_mapping.get("entailment", 0))
                contra_idx = self.label_mapping.get("CONTRADICTION", self.label_mapping.get("contradiction", 2))

                for j, (passage, facet) in enumerate(batch_to_score):
                    entailment_score = probs[j, entail_idx].item()
                    contradiction_score = probs[j, contra_idx].item()
                    sufficiency_score = entailment_score - 0.5 * contradiction_score

                    # Update cache
                    cache_key = (passage.pid, facet.facet_id)
                    if len(self.cache) >= self.cache_size:
                        self.cache.popitem(last=False)
                    self.cache[cache_key] = sufficiency_score

            # Collect all scores in order
            for passage, facet in batch:
                cache_key = (passage.pid, facet.facet_id)
                if cache_key in cached_scores:
                    scores.append(cached_scores[cache_key])
                else:
                    scores.append(self.cache[cache_key])

        return scores

    def score_with_details(self, passage: Passage, facet: Facet) -> NLIScore:
        """Get detailed NLI scores."""
        premise = passage.text
        hypothesis = facet.to_hypothesis()

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1)

        entail_idx = self.label_mapping.get("ENTAILMENT", self.label_mapping.get("entailment", 0))
        neutral_idx = self.label_mapping.get("NEUTRAL", self.label_mapping.get("neutral", 1))
        contra_idx = self.label_mapping.get("CONTRADICTION", self.label_mapping.get("contradiction", 2))

        entailment_score = probs[entail_idx].item()
        neutral_score = probs[neutral_idx].item()
        contradiction_score = probs[contra_idx].item()
        sufficiency_score = entailment_score - 0.5 * contradiction_score

        return NLIScore(
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
            final_score=sufficiency_score
        )

    def clear_cache(self) -> None:
        """Clear the score cache."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size,
            'cache_hit_rate': 0.0  # Would need to track hits/misses for this
        }


class CrossEncoderReranker:
    """Cross-encoder for reranking passages."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda:0"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  # nosec B615
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)  # nosec B615
        self.model.eval()

    def rerank(
        self,
        query: str,
        passages: List[Passage],
        top_k: int = 20
    ) -> List[Tuple[Passage, float]]:
        """Rerank passages for a query."""
        if not passages:
            return []

        # Score all passages
        scores = []

        for passage in passages:
            inputs = self.tokenizer(
                query,
                passage.text,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits[0, 0].item()  # Relevance score
                scores.append((passage, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def batch_rerank(
        self,
        query: str,
        passages: List[Passage],
        batch_size: int = 32,
        top_k: int = 20
    ) -> List[Tuple[Passage, float]]:
        """Batch reranking for efficiency."""
        if not passages:
            return []

        all_scores = []

        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            queries = [query] * len(batch_passages)
            texts = [p.text for p in batch_passages]

            inputs = self.tokenizer(
                queries,
                texts,
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits[:, 0].cpu().numpy()

            for passage, score in zip(batch_passages, scores):
                all_scores.append((passage, float(score)))

        # Sort and return top-k
        all_scores.sort(key=lambda x: x[1], reverse=True)
        return all_scores[:top_k]
