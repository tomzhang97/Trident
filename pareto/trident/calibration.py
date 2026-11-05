"""
Enhanced calibration utilities with selection-conditional Mondrian conformal prediction.

Per TRIDENT Framework Sections 3.3-3.5:
- Selection-conditional Mondrian calibration with bins:
  b(p,f) = (facet_type(f), length_bucket(p), retriever_score_bucket(p))
- Typical grid: 6 × 3 × 3 = 54 bins
- Bin merging when insufficient negatives (N_min threshold)
- Deterministic and randomized conformal p-values
- Label-noise robustness with denominator-inflated p-values
"""

from __future__ import annotations

import bisect
import hashlib
import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Any, Set

import numpy as np


class PValueMode(Enum):
    """Mode for p-value computation."""
    DETERMINISTIC = "deterministic"  # Conservative: (1 + #{s' >= s}) / (1 + n)
    RANDOMIZED = "randomized"  # Exact: (1 + #{s' > s} + U * #{s' = s}) / (1 + n)


@dataclass
class BinSpec:
    """
    Specification for Mondrian calibration bins.

    Per Section 3.3:
    b(p,f) = (facet_type(f), length_bucket(p), retriever_score_bucket(p))
    Typical grid: 6 × 3 × 3 = 54 bins
    """
    # Facet types (from framework Section 2.1)
    facet_types: Tuple[str, ...] = (
        "ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE_HOP1", "BRIDGE_HOP2"
    )

    # Length buckets: short (0-50), medium (50-150), long (150+)
    length_buckets: Tuple[Tuple[int, int, str], ...] = (
        (0, 50, "short"),
        (50, 150, "medium"),
        (150, float('inf'), "long"),
    )

    # Retriever score buckets
    retriever_score_buckets: Tuple[Tuple[float, float, str], ...] = (
        (0.0, 0.33, "low"),
        (0.33, 0.67, "medium"),
        (0.67, 1.0, "high"),
    )

    def get_bin_key(
        self,
        facet_type: str,
        text_length: int,
        retriever_score: float = 0.5
    ) -> str:
        """Get the bin key for given features."""
        # Normalize facet type
        facet_type_str = facet_type.upper() if isinstance(facet_type, str) else str(facet_type)
        if facet_type_str not in self.facet_types:
            # Use closest match or default
            facet_type_str = "ENTITY"

        # Determine length bucket
        length_bin = "medium"
        for min_len, max_len, bin_name in self.length_buckets:
            if min_len <= text_length < max_len:
                length_bin = bin_name
                break

        # Determine retriever score bucket
        retriever_bin = "medium"
        for min_score, max_score, bin_name in self.retriever_score_buckets:
            if min_score <= retriever_score < max_score:
                retriever_bin = bin_name
                break

        return f"{facet_type_str}_{length_bin}_{retriever_bin}"

    def get_all_bin_keys(self) -> List[str]:
        """Get all possible bin keys (6 × 3 × 3 = 54)."""
        keys = []
        for ft in self.facet_types:
            for _, _, lb in self.length_buckets:
                for _, _, rb in self.retriever_score_buckets:
                    keys.append(f"{ft}_{lb}_{rb}")
        return keys

    def get_coarser_bin(self, bin_key: str) -> Optional[str]:
        """
        Get a coarser bin by collapsing dimensions.

        Per Section 3.3: First collapse retriever-score, then length.
        """
        parts = bin_key.split("_")
        if len(parts) != 3:
            return None

        facet_type, length_bin, retriever_bin = parts

        # First, collapse retriever score bucket
        if retriever_bin != "all":
            return f"{facet_type}_{length_bin}_all"

        # Then, collapse length bucket
        if length_bin != "all":
            return f"{facet_type}_all_all"

        # Finally, collapse to just facet type
        return facet_type


@dataclass
class ConformalBin:
    """
    A bin for selection-conditional conformal calibration.

    Stores negative scores (Σ(p,f) = 0) for computing conformal p-values.
    """
    bin_key: str
    negative_scores: List[float] = field(default_factory=list)
    n_positives: int = 0
    is_merged: bool = False
    merged_from: List[str] = field(default_factory=list)

    def add_negative(self, score: float) -> None:
        """Add a negative sample score."""
        self.negative_scores.append(score)

    def add_positive(self) -> None:
        """Track positive sample count."""
        self.n_positives += 1

    def finalize(self) -> None:
        """Sort scores descending for efficient p-value computation."""
        self.negative_scores.sort(reverse=True)

    @property
    def n_negatives(self) -> int:
        """Number of negative samples."""
        return len(self.negative_scores)

    def compute_pvalue_deterministic(self, score: float) -> float:
        """
        Compute deterministic (conservative) conformal p-value.

        Per Section 3.4:
        π(p,f) = (1 + #{s' ∈ N_b : s' >= s(p,f)}) / (1 + n)

        Super-uniform: Pr[π ≤ t | ¬Σ] ≤ t + 1/(n+1)
        """
        if not self.negative_scores:
            return 1.0

        n = len(self.negative_scores)
        # Count scores >= test score (since sorted descending, use bisect)
        # Find first index where score < test_score
        count_geq = bisect.bisect_left(
            [-s for s in self.negative_scores],
            -score
        )

        return (1 + count_geq) / (1 + n)

    def compute_pvalue_randomized(self, score: float, u: Optional[float] = None) -> float:
        """
        Compute randomized (exact) conformal p-value.

        Per Section 3.4:
        π_rand(p,f) = (1 + #{s' > s} + U · #{s' = s}) / (1 + n), U ~ Unif(0,1)

        Exact super-uniformity: Pr[π_rand ≤ t | ¬Σ] = t
        """
        if not self.negative_scores:
            return 1.0

        if u is None:
            u = random.random()

        n = len(self.negative_scores)
        count_greater = sum(1 for s in self.negative_scores if s > score)
        count_equal = sum(1 for s in self.negative_scores if abs(s - score) < 1e-9)

        return (1 + count_greater + u * count_equal) / (1 + n)

    def compute_pvalue_robust(
        self,
        score: float,
        epsilon: float = 0.0
    ) -> float:
        """
        Compute label-noise robust p-value.

        Per Section 3.5:
        π_robust = (1 + #{s' >= s}) / (1 + (1 - ε) * n)

        If up to ε of negatives are mislabeled.
        """
        if not self.negative_scores:
            return 1.0

        n = len(self.negative_scores)
        count_geq = sum(1 for s in self.negative_scores if s >= score)

        effective_n = (1 - epsilon) * n
        return (1 + count_geq) / (1 + effective_n)

    def min_achievable_pvalue(self) -> float:
        """
        Get minimum achievable p-value for this bin.

        Per Section 3.4 (Feasibility guard):
        Deterministic p-values cannot go below 1/(n_b + 1)
        """
        if not self.negative_scores:
            return 1.0
        return 1.0 / (len(self.negative_scores) + 1)


@dataclass
class SelectionConditionalCalibrator:
    """
    Selection-conditional Mondrian calibrator.

    Per Section 3.3:
    Replays the exact retriever and shortlister used at test time on a separate
    labeled calibration corpus to build negative score pools per bin.
    """
    bins: Dict[str, ConformalBin] = field(default_factory=dict)
    bin_spec: BinSpec = field(default_factory=BinSpec)
    n_min: int = 50  # Minimum per-bin negatives
    version: str = "v1.0"
    pvalue_mode: PValueMode = PValueMode.DETERMINISTIC
    epsilon: float = 0.0  # Label noise rate for robust p-values
    use_mondrian: bool = True

    # Version hashes for reproducibility (Section 10)
    retriever_hash: str = ""
    shortlister_hash: str = ""
    verifier_hash: str = ""
    corpus_hash: str = ""

    # Merged bin tracking
    merged_bins: Dict[str, str] = field(default_factory=dict)  # original -> merged

    def add_calibration_sample(
        self,
        score: float,
        is_sufficient: bool,
        facet_type: str,
        text_length: int,
        retriever_score: float = 0.5
    ) -> None:
        """Add a calibration sample to the appropriate bin."""
        bin_key = self.bin_spec.get_bin_key(facet_type, text_length, retriever_score)

        if bin_key not in self.bins:
            self.bins[bin_key] = ConformalBin(bin_key=bin_key)

        if is_sufficient:
            self.bins[bin_key].add_positive()
        else:
            self.bins[bin_key].add_negative(score)

    def finalize(self) -> None:
        """
        Finalize calibration: merge bins and sort scores.

        Per Section 3.3:
        If fewer than N_min negatives, merge to a coarser bin
        (first collapse retriever-score, then length).
        """
        # First, finalize all bins
        for bin_obj in self.bins.values():
            bin_obj.finalize()

        # Then, handle bin merging
        self._merge_small_bins()

    def _merge_small_bins(self) -> None:
        """Merge bins with insufficient negatives."""
        bins_to_merge = []

        for bin_key, bin_obj in self.bins.items():
            if bin_obj.n_negatives < self.n_min:
                bins_to_merge.append(bin_key)

        for bin_key in bins_to_merge:
            self._merge_bin(bin_key)

    def _merge_bin(self, bin_key: str) -> None:
        """
        Merge a bin into a coarser bin.

        Per Section 3.3: First collapse retriever-score, then length.
        """
        coarser_key = self.bin_spec.get_coarser_bin(bin_key)

        if coarser_key is None or coarser_key == bin_key:
            # Can't merge further; this bin will have limited resolution
            return

        # Create or get the coarser bin
        if coarser_key not in self.bins:
            self.bins[coarser_key] = ConformalBin(bin_key=coarser_key, is_merged=True)

        # Merge scores
        original_bin = self.bins[bin_key]
        self.bins[coarser_key].negative_scores.extend(original_bin.negative_scores)
        self.bins[coarser_key].n_positives += original_bin.n_positives
        self.bins[coarser_key].merged_from.append(bin_key)
        self.bins[coarser_key].finalize()

        # Track the merge
        self.merged_bins[bin_key] = coarser_key

        # Check if merged bin still needs further merging
        if self.bins[coarser_key].n_negatives < self.n_min:
            self._merge_bin(coarser_key)

    def get_effective_bin(self, bin_key: str) -> str:
        """Get the effective bin after merging."""
        while bin_key in self.merged_bins:
            bin_key = self.merged_bins[bin_key]
        return bin_key

    def compute_pvalue(
        self,
        score: float,
        facet_type: str,
        text_length: int,
        retriever_score: float = 0.5,
        u: Optional[float] = None
    ) -> Tuple[float, str]:
        """
        Compute conformal p-value for a score.

        Returns: (p_value, bin_key_used)
        """
        bin_key = self.bin_spec.get_bin_key(facet_type, text_length, retriever_score)
        effective_key = self.get_effective_bin(bin_key)

        if effective_key not in self.bins:
            # No calibration data; return conservative p-value
            return 1.0 - score, bin_key

        bin_obj = self.bins[effective_key]

        if self.epsilon > 0:
            pvalue = bin_obj.compute_pvalue_robust(score, self.epsilon)
        elif self.pvalue_mode == PValueMode.RANDOMIZED:
            pvalue = bin_obj.compute_pvalue_randomized(score, u)
        else:
            pvalue = bin_obj.compute_pvalue_deterministic(score)

        return pvalue, effective_key

    def check_feasibility(
        self,
        alpha_bar: float,
        facet_type: str,
        text_length: int,
        retriever_score: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Check if threshold is feasible for given bin.

        Per Section 3.4 (Feasibility guard):
        If ᾱ_f < 1/(n_b+1) for any used bin of facet f, we either enable
        randomized p-values for f or merge bins until feasible.

        Returns: (is_feasible, reason)
        """
        bin_key = self.bin_spec.get_bin_key(facet_type, text_length, retriever_score)
        effective_key = self.get_effective_bin(bin_key)

        if effective_key not in self.bins:
            return True, "no_calibration_data"

        bin_obj = self.bins[effective_key]
        min_pvalue = bin_obj.min_achievable_pvalue()

        if alpha_bar < min_pvalue:
            if self.pvalue_mode == PValueMode.RANDOMIZED:
                return True, "randomized_mode"
            else:
                return False, f"threshold_too_strict: {alpha_bar} < {min_pvalue}"

        return True, "feasible"

    def to_pvalue(
        self,
        score: float,
        bucket_key: str,
        text_length: Optional[int] = None
    ) -> float:
        """
        Legacy compatibility method for mapping score to p-value.

        Parses bucket_key to extract facet type and uses defaults for other params.
        """
        # Parse facet type from bucket key
        facet_type = bucket_key.split('_')[0] if '_' in bucket_key else bucket_key
        length = text_length if text_length is not None else 100

        pvalue, _ = self.compute_pvalue(score, facet_type, length)
        return pvalue

    def get_bin_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all bins."""
        stats = {}
        for bin_key, bin_obj in self.bins.items():
            stats[bin_key] = {
                'n_negatives': bin_obj.n_negatives,
                'n_positives': bin_obj.n_positives,
                'is_merged': bin_obj.is_merged,
                'merged_from': bin_obj.merged_from,
                'min_pvalue': bin_obj.min_achievable_pvalue(),
            }
        return stats

    def save(self, path: str) -> None:
        """Save calibrator to file."""
        data = {
            'version': self.version,
            'n_min': self.n_min,
            'pvalue_mode': self.pvalue_mode.value,
            'epsilon': self.epsilon,
            'use_mondrian': self.use_mondrian,
            'retriever_hash': self.retriever_hash,
            'shortlister_hash': self.shortlister_hash,
            'verifier_hash': self.verifier_hash,
            'corpus_hash': self.corpus_hash,
            'merged_bins': self.merged_bins,
            'bins': {}
        }

        for bin_key, bin_obj in self.bins.items():
            data['bins'][bin_key] = {
                'negative_scores': bin_obj.negative_scores,
                'n_positives': bin_obj.n_positives,
                'is_merged': bin_obj.is_merged,
                'merged_from': bin_obj.merged_from,
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SelectionConditionalCalibrator":
        """Load calibrator from file."""
        with open(path) as f:
            data = json.load(f)

        calibrator = cls(
            version=data['version'],
            n_min=data['n_min'],
            pvalue_mode=PValueMode(data['pvalue_mode']),
            epsilon=data.get('epsilon', 0.0),
            use_mondrian=data.get('use_mondrian', True),
            retriever_hash=data.get('retriever_hash', ''),
            shortlister_hash=data.get('shortlister_hash', ''),
            verifier_hash=data.get('verifier_hash', ''),
            corpus_hash=data.get('corpus_hash', ''),
            merged_bins=data.get('merged_bins', {}),
        )

        for bin_key, bin_data in data.get('bins', {}).items():
            bin_obj = ConformalBin(
                bin_key=bin_key,
                negative_scores=bin_data['negative_scores'],
                n_positives=bin_data['n_positives'],
                is_merged=bin_data.get('is_merged', False),
                merged_from=bin_data.get('merged_from', []),
            )
            calibrator.bins[bin_key] = bin_obj

        return calibrator

    def get_version_hash(self) -> str:
        """Get hash of calibrator for versioning."""
        content = json.dumps({
            'version': self.version,
            'n_bins': len(self.bins),
            'total_negatives': sum(b.n_negatives for b in self.bins.values()),
            'retriever_hash': self.retriever_hash,
            'verifier_hash': self.verifier_hash,
        }, sort_keys=True)
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]


def build_calibrator(
    calibration_data: List[Dict[str, Any]],
    retriever: Any,
    shortlister: Any,
    verifier: Any,
    bin_spec: BinSpec,
    t_f: int = 10,
    n_min: int = 50,
    pvalue_mode: PValueMode = PValueMode.DETERMINISTIC,
    epsilon: float = 0.0
) -> SelectionConditionalCalibrator:
    """
    Build selection-conditional calibrator.

    Per Section 9.1 (BuildCalibrator):
    Input: D_cal (labeled (q,p,f,Σ)), retriever R, shortlister S, verifier V,
           BinSpec B, caps {T_f}, N_min
    For each bin b: N_b ← []
    For each q in D_cal:
      P_ret ← R(q)
      For each f in facets(q):
        P_f ← S(f, P_ret, T_f)   // exact replay, deterministic
        For p in P_f:
          s ← V.score(p,f)
          if Σ(p,f) = 0: N_{B(p,f)}.append(s)
    Merge bins until |N_b| ≥ N_min per policy; sort N_b desc; store hashes.
    Return Calibrator C = {N_b, versions}
    """
    calibrator = SelectionConditionalCalibrator(
        bin_spec=bin_spec,
        n_min=n_min,
        pvalue_mode=pvalue_mode,
        epsilon=epsilon,
    )

    # Compute version hashes
    calibrator.retriever_hash = _compute_hash(retriever)
    calibrator.shortlister_hash = _compute_hash(shortlister)
    calibrator.verifier_hash = _compute_hash(verifier)

    # Process calibration data
    for sample in calibration_data:
        query = sample['query']
        facets = sample['facets']

        # Retrieve passages (exact replay)
        if retriever is not None:
            retrieved = retriever.retrieve(query)
            passages = retrieved.passages
            retriever_scores = {p.pid: s for p, s in zip(passages, retrieved.scores)}
        else:
            passages = sample.get('passages', [])
            retriever_scores = sample.get('retriever_scores', {})

        # For each facet
        for facet in facets:
            facet_id = facet.get('facet_id', str(facet))
            facet_type = facet.get('facet_type', 'ENTITY')

            # Shortlist passages for this facet (deterministic)
            if shortlister is not None:
                shortlisted = shortlister.shortlist(facet, passages, t_f)
            else:
                shortlisted = passages[:t_f]

            # Score each shortlisted passage
            for passage in shortlisted:
                pid = passage.pid if hasattr(passage, 'pid') else str(passage)

                # Get verifier score
                if verifier is not None:
                    score = verifier.score(passage, facet)
                else:
                    score = sample.get('scores', {}).get((pid, facet_id), 0.5)

                # Get sufficiency label
                is_sufficient = sample.get('labels', {}).get((pid, facet_id), False)

                # Get bin features
                text_length = len(passage.text.split()) if hasattr(passage, 'text') else 100
                retriever_score = retriever_scores.get(pid, 0.5)

                # Add to calibrator
                calibrator.add_calibration_sample(
                    score=score,
                    is_sufficient=is_sufficient,
                    facet_type=facet_type,
                    text_length=text_length,
                    retriever_score=retriever_score,
                )

    # Finalize (merge and sort)
    calibrator.finalize()

    return calibrator


def _compute_hash(obj: Any) -> str:
    """Compute a hash for versioning an object."""
    if obj is None:
        return "none"
    if hasattr(obj, 'get_version_hash'):
        return obj.get_version_hash()
    if hasattr(obj, '__class__'):
        return hashlib.md5(
            obj.__class__.__name__.encode(), usedforsecurity=False
        ).hexdigest()[:8]
    return hashlib.md5(str(obj).encode(), usedforsecurity=False).hexdigest()[:8]


# =============================================================================
# Legacy compatibility classes
# =============================================================================

def _pava(scores: Sequence[float], failures: Sequence[float]) -> List[float]:
    """Perform isotonic regression using pool-adjacent-violators algorithm."""

    if len(scores) != len(failures):
        raise ValueError("scores and failures must be the same length")

    # Sort by score ascending to enforce monotonicity
    pairs = sorted(zip(scores, failures), key=lambda x: x[0])
    values = [pair[1] for pair in pairs]
    weights = [1.0] * len(values)

    idx = 0
    while idx < len(values) - 1:
        if values[idx] <= values[idx + 1]:
            idx += 1
            continue

        # Violates monotonicity, pool adjacent values
        total_weight = weights[idx] + weights[idx + 1]
        avg = (values[idx] * weights[idx] + values[idx + 1] * weights[idx + 1]) / total_weight
        values[idx] = avg
        weights[idx] = total_weight
        del values[idx + 1]
        del weights[idx + 1]
        idx = max(idx - 1, 0)

    # Expand back to original length
    expanded: List[float] = []
    for weight, value in zip(weights, values):
        expanded.extend([value] * int(weight))

    # Pad if needed
    while len(expanded) < len(pairs):
        expanded.append(values[-1] if values else 0.0)

    return expanded[:len(pairs)]


@dataclass
class MondrianBin:
    """A bin in Mondrian conformal calibration (legacy compatibility)."""
    facet_type: str
    length_bin: str
    score_range: Tuple[float, float]
    samples: List[Tuple[float, bool]] = field(default_factory=list)
    calibration_map: Optional[List[Tuple[float, float]]] = None

    def add_sample(self, score: float, is_correct: bool) -> None:
        """Add a calibration sample."""
        self.samples.append((score, is_correct))

    def fit_isotonic(self) -> None:
        """Fit isotonic regression on this bin."""
        if not self.samples:
            return

        scores = [s for s, _ in self.samples]
        failures = [1.0 - int(correct) for _, correct in self.samples]

        iso_values = _pava(scores, failures)
        sorted_pairs = sorted(zip(scores, iso_values), key=lambda x: x[0])

        # Create calibration map
        stride = max(1, len(sorted_pairs) // 20)
        self.calibration_map = [sorted_pairs[i] for i in range(0, len(sorted_pairs), stride)]
        if sorted_pairs and self.calibration_map[-1][0] != sorted_pairs[-1][0]:
            self.calibration_map.append(sorted_pairs[-1])

    def get_pvalue(self, score: float) -> float:
        """Get calibrated p-value for score."""
        if not self.calibration_map:
            # Uncalibrated fallback
            return max(0.0, 1.0 - score)

        for threshold, p_value in self.calibration_map:
            if score <= threshold:
                return min(max(p_value, 0.0), 1.0)

        return min(max(self.calibration_map[-1][1], 0.0), 1.0)


@dataclass
class ReliabilityCalibrator:
    """
    Enhanced calibrator with Mondrian/split conformal prediction.

    This class now wraps SelectionConditionalCalibrator for new features
    while maintaining backward compatibility with existing code.
    """

    reliability_table_size: int = 20
    tables: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    mondrian_bins: Dict[str, MondrianBin] = field(default_factory=dict)
    version: str = "v1.0"
    use_mondrian: bool = True

    # New: Selection-conditional calibrator
    _conformal_calibrator: Optional[SelectionConditionalCalibrator] = None

    # Length bins for Mondrian
    length_bins = [(0, 50, "short"), (50, 150, "medium"), (150, float('inf'), "long")]

    # Score bins for additional stratification
    score_bins = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]

    def get_bin_key(
        self,
        facet_type: str,
        text_length: int,
        score: Optional[float] = None
    ) -> str:
        """Get Mondrian bin key for given features."""
        # Determine length bin
        length_bin = "medium"
        for min_len, max_len, bin_name in self.length_bins:
            if min_len <= text_length < max_len:
                length_bin = bin_name
                break

        # Optionally include score bin
        if score is not None and self.use_mondrian:
            for min_score, max_score in self.score_bins:
                if min_score <= score < max_score:
                    return f"{facet_type}_{length_bin}_{min_score:.1f}_{max_score:.1f}"

        return f"{facet_type}_{length_bin}"

    def fit(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        bucket_key: str,
        text_lengths: Optional[Sequence[int]] = None
    ) -> None:
        """
        Fit calibration for a bucket.

        If text_lengths provided and use_mondrian=True, uses Mondrian bins.
        Otherwise falls back to simple isotonic calibration.
        """
        if len(scores) == 0:
            raise ValueError("Cannot fit calibrator on empty data")

        if self.use_mondrian and text_lengths is not None:
            # Mondrian calibration
            self._fit_mondrian(scores, sufficiency_labels, bucket_key, text_lengths)
        else:
            # Simple isotonic calibration
            self._fit_isotonic(scores, sufficiency_labels, bucket_key)

    def _fit_isotonic(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        bucket_key: str
    ) -> None:
        """Fit simple isotonic regression."""
        failures = [1.0 - label for label in sufficiency_labels]
        iso_values = _pava(scores, failures)
        sorted_pairs = sorted(zip(scores, iso_values), key=lambda x: x[0])

        stride = max(1, len(sorted_pairs) // self.reliability_table_size)
        table = [sorted_pairs[i] for i in range(0, len(sorted_pairs), stride)]
        if table[-1][0] != sorted_pairs[-1][0]:
            table.append(sorted_pairs[-1])

        self.tables[bucket_key] = table

    def _fit_mondrian(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        facet_type: str,
        text_lengths: Sequence[int]
    ) -> None:
        """Fit Mondrian conformal calibration."""
        # Create bins and add samples
        for score, label, text_len in zip(scores, sufficiency_labels, text_lengths):
            bin_key = self.get_bin_key(facet_type, text_len, score)

            if bin_key not in self.mondrian_bins:
                # Determine bins
                length_bin = "medium"
                for min_len, max_len, bin_name in self.length_bins:
                    if min_len <= text_len < max_len:
                        length_bin = bin_name
                        break

                score_range = (0.0, 1.0)
                for min_score, max_score in self.score_bins:
                    if min_score <= score < max_score:
                        score_range = (min_score, max_score)
                        break

                self.mondrian_bins[bin_key] = MondrianBin(
                    facet_type=facet_type,
                    length_bin=length_bin,
                    score_range=score_range
                )

            self.mondrian_bins[bin_key].add_sample(score, bool(label))

        # Fit isotonic regression for each bin
        for bin_key in self.mondrian_bins:
            self.mondrian_bins[bin_key].fit_isotonic()

    def to_pvalue(
        self,
        score: float,
        bucket_key: str,
        text_length: Optional[int] = None
    ) -> float:
        """
        Map a score to a calibrated p-value.

        Uses conformal calibrator if available, otherwise falls back to legacy.
        """
        # Try new conformal calibrator first
        if self._conformal_calibrator is not None:
            facet_type = bucket_key.split('_')[0] if '_' in bucket_key else bucket_key
            length = text_length if text_length is not None else 100
            pvalue, _ = self._conformal_calibrator.compute_pvalue(
                score, facet_type, length
            )
            return pvalue

        # Try Mondrian first if enabled
        if self.use_mondrian and text_length is not None:
            # Parse facet type from bucket_key
            facet_type = bucket_key.split('_')[0] if '_' in bucket_key else bucket_key
            bin_key = self.get_bin_key(facet_type, text_length, score)

            if bin_key in self.mondrian_bins:
                return self.mondrian_bins[bin_key].get_pvalue(score)

        # Fall back to simple calibration
        table = self.tables.get(bucket_key)
        if not table:
            # Default monotone mapping
            return max(0.0, 1.0 - score)

        for threshold, p_value in table:
            if score <= threshold:
                return min(max(p_value, 0.0), 1.0)

        return min(max(table[-1][1], 0.0), 1.0)

    def set_conformal_calibrator(
        self,
        calibrator: SelectionConditionalCalibrator
    ) -> None:
        """Set the conformal calibrator for p-value computation."""
        self._conformal_calibrator = calibrator

    def merge(self, other: "ReliabilityCalibrator") -> None:
        """Merge calibration tables from another calibrator."""
        for key, table in other.tables.items():
            self.tables[key] = table

        for key, bin_obj in other.mondrian_bins.items():
            self.mondrian_bins[key] = bin_obj

    def save(self, path: str) -> None:
        """Save calibrator to file."""
        data = {
            'version': self.version,
            'use_mondrian': self.use_mondrian,
            'tables': dict(self.tables),
            'mondrian_bins': {}
        }

        # Serialize Mondrian bins
        for key, bin_obj in self.mondrian_bins.items():
            data['mondrian_bins'][key] = {
                'facet_type': bin_obj.facet_type,
                'length_bin': bin_obj.length_bin,
                'score_range': bin_obj.score_range,
                'calibration_map': bin_obj.calibration_map
            }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ReliabilityCalibrator":
        """Load calibrator from file."""
        with open(path) as f:
            data = json.load(f)

        calibrator = cls(
            version=data['version'],
            use_mondrian=data.get('use_mondrian', True)
        )
        calibrator.tables = data['tables']

        # Reconstruct Mondrian bins
        for key, bin_data in data.get('mondrian_bins', {}).items():
            bin_obj = MondrianBin(
                facet_type=bin_data['facet_type'],
                length_bin=bin_data['length_bin'],
                score_range=tuple(bin_data['score_range'])
            )
            bin_obj.calibration_map = bin_data['calibration_map']
            calibrator.mondrian_bins[key] = bin_obj

        return calibrator

    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, List[Tuple[float, float]]],
        version: str = "v1.0"
    ) -> "ReliabilityCalibrator":
        """Create calibrator from mapping."""
        calibrator = cls()
        calibrator.tables = mapping
        calibrator.version = version
        return calibrator

    def get_version_hash(self) -> str:
        """Get hash of calibration tables for versioning."""
        content = json.dumps({
            'tables': dict(self.tables),
            'mondrian_bins': len(self.mondrian_bins)
        }, sort_keys=True)
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]


@dataclass
class CalibrationMonitor:
    """Monitor calibration quality over time."""

    target_alpha: float
    window_size: int = 100
    history: List[Tuple[float, bool]] = field(default_factory=list)

    def update(self, p_value: float, is_correct: bool) -> None:
        """Update with new observation."""
        self.history.append((p_value, is_correct))

        # Keep only recent history
        if len(self.history) > self.window_size * 10:
            self.history = self.history[-self.window_size * 5:]

    def get_violation_rate(self, alpha: float) -> float:
        """Get empirical violation rate at given alpha."""
        if not self.history:
            return 0.0

        recent = self.history[-self.window_size:] if len(self.history) > self.window_size else self.history

        violations = sum(
            1 for p, correct in recent
            if p <= alpha and not correct
        )
        total = sum(1 for p, _ in recent if p <= alpha)

        return violations / max(total, 1)

    def is_calibrated(self, tolerance: float = 0.05) -> bool:
        """Check if calibration is within tolerance."""
        violation_rate = self.get_violation_rate(self.target_alpha)
        return abs(violation_rate - self.target_alpha) <= tolerance
