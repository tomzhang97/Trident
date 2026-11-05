"""
Enhanced Safe-Cover implementation with RC-MCFC algorithm.

Per TRIDENT Framework Section 4:
- Risk-Controlled Min-Cost Facet Cover (RC-MCFC)
- Query-level FWER control via Bonferroni
- Episode isolation with frozen knobs
- Deterministic shortlisting with tie-breaks
- Sound dual lower bound and infeasibility detection
- Comprehensive certificates with full audit trail
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from .candidates import Passage
from .facets import Facet, FacetType
from .config import SafeCoverConfig


class AbstentionReason(Enum):
    """Reasons for abstention per Section 4.6."""
    NONE = "none"
    NO_FACETS = "no_facets"  # Facet miner returned zero facets
    NO_PASSAGES = "no_passages"  # Retrieval returned zero passages
    NO_COVERING_PASSAGES = "no_covering_passages"  # ∃f ∈ F_uncov with no p : f ∈ C(p)
    INFEASIBILITY_PROVEN = "infeasibility_proven"  # LB_dual > B_ctx - cost_ctx
    BUDGET_EXHAUSTED = "budget_exhausted"  # No candidates fit within remaining budget


@dataclass
class CoverageCertificate:
    """
    Certificate for facet coverage.

    Per Section 4.7, emitted only for facets that enter Cov via selected passages.
    Contains full audit trail for reproducibility.
    """
    facet_id: str
    facet_type: str
    passage_id: str
    p_value: float
    threshold: float  # ᾱ_f used for this coverage decision
    alpha_facet: float  # α_f (per-facet error budget)
    alpha_query: float  # α_query (query-level FWER target)
    t_f: int  # Max tests per facet (Bonferroni budget)
    bin: str  # Calibration bin used
    calibrator_version: str
    verifier_version: str
    retriever_version: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'facet_id': self.facet_id,
            'facet_type': self.facet_type,
            'passage_id': self.passage_id,
            'p_value': self.p_value,
            'threshold': self.threshold,
            'alpha_facet': self.alpha_facet,
            'alpha_query': self.alpha_query,
            't_f': self.t_f,
            'bin': self.bin,
            'calibrator_version': self.calibrator_version,
            'verifier_version': self.verifier_version,
            'retriever_version': self.retriever_version,
            'timestamp': self.timestamp
        }


@dataclass
class EpisodeKnobs:
    """
    Frozen knobs for episode isolation.

    Per Section 4.1: Per query, freeze these parameters.
    No mid-episode adaptation.
    """
    t_f: int  # Max tests per facet
    alpha_query: float  # Query-level FWER target
    alpha_f: Dict[str, float]  # Per-facet alpha (α_query / |F(q)|)
    alpha_bar_f: Dict[str, float]  # Per-test alpha (α_f / T_f)
    calibrator_version: str
    verifier_version: str
    retriever_version: str
    frozen_at: float = field(default_factory=time.time)

    def get_alpha_bar(self, facet_id: str) -> float:
        """Get the Bonferroni-corrected threshold for a facet."""
        return self.alpha_bar_f.get(facet_id, self.alpha_query / self.t_f)


@dataclass
class SafeCoverResult:
    """Result from Safe-Cover algorithm."""
    selected_passages: List[Passage]
    certificates: List[CoverageCertificate]
    covered_facets: List[str]
    uncovered_facets: List[str]
    dual_lower_bound: float
    abstained: bool
    abstention_reason: AbstentionReason
    coverage_map: Dict[str, List[str]]  # passage_id -> covered facet_ids
    total_cost: int
    infeasible: bool = False
    episode_knobs: Optional[EpisodeKnobs] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'selected_passages': [p.__dict__ for p in self.selected_passages],
            'certificates': [c.to_dict() for c in self.certificates],
            'covered_facets': self.covered_facets,
            'uncovered_facets': self.uncovered_facets,
            'dual_lower_bound': self.dual_lower_bound,
            'abstained': self.abstained,
            'abstention_reason': self.abstention_reason.value,
            'total_cost': self.total_cost,
            'infeasible': self.infeasible,
        }


@dataclass
class ShortlistEntry:
    """Entry in the per-facet shortlist with deterministic ordering."""
    passage: Passage
    score: float
    p_value: float
    bin_key: str

    def sort_key(self) -> Tuple[float, int, str]:
        """
        Deterministic tie-break key.

        Per Section 4.4: score↑, then passage ID.
        """
        return (-self.score, self.passage.cost, self.passage.pid)


class SafeCoverAlgorithm:
    """
    Risk-Controlled Min-Cost Facet Cover (RC-MCFC) algorithm.

    Implements the full Safe-Cover algorithm per Sections 4 and 9.2.
    """

    def __init__(self, config: SafeCoverConfig, calibrator: Any):
        self.config = config
        self.calibrator = calibrator
        self.fallback_active = False
        self.fallback_scale = 0.5  # Conservative fallback multiplier (ρ ∈ [0.5, 0.9])

        # Version tracking for certificates
        self.verifier_version = getattr(calibrator, 'verifier_hash', 'v1.0')
        self.retriever_version = getattr(calibrator, 'retriever_hash', 'v1.0')

    def _get_budget_cap(self) -> Optional[int]:
        """Get the effective budget cap, preferring max_evidence_tokens over token_cap."""
        return (
            self.config.max_evidence_tokens
            if self.config.max_evidence_tokens is not None
            else self.config.token_cap
        )

    def _freeze_episode_knobs(
        self,
        facets: List[Facet],
        alpha_query: Optional[float] = None
    ) -> EpisodeKnobs:
        """
        Freeze episode knobs per Section 4.1.

        Per query, freeze: T_f, α_query, α_f, ᾱ_f, calibrator version, coverage sets.
        No mid-episode adaptation.
        """
        alpha_query = alpha_query or self.config.per_facet_alpha
        n_facets = len(facets)

        # Per-facet configs
        t_f = 10  # Default max tests per facet
        alpha_f = {}
        alpha_bar_f = {}

        for facet in facets:
            # Get facet-specific config if available
            if facet.facet_id in self.config.per_facet_configs:
                facet_config = self.config.per_facet_configs[facet.facet_id]
                facet_t_f = facet_config.max_tests
                facet_alpha = facet_config.alpha
            else:
                facet_t_f = t_f
                # Bonferroni allocation: α_f = α_query / |F(q)|
                facet_alpha = alpha_query / max(n_facets, 1)

            # ᾱ_f = α_f / T_f
            facet_alpha_bar = facet_alpha / max(facet_t_f, 1)

            # Apply fallback if active
            if self.fallback_active:
                facet_alpha_bar *= self.fallback_scale

            alpha_f[facet.facet_id] = facet_alpha
            alpha_bar_f[facet.facet_id] = facet_alpha_bar

        return EpisodeKnobs(
            t_f=t_f,
            alpha_query=alpha_query,
            alpha_f=alpha_f,
            alpha_bar_f=alpha_bar_f,
            calibrator_version=getattr(self.calibrator, 'version', 'v1.0'),
            verifier_version=self.verifier_version,
            retriever_version=self.retriever_version,
        )

    def run(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        scores: Optional[Dict[Tuple[str, str], float]] = None,
        bins: Optional[Dict[Tuple[str, str], str]] = None
    ) -> SafeCoverResult:
        """
        Run RC-MCFC algorithm with certificates.

        Per Section 9.2 (Safe-Cover algorithm):
        1. Freeze episode knobs
        2. Build fixed coverage sets using Bonferroni thresholds
        3. Run greedy set cover with cost-effectiveness
        4. Compute dual lower bound for early abstention
        5. Generate certificates for covered facets
        """
        # CRITICAL FIX: Handle zero facets case
        if not facets:
            return SafeCoverResult(
                selected_passages=[],
                certificates=[],
                covered_facets=[],
                uncovered_facets=[],
                dual_lower_bound=0.0,
                abstained=True,
                abstention_reason=AbstentionReason.NO_FACETS,
                coverage_map={},
                total_cost=0,
                infeasible=True,
                episode_knobs=None,
            )

        # CRITICAL FIX: Handle zero passages case
        if not passages:
            return SafeCoverResult(
                selected_passages=[],
                certificates=[],
                covered_facets=[],
                uncovered_facets=[f.facet_id for f in facets],
                dual_lower_bound=float('inf'),
                abstained=True,
                abstention_reason=AbstentionReason.NO_PASSAGES,
                coverage_map={},
                total_cost=0,
                infeasible=True,
                episode_knobs=None,
            )

        # Step 1: Freeze episode knobs (Section 4.1)
        knobs = self._freeze_episode_knobs(facets)

        # Step 2: Build fixed coverage sets (Section 4.2)
        coverage_sets, coverage_info = self._build_coverage_sets(
            facets, passages, p_values, knobs, scores, bins
        )

        # Check for uncoverable facets before greedy
        uncoverable = self._find_uncoverable_facets(facets, coverage_sets)
        if uncoverable:
            return self._create_abstain_result(
                facets=facets,
                passages=[],
                reason=AbstentionReason.NO_COVERING_PASSAGES,
                uncovered=uncoverable,
                knobs=knobs,
                coverage_map={}
            )

        # Step 3: Compute initial dual lower bound (Section 4.6)
        initial_dual_lb = self._compute_dual_lower_bound(
            facets, passages, coverage_sets, set(f.facet_id for f in facets)
        )
        budget_cap = self._get_budget_cap()

        # Early infeasibility check
        if budget_cap is not None and initial_dual_lb > budget_cap:
            return self._create_abstain_result(
                facets=facets,
                passages=[],
                reason=AbstentionReason.INFEASIBILITY_PROVEN,
                uncovered=[f.facet_id for f in facets],
                knobs=knobs,
                dual_lb=initial_dual_lb,
                coverage_map={}
            )

        # Step 4: Run greedy set cover (Section 4.5)
        selected, certificates, coverage_map, final_uncovered = self._greedy_cover(
            facets, passages, coverage_sets, p_values, knobs, coverage_info, budget_cap
        )

        # Step 5: Compute final dual lower bound
        covered_facets = set()
        for facet_list in coverage_map.values():
            covered_facets.update(facet_list)
        uncovered_facets = [f.facet_id for f in facets if f.facet_id not in covered_facets]

        final_dual_lb = self._compute_dual_lower_bound(
            facets, passages, coverage_sets,
            set(uncovered_facets)
        )

        # Step 6: Determine abstention
        total_cost = sum(p.cost for p in selected)
        abstained, abstention_reason = self._determine_abstention(
            uncovered_facets, selected, final_dual_lb, budget_cap, total_cost
        )

        return SafeCoverResult(
            selected_passages=selected,
            certificates=certificates,
            covered_facets=list(covered_facets),
            uncovered_facets=uncovered_facets,
            dual_lower_bound=final_dual_lb,
            abstained=abstained,
            abstention_reason=abstention_reason,
            coverage_map=coverage_map,
            total_cost=total_cost,
            infeasible=len(uncovered_facets) > 0,
            episode_knobs=knobs,
        )

    def _build_coverage_sets(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        knobs: EpisodeKnobs,
        scores: Optional[Dict[Tuple[str, str], float]] = None,
        bins: Optional[Dict[Tuple[str, str], str]] = None
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict[str, Any]]]:
        """
        Build fixed coverage sets using Bonferroni thresholds.

        Per Section 4.2:
        C(p) = { f ∈ F(q) : π(p,f) ≤ ᾱ_f }
        """
        coverage_sets = {p.pid: set() for p in passages}
        coverage_info = {}  # (pid, fid) -> {p_value, alpha_bar, bin, ...}

        for facet in facets:
            alpha_bar = knobs.get_alpha_bar(facet.facet_id)

            # Get per-facet test limit
            if facet.facet_id in self.config.per_facet_configs:
                max_tests = self.config.per_facet_configs[facet.facet_id].max_tests
            else:
                max_tests = knobs.t_f

            # Shortlist passages for this facet (Section 4.4)
            shortlist = self._shortlist_for_facet(
                passages, facet, p_values, scores, max_tests
            )

            # Test coverage for each shortlisted passage
            for entry in shortlist:
                key = (entry.passage.pid, facet.facet_id)

                if key in p_values:
                    pv = p_values[key]

                    if pv <= alpha_bar:
                        coverage_sets[entry.passage.pid].add(facet.facet_id)

                    # Store coverage info for certificate generation
                    coverage_info[key] = {
                        'p_value': pv,
                        'alpha_bar': alpha_bar,
                        'alpha_facet': knobs.alpha_f.get(facet.facet_id, knobs.alpha_query),
                        'bin': bins.get(key, 'default') if bins else entry.bin_key,
                        'score': scores.get(key, 0.0) if scores else entry.score,
                        'facet_type': facet.facet_type.value if isinstance(facet.facet_type, FacetType) else str(facet.facet_type),
                    }

        return coverage_sets, coverage_info

    def _shortlist_for_facet(
        self,
        passages: List[Passage],
        facet: Facet,
        p_values: Dict[Tuple[str, str], float],
        scores: Optional[Dict[Tuple[str, str], float]],
        max_candidates: int
    ) -> List[ShortlistEntry]:
        """
        Deterministic shortlisting per Section 4.4.

        Top-T_f per facet from a cheap facet-aware ranker;
        deterministic tie-breaks (score↑, then passage ID).
        """
        entries = []

        for passage in passages:
            key = (passage.pid, facet.facet_id)
            score = scores.get(key, 0.0) if scores else 0.0
            pv = p_values.get(key, 1.0)

            entry = ShortlistEntry(
                passage=passage,
                score=score,
                p_value=pv,
                bin_key='default'
            )
            entries.append(entry)

        # Sort by score descending, cost ascending, pid for determinism
        entries.sort(key=lambda e: e.sort_key())

        return entries[:max_candidates]

    def _find_uncoverable_facets(
        self,
        facets: List[Facet],
        coverage_sets: Dict[str, Set[str]]
    ) -> List[str]:
        """
        Find facets that have no covering passages.

        Per Section 4.6 (Uncoverable facets):
        If ∃f ∈ F_uncov with no p such that f ∈ C(p), set LB_dual = ∞
        """
        coverable_facets = set()
        for pid, covered in coverage_sets.items():
            coverable_facets.update(covered)

        uncoverable = []
        for facet in facets:
            if facet.facet_id not in coverable_facets:
                uncoverable.append(facet.facet_id)

        return uncoverable

    def _greedy_cover(
        self,
        facets: List[Facet],
        passages: List[Passage],
        coverage_sets: Dict[str, Set[str]],
        p_values: Dict[Tuple[str, str], float],
        knobs: EpisodeKnobs,
        coverage_info: Dict[Tuple[str, str], Dict[str, Any]],
        budget_cap: Optional[int]
    ) -> Tuple[List[Passage], List[CoverageCertificate], Dict[str, List[str]], List[str]]:
        """
        Greedy set cover with cost-effectiveness.

        Per Section 4.5:
        p* ∈ argmax_{p ∉ S} |C(p) ∩ F_uncov| / c(p)
        Ties: cost↓, mean-π↓, pid

        Per Section 4.7: Emit certificates for facets as they enter Cov.
        """
        selected_passages = []
        certificates = []
        coverage_map = {}

        uncovered = {f.facet_id for f in facets}
        remaining_passages = {p.pid: p for p in passages}
        max_units = self.config.max_units or len(passages)
        total_cost = 0

        while uncovered and remaining_passages and len(selected_passages) < max_units:
            # Find most cost-effective passage
            best_passage = None
            best_score = (-float('inf'), float('inf'), float('inf'), "")

            for pid, passage in remaining_passages.items():
                newly_covered = coverage_sets[pid] & uncovered

                if not newly_covered:
                    continue

                # Check budget constraint
                if budget_cap is not None and self.config.stop_on_budget:
                    if total_cost + passage.cost > budget_cap:
                        continue

                # Cost-effectiveness ratio (Section 4.5)
                effectiveness = len(newly_covered) / max(passage.cost, 1)

                # Mean p-value for tie-breaking
                mean_p = np.mean([
                    p_values.get((pid, fid), 1.0)
                    for fid in newly_covered
                ])

                # Score tuple: (effectiveness↑, cost↓, mean_p↓, pid for determinism)
                score = (effectiveness, -passage.cost, -mean_p, pid)

                if score > best_score:
                    best_score = score
                    best_passage = passage

            # No more passages can cover uncovered facets
            if best_passage is None:
                break

            # Add best passage to selection
            selected_passages.append(best_passage)
            total_cost += best_passage.cost
            newly_covered = coverage_sets[best_passage.pid] & uncovered
            coverage_map[best_passage.pid] = list(newly_covered)

            # Generate certificates for newly covered facets (Section 4.7)
            for facet_id in newly_covered:
                key = (best_passage.pid, facet_id)
                info = coverage_info.get(key, {})

                certificate = CoverageCertificate(
                    facet_id=facet_id,
                    facet_type=info.get('facet_type', 'unknown'),
                    passage_id=best_passage.pid,
                    p_value=info.get('p_value', p_values.get(key, 0.0)),
                    threshold=info.get('alpha_bar', knobs.get_alpha_bar(facet_id)),
                    alpha_facet=info.get('alpha_facet', knobs.alpha_f.get(facet_id, knobs.alpha_query)),
                    alpha_query=knobs.alpha_query,
                    t_f=knobs.t_f,
                    bin=info.get('bin', 'default'),
                    calibrator_version=knobs.calibrator_version,
                    verifier_version=knobs.verifier_version,
                    retriever_version=knobs.retriever_version,
                )
                certificates.append(certificate)

            # Update uncovered set
            uncovered -= newly_covered

            # Remove selected passage from remaining
            del remaining_passages[best_passage.pid]

        return selected_passages, certificates, coverage_map, list(uncovered)

    def _compute_dual_lower_bound(
        self,
        facets: List[Facet],
        passages: List[Passage],
        coverage_sets: Dict[str, Set[str]],
        uncovered_facets: Set[str]
    ) -> float:
        """
        Compute dual lower bound for the residual uncovered facets.

        Per Section 4.6:
        max Σ_{f ∈ F_uncov} y_f
        s.t. Σ_{f ∈ C(p) ∩ F_uncov} y_f ≤ c(p), y_f ≥ 0

        Uniformly raise y_f until some passage constraint becomes tight.
        """
        if not uncovered_facets:
            return 0.0

        # Check for uncoverable facets
        coverable_by_any = set()
        for pid, covered in coverage_sets.items():
            coverable_by_any.update(covered & uncovered_facets)

        if len(coverable_by_any) < len(uncovered_facets):
            # Some facets have no covering passages
            return float('inf')

        # Initialize dual variables
        dual_vars = {fid: 0.0 for fid in uncovered_facets}
        tight_passages = set()

        epsilon = self.config.dual_tolerance
        max_iterations = 100

        remaining_uncovered = set(uncovered_facets)

        for _ in range(max_iterations):
            if not remaining_uncovered:
                break

            # Find minimum slack
            min_slack = float('inf')
            min_passage = None
            min_facets = set()

            for passage in passages:
                if passage.pid in tight_passages:
                    continue

                # Facets this passage can cover that are still uncovered
                coverable = coverage_sets[passage.pid] & remaining_uncovered
                if not coverable:
                    continue

                # Calculate dual sum for this passage
                dual_sum = sum(dual_vars[fid] for fid in coverable)
                slack = passage.cost - dual_sum

                if slack < min_slack:
                    min_slack = slack
                    min_passage = passage
                    min_facets = coverable

            if min_passage is None or min_slack == float('inf'):
                # No feasible solution
                return float('inf')

            if min_slack <= epsilon:
                # Passage is tight
                tight_passages.add(min_passage.pid)
                remaining_uncovered -= min_facets
                continue

            # Raise dual variables uniformly for facets covered by min_passage
            if min_facets:
                raise_amount = min_slack / len(min_facets)
                for fid in min_facets:
                    dual_vars[fid] += raise_amount

                tight_passages.add(min_passage.pid)
                remaining_uncovered -= min_facets

        return sum(dual_vars.values())

    def _determine_abstention(
        self,
        uncovered_facets: List[str],
        selected: List[Passage],
        dual_lb: float,
        budget_cap: Optional[int],
        total_cost: int
    ) -> Tuple[bool, AbstentionReason]:
        """
        Determine whether to abstain and the reason.

        Per Section 4.6:
        - no_covering_passages: ∃f ∈ F_uncov with no p : f ∈ C(p)
        - infeasibility_proven: LB_dual > B_ctx - cost_ctx
        - budget_exhausted: No candidates fit within remaining budget
        """
        # Check for uncovered facets
        if uncovered_facets and not selected:
            return True, AbstentionReason.NO_COVERING_PASSAGES

        # Check dual lower bound
        if budget_cap is not None:
            remaining_budget = budget_cap - total_cost
            if dual_lb > remaining_budget and uncovered_facets:
                if self.config.early_abstain:
                    return True, AbstentionReason.INFEASIBILITY_PROVEN

        # Check if we have uncovered facets (budget exhausted or no covering passages)
        if uncovered_facets:
            if self.config.abstain_on_infeasible:
                # Determine the specific reason
                if dual_lb == float('inf'):
                    return True, AbstentionReason.NO_COVERING_PASSAGES
                else:
                    return True, AbstentionReason.BUDGET_EXHAUSTED

        # No abstention
        return False, AbstentionReason.NONE

    def _create_abstain_result(
        self,
        facets: List[Facet],
        passages: List[Passage],
        reason: AbstentionReason,
        uncovered: List[str],
        knobs: EpisodeKnobs,
        dual_lb: float = float('inf'),
        coverage_map: Dict[str, List[str]] = None
    ) -> SafeCoverResult:
        """Create an abstention result."""
        return SafeCoverResult(
            selected_passages=passages,
            certificates=[],
            covered_facets=[f.facet_id for f in facets if f.facet_id not in uncovered],
            uncovered_facets=uncovered,
            dual_lower_bound=dual_lb,
            abstained=True,
            abstention_reason=reason,
            coverage_map=coverage_map or {},
            total_cost=sum(p.cost for p in passages),
            infeasible=True,
            episode_knobs=knobs,
        )

    def apply_fallback(self) -> None:
        """Apply fallback thresholds when drift is detected."""
        self.fallback_active = True

    def reset_fallback(self) -> None:
        """Reset fallback mode."""
        self.fallback_active = False

    def get_spurious_facet_prepass(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        spurious_alpha: float = 1e-4,
        max_tests_per_facet: int = 3
    ) -> List[Facet]:
        """
        Pre-pass to filter spurious facets with very strict threshold.

        This helps remove noisy facets from the mining stage.
        """
        valid_facets = []

        for facet in facets:
            tests = 0
            found_support = False

            for passage in passages:
                if tests >= max_tests_per_facet:
                    break

                key = (passage.pid, facet.facet_id)
                if key in p_values:
                    tests += 1
                    if p_values[key] <= spurious_alpha:
                        found_support = True
                        break

            if found_support:
                valid_facets.append(facet)

        return valid_facets


def compute_holm_adjusted_pvalues(
    facet_pvalues: Dict[str, float],
    t_f: int
) -> Dict[str, float]:
    """
    Compute Holm step-down adjusted p-values.

    Per Section 4.7 (Holm, optional, reporting-only):
    Compute per-facet p_f = min(1, min_{p ∈ P_f} π(p,f) · T_f)
    and apply Holm step-down post-selection to report a tighter bound.

    Does not change C(p) or greedy decisions.
    """
    # Sort facets by p-value
    sorted_facets = sorted(facet_pvalues.items(), key=lambda x: x[1])
    n = len(sorted_facets)

    adjusted = {}
    cummax = 0.0

    for i, (fid, pv) in enumerate(sorted_facets):
        # Holm adjustment: multiply by (n - i)
        adj_pv = pv * (n - i)
        # Enforce monotonicity
        adj_pv = max(adj_pv, cummax)
        cummax = adj_pv
        adjusted[fid] = min(adj_pv, 1.0)

    return adjusted
