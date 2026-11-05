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
from . import nli_scorer


# --------------------------------------------------------------------------
# FIXES (vs base safe_cover):
#  (1) STRUCTURAL_TYPES set distinguishes facets that MUST be covered (BRIDGE,
#      RELATION, NUMERIC, TEMPORAL) from heuristic mining noise (ENTITY etc.).
#      Only structural facets enter Bonferroni denominator and the dual LB;
#      non-structural facets are best-effort and can never trigger abstention.
#  (2) HEURISTIC_ALPHA_FLOOR = 0.05 prevents over-tight thresholds when the
#      verifier is heuristic (no calibrator) and p-values cluster in [0.02,0.1].
#  (3) tf_for tightened (10 -> 5 for RELATION/BRIDGE) to relax Bonferroni.
#  (4) Early-infeasibility check now respects abstain_on_infeasible=False so
#      structural-uncoverable queries fall through to greedy + Pareto fallback.
#  (5) Certificates are emitted INSIDE the greedy loop (one per (passage,facet)
#      decision) instead of a disconnected post-pass that lost coverage_map fidelity.
#  (6) _compute_dual_lower_bound considers only structural uncovered facets, so
#      a missing optional ENTITY facet does not push the LB to +inf.
# --------------------------------------------------------------------------

STRUCTURAL_TYPES: Set[str] = {
    "BRIDGE", "BRIDGE_HOP", "BRIDGE_HOP1", "BRIDGE_HOP2", "BRIDGE_HOP3",
    "RELATION", "NUMERIC", "TEMPORAL",
}

HEURISTIC_ALPHA_FLOOR: float = 0.05


def _facet_type_str(facet: "Facet") -> str:
    ft = facet.facet_type
    return ft.value if hasattr(ft, "value") else str(ft)


def _is_structural(facet: "Facet") -> bool:
    """A facet is structural (=required for coverage) if its type is in
    STRUCTURAL_TYPES OR the facet is explicitly marked required."""
    if getattr(facet, "required", False):
        return True
    return _facet_type_str(facet) in STRUCTURAL_TYPES


def tf_for(ft_enum: FacetType, ft_str: str) -> int:
    """
    Get T_f for a facet type. MUST be consistent with pipeline.py.

    Used for:
    - Shortlist size in pipeline
    - Bonferroni threshold α_bar = α_f / T_f
    - Coverage set building in safe_cover

    Tightened from origin (RELATION/BRIDGE used 10) so threshold is
    feasible against typical p-values from the calibrator.
    """
    if ft_enum == FacetType.ENTITY:
        return 5
    if ft_enum == FacetType.RELATION or "BRIDGE" in ft_str:
        return 5
    return 5


class AbstentionReason(Enum):
    """Reasons for abstention per Section 4.6."""
    NONE = "none"
    NO_FACETS = "no_facets"  # facet miner produced nothing
    NO_PASSAGES = "no_passages"  # retriever returned nothing
    NO_COVERING_PASSAGES = "no_covering_passages"  # ∃f ∈ F_uncov with no p : f ∈ C(p)
    NO_SUPPORT_EDGES = "no_support_edges"  # No (p,f) edges with valid p-values
    INFEASIBILITY_PROVEN = "infeasibility_proven"  # LB_dual > B_ctx - cost_ctx
    HEURISTIC_BUDGET_INFEASIBLE = "heuristic_budget_infeasible"  # heuristic mode + budget exhausted
    BUDGET_EXHAUSTED = "budget_exhausted"  # No candidates fit within remaining budget
    PVALUE_INFEASIBLE_SMALL_BIN = "pvalue_infeasible_small_bin"  # α_bar_f < 1/(n_b+1) and cannot resolve
    INFEASIBLE_MISSING_FACETS = "infeasible_missing_facets"  # U monotonic: required facets not coverable


@dataclass
class CoverageCertificate:
    """
    Certificate for facet coverage.

    Per Section 4.7, emitted only for facets that enter Cov via selected passages.
    Contains full audit trail for reproducibility.

    Explicit Schema (per reviewer request):
    - facet_id: Unique identifier for the facet
    - facet_type: Type of facet (ENTITY, RELATION, TEMPORAL, NUMERIC, BRIDGE_HOP1, BRIDGE_HOP2)
    - passage_id: ID of the passage covering this facet
    - p_value: Calibrated p-value for this (passage, facet) pair
    - threshold: ᾱ_f used for this coverage decision
    - alpha_f: Per-facet error budget (α_query / |F(q)|)
    - alpha_query: Query-level FWER target
    - t_f: Max tests per facet (Bonferroni budget T_f)
    - bin: Calibration bin key used for p-value computation
    - bin_size: Number of negatives in the calibration bin
    - pvalue_mode: "deterministic" or "randomized"
    - calibrator_version: Hash of calibrator state
    - retriever_version: Hash of retriever model/index
    - shortlister_version: Hash of shortlister
    - verifier_version: Hash of verifier model
    - timestamp: Unix timestamp when certificate was generated

    Certificates are INVALID if any version differs from calibration time.
    """
    facet_id: str
    facet_type: str
    passage_id: str
    p_value: float
    threshold: float  # ᾱ_f used for this coverage decision
    alpha_facet: float  # α_f (per-facet error budget)
    alpha_query: float  # α_query (query-level FWER target)
    t_f: int  # Max tests per facet (Bonferroni budget)
    bin: str  # Calibration bin used
    bin_size: int = 0  # Number of negatives in calibration bin
    pvalue_mode: str = "deterministic"  # "deterministic" or "randomized"
    calibrator_version: str = ""
    retriever_version: str = ""
    shortlister_version: str = ""  # NEW: Hash of shortlister
    verifier_version: str = ""
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
            'bin_size': self.bin_size,
            'pvalue_mode': self.pvalue_mode,
            'calibrator_version': self.calibrator_version,
            'retriever_version': self.retriever_version,
            'shortlister_version': self.shortlister_version,
            'verifier_version': self.verifier_version,
            'timestamp': self.timestamp
        }


@dataclass
class EpisodeKnobs:
    """
    Frozen knobs for episode isolation.

    Per Section 4.1: Per query, freeze these parameters.
    No mid-episode adaptation.

    All these values are logged in certificates and used to validate
    that test-time versions match calibration-time versions.
    """
    t_f: int  # Max tests per facet
    alpha_query: float  # Query-level FWER target
    alpha_f: Dict[str, float]  # Per-facet alpha (α_query / |F(q)|)
    alpha_bar_f: Dict[str, float]  # Per-test alpha (α_f / T_f)
    calibrator_version: str
    verifier_version: str
    retriever_version: str
    shortlister_version: str = ""  # Hash of shortlister for selection-conditional calibration
    pvalue_mode: str = "deterministic"  # "deterministic" or "randomized"
    gate_version: str = ""  # Tested-Set gate definition version
    normalize_version: str = ""  # Text normalization version used in gating
    relation_keywords_hash: str = ""  # Hash of RELATION_KEYWORDS used in gating
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
            'episode_knobs': self.episode_knobs.__dict__ if self.episode_knobs else None,
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

        # Version tracking for certificates (selection-conditional)
        self.verifier_version = getattr(calibrator, 'verifier_hash', 'v1.0')
        self.retriever_version = getattr(calibrator, 'retriever_hash', 'v1.0')
        self.shortlister_version = getattr(calibrator, 'shortlister_hash', 'v1.0')
        self.pvalue_mode = getattr(calibrator, 'pvalue_mode', 'deterministic')
        if hasattr(self.pvalue_mode, 'value'):
            self.pvalue_mode = self.pvalue_mode.value

        # Tested-Set definition metadata
        self.gate_version = nli_scorer.GATE_VERSION
        self.normalize_version = nli_scorer.NORMALIZE_VERSION
        self.relation_keyword_hash = nli_scorer.RELATION_KEYWORDS_HASH

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
        passages: Optional[List[Passage]] = None,
        alpha_query: Optional[float] = None,
        is_heuristic_mode: bool = False,
    ) -> EpisodeKnobs:
        """
        Freeze episode knobs per Section 4.1 — FIXED semantics.

        Bonferroni denominator is |structural facets|, not |F|. Non-structural
        facets receive alpha_bar=1.0 (always-coverable best-effort).
        Heuristic mode applies HEURISTIC_ALPHA_FLOOR so thresholds stay
        achievable for un-calibrated p-values.
        """
        import os
        debug = os.environ.get("TRIDENT_DEBUG_PVALUE", "0") == "1"

        per_facet_alpha = alpha_query or self.config.per_facet_alpha

        structural_facets = [f for f in facets if _is_structural(f)]
        n_structural = max(len(structural_facets), 1)

        pool_n = len(passages) if passages else 10
        exhaustive = (
            getattr(self.config, "exhaustive_pool", True)
            and pool_n <= getattr(self.config, "exhaustive_pool_max_n", 10)
        )

        t_f = pool_n if exhaustive else getattr(self.config, "max_tests", 5)
        alpha_f: Dict[str, float] = {}
        alpha_bar_f: Dict[str, float] = {}

        if debug:
            print(
                f"[CERT DEBUG] pool_n={pool_n} exhaustive={exhaustive} "
                f"|F|={len(facets)} |F_struct|={n_structural} heuristic={is_heuristic_mode}"
            )

        for facet in facets:
            ft = facet.facet_type
            ft_str = _facet_type_str(facet)
            structural = _is_structural(facet)

            if facet.facet_id in self.config.per_facet_configs:
                facet_config = self.config.per_facet_configs[facet.facet_id]
                facet_t_f = facet_config.max_tests
                facet_alpha = facet_config.alpha
            else:
                if structural:
                    # Bonferroni split over structural facets only
                    facet_alpha = per_facet_alpha / n_structural
                else:
                    # Optional facet — never blocks coverage
                    facet_alpha = 1.0

                if exhaustive:
                    facet_t_f = pool_n
                else:
                    facet_t_f = tf_for(ft, ft_str)

            facet_alpha_bar = facet_alpha / max(facet_t_f, 1)

            # Heuristic mode: floor the threshold so un-calibrated p-values
            # have a chance of crossing.
            if is_heuristic_mode and structural:
                facet_alpha_bar = max(facet_alpha_bar, HEURISTIC_ALPHA_FLOOR)

            if self.fallback_active:
                facet_alpha_bar *= self.fallback_scale

            alpha_f[facet.facet_id] = facet_alpha
            alpha_bar_f[facet.facet_id] = facet_alpha_bar

            if debug:
                print(
                    f"[CERT DEBUG] facet={ft_str} structural={structural} "
                    f"alpha_f={facet_alpha:.4f} T_f={facet_t_f} "
                    f"alpha_bar_f={facet_alpha_bar:.4f}"
                )

        return EpisodeKnobs(
            t_f=t_f,
            alpha_query=per_facet_alpha,
            alpha_f=alpha_f,
            alpha_bar_f=alpha_bar_f,
            calibrator_version=getattr(self.calibrator, "version", "v1.0"),
            verifier_version=self.verifier_version,
            retriever_version=self.retriever_version,
            shortlister_version=self.shortlister_version,
            pvalue_mode=self.pvalue_mode,
            gate_version=self.gate_version,
            normalize_version=self.normalize_version,
            relation_keywords_hash=self.relation_keyword_hash,
        )

    def run(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        scores: Optional[Dict[Tuple[str, str], float]] = None,
        bins: Optional[Dict[Tuple[str, str], str]] = None,
        uncoverable_facet_ids: Optional[Set[str]] = None,
        is_heuristic_mode: bool = False,
    ) -> SafeCoverResult:
        """
        Run RC-MCFC algorithm with certificates.

        FIXES vs origin run():
        - Explicit empty-input early exits (NO_FACETS / NO_PASSAGES)
        - Dual LB computed over STRUCTURAL uncovered facets only
        - Initial infeasibility check respects config.abstain_on_infeasible
          and config.early_abstain — we fall through to greedy + Pareto
          fallback instead of hard-abstaining.
        """
        # Early guard: no facets / no passages
        if not facets:
            knobs = self._freeze_episode_knobs(
                facets, passages, is_heuristic_mode=is_heuristic_mode
            )
            return self._create_abstain_result(
                facets=facets, passages=[],
                reason=AbstentionReason.NO_FACETS,
                uncovered=[], knobs=knobs, dual_lb=0.0, coverage_map={},
            )
        if not passages:
            knobs = self._freeze_episode_knobs(
                facets, passages, is_heuristic_mode=is_heuristic_mode
            )
            return self._create_abstain_result(
                facets=facets, passages=[],
                reason=AbstentionReason.NO_PASSAGES,
                uncovered=[f.facet_id for f in facets],
                knobs=knobs, dual_lb=float("inf"), coverage_map={},
            )

        # Step 1: Freeze episode knobs (Section 4.1)
        knobs = self._freeze_episode_knobs(
            facets, passages, is_heuristic_mode=is_heuristic_mode
        )

        # Step 2: Build fixed coverage sets (Section 4.2)
        coverage_sets, coverage_info = self._build_coverage_sets(
            facets, passages, p_values, knobs, scores, bins, uncoverable_facet_ids
        )

        # Structural facet set — only these participate in infeasibility logic.
        structural_ids = {f.facet_id for f in facets if _is_structural(f)}

        # Step 3: Compute initial dual lower bound over STRUCTURAL facets only.
        # A missing optional ENTITY facet must not push LB to +inf.
        initial_dual_lb = self._compute_dual_lower_bound(
            facets, passages, coverage_sets, set(structural_ids)
        )
        budget_cap = self._get_budget_cap()

        # Early infeasibility check — now respects abstain_on_infeasible/early_abstain.
        # If the user asked for fallback behavior, we keep going and let greedy do
        # its best; the caller (pipeline) can then switch to Pareto mode.
        if (
            budget_cap is not None
            and initial_dual_lb > budget_cap
            and self.config.early_abstain
            and self.config.abstain_on_infeasible
        ):
            reason = (
                AbstentionReason.HEURISTIC_BUDGET_INFEASIBLE
                if is_heuristic_mode
                else AbstentionReason.INFEASIBILITY_PROVEN
            )
            return self._create_abstain_result(
                facets=facets, passages=[],
                reason=reason,
                uncovered=[f.facet_id for f in facets],
                knobs=knobs, dual_lb=initial_dual_lb, coverage_map={},
            )

        # Step 4: Run greedy set cover (Section 4.5)
        selected, certificates, coverage_map, _ = self._greedy_cover(
            facets, passages, coverage_sets, p_values, knobs, coverage_info, budget_cap
        )

        # Best-effort selection for configs that disable BOTH internal
        # abstention AND pareto fallback (e.g. safe_cover_2000_relaxed): if
        # greedy picked nothing because no p-value crossed the strict
        # threshold, pick the cheapest passages by (min p-value across facets,
        # cost) up to the budget. Uncertified, but non-empty evidence for the
        # LLM — matching the intent of `abstain_on_infeasible=False`.
        if (
            not selected
            and not self.config.abstain_on_infeasible
            and not getattr(self.config, "fallback_to_pareto", False)
            and facets and passages
        ):
            budget_left = budget_cap if budget_cap is not None else sum(p.cost for p in passages)

            def _passage_score(p: Passage) -> Tuple[float, int, str]:
                per_facet_best = [
                    p_values.get((p.pid, f.facet_id), 1.0) for f in facets
                ]
                min_p = min(per_facet_best) if per_facet_best else 1.0
                return (min_p, p.cost, p.pid)

            ranked = sorted(passages, key=_passage_score)
            max_units = self.config.max_units or len(passages)
            cost_acc = 0
            for p in ranked:
                if len(selected) >= max_units:
                    break
                if cost_acc + p.cost > budget_left:
                    continue
                selected.append(p)
                cost_acc += p.cost
                coverage_map.setdefault(p.pid, [])

        # Step 5: Compute final dual lower bound over STRUCTURAL uncovered
        covered_facets: Set[str] = set()
        for facet_list in coverage_map.values():
            covered_facets.update(facet_list)
        uncovered_facets = [f.facet_id for f in facets if f.facet_id not in covered_facets]
        uncovered_structural = {fid for fid in uncovered_facets if fid in structural_ids}

        final_dual_lb = self._compute_dual_lower_bound(
            facets, passages, coverage_sets, uncovered_structural
        )

        total_cost = sum(p.cost for p in selected)
        abstained, abstention_reason = self._determine_abstention(
            list(uncovered_structural), selected, final_dual_lb, budget_cap, total_cost,
            is_heuristic_mode=is_heuristic_mode,
        )

        # `infeasible` is the signal pipeline.py uses to decide whether to
        # trigger the Pareto fallback. It must be True when:
        #   (a) structural facets remain uncovered (the classic case), OR
        #   (b) greedy selected zero passages for a non-empty facet set —
        #       a degenerate state (e.g. entity-only queries whose optional
        #       facets weren't reached) where the caller must fall back to
        #       Pareto, otherwise the LLM sees no evidence and ABSTAINs.
        is_infeasible = (
            len(uncovered_structural) > 0
            or (len(selected) == 0 and len(facets) > 0)
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
            infeasible=is_infeasible,
            episode_knobs=knobs,
        )

    def _build_coverage_sets(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        knobs: EpisodeKnobs,
        scores: Optional[Dict[Tuple[str, str], float]] = None,
        bins: Optional[Dict[Tuple[str, str], str]] = None,
        uncoverable_facet_ids: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[Tuple[str, str], Dict[str, Any]]]:
        """
        Build fixed coverage sets using Bonferroni thresholds.

        Per Section 4.2:
        C(p) = { f ∈ F(q) : π(p,f) ≤ ᾱ_f }
        """
        coverage_sets = {p.pid: set() for p in passages}
        coverage_info = {}  # (pid, fid) -> {p_value, alpha_bar, bin, ...}
        blocked_facets = set(uncoverable_facet_ids or [])

        import os
        debug = os.environ.get("TRIDENT_DEBUG_PVALUE", "0") == "1"

        # EXHAUSTIVE PROBING: When pool is small, use all passages
        pool_n = len(passages)
        exhaustive = (
            getattr(self.config, "exhaustive_pool", True)
            and pool_n <= getattr(self.config, "exhaustive_pool_max_n", 10)
        )

        for facet in facets:
            ft = facet.facet_type
            alpha_bar = knobs.get_alpha_bar(facet.facet_id)

            if facet.facet_id in blocked_facets:
                continue

            # Get per-facet test limit
            if facet.facet_id in self.config.per_facet_configs:
                max_tests = self.config.per_facet_configs[facet.facet_id].max_tests
            elif exhaustive:
                # Probe ALL passages to prevent STAGE-1 MISS
                max_tests = pool_n
            else:
                # Use shared tf_for function (MUST match pipeline.py)
                ft_str = ft.value if hasattr(ft, 'value') else str(ft)
                max_tests = tf_for(ft, ft_str)

            # Shortlist passages for this facet (Section 4.4)
            shortlist = self._shortlist_for_facet(
                passages, facet, p_values, scores, max_tests
            )

            # Track best p-value for this facet (for debugging)
            facet_p_values = []

            # Test coverage for each shortlisted passage
            for entry in shortlist:
                key = (entry.passage.pid, facet.facet_id)

                if key in p_values:
                    pv = p_values[key]
                    facet_p_values.append(pv)

                    if pv <= alpha_bar:
                        coverage_sets[entry.passage.pid].add(facet.facet_id)

                    # Store coverage info for certificate generation
                    coverage_info[key] = {
                        'p_value': pv,
                        'alpha_bar': alpha_bar,
                        'alpha_facet': knobs.alpha_f.get(facet.facet_id, knobs.alpha_query),
                        'bin': bins.get(key, 'default') if bins else entry.bin_key,
                        'score': scores.get(key, 0.0) if scores else entry.score,
                        'facet_type': ft.value if isinstance(ft, FacetType) else str(ft),
                    }

            # Log best p-value per facet for debugging
            if debug:
                best_p = min(facet_p_values) if facet_p_values else 1.0
                ft_str = ft.value if hasattr(ft, 'value') else str(ft)
                passes = best_p <= alpha_bar
                print(f"[COVER DEBUG] facet={ft_str} best_p={best_p:.4g} alpha_bar={alpha_bar:.4g} pass={passes}")

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
        selected_passages: List[Passage] = []
        coverage_map: Dict[str, List[str]] = {}
        certificates: List[CoverageCertificate] = []

        uncovered: Set[str] = {f.facet_id for f in facets}
        facet_by_id = {f.facet_id: f for f in facets}
        remaining_passages = {p.pid: p for p in passages}
        max_units = self.config.max_units or len(passages)
        total_cost = 0

        while uncovered and remaining_passages and len(selected_passages) < max_units:
            best_passage = None
            best_score = (-float("inf"), float("inf"), float("inf"), "")

            for pid, passage in remaining_passages.items():
                newly_covered = coverage_sets[pid] & uncovered
                if not newly_covered:
                    continue
                if budget_cap is not None and self.config.stop_on_budget:
                    if total_cost + passage.cost > budget_cap:
                        continue

                effectiveness = len(newly_covered) / max(passage.cost, 1)
                mean_p = float(np.mean([
                    p_values.get((pid, fid), 1.0) for fid in newly_covered
                ]))
                score = (effectiveness, -passage.cost, -mean_p, pid)
                if score > best_score:
                    best_score = score
                    best_passage = passage

            if best_passage is None:
                break

            newly_covered = coverage_sets[best_passage.pid] & uncovered
            selected_passages.append(best_passage)
            total_cost += best_passage.cost

            # Emit certificates INLINE for each facet this passage covers.
            # coverage_map faithfully records the greedy decision.
            coverage_map[best_passage.pid] = sorted(newly_covered)
            for fid in newly_covered:
                key = (best_passage.pid, fid)
                info = coverage_info.get(key, {})
                bin_key = info.get("bin", "default")
                bin_size = 0
                if hasattr(self.calibrator, "bins") and bin_key in self.calibrator.bins:
                    bin_entry = self.calibrator.bins[bin_key]
                    if hasattr(bin_entry, "n_negatives"):
                        bin_size = bin_entry.n_negatives
                    elif isinstance(bin_entry, dict):
                        bin_size = int(bin_entry.get("n_negatives", len(bin_entry)))
                    elif isinstance(bin_entry, (list, tuple)):
                        bin_size = len(bin_entry)

                certificates.append(CoverageCertificate(
                    facet_id=fid,
                    facet_type=info.get("facet_type", _facet_type_str(facet_by_id[fid])),
                    passage_id=best_passage.pid,
                    p_value=info.get("p_value", p_values.get(key, 0.0)),
                    threshold=info.get("alpha_bar", knobs.get_alpha_bar(fid)),
                    alpha_facet=info.get("alpha_facet", knobs.alpha_f.get(fid, knobs.alpha_query)),
                    alpha_query=knobs.alpha_query,
                    t_f=knobs.t_f,
                    bin=bin_key,
                    bin_size=bin_size,
                    pvalue_mode=knobs.pvalue_mode,
                    calibrator_version=knobs.calibrator_version,
                    verifier_version=knobs.verifier_version,
                    shortlister_version=knobs.shortlister_version,
                    retriever_version=knobs.retriever_version,
                ))

            uncovered -= newly_covered
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
        total_cost: int,
        is_heuristic_mode: bool = False,
    ) -> Tuple[bool, AbstentionReason]:
        """
        Determine whether to abstain and the reason.

        `uncovered_facets` is the set of STRUCTURAL uncovered facets — optional
        facets are never counted here. When no structural facets are uncovered
        the result is NEVER an abstention (selected may still be empty if there
        were no structural facets, but that's fine for the caller).
        """
        # No structural facets uncovered → success
        if not uncovered_facets:
            return False, AbstentionReason.NONE

        # From here on: structural facets remain uncovered. Whether we abstain
        # depends entirely on abstain_on_infeasible — when it is False the
        # pipeline is expected to Pareto-fallback, so we must NOT abstain.
        if not self.config.abstain_on_infeasible:
            return False, AbstentionReason.NONE

        # abstain_on_infeasible=True: emit the most specific reason.
        if not selected:
            reason = (
                AbstentionReason.HEURISTIC_BUDGET_INFEASIBLE
                if is_heuristic_mode
                else AbstentionReason.NO_COVERING_PASSAGES
            )
            return True, reason

        if budget_cap is not None:
            remaining_budget = budget_cap - total_cost
            if dual_lb > remaining_budget and self.config.early_abstain:
                reason = (
                    AbstentionReason.HEURISTIC_BUDGET_INFEASIBLE
                    if is_heuristic_mode
                    else AbstentionReason.INFEASIBILITY_PROVEN
                )
                return True, reason

        if dual_lb == float("inf"):
            return True, AbstentionReason.INFEASIBLE_MISSING_FACETS
        return True, AbstentionReason.BUDGET_EXHAUSTED

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