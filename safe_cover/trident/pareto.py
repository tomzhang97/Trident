"""Pareto-Knapsack optimizer for relaxed coverage utility."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from .candidates import Passage
from .facets import Facet
from .config import ParetoConfig


@dataclass
class ParetoResult:
    """Result from Pareto-Knapsack optimization."""
    selected_passages: List[Passage]
    covered_facets: List[str]
    uncovered_facets: List[str]
    achieved_utility: float
    total_cost: int
    trace: List[Dict[str, Any]] = field(default_factory=list)
    pareto_points: List[Tuple[float, int]] = field(default_factory=list)  # (utility, cost) pairs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'selected_passages': [p.__dict__ for p in self.selected_passages],
            'covered_facets': self.covered_facets,
            'uncovered_facets': self.uncovered_facets,
            'achieved_utility': self.achieved_utility,
            'total_cost': self.total_cost,
            'trace': self.trace,
            'pareto_points': self.pareto_points
        }


class ParetoKnapsackOptimizer:
    """
    Pareto-Knapsack optimizer for submodular coverage utility.

    This implements the relaxed optimization mode from TRIDENT spec,
    using lazy greedy with cost-aware selection.
    """

    def __init__(self, config: ParetoConfig):
        self.config = config
        self.default_weight = config.weight_default

    def optimize(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        budget: Optional[int] = None
    ) -> ParetoResult:
        """
        Optimize passage selection for coverage utility under budget.

        Uses lazy greedy algorithm for submodular maximization.
        Enforces max_evidence_tokens and max_units constraints if configured.
        """
        # Use config-specified limits if available, otherwise fall back to budget parameter
        if budget is None:
            if self.config.budget is None or self.config.budget <= 0:
                raise ValueError("Pareto budget must be specified and positive")
            budget = self.config.budget

        max_evidence_tokens = self.config.max_evidence_tokens or budget
        max_units = self.config.max_units or len(passages)

        # Initialize
        selected = []
        covered = set()
        uncovered = {f.facet_id for f in facets}
        total_cost = 0
        trace = []

        # Lazy evaluation with priority queue
        # Heap elements: (-marginal_gain_per_cost, passage)
        heap = []
        passage_gains = {}  # Cache marginal gains

        # Initialize heap with all passages
        for passage in passages:
            gain = self._compute_marginal_gain(
                passage, facets, covered, p_values
            )
            gain_per_cost = gain / max(passage.cost, 1)
            heapq.heappush(heap, (-gain_per_cost, id(passage), passage))
            passage_gains[passage.pid] = gain

        # Lazy greedy selection with budget and unit constraints
        while heap and total_cost < max_evidence_tokens and len(selected) < max_units:
            # Get best candidate
            neg_gain_per_cost, _, passage = heapq.heappop(heap)
            gain_per_cost = -neg_gain_per_cost

            # Recompute marginal gain (lazy evaluation)
            current_gain = self._compute_marginal_gain(
                passage, facets, covered, p_values
            )
            current_gain_per_cost = current_gain / max(passage.cost, 1)

            # Check if this is still the best option
            if current_gain_per_cost < gain_per_cost - 1e-9:
                # Gain has decreased, re-insert with updated value
                heapq.heappush(heap, (-current_gain_per_cost, id(passage), passage))
                passage_gains[passage.pid] = current_gain
                continue

            # Check budget constraint
            if self.config.stop_on_budget and total_cost + passage.cost > max_evidence_tokens:
                # Skip this passage if it would exceed budget
                continue

            # Check unit limit
            if len(selected) >= max_units:
                break

            # Select this passage
            selected.append(passage)
            total_cost += passage.cost

            # Update covered facets
            newly_covered = self._get_covered_facets(
                passage, facets, p_values
            ) - covered
            covered.update(newly_covered)
            uncovered -= newly_covered

            # Log trace
            trace.append({
                'passage_id': passage.pid,
                'cost': passage.cost,
                'marginal_gain': current_gain,
                'gain_per_cost': current_gain_per_cost,
                'newly_covered': list(newly_covered),
                'total_cost': total_cost,
                'utility': self._compute_utility(covered, facets)
            })

        # Check best singleton (in case it's better than greedy solution)
        best_singleton = self._find_best_singleton(
            passages, facets, p_values, budget
        )

        if best_singleton:
            singleton_utility = self._compute_utility(
                self._get_covered_facets(best_singleton, facets, p_values),
                facets
            )
            greedy_utility = self._compute_utility(covered, facets)

            if singleton_utility > greedy_utility:
                # Use singleton instead
                selected = [best_singleton]
                covered = self._get_covered_facets(best_singleton, facets, p_values)
                uncovered = {f.facet_id for f in facets} - covered
                total_cost = best_singleton.cost

        # Compute Pareto frontier points
        pareto_points = self._compute_pareto_frontier(
            passages, facets, p_values, budget
        )

        return ParetoResult(
            selected_passages=selected,
            covered_facets=list(covered),
            uncovered_facets=list(uncovered),
            achieved_utility=self._compute_utility(covered, facets),
            total_cost=total_cost,
            trace=trace,
            pareto_points=pareto_points
        )

    def _compute_marginal_gain(
        self,
        passage: Passage,
        facets: List[Facet],
        covered: Set[str],
        p_values: Dict[Tuple[str, str], float]
    ) -> float:
        """Compute marginal utility gain from adding passage."""
        gain = 0.0

        for facet in facets:
            if facet.facet_id in covered:
                continue

            key = (passage.pid, facet.facet_id)
            if key in p_values:
                # Use relaxed threshold (possibly from BH-FDR)
                if p_values[key] <= self.config.relaxed_alpha:
                    gain += facet.weight

        return gain

    def _get_covered_facets(
        self,
        passage: Passage,
        facets: List[Facet],
        p_values: Dict[Tuple[str, str], float]
    ) -> Set[str]:
        """Get facets covered by a passage."""
        covered = set()

        for facet in facets:
            key = (passage.pid, facet.facet_id)
            if key in p_values:
                if p_values[key] <= self.config.relaxed_alpha:
                    covered.add(facet.facet_id)

        return covered

    def _compute_utility(
        self,
        covered: Set[str],
        facets: List[Facet]
    ) -> float:
        """Compute total utility from covered facets."""
        utility = 0.0

        for facet in facets:
            if facet.facet_id in covered:
                utility += facet.weight

        return utility

    def _find_best_singleton(
        self,
        passages: List[Passage],
        facets: List[Facet],
        p_values: Dict[Tuple[str, str], float],
        budget: int
    ) -> Optional[Passage]:
        """Find best single passage within budget."""
        best_passage = None
        best_utility = 0.0

        for passage in passages:
            if passage.cost > budget:
                continue

            covered = self._get_covered_facets(passage, facets, p_values)
            utility = self._compute_utility(covered, facets)

            if utility > best_utility:
                best_utility = utility
                best_passage = passage

        return best_passage

    def _compute_pareto_frontier(
        self,
        passages: List[Passage],
        facets: List[Facet],
        p_values: Dict[Tuple[str, str], float],
        max_budget: int,
        num_points: int = 20
    ) -> List[Tuple[float, int]]:
        """
        Compute Pareto frontier of utility vs cost.

        This helps visualize the quality-cost tradeoff.
        """
        pareto_points = []

        # Sample different budget levels
        budget_levels = np.linspace(
            min(p.cost for p in passages),
            min(max_budget, sum(p.cost for p in passages[:10])),
            num_points
        )

        for budget in budget_levels:
            # Run optimization at this budget
            result = self._optimize_at_budget(
                passages, facets, p_values, int(budget)
            )
            pareto_points.append((result['utility'], result['cost']))

        # Remove dominated points
        pareto_points = self._filter_dominated_points(pareto_points)

        return pareto_points

    def _optimize_at_budget(
        self,
        passages: List[Passage],
        facets: List[Facet],
        p_values: Dict[Tuple[str, str], float],
        budget: int
    ) -> Dict[str, float]:
        """Quick optimization at specific budget (for Pareto curve)."""
        covered = set()
        total_cost = 0

        # Simple greedy for speed
        for passage in sorted(passages, key=lambda p: p.cost):
            if total_cost + passage.cost > budget:
                continue

            newly_covered = self._get_covered_facets(
                passage, facets, p_values
            ) - covered

            if newly_covered:
                covered.update(newly_covered)
                total_cost += passage.cost

        return {
            'utility': self._compute_utility(covered, facets),
            'cost': total_cost
        }

    def _filter_dominated_points(
        self,
        points: List[Tuple[float, int]]
    ) -> List[Tuple[float, int]]:
        """Remove dominated points from Pareto frontier."""
        if not points:
            return []

        # Sort by cost
        points.sort(key=lambda x: x[1])

        pareto = []
        max_utility = -float('inf')

        for utility, cost in points:
            if utility > max_utility:
                pareto.append((utility, cost))
                max_utility = utility

        return pareto

    def continuous_greedy_with_rounding(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        budget: int,
        num_iterations: int = 100
    ) -> ParetoResult:
        """
        Continuous greedy with rounding (Sviridenko's algorithm).

        Achieves (1-1/e) approximation for monotone submodular functions.
        This is more theoretically principled but slower than lazy greedy.
        """
        n = len(passages)

        # Initialize fractional solution
        x = np.zeros(n)

        # Continuous greedy
        for t in range(num_iterations):
            # Compute gradient
            gradient = np.zeros(n)

            for i, passage in enumerate(passages):
                if x[i] >= 1.0:
                    continue

                # Estimate marginal gain
                marginal = self._estimate_multilinear_extension_gradient(
                    i, x, passages, facets, p_values
                )
                gradient[i] = marginal / max(passage.cost, 1)

            # Find feasible direction
            best_idx = np.argmax(gradient)

            if gradient[best_idx] <= 0:
                break

            # Update fractional solution
            step_size = min(1.0 / num_iterations, 1.0 - x[best_idx])
            x[best_idx] += step_size

        # Rounding: select passages with highest fractional values
        selected_indices = []
        total_cost = 0

        for idx in np.argsort(-x):
            if total_cost + passages[idx].cost <= budget:
                selected_indices.append(idx)
                total_cost += passages[idx].cost

        # Compute result
        selected = [passages[i] for i in selected_indices]
        covered = set()

        for passage in selected:
            covered.update(self._get_covered_facets(passage, facets, p_values))

        uncovered = {f.facet_id for f in facets} - covered

        return ParetoResult(
            selected_passages=selected,
            covered_facets=list(covered),
            uncovered_facets=list(uncovered),
            achieved_utility=self._compute_utility(covered, facets),
            total_cost=total_cost,
            trace=[],
            pareto_points=[]
        )

    def _estimate_multilinear_extension_gradient(
        self,
        idx: int,
        x: np.ndarray,
        passages: List[Passage],
        facets: List[Facet],
        p_values: Dict[Tuple[str, str], float],
        num_samples: int = 10
    ) -> float:
        """
        Estimate gradient of multilinear extension via sampling.

        This is used in the continuous greedy algorithm.
        """
        marginal_sum = 0.0

        for _ in range(num_samples):
            # Sample random set based on x
            sampled = set()
            for i, prob in enumerate(x):
                if i != idx and np.random.random() < prob:
                    sampled.add(i)

            # Compute marginal gain
            covered_without = set()
            for i in sampled:
                covered_without.update(
                    self._get_covered_facets(passages[i], facets, p_values)
                )

            covered_with = covered_without.copy()
            covered_with.update(
                self._get_covered_facets(passages[idx], facets, p_values)
            )

            utility_without = self._compute_utility(covered_without, facets)
            utility_with = self._compute_utility(covered_with, facets)

            marginal_sum += utility_with - utility_without

        return marginal_sum / num_samples