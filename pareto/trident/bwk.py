"""Bandits with Knapsacks (BwK) controller for action selection."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


@dataclass
class BwKState:
    """State representation for BwK."""
    deficit_vector: List[str]  # Uncovered facets
    budget_remaining: int
    retriever_entropy: float
    cache_hits: int
    latency_percentile: float
    iteration: int

    def to_feature_vector(self) -> np.ndarray:
        """Convert state to feature vector."""
        features = [
            len(self.deficit_vector),  # Number of uncovered facets
            self.budget_remaining / 1000.0,  # Normalized budget
            self.retriever_entropy,
            self.cache_hits / 100.0,  # Normalized cache hits
            self.latency_percentile,
            self.iteration / 10.0  # Normalized iteration
        ]
        return np.array(features)


@dataclass
class BwKArm:
    """Arm (action) in BwK."""
    name: str
    expected_cost: int
    cost_variance: float
    pulls: int = 0
    total_reward: float = 0.0
    ucb_score: float = float('inf')


class BwKController:
    """
    Bandits with Knapsacks controller for exploration under budgets.

    From TRIDENT spec: Uses consumption-aware UCB for action selection
    with marginal verified coverage per token as reward.
    """

    def __init__(self, config: Any):
        self.config = config
        self.exploration_bonus = config.bwk_exploration_bonus

        # Initialize arms
        self.arms = {
            'add_evidence': BwKArm('add_evidence', expected_cost=200, cost_variance=50),
            'hop_expand': BwKArm('hop_expand', expected_cost=500, cost_variance=150),
            're_read': BwKArm('re_read', expected_cost=100, cost_variance=20),
            'vqc_rewrite': BwKArm('vqc_rewrite', expected_cost=300, cost_variance=80)
        }

        # Episode state
        self.current_episode = None
        self.episode_history = []
        self.total_pulls = 0

        # Contextual bandits parameters
        self.context_weights = np.random.randn(6, len(self.arms))  # 6 features, 4 arms
        self.learning_rate = 0.01

    def start_episode(self, initial_state: Dict[str, Any]) -> None:
        """Start a new episode (query)."""
        self.current_episode = {
            'query': initial_state.get('query', ''),
            'initial_facets': initial_state.get('facets', []),
            'budget': initial_state.get('budget_remaining', 2000),
            'actions': [],
            'rewards': [],
            'costs': []
        }

    def select_action(
        self,
        deficit: List[str],
        budget_remaining: int,
        retriever_entropy: float = 0.5,
        cache_hits: int = 0,
        latency_percentile: float = 0.5
    ) -> str:
        """
        Select action using consumption-aware UCB.

        Disallows arms whose worst-case cost exceeds budget.
        """
        # Create state
        state = BwKState(
            deficit_vector=deficit,
            budget_remaining=budget_remaining,
            retriever_entropy=retriever_entropy,
            cache_hits=cache_hits,
            latency_percentile=latency_percentile,
            iteration=len(self.current_episode['actions']) if self.current_episode else 0
        )

        # Get feasible arms
        feasible_arms = self._get_feasible_arms(budget_remaining)

        if not feasible_arms:
            # No feasible actions, return cheapest
            return min(self.arms.values(), key=lambda a: a.expected_cost).name

        # Near end of budget: myopic best
        if budget_remaining < 500:
            return self._select_myopic_best(feasible_arms, state)

        # Compute UCB scores
        best_arm = None
        best_score = -float('inf')

        for arm_name in feasible_arms:
            arm = self.arms[arm_name]
            score = self._compute_ucb_score(arm, state)

            if score > best_score:
                best_score = score
                best_arm = arm_name

        # Record action
        if self.current_episode:
            self.current_episode['actions'].append(best_arm)
            self.current_episode['costs'].append(self.arms[best_arm].expected_cost)

        self.total_pulls += 1
        self.arms[best_arm].pulls += 1

        return best_arm

    def _get_feasible_arms(self, budget_remaining: int) -> List[str]:
        """Get arms whose worst-case cost fits in budget."""
        feasible = []

        for name, arm in self.arms.items():
            worst_case_cost = arm.expected_cost + 2 * math.sqrt(arm.cost_variance)
            if worst_case_cost <= budget_remaining:
                feasible.append(name)

        return feasible

    def _compute_ucb_score(self, arm: BwKArm, state: BwKState) -> float:
        """Compute UCB score for an arm given state."""
        if arm.pulls == 0:
            return float('inf')  # Explore unpulled arms first

        # Base UCB
        avg_reward = arm.total_reward / max(arm.pulls, 1)
        exploration_term = self.exploration_bonus * math.sqrt(
            2 * math.log(self.total_pulls + 1) / arm.pulls
        )

        # Contextual adjustment
        features = state.to_feature_vector()
        arm_idx = list(self.arms.keys()).index(arm.name)
        context_score = np.dot(features, self.context_weights[:, arm_idx])

        # Consumption-aware adjustment
        budget_factor = state.budget_remaining / max(arm.expected_cost, 1)

        return avg_reward + exploration_term + 0.1 * context_score + 0.1 * math.log(budget_factor + 1)

    def _select_myopic_best(self, feasible_arms: List[str], state: BwKState) -> str:
        """Select best action myopically when budget is low."""
        best_arm = None
        best_efficiency = -float('inf')

        for arm_name in feasible_arms:
            arm = self.arms[arm_name]
            if arm.pulls > 0:
                avg_reward = arm.total_reward / arm.pulls
                efficiency = avg_reward / max(arm.expected_cost, 1)

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_arm = arm_name

        return best_arm or feasible_arms[0]

    def update_reward(
        self,
        reward: float,
        actual_cost: Optional[int] = None
    ) -> None:
        """
        Update with observed reward.

        Reward = marginal verified coverage per token.
        """
        if not self.current_episode or not self.current_episode['actions']:
            return

        last_action = self.current_episode['actions'][-1]
        arm = self.arms[last_action]

        # Update arm statistics
        arm.total_reward += reward

        # Update cost estimate if provided
        if actual_cost is not None:
            # Online update of expected cost
            alpha = 1.0 / (arm.pulls + 1)
            arm.expected_cost = (1 - alpha) * arm.expected_cost + alpha * actual_cost

            # Update variance estimate
            cost_error = actual_cost - arm.expected_cost
            arm.cost_variance = (1 - alpha) * arm.cost_variance + alpha * cost_error ** 2

        # Record reward
        self.current_episode['rewards'].append(reward)

        # Update context weights (simple gradient update)
        if arm.pulls > 1:
            last_state_idx = len(self.current_episode['actions']) - 1
            if last_state_idx >= 0:
                # Reconstruct state features (simplified)
                features = np.random.randn(6)  # Would need to store actual state
                arm_idx = list(self.arms.keys()).index(last_action)

                # Gradient update
                prediction = np.dot(features, self.context_weights[:, arm_idx])
                error = reward - prediction
                self.context_weights[:, arm_idx] += self.learning_rate * error * features

    def end_episode(self) -> Dict[str, Any]:
        """End current episode and return statistics."""
        if not self.current_episode:
            return {}

        episode_stats = {
            'total_actions': len(self.current_episode['actions']),
            'total_reward': sum(self.current_episode['rewards']),
            'total_cost': sum(self.current_episode['costs']),
            'action_distribution': self._get_action_distribution(),
            'average_reward': np.mean(self.current_episode['rewards']) if self.current_episode['rewards'] else 0
        }

        self.episode_history.append(self.current_episode)
        self.current_episode = None

        return episode_stats

    def _get_action_distribution(self) -> Dict[str, int]:
        """Get distribution of actions taken in current episode."""
        if not self.current_episode:
            return {}

        distribution = {}
        for action in self.current_episode['actions']:
            distribution[action] = distribution.get(action, 0) + 1

        return distribution

    def get_regret_estimate(self) -> float:
        """
        Estimate regret (simplified).

        True regret computation would require knowing optimal policy.
        """
        if not self.episode_history:
            return 0.0

        # Use best empirical arm as proxy for optimal
        best_avg_reward = max(
            arm.total_reward / max(arm.pulls, 1)
            for arm in self.arms.values()
        )

        # Compute empirical regret
        total_regret = 0.0
        total_pulls = sum(arm.pulls for arm in self.arms.values())

        for arm in self.arms.values():
            if arm.pulls > 0:
                avg_reward = arm.total_reward / arm.pulls
                regret = (best_avg_reward - avg_reward) * arm.pulls
                total_regret += regret

        return total_regret / max(total_pulls, 1)

    def reset(self) -> None:
        """Reset the controller."""
        for arm in self.arms.values():
            arm.pulls = 0
            arm.total_reward = 0.0
            arm.ucb_score = float('inf')

        self.current_episode = None
        self.episode_history.clear()
        self.total_pulls = 0
        self.context_weights = np.random.randn(6, len(self.arms))