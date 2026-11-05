"""
Enhanced monitoring utilities for drift detection and calibration.

Per TRIDENT Framework Section 5 (Shift Monitoring):
- Per bin, log score stats, rejections π ≤ ᾱ_f, and near-threshold counts
- Post-episode, compute PSI/KL vs calibration and empirical violation on a labeled buffer
- Alarms trigger for future episodes only (no in-episode action):
  - Shrink thresholds: ᾱ_f ← ρ·ᾱ_f, ρ ∈ (0.5, 0.9)
  - Schedule recalibration when buffer reaches N_recal
- All versions (retriever/shortlister/verifier/calibrator) are hashed and logged
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np


class AlarmType(Enum):
    """Types of monitoring alarms."""
    NONE = "none"
    PSI_EXCEEDED = "psi_exceeded"
    KL_EXCEEDED = "kl_exceeded"
    VIOLATION_RATE_HIGH = "violation_rate_high"
    THRESHOLD_INFEASIBLE = "threshold_infeasible"


@dataclass
class BinStats:
    """
    Per-bin statistics for shift monitoring.

    Per Section 5: Log score stats, rejections, and near-threshold counts.
    """
    bin_key: str
    scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    rejections: int = 0  # Count of π ≤ ᾱ_f
    near_threshold: int = 0  # Count of ᾱ_f < π ≤ 2·ᾱ_f
    total_tests: int = 0
    violations: int = 0  # False positives (π ≤ ᾱ_f but ¬Σ)

    def update(
        self,
        score: float,
        p_value: float,
        alpha_bar: float,
        is_sufficient: Optional[bool] = None
    ) -> None:
        """Update bin statistics with new observation."""
        self.scores.append(score)
        self.total_tests += 1

        if p_value <= alpha_bar:
            self.rejections += 1
            if is_sufficient is not None and not is_sufficient:
                self.violations += 1
        elif p_value <= 2 * alpha_bar:
            self.near_threshold += 1

    @property
    def rejection_rate(self) -> float:
        """Get rejection rate for this bin."""
        if self.total_tests == 0:
            return 0.0
        return self.rejections / self.total_tests

    @property
    def violation_rate(self) -> float:
        """Get violation rate (false positive rate) for this bin."""
        if self.rejections == 0:
            return 0.0
        return self.violations / self.rejections

    def get_score_distribution(self, n_bins: int = 10) -> np.ndarray:
        """Get score distribution as histogram."""
        if len(self.scores) < n_bins:
            return np.zeros(n_bins)
        hist, _ = np.histogram(list(self.scores), bins=n_bins, range=(0, 1))
        hist = hist + 1e-10  # Avoid zeros
        return hist / hist.sum()


@dataclass
class ShiftAlarm:
    """Alarm triggered by shift detection."""
    alarm_type: AlarmType
    bin_key: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alarm_type': self.alarm_type.value,
            'bin_key': self.bin_key,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'recommended_action': self.recommended_action,
        }


@dataclass
class ShiftMonitorConfig:
    """Configuration for shift monitoring."""
    psi_threshold: float = 0.25  # PSI alarm threshold
    kl_threshold: float = 0.5  # KL divergence alarm threshold
    violation_multiplier: float = 2.0  # Violation rate > 2·α_query triggers alarm
    threshold_shrink_factor: float = 0.7  # ρ ∈ (0.5, 0.9) for threshold shrinking
    recalibration_buffer_size: int = 1000  # N_recal for scheduling recalibration
    baseline_window: int = 100  # Samples for establishing baseline
    monitoring_window: int = 50  # Samples for computing current stats


@dataclass
class DriftMonitor:
    """
    Enhanced drift monitor with PSI/KL and per-bin tracking.

    Per Section 5: Post-episode monitoring with no in-episode actions.
    """
    config: ShiftMonitorConfig = field(default_factory=ShiftMonitorConfig)

    # Per-bin statistics
    bin_stats: Dict[str, BinStats] = field(default_factory=dict)

    # Baseline distributions (from calibration corpus)
    baseline_distributions: Dict[str, np.ndarray] = field(default_factory=dict)

    # Alarm history
    alarms: List[ShiftAlarm] = field(default_factory=list)
    alarm_triggered: bool = False

    # Recalibration scheduling
    labeled_buffer: List[Dict[str, Any]] = field(default_factory=list)
    recalibration_scheduled: bool = False

    # Version tracking
    version_hashes: Dict[str, str] = field(default_factory=dict)

    # Global score history (for backward compatibility)
    score_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    psi_history: List[float] = field(default_factory=list)

    def set_baseline(self, bin_key: str, distribution: np.ndarray) -> None:
        """Set baseline distribution for a bin."""
        self.baseline_distributions[bin_key] = distribution

    def update(
        self,
        bin_key: str,
        score: float,
        p_value: float,
        alpha_bar: float,
        is_sufficient: Optional[bool] = None
    ) -> None:
        """
        Update monitor with new observation.

        Per Section 5: Log score stats, rejections, and near-threshold counts per bin.
        """
        if bin_key not in self.bin_stats:
            self.bin_stats[bin_key] = BinStats(bin_key=bin_key)

        self.bin_stats[bin_key].update(score, p_value, alpha_bar, is_sufficient)

        # Also update global history
        self.score_history.append(score)

        # Add to labeled buffer if we have ground truth
        if is_sufficient is not None:
            self.labeled_buffer.append({
                'bin_key': bin_key,
                'score': score,
                'p_value': p_value,
                'alpha_bar': alpha_bar,
                'is_sufficient': is_sufficient,
                'timestamp': time.time(),
            })

            # Check recalibration trigger
            if len(self.labeled_buffer) >= self.config.recalibration_buffer_size:
                self.recalibration_scheduled = True

    def check_drift(self, scores: Dict[Tuple[str, str], float] = None) -> bool:
        """
        Check for distribution drift (backward compatible interface).

        Returns True if any alarm is triggered.
        """
        if scores:
            for score in scores.values():
                self.score_history.append(score)

        # Need baseline to compare
        if len(self.score_history) < self.config.baseline_window:
            return False

        # Initialize baseline if not set
        if 'global' not in self.baseline_distributions:
            baseline_scores = list(self.score_history)[:self.config.baseline_window]
            self.baseline_distributions['global'] = self._compute_distribution(baseline_scores)

        # Check global PSI
        if len(self.score_history) >= self.config.monitoring_window:
            recent_scores = list(self.score_history)[-self.config.monitoring_window:]
            recent_dist = self._compute_distribution(recent_scores)
            psi = self._compute_psi(self.baseline_distributions['global'], recent_dist)
            self.psi_history.append(psi)

            if psi > self.config.psi_threshold:
                self.alarm_triggered = True
                self.alarms.append(ShiftAlarm(
                    alarm_type=AlarmType.PSI_EXCEEDED,
                    bin_key='global',
                    metric_value=psi,
                    threshold=self.config.psi_threshold,
                    recommended_action='shrink_thresholds',
                ))
                return True

        return False

    def run_post_episode_check(self, alpha_query: float) -> List[ShiftAlarm]:
        """
        Run post-episode shift checks.

        Per Section 5: Post-episode, compute PSI/KL vs calibration and
        empirical violation on a labeled buffer.
        """
        new_alarms = []

        for bin_key, stats in self.bin_stats.items():
            # Check PSI if we have baseline
            if bin_key in self.baseline_distributions and len(stats.scores) >= 20:
                current_dist = stats.get_score_distribution()
                baseline_dist = self.baseline_distributions[bin_key]

                psi = self._compute_psi(baseline_dist, current_dist)
                if psi > self.config.psi_threshold:
                    alarm = ShiftAlarm(
                        alarm_type=AlarmType.PSI_EXCEEDED,
                        bin_key=bin_key,
                        metric_value=psi,
                        threshold=self.config.psi_threshold,
                        recommended_action=f'shrink_threshold_by_{self.config.threshold_shrink_factor}',
                    )
                    new_alarms.append(alarm)

                kl = self._compute_kl(baseline_dist, current_dist)
                if kl > self.config.kl_threshold:
                    alarm = ShiftAlarm(
                        alarm_type=AlarmType.KL_EXCEEDED,
                        bin_key=bin_key,
                        metric_value=kl,
                        threshold=self.config.kl_threshold,
                        recommended_action='schedule_recalibration',
                    )
                    new_alarms.append(alarm)

            # Check violation rate
            if stats.violations > 0:
                violation_rate = stats.violation_rate
                expected_rate = alpha_query
                if violation_rate > self.config.violation_multiplier * expected_rate:
                    alarm = ShiftAlarm(
                        alarm_type=AlarmType.VIOLATION_RATE_HIGH,
                        bin_key=bin_key,
                        metric_value=violation_rate,
                        threshold=self.config.violation_multiplier * expected_rate,
                        recommended_action='shrink_threshold',
                    )
                    new_alarms.append(alarm)

        self.alarms.extend(new_alarms)
        if new_alarms:
            self.alarm_triggered = True

        return new_alarms

    def get_threshold_adjustments(self) -> Dict[str, float]:
        """
        Get recommended threshold adjustments based on alarms.

        Per Section 5: ᾱ_f ← ρ·ᾱ_f, ρ ∈ (0.5, 0.9)
        """
        adjustments = {}
        rho = self.config.threshold_shrink_factor

        for alarm in self.alarms:
            if alarm.alarm_type in [AlarmType.PSI_EXCEEDED, AlarmType.VIOLATION_RATE_HIGH]:
                bin_key = alarm.bin_key
                if bin_key not in adjustments:
                    adjustments[bin_key] = rho
                else:
                    # Multiple alarms: compound shrinkage
                    adjustments[bin_key] *= rho

        return adjustments

    def _compute_distribution(self, scores: List[float], n_bins: int = 10) -> np.ndarray:
        """Compute score distribution as histogram."""
        hist, _ = np.histogram(scores, bins=n_bins, range=(0, 1))
        hist = hist + 1e-10  # Avoid zeros
        return hist / hist.sum()

    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """
        Compute Population Stability Index.

        PSI = Σ (current_i - baseline_i) * ln(current_i / baseline_i)
        """
        psi = 0.0
        for b, c in zip(baseline, current):
            if b > 0 and c > 0:
                psi += (c - b) * math.log(c / b)
        return abs(psi)

    def _compute_kl(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """
        Compute KL divergence from baseline to current.

        KL(baseline || current) = Σ baseline_i * ln(baseline_i / current_i)
        """
        kl = 0.0
        for b, c in zip(baseline, current):
            if b > 0 and c > 0:
                kl += b * math.log(b / c)
        return abs(kl)

    def reset_baseline(self) -> None:
        """Reset baseline distributions from current data."""
        for bin_key, stats in self.bin_stats.items():
            if len(stats.scores) >= self.config.baseline_window:
                self.baseline_distributions[bin_key] = stats.get_score_distribution()

        # Global baseline
        if len(self.score_history) >= self.config.baseline_window:
            self.baseline_distributions['global'] = self._compute_distribution(
                list(self.score_history)[-self.config.baseline_window:]
            )

        self.alarm_triggered = False

    def reset_for_episode(self) -> None:
        """Reset per-episode state while preserving baselines."""
        self.alarm_triggered = False
        # Note: We don't clear bin_stats as we want to accumulate for post-episode checks

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            'alarm_triggered': self.alarm_triggered,
            'num_alarms': len(self.alarms),
            'latest_psi': self.psi_history[-1] if self.psi_history else 0,
            'recalibration_scheduled': self.recalibration_scheduled,
            'labeled_buffer_size': len(self.labeled_buffer),
            'bins_monitored': len(self.bin_stats),
            'per_bin_stats': {
                bin_key: {
                    'rejection_rate': stats.rejection_rate,
                    'violation_rate': stats.violation_rate,
                    'total_tests': stats.total_tests,
                }
                for bin_key, stats in self.bin_stats.items()
            }
        }

    def set_version_hashes(
        self,
        retriever_hash: str = '',
        shortlister_hash: str = '',
        verifier_hash: str = '',
        calibrator_hash: str = ''
    ) -> None:
        """Set version hashes for telemetry."""
        self.version_hashes = {
            'retriever': retriever_hash,
            'shortlister': shortlister_hash,
            'verifier': verifier_hash,
            'calibrator': calibrator_hash,
        }


@dataclass
class CalibrationMonitor:
    """Monitor calibration quality over time."""

    target_alpha: float
    window_size: int = 100
    history: deque = field(default_factory=lambda: deque(maxlen=1000))

    def update(self, p_value: float, is_correct: bool) -> None:
        """Update with new observation."""
        self.history.append((p_value, is_correct))

    def get_violation_rate(self, alpha: float) -> float:
        """Get empirical violation rate at given alpha."""
        if not self.history:
            return 0.0

        violations = sum(
            1 for p, correct in self.history
            if p <= alpha and not correct
        )
        total = sum(1 for p, _ in self.history if p <= alpha)

        return violations / max(total, 1)

    def is_calibrated(self, tolerance: float = 0.05) -> bool:
        """Check if calibration is within tolerance."""
        violation_rate = self.get_violation_rate(self.target_alpha)
        return abs(violation_rate - self.target_alpha) <= tolerance

    def get_calibration_curve(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibration curve."""
        if len(self.history) < n_bins:
            return np.array([]), np.array([])

        alphas = np.linspace(0, 1, n_bins + 1)
        expected = []
        observed = []

        for i in range(n_bins):
            alpha_low = alphas[i]
            alpha_high = alphas[i + 1]

            in_bin = [
                (p, correct) for p, correct in self.history
                if alpha_low <= p < alpha_high
            ]

            if in_bin:
                expected.append((alpha_low + alpha_high) / 2)
                observed.append(sum(correct for _, correct in in_bin) / len(in_bin))

        return np.array(expected), np.array(observed)


@dataclass
class CoverageMonitor:
    """Monitor facet coverage rates."""

    facet_coverage: Dict[str, List[bool]] = field(default_factory=dict)
    passage_efficiency: List[float] = field(default_factory=list)

    def update_coverage(
        self,
        facets: List[Any],
        covered_facets: List[str],
        tokens_used: int
    ) -> None:
        """Update coverage statistics."""
        for facet in facets:
            facet_id = facet.facet_id if hasattr(facet, 'facet_id') else str(facet)
            is_covered = facet_id in covered_facets

            if facet_id not in self.facet_coverage:
                self.facet_coverage[facet_id] = []
            self.facet_coverage[facet_id].append(is_covered)

        # Track efficiency
        coverage_rate = len(covered_facets) / max(len(facets), 1)
        efficiency = coverage_rate / max(tokens_used, 1) * 1000  # Coverage per 1k tokens
        self.passage_efficiency.append(efficiency)

    def get_facet_success_rates(self) -> Dict[str, float]:
        """Get success rate for each facet type."""
        success_rates = {}
        for facet_id, coverage_history in self.facet_coverage.items():
            if coverage_history:
                success_rates[facet_id] = sum(coverage_history) / len(coverage_history)
        return success_rates

    def get_average_efficiency(self) -> float:
        """Get average passage efficiency."""
        if not self.passage_efficiency:
            return 0.0
        return float(np.mean(self.passage_efficiency))

    def get_coverage_trend(self, window: int = 50) -> List[float]:
        """Get coverage trend over time."""
        all_coverage = []

        # Aggregate all facet coverage
        max_len = max(len(v) for v in self.facet_coverage.values()) if self.facet_coverage else 0

        for i in range(max_len):
            covered = 0
            total = 0
            for facet_history in self.facet_coverage.values():
                if i < len(facet_history):
                    covered += facet_history[i]
                    total += 1

            if total > 0:
                all_coverage.append(covered / total)

        # Compute moving average
        if len(all_coverage) < window:
            return all_coverage

        trend = []
        for i in range(len(all_coverage) - window + 1):
            window_avg = float(np.mean(all_coverage[i:i + window]))
            trend.append(window_avg)

        return trend


class SystemMonitor:
    """Comprehensive system monitoring."""

    def __init__(self, config: Any):
        self.config = config
        self.drift_monitor = DriftMonitor(
            config=ShiftMonitorConfig(
                psi_threshold=getattr(config, 'psi_threshold', 0.25),
            )
        )
        self.calibration_monitors: Dict[str, CalibrationMonitor] = {}
        self.coverage_monitor = CoverageMonitor()
        self.latency_tracker = LatencyTracker()
        self.memory_tracker = MemoryTracker() if getattr(config.telemetry, 'profile_memory', False) else None

    def update(self, pipeline_output: Dict[str, Any]) -> None:
        """Update all monitors with pipeline output."""
        # Update drift monitor
        if 'scores' in pipeline_output:
            self.drift_monitor.check_drift(pipeline_output['scores'])

        # Update calibration monitors
        if 'certificates' in pipeline_output:
            for cert in pipeline_output['certificates']:
                facet_type = cert.get('facet_type', 'default')
                if facet_type not in self.calibration_monitors:
                    self.calibration_monitors[facet_type] = CalibrationMonitor(
                        target_alpha=cert.get('alpha_bar', 0.01)
                    )

                self.calibration_monitors[facet_type].update(
                    p_value=cert['p_value'],
                    is_correct=cert.get('is_correct', True)
                )

        # Update coverage
        if 'facets' in pipeline_output and 'covered_facets' in pipeline_output:
            self.coverage_monitor.update_coverage(
                facets=pipeline_output['facets'],
                covered_facets=pipeline_output['covered_facets'],
                tokens_used=pipeline_output.get('tokens_used', 0)
            )

        # Update latency
        if 'latency_ms' in pipeline_output:
            self.latency_tracker.record(pipeline_output['latency_ms'])

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        status = {
            'drift': self.drift_monitor.get_status(),
            'calibration': {
                facet_type: {
                    'is_calibrated': monitor.is_calibrated(),
                    'violation_rate': monitor.get_violation_rate(monitor.target_alpha)
                }
                for facet_type, monitor in self.calibration_monitors.items()
            },
            'coverage': {
                'average_efficiency': self.coverage_monitor.get_average_efficiency(),
                'success_rates': self.coverage_monitor.get_facet_success_rates()
            },
            'latency': self.latency_tracker.get_stats()
        }

        if self.memory_tracker:
            status['memory'] = self.memory_tracker.get_stats()

        return status


@dataclass
class LatencyTracker:
    """Track latency statistics."""

    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latencies.append(latency_ms)

    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0}

        latencies = sorted(self.latencies)
        n = len(latencies)

        return {
            'mean': float(np.mean(latencies)),
            'p50': latencies[n // 2],
            'p95': latencies[int(n * 0.95)] if n > 20 else latencies[-1],
            'p99': latencies[int(n * 0.99)] if n > 100 else latencies[-1]
        }


class MemoryTracker:
    """Track memory usage."""

    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            self.psutil = None
            self.process = None

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.psutil:
            return {'error': 'psutil not installed'}

        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
