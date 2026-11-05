"""Logging and telemetry utilities for TRIDENT."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logger(
    output_dir: str,
    name: str = "trident",
    level: str = "INFO"
) -> logging.Logger:
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)

    # File handler
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f"{output_dir}/trident.log")
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(console_format)

    # Add handlers
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def log_metrics(
    metrics: Dict[str, Any],
    output_path: str,
    append: bool = True
) -> None:
    """Log metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if append and Path(output_path).exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(metrics)
            metrics = existing
        else:
            metrics = [existing, metrics]

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


@dataclass
class TelemetryTracker:
    """Track telemetry data for pipeline execution."""

    config: Any
    events: List[Dict[str, Any]] = field(default_factory=list)
    timers: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    current_query: Optional[str] = None
    query_start_time: Optional[float] = None

    def start_query(self, query: str) -> None:
        """Start tracking a new query."""
        self.current_query = query
        self.query_start_time = time.time()
        self.log("query_start", {"query": query})

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[f"{name}_start"] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        start_key = f"{name}_start"
        if start_key not in self.timers:
            return 0.0

        elapsed = time.time() - self.timers[start_key]
        self.timers[name] = elapsed
        del self.timers[start_key]

        self.log(f"{name}_complete", {"elapsed_ms": elapsed * 1000})
        return elapsed

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value

    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.enable:
            return

        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }

        if self.current_query:
            event["query"] = self.current_query

        self.events.append(event)

    def get_trace(self) -> Dict[str, Any]:
        """Get complete telemetry trace."""
        return {
            "events": self.events,
            "timers": self.timers,
            "counters": self.counters,
            "total_time": time.time() - self.query_start_time if self.query_start_time else 0
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "num_events": len(self.events),
            "total_time": time.time() - self.query_start_time if self.query_start_time else 0,
            "timers": dict(self.timers),
            "counters": dict(self.counters)
        }

    def save_trace(self, output_path: str) -> None:
        """Save telemetry trace to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.get_trace(), f, indent=2)

    def reset(self) -> None:
        """Reset telemetry for new query."""
        self.events.clear()
        self.timers.clear()
        self.counters.clear()
        self.current_query = None
        self.query_start_time = None


@dataclass
class SafeCoverDebugger:
    """
    Debug helper for Safe-Cover pipeline.

    Use this to diagnose common issues:
    - Zero evidence selection
    - Always abstaining
    - P-value/calibration issues
    """

    enabled: bool = True
    log_level: str = "INFO"
    _logger: Optional[logging.Logger] = None

    def __post_init__(self):
        if self.enabled:
            self._logger = logging.getLogger("safe_cover_debug")
            self._logger.setLevel(getattr(logging, self.log_level))
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s [DEBUG] %(message)s'
                ))
                self._logger.addHandler(handler)

    def log(self, msg: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Log a debug message."""
        if self.enabled and self._logger:
            if data:
                self._logger.info(f"{msg}: {json.dumps(data, default=str)}")
            else:
                self._logger.info(msg)

    def log_retrieval(self, num_passages: int, retriever_type: str) -> None:
        """Log retrieval stage results."""
        self.log("RETRIEVAL", {
            "num_passages": num_passages,
            "retriever_type": retriever_type,
            "status": "OK" if num_passages > 0 else "EMPTY"
        })
        if num_passages == 0:
            self.log("WARNING: Zero passages retrieved. Check corpus_path/index_path config.")

    def log_facets(self, facets: List[Any]) -> None:
        """Log facet mining results."""
        self.log("FACET_MINING", {
            "num_facets": len(facets),
            "facet_types": [getattr(f, 'facet_type', 'unknown') for f in facets] if facets else [],
            "status": "OK" if len(facets) > 0 else "EMPTY"
        })
        if not facets:
            self.log("WARNING: Zero facets mined. Will abstain with reason=no_facets.")

    def log_scoring(self, num_scores: int, p_values: Dict[Any, float]) -> None:
        """Log scoring stage results."""
        if p_values:
            min_pv = min(p_values.values())
            max_pv = max(p_values.values())
            avg_pv = sum(p_values.values()) / len(p_values)
        else:
            min_pv = max_pv = avg_pv = 0.0

        self.log("SCORING", {
            "num_scores": num_scores,
            "min_pvalue": min_pv,
            "max_pvalue": max_pv,
            "avg_pvalue": avg_pv,
        })

    def log_selection(
        self,
        selected: List[Any],
        covered: List[str],
        uncovered: List[str],
        alpha: float
    ) -> None:
        """Log selection stage results."""
        self.log("SELECTION", {
            "num_selected": len(selected),
            "num_covered": len(covered),
            "num_uncovered": len(uncovered),
            "alpha": alpha,
            "status": "OK" if len(selected) > 0 else "EMPTY"
        })
        if not selected:
            self.log("WARNING: Zero passages selected. Check p-value thresholds vs alpha.")

    def log_abstention(self, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log abstention event."""
        self.log("ABSTENTION", {
            "reason": reason,
            "details": details or {}
        })

    def diagnose_empty_selection(
        self,
        num_passages: int,
        num_facets: int,
        p_values: Dict[Any, float],
        alpha: float
    ) -> str:
        """Diagnose why selection is empty and return recommendations."""
        issues = []

        if num_passages == 0:
            issues.append("ISSUE: Zero passages retrieved.")
            issues.append("  FIX: Check retrieval.corpus_path and retrieval.index_path in config.")

        if num_facets == 0:
            issues.append("ISSUE: Zero facets mined.")
            issues.append("  FIX: Check query format or facet miner configuration.")

        if p_values:
            min_pv = min(p_values.values())
            if min_pv > alpha:
                issues.append(f"ISSUE: Minimum p-value ({min_pv:.4f}) > alpha ({alpha:.4f}).")
                issues.append("  FIX: Increase safe_cover.per_facet_alpha or use pvalue_mode='randomized'.")
                issues.append("  FIX: Decrease nli.score_threshold if verifier is too strict.")

            below_alpha = sum(1 for pv in p_values.values() if pv <= alpha)
            if below_alpha == 0:
                issues.append(f"ISSUE: No p-values below threshold.")
                issues.append(f"  INFO: {len(p_values)} scores, alpha={alpha}")

        return "\n".join(issues) if issues else "No obvious issues detected."


def create_debug_config() -> Dict[str, Any]:
    """
    Create a debug-friendly config for troubleshooting Safe-Cover.

    Use this config to validate the pipeline before tightening thresholds.
    """
    return {
        "mode": "safe_cover",
        "safe_cover": {
            "per_facet_alpha": 0.1,  # Relaxed threshold
            "early_abstain": False,  # Don't abstain early
            "abstain_on_infeasible": False,  # Allow partial coverage
            "pvalue_mode": "randomized",  # Avoid deterministic discretization
            "fallback_to_pareto": True,  # Fallback if Safe-Cover fails
        },
        "calibration": {
            "use_mondrian": False,  # Simpler calibration
            "pvalue_mode": "randomized",
            "n_min": 30,  # Lower bin requirement
        },
        "nli": {
            "score_threshold": 0.8,  # Less strict scoring
        },
        "telemetry": {
            "enable": True,
            "log_level": "DEBUG",
        }
    }