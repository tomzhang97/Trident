"""Validated runtime presets for the final Safe-Cover implementation."""

from __future__ import annotations

from .config import SafeCoverConfig

_PRESETS = {
    "safe_cover_equal_2000": {
        "per_facet_alpha": 0.02,
        "max_evidence_tokens": 2000,
        "max_units": 10,
        "stop_on_budget": True,
        "abstain_on_infeasible": False,
        "fallback_to_pareto": True,
        "coverage_threshold": 0.15,
        "token_cap": 2000,
        "dual_tolerance": 1e-6,
        "early_abstain": True,
        "use_certificates": True,
        "monitor_drift": False,
        "psi_threshold": 0.5,
    },
    "safe_cover_loose_4000": {
        "per_facet_alpha": 0.05,
        "max_evidence_tokens": 4000,
        "max_units": 16,
        "stop_on_budget": True,
        "abstain_on_infeasible": False,
        "fallback_to_pareto": True,
        "coverage_threshold": 0.15,
        "token_cap": 4000,
        "dual_tolerance": 1e-6,
        "early_abstain": True,
        "use_certificates": True,
        "monitor_drift": False,
        "psi_threshold": 0.5,
    },
}


def available_presets() -> tuple[str, ...]:
    """Return the names of the validated Safe-Cover presets."""
    return tuple(_PRESETS)


def get_preset(name: str = "safe_cover_equal_2000") -> SafeCoverConfig:
    """Return a fresh configuration for a validated Safe-Cover setup."""
    try:
        values = _PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(available_presets())
        raise ValueError(
            f"Unknown Safe-Cover preset {name!r}; choose one of: {choices}"
        ) from exc
    return SafeCoverConfig(**values)
