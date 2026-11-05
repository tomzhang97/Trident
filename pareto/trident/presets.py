"""Validated runtime presets for the final Pareto implementation."""

from __future__ import annotations

from .config import ParetoConfig

_PRESETS = {
    "pareto_match_1500_alpha06": {
        "budget": 1000,
        "max_evidence_tokens": 1000,
        "max_units": 8,
        "stop_on_budget": True,
        "relaxed_alpha": 0.6,
        "weight_default": 1.0,
        "use_vqc": False,
        "use_bwk": False,
        "max_vqc_iterations": 0,
        "bwk_exploration_bonus": 0.1,
    },
    "pareto_match_500_alpha06": {
        "budget": 500,
        "max_evidence_tokens": 500,
        "max_units": 5,
        "stop_on_budget": True,
        "relaxed_alpha": 0.6,
        "weight_default": 1.0,
        "use_vqc": False,
        "use_bwk": False,
        "max_vqc_iterations": 0,
        "bwk_exploration_bonus": 0.1,
    },
    "pareto_match_400": {
        "budget": 400,
        "max_evidence_tokens": 400,
        "max_units": 4,
        "stop_on_budget": True,
        "relaxed_alpha": 0.5,
        "weight_default": 1.0,
        "use_vqc": True,
        "use_bwk": True,
        "max_vqc_iterations": 0,
        "bwk_exploration_bonus": 0.1,
    },
}


def available_presets() -> tuple[str, ...]:
    """Return the names of the validated Pareto presets."""
    return tuple(_PRESETS)


def get_preset(name: str = "pareto_match_1500_alpha06") -> ParetoConfig:
    """Return a fresh configuration for a validated Pareto setup."""
    try:
        values = _PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(available_presets())
        raise ValueError(
            f"Unknown Pareto preset {name!r}; choose one of: {choices}"
        ) from exc
    return ParetoConfig(**values)
