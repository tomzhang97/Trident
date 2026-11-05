"""TRIDENT package exposing risk-controlled retrieval pipelines."""

from .config import (
    FacetConfig,
    SafeCoverConfig,
    ParetoConfig,
    CalibrationConfig,
)
from .facets import Facet
from .candidates import Passage
from .pipeline import TridentPipeline

__all__ = [
    "CalibrationConfig",
    "Facet",
    "FacetConfig",
    "ParetoConfig",
    "Passage",
    "SafeCoverConfig",
    "TridentPipeline",
]
