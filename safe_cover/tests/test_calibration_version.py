"""Tests for calibration version metadata handling."""

import importlib.util

import pytest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
pytestmark = pytest.mark.skipif(
    not HAS_NUMPY, reason="numpy is required for calibrator imports"
)

if HAS_NUMPY:
    from trident.calibration import ReliabilityCalibrator
else:  # pragma: no cover - placeholders for type checking
    ReliabilityCalibrator = None  # type: ignore


def test_calibrator_load_preserves_version(tmp_path):
    """Loading a calibrator should restore its version metadata."""
    path = tmp_path / "calibrator.json"

    calibrator = ReliabilityCalibrator(use_mondrian=False, version="v-test")
    calibrator.save(path)

    loaded = ReliabilityCalibrator.load(path)

    assert loaded.version == "v-test"
