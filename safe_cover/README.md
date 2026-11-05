# Safe-Cover

The Safe-Cover implementation of [TRIDENT](../README.md), accepted at **NeurIPS 2026**.

This package evaluates evidence support for required facets using calibrated certificates. It includes the multi-hop and abstention logic used in the paper.

## Install

From this directory, in a dedicated Python environment:

```bash
python -m pip install -e '.[retrieval,monitoring]'
```

## Configuration

```python
from trident.presets import get_preset

config = get_preset("safe_cover_equal_2000")
```

The higher-budget `safe_cover_loose_4000` preset is also available. The reference environment used Python 3.10, NumPy 2.2.6, PyTorch 2.11.0, Transformers 5.10.2, and Sentence Transformers 5.3.0.

A certificate addresses evidence-facet coverage, not answer correctness. Changing the calibration or verification procedure requires recalibration. Model weights and calibration artifacts are not included.

Install this package separately from [Pareto-Knapsack](../pareto/), which uses the same Python import name.
