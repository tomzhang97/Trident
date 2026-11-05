# Pareto-Knapsack

The Pareto-Knapsack implementation of [TRIDENT](../README.md), accepted at **NeurIPS 2026**.

This package selects evidence under a token budget, balancing estimated utility and cost. It preserves the Pareto pipeline used in the paper.

## Install

From this directory, in a dedicated Python environment:

```bash
python -m pip install -e '.[retrieval,monitoring]'
```

## Configuration

```python
from trident.presets import get_preset

config = get_preset("pareto_match_1500_alpha06")
```

The package also provides lower-budget presets. The reference environment used Python 3.10, NumPy 1.26.4, PyTorch 2.9.0, Transformers 4.57.3, spaCy 3.8.11, and Sentence Transformers 5.1.2.

Model weights and experimental data are not included. Install this package separately from [Safe-Cover](../safe_cover/), which uses the same Python import name.
