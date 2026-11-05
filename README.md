# TRIDENT

**TRIDENT: Post-Selection Evidence Accountability for Token-Efficient RAG**  
**Accepted at NeurIPS 2026** · [Paper](https://openreview.net/forum?id=2UTqijjMAy)

TRIDENT addresses evidence selection in retrieval-augmented generation (RAG) under token budgets. The paper develops two complementary methods: one for efficient selection and one for controlling unsupported evidence-coverage claims.

## Implementations

| Method | Purpose | Code |
| --- | --- | --- |
| **Pareto-Knapsack** | Select evidence under a token budget, balancing utility and cost. | [`pareto/`](pareto/) |
| **Safe-Cover** | Assess support for required evidence facets using calibrated certificates, with abstention when support cannot be established. | [`safe_cover/`](safe_cover/) |

These are separate implementations with different pipelines and configurations. Both provide a Python package named `trident`, so install them in **separate environments**.

## Installation

Choose one method:

```bash
python -m venv .venv
source .venv/bin/activate
cd pareto  # or safe_cover
python -m pip install -e '.[retrieval,monitoring]'
```

## Configuration

Each package includes the configurations used for the final release:

```python
from trident.presets import get_preset

config = get_preset()
```

- **Pareto-Knapsack:** `pareto_match_1500_alpha06` (default), plus lower-budget presets.
- **Safe-Cover:** `safe_cover_equal_2000` (default) or `safe_cover_loose_4000`.

See the README in each method's directory for details.

## Notes

This repository contains the framework code and tests. Datasets, evaluation runners, baseline implementations, model weights, calibration artifacts, and experimental outputs are not included. Release details are in [`RELEASE.md`](RELEASE.md).

A Safe-Cover certificate concerns the support of evidence facets, **not a guarantee that the generated answer is correct**. Changes to its retrieval, verification, or calibration procedure may invalidate an existing calibration and require recalibration.

## Citation

If you use TRIDENT in your research, please cite:

**TRIDENT: Post-Selection Evidence Accountability for Token-Efficient RAG.**  
*Advances in Neural Information Processing Systems (NeurIPS), 2026.*  
[Paper on OpenReview](https://openreview.net/forum?id=2UTqijjMAy)

## License

[Apache License 2.0](LICENSE).
