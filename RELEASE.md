# Release notes

This repository contains the implementations accompanying **TRIDENT: Post-Selection Evidence Accountability for Token-Efficient RAG**, accepted at **NeurIPS 2026** ([paper](https://openreview.net/forum?id=2UTqijjMAy)).

## Code versions

| Package | Source snapshot | Scope |
| --- | --- | --- |
| [`pareto/`](pareto/) | `trident-main1216`, including final prompt changes | Pareto-Knapsack selection |
| [`safe_cover/`](safe_cover/) | `trident-safecover-fix` | Calibrated evidence coverage and abstention |

The packages remain separate because they use different retrieval, facet extraction, verification, prompting, and selection procedures. Merging their execution paths would change the behavior of the evaluated implementations.

## Included

Both packages contain their framework code, supported presets, and tests. The repository excludes benchmark runners, datasets, baselines, pretrained weights, calibration data, experiment logs, and paper build artifacts.

## Compatibility and calibration

- Dense-index serialization uses NumPy archives with `allow_pickle=False`. Indexes saved in the previous pickle format must be rebuilt.
- Safe-Cover certificates are tied to the retrieval, shortlisting, verification, serialization, tie-breaking, and calibration procedure used to derive them. Changes to that procedure require recalibration.
- Each package uses the import name `trident`; install them in separate environments.

## Verification

The release includes tests covering Pareto selection and budget behavior, Safe-Cover certification and abstention, calibration metadata, and retrieval serialization. The GitHub Actions workflow runs the test suite for each package independently.
