# Contributing

Contributions are welcome through focused pull requests. Keep the Pareto and
Safe-Cover implementations isolated unless a change has been independently
validated for both.

1. Choose `pareto/` or `safe_cover/` and create a dedicated virtual
   environment.
2. Install `python -m pip install -e '.[dev]'` from that directory.
3. Add or update tests for behavior changes.
4. Run `python -m pytest` before opening a pull request.

Do not add datasets, model weights, calibration pools, baseline systems,
credentials, logs, or generated experiment results. Changes to retrieval,
shortlisting, verification, calibration bins, serialization, prompts, or
tie-breaking must state whether recalibration or performance revalidation is
required.

Use Conventional Commit subjects such as
`fix(safe-cover): preserve certificate metadata`.
