## OOV / lexical coverage analysis

The reproducible analysis is implemented in `scripts/oov_analysis.py`. It derives the training vocabulary from clean VoxPopuli train references, measures exact per-clip IV/OOV reference-token occurrence coverage for every available test prediction file, and recomputes WER/CER with the project's scoring functions.

Run from the repository root with `python scripts/oov_analysis.py`; the generated report is `results/oov_analysis.md`.

Coverage is descriptive exact-token overlap, not a causal estimate of the effect of OOV status.