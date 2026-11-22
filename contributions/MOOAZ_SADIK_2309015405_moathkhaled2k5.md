# MOOAZ SADIK

- Student ID: 2309015405
- GitHub username: `moathkhaled2k5`

## Responsibility area

Issue #7: the justification of the evaluation metrics in the README, and a
check that the README "Reproduce" commands work in a fresh clone.

## Work done

- **22 November 2026, metric justification.** Wrote the paragraph "Why these
  metrics" in the README section "Metrics and why they fit Hungarian", citing
  chapter 5 "Evaluation" of the Hugging Face Audio Course. It explains that
  WER counts word substitutions, deletions and insertions, that one wrong
  Hungarian suffix makes a whole word wrong, that CER shows these partial
  errors, and that the same text normalisation (`src/huasr/text.py`) is
  applied to every system. Files: `README.md`.
- **22 November 2026, fresh-clone reproducibility check.** Cloned the
  repository into a new folder on Windows 11 with Python 3.14.0 and ran every
  "Reproduce" step that needs no GPU: `pip install -r requirements.txt`,
  `pip install -e .`, `python -m unittest` (383 tests, 36 skipped, no
  failures), `python -m huasr.report --check`, `python scripts/oov_analysis.py`
  (output identical to the committed `results/oov_analysis.md`) and
  `python -m huasr.subset unpack --raw data/raw/voxpopuli`. All worked.
  In a deeply nested folder the clone failed to check out one file under
  `results/` because its path is longer than 260 characters, and the tests
  then failed. I added a note on this to the "Reproduce" section, together
  with the `report --check` and `oov_analysis.py` commands, which were
  missing from it. The results are reported in issue #7. Files: `README.md`.

## Commits

- [896e66f](https://github.com/kaankalkan03/NLP-project/commit/896e66f) "Why these metrics" paragraph in the README
- [8040260](https://github.com/kaankalkan03/NLP-project/commit/8040260) fixes to the README "Reproduce" section after the fresh-clone check
- This contribution record
- [Issue #7](https://github.com/kaankalkan03/NLP-project/issues/7)
