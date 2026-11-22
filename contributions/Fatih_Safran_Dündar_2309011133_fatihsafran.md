# Fatih Safran Dündar

- Student ID: 2309011133
- GitHub username: `fatihsafran`

## Responsibility area

Issue #9: Analyze the errors in the saved Hungarian speech recognition predictions.

## Work done

- 2026-11-22: Reviewed the AI-assisted analysis draft in `notebooks/error_analysis.ipynb` and `scripts/error_analysis.py`. The analysis uses saved predictions and keeps validation and test results separate.
- 2026-11-22: Reviewed the results in `results/error_analysis.md`, which is linked from `README.md`. What stood out to me was that fine-tuning helped long words more, but they were still harder to transcribe. On the test set, the error rate for words with at least 12 letters fell from 49.46% to 27.86%. For shorter words, it fell from 22.70% to 16.16%.
- 2026-11-22: Looked at which errors remained after fine-tuning. The number of possible suffix errors fell from 880 to 662, but 363 reference word positions were flagged in both models. I would treat these as estimates because the rule compares spelling rather than identifying grammatical suffixes.

## Commits

- [Analysis notebook and code](https://github.com/kaankalkan03/NLP-project/commit/5156b2df1505a27ca9246c092daa148b37d749fd)
- [Generated report and README](https://github.com/kaankalkan03/NLP-project/commit/fdc00a5db3a06972a6481c84c3c29daf7bd9081c)

