# Berkay Hakan Sadi

- Student ID: 2309011065
- GitHub username: `hakanstarlord`

## Responsibility area

Data pipeline and split, reference cleaning, the scoring and evaluation code,
the GMM-HMM baseline, the pretrained baselines, and the group's own model
(fine-tuning Whisper small). I also set up the repository and the task
issues for the team.

## Work done

Result figures below are copied from the generated tables in
[`results/README.md`](../results/README.md); that file is the reference.

- **16–18 November, data.** Chose Hungarian and, after finding that the
  Common Voice dataset page forbids re-hosting, moved the training data to
  VoxPopuli Hungarian (CC0). Wrote the download and subset code. Found that
  the official VoxPopuli split is not speaker-disjoint, and rebuilt the
  split so that no speaker and no recording session is in two splits
  (train 14.0 h, validation 2.5 h, test 2.4 h). Audio is stored as FLAC
  shards with a manifest and checksums.
  Files: `src/huasr/download.py`, `subset.py`, `audio.py`, `shards.py`,
  `data/raw/voxpopuli/`.
- **17–18 November, cleaning.** Found two defects in the shipped reference
  text: every "í" was missing from `normalized_text`, and digits had been
  written as English number words inside Hungarian sentences. Restored the
  missing letter by aligning with `raw_text` and logged every change;
  flagged the clips with English number words so that scores are reported
  with and without them. Clips without a speaker id are removed and logged.
  Files: `data/clean/voxpopuli/` (manifest, repair log, removal log).
- **18 November, scoring and evaluation.** Wrote the text normalisation, WER
  and CER with speaker-level bootstrap intervals, the paired comparison of
  two systems, the runner that writes a run record and a prediction file
  for every system, and the script that generates the result tables. The
  runner refuses the test split without `--final-test`.
  Files: `src/huasr/text.py`, `metrics.py`, `evaluate.py`, `score.py`,
  `compare.py`, `report.py`, `systems.py`, `tests/`.
- **18 and 22 November, Whisper baselines.** Ran Whisper base and small
  (faster-whisper) on validation and, once, on test.
  Files: `results/baselines/faster-whisper-base/`, `faster-whisper-small/`.
- **20–21 November, GMM-HMM baseline.** Trained an MFCC + GMM-HMM system on
  our training split with Montreal Forced Aligner, a graphemic lexicon built
  from the training words and a trigram language model from the training
  transcripts. It is very weak (test WER 87.9%); about 16% of the test words
  are outside its vocabulary, and the cause of the remaining gap was not
  investigated.
  Files: `src/huasr/gmm_hmm.py`, `environment-gmm-hmm.yml`,
  `results/baselines/gmm-hmm-mfa/`.
- **21–22 November, our own model.** Fine-tuned `openai/whisper-small` on
  the training split on Google Colab (NVIDIA L4, 2,000 updates, checkpoint
  chosen on validation). On the test split the word error rate went from
  29.6% (the same checkpoint without our training) to 20.7%; the paired
  difference is -8.9 points with a 95% interval of -10.2 to -7.7.
  Files: `notebooks/finetune_whisper_small.ipynb`, `src/huasr/finetune.py`,
  `whisper_hf.py`, `results/own/hf-whisper-small-finetuned/`.
- **22 November, pretrained baselines.** Wrote the adapters and the Colab
  notebook for wav2vec 2.0 + CTC, Canary, Parakeet and Qwen3-ASR, and ran
  them on validation and test. Canary (9.1% test WER) is better than our
  fine-tuned model. It is a much larger model; its model card lists the
  Granary corpus as training data, and Granary has a Hungarian VoxPopuli
  subset. Whether it includes the recordings of our test split has not been
  checked.
  Files: `notebooks/baselines_pretrained.ipynb`, `src/huasr/pretrained.py`,
  `results/baselines/`.
- **20–22 November, coordination.** Prepared the repository, the team task
  list and the GitHub issues #1–#11.

## Commits

- [986f5e9](https://github.com/kaankalkan03/NLP-project/commit/986f5e9) data pipeline, scoring and evaluation code with tests
- [86bbc8e](https://github.com/kaankalkan03/NLP-project/commit/86bbc8e) cleaned references with repair and removal logs
- [21ab520](https://github.com/kaankalkan03/NLP-project/commit/21ab520), [bc5876d](https://github.com/kaankalkan03/NLP-project/commit/bc5876d), [9f38f22](https://github.com/kaankalkan03/NLP-project/commit/9f38f22) raw audio shards and manifests
- [80de01c](https://github.com/kaankalkan03/NLP-project/commit/80de01c) Whisper base and small on validation
- [b7cf3d7](https://github.com/kaankalkan03/NLP-project/commit/b7cf3d7), [27c1edf](https://github.com/kaankalkan03/NLP-project/commit/27c1edf) fine-tuning code and notebook
- [3d3d166](https://github.com/kaankalkan03/NLP-project/commit/3d3d166) GMM-HMM baseline
- [fe089cb](https://github.com/kaankalkan03/NLP-project/commit/fe089cb) fine-tuned Whisper small, validation results
- [b6e84b5](https://github.com/kaankalkan03/NLP-project/commit/b6e84b5) final test results
- [7bca06a](https://github.com/kaankalkan03/NLP-project/commit/7bca06a) adapters and notebook for the pretrained baselines
- [06116f7](https://github.com/kaankalkan03/NLP-project/commit/06116f7), [abf91f5](https://github.com/kaankalkan03/NLP-project/commit/abf91f5) pretrained baseline results
