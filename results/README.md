# Results

What every run leaves behind, how it is scored, and the result tables. The
tables are written by `python -m huasr.report` from the files of this folder;
no number in them is typed by hand.

**Both splits have been scored.** Every system was run on the whole
validation split and, once, on the whole test split. The test split was kept
for the final evaluation, after every setting was frozen:
`python -m huasr.evaluate` refuses `--split test` unless `--final-test` is
given, and the run manifest records that the flag was used. The test runs
were made on 22 November 2026. The validation figures are development
figures; the test figures are the results of the project. The main README
shows the test split alone, sorted by WER, with the discussion
([Results and discussion](../README.md#results-and-discussion)); this file
has both splits and the notes on each system.

## Layout

```text
results/
  README.md                      this file
  baselines/<system>/<split>/    baselines: checkpoints used as they were published, and the GMM-HMM
                                 system (gmm-hmm-mfa), trained on our training split with a standard recipe
  own/<system>/<split>/          models trained in this project
    run.json                     written by huasr.evaluate (for gmm-hmm-mfa: by huasr.gmm_hmm)
    predictions.csv              written by huasr.evaluate (for gmm-hmm-mfa: by huasr.gmm_hmm)
    scores.json                  written by huasr.score from the two files above
    comparison_with_<baseline>.json   only for our own models: written by huasr.compare from the
                                      prediction files of this run and of the baseline's run
  own/hf-whisper-small-finetuned/training.json    written by huasr.finetune: the record of the training run
```

`<split>` is `validation` or `test`. One folder holds one run of one system
on one split. A system belongs to the group it declares (`group` in
`src/huasr/systems.py`), so a baseline can never appear among our own
models.

| File | Content |
|---|---|
| `run.json` | system and group; checkpoint (repository id, revision, the checkpoint it was converted from, size and SHA-256 of the weight file); runtime and decoding settings; seed; split, number of clips and whether the whole split was run; whether the run was made with `--final-test`; SHA-256 of the raw and of the clean manifest; audio and processing time, loading time; processor, operating system; package versions; start and end time (UTC) |
| `predictions.csv` | one row per clip, in the order of the clean manifest: `clip_id`, `speaker_id`, `session_id`, `duration_ms`, `english_number_words` (the flag of the clean manifest), `reference`, `hypothesis` (exactly what the system wrote, before any normalisation), `processing_seconds` |
| `scores.json` | corpus-level WER, CER and diacritic-insensitive WER, each with its 95% interval and the numbers of substitutions, deletions, insertions and reference units; real-time factor; clips with an empty hypothesis; all of it for all clips and again without the clips flagged for English number words |
| `comparison_with_<baseline>.json` | the two systems and their run folders; for WER, CER and diacritic-insensitive WER the rate of both, the difference, its 95% interval and a one-sided p-value from the paired bootstrap over speakers; all of it for all clips and again without the clips flagged for English number words |
| `training.json` | the fine-tuning run: clips used and left out, settings, seed, loss and validation scores of every evaluated step, the kept step, SHA-256 of the kept weights, time, GPU and package versions |

## Runs in this folder

Every run folder, with the start of the run, the machine and, for the test
split, whether the run was made as a final evaluation:

<!-- generated: results-runs -->
| System | Group | Split | Started (UTC) | Clips | Whole split | Made with `--final-test` | Scored | Run on |
|---|---|---|---|---|---|---|---|---|
| canary-1b-v2 | baselines | validation | 2026-11-22 06:25 | 826 | yes | - | yes | GPU: NVIDIA L4 |
| faster-whisper-base | baselines | validation | 2026-11-18 09:17 | 826 | yes | - | yes | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 4 threads |
| faster-whisper-small | baselines | validation | 2026-11-18 09:47 | 826 | yes | - | yes | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 4 threads |
| gmm-hmm-mfa | baselines | validation | 2026-11-21 11:08 | 826 | yes | - | yes | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 8 parallel jobs |
| hf-whisper-small | baselines | validation | 2026-11-22 02:15 | 826 | yes | - | yes | GPU: NVIDIA L4 |
| parakeet-tdt-0.6b-v3 | baselines | validation | 2026-11-22 06:41 | 826 | yes | - | yes | GPU: NVIDIA L4 |
| qwen3-asr-1.7b | baselines | validation | 2026-11-22 06:48 | 826 | yes | - | yes | GPU: NVIDIA L4 |
| wav2vec2-xlsr-53-hungarian | baselines | validation | 2026-11-22 06:21 | 826 | yes | - | yes | GPU: NVIDIA L4 |
| hf-whisper-small-finetuned | own | validation | 2026-11-22 03:42 | 826 | yes | - | yes | GPU: NVIDIA L4 |
| canary-1b-v2 | baselines | test | 2026-11-22 06:33 | 897 | yes | yes | yes | GPU: NVIDIA L4 |
| faster-whisper-base | baselines | test | 2026-11-22 04:11 | 897 | yes | yes | yes | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 4 threads |
| faster-whisper-small | baselines | test | 2026-11-22 04:41 | 897 | yes | yes | yes | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 4 threads |
| gmm-hmm-mfa | baselines | test | 2026-11-22 04:56 | 897 | yes | yes | yes | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 8 parallel jobs |
| hf-whisper-small | baselines | test | 2026-11-22 04:12 | 897 | yes | yes | yes | GPU: NVIDIA L4 |
| parakeet-tdt-0.6b-v3 | baselines | test | 2026-11-22 06:44 | 897 | yes | yes | yes | GPU: NVIDIA L4 |
| qwen3-asr-1.7b | baselines | test | 2026-11-22 07:32 | 897 | yes | yes | yes | GPU: NVIDIA L4 |
| wav2vec2-xlsr-53-hungarian | baselines | test | 2026-11-22 06:22 | 897 | yes | yes | yes | GPU: NVIDIA L4 |
| hf-whisper-small-finetuned | own | test | 2026-11-22 04:33 | 897 | yes | yes | yes | GPU: NVIDIA L4 |

Source: `results/<group>/<system>/<split>/run.json`.
<!-- end generated -->

## How a run is made and scored

```bash
python -m huasr.evaluate --system faster-whisper-base     # validation split by default
python -m huasr.evaluate --system faster-whisper-small
python -m huasr.evaluate --system faster-whisper-small --split test --final-test    # the final evaluation, once
python -m huasr.score --all                               # scores.json for every run
python -m huasr.report                                    # the tables below
```

The commands of the other systems are in the main README: the GMM-HMM
baseline under [Reproduce](../README.md#reproduce), the Whisper small systems
of the Hugging Face implementation and the fine-tuning under
[Fine-tuning Whisper small](../README.md#fine-tuning-whisper-small-our-own-model),
the four pretrained baselines under
[How to run the pretrained baselines](../README.md#how-to-run-the-pretrained-baselines).

- **Same input for every system.** The runner reads the clips of the split
  from the shards of `data/raw/voxpopuli/`, checks each one against the
  SHA-256 of the manifest, and hands the system the decoded samples, one clip
  at a time. The reference is the column `reference` of
  `data/clean/voxpopuli/manifest.csv`. A system never sees a reference. The
  GMM-HMM system gets the same clips, checked in the same way, as a folder,
  because its toolkit decodes a folder at once.
- **Scoring is separate from running.** `huasr.score` reads only the
  prediction file and the run manifest, so the scoring can be changed without
  running a model again, and every system is scored by the same code.
- **Rates** are corpus-level, after the scoring policy of the main README has
  been applied to both sides.
- **Intervals** are 95% percentile intervals from a bootstrap that resamples
  speakers (`bootstrap_interval` in `src/huasr/metrics.py`; number of
  resamples and seed are recorded in `scores.json`). Both splits have few
  speakers, the validation split fewer than the test split
  ([`data/clean/README.md`](../data/clean/README.md)), so the intervals are
  wide, and wider on the validation split.
- **WER without diacritics** applies the same policy with the accents removed
  on both sides. The distance to the WER is the share of words that are wrong
  only by an accent.
- **RTF** (real-time factor) is the processing time divided by the audio
  duration, summed over the clips. Loading the model is not included; clips
  are processed one at a time (the GMM-HMM system is the exception, see
  below). It depends on the machine and on what else the machine is doing.
  The runs were made on different machines, which the table of runs above
  names: **RTF values can only be compared between runs on the same
  machine.**
- **English number words.** The corpus wrote digits out in English in some
  references (see [`data/clean/README.md`](../data/clean/README.md)), which no
  system can get right. The last three columns repeat the figures without
  these clips. The flag only splits the report: every clip stays in the
  prediction file.
- A run on a part of a split (`--limit N`) is marked as incomplete and is not
  shown in the result tables; the table of runs above would list it with
  "no" under "Whole split". No such run is committed.

## Baselines

Seven of the baselines are checkpoints used as published, with no training
on our data: Whisper base and small through faster-whisper, Whisper small
through `transformers` (`hf-whisper-small`, the checkpoint our own model
starts from), a wav2vec 2.0 encoder with a CTC head, Canary, Parakeet TDT
and Qwen3-ASR. `gmm-hmm-mfa` has no published checkpoint: it is a standard
MFCC + GMM-HMM recipe trained on our training split (see
[GMM-HMM baseline](#gmm-hmm-baseline) below). The table has the validation
rows first, then the test rows, each in the order of the system names; the
main README has the test rows sorted by WER.

<!-- generated: results-baselines -->
| System | Checkpoint and revision | Split | Clips | WER % [95%] | CER % [95%] | WER % without diacritics | RTF | Clips without English number words | their WER % | their CER % |
|---|---|---|---|---|---|---|---|---|---|---|
| canary-1b-v2 | `nvidia/canary-1b-v2` at `d455706339a6` | validation | 826 | 10.59 [7.89, 14.77] | 4.96 [2.82, 8.55] | 10.47 | 0.051 | 771 | 8.40 | 3.32 |
| faster-whisper-base | `Systran/faster-whisper-base` at `ebe41f70d5b6` | validation | 826 | 54.41 [49.95, 59.16] | 16.52 [13.59, 20.16] | 52.99 | 0.196 | 771 | 52.66 | 15.05 |
| faster-whisper-small | `Systran/faster-whisper-small` at `536b0662742c` | validation | 826 | 31.55 [28.01, 36.13] | 10.12 [7.64, 13.82] | 30.64 | 0.259 | 771 | 29.40 | 8.48 |
| gmm-hmm-mfa | trained here on the training split; no published checkpoint | validation | 826 | 86.85 [86.02, 87.68] | 71.97 [70.70, 73.18] | 86.84 | 0.019 | 771 | 86.54 | 71.64 |
| hf-whisper-small | `openai/whisper-small` at `973afd24965f` | validation | 826 | 33.65 [27.78, 41.32] | 12.04 [7.36, 19.09] | 32.73 | 0.146 | 771 | 31.04 | 9.78 |
| parakeet-tdt-0.6b-v3 | `nvidia/parakeet-tdt-0.6b-v3` at `541d1f99c6b0` | validation | 826 | 19.39 [10.61, 32.87] | 9.63 [3.36, 18.85] | 19.23 | 0.019 | 771 | 17.79 | 8.44 |
| qwen3-asr-1.7b | `Qwen/Qwen3-ASR-1.7B-hf` at `bcd2b5b7f32b` | validation | 826 | 32.09 [24.96, 42.17] | 14.39 [8.66, 22.81] | 31.33 | 0.288 | 771 | 30.10 | 12.73 |
| wav2vec2-xlsr-53-hungarian | `jonatasgrosman/wav2vec2-large-xlsr-53-hungarian` at `2bd0786e71ca` | validation | 826 | 49.40 [46.35, 52.76] | 13.38 [11.04, 16.90] | 47.55 | 0.006 | 771 | 48.01 | 11.85 |
| canary-1b-v2 | `nvidia/canary-1b-v2` at `d455706339a6` | test | 897 | 9.12 [8.16, 10.10] | 3.77 [3.04, 4.59] | 8.97 | 0.051 | 862 | 8.47 | 3.29 |
| faster-whisper-base | `Systran/faster-whisper-base` at `ebe41f70d5b6` | test | 897 | 52.05 [50.30, 53.98] | 15.03 [14.00, 16.11] | 50.47 | 0.209 | 862 | 51.66 | 14.58 |
| faster-whisper-small | `Systran/faster-whisper-small` at `536b0662742c` | test | 897 | 29.43 [27.70, 31.31] | 8.53 [7.52, 9.65] | 28.54 | 0.279 | 862 | 28.91 | 7.99 |
| gmm-hmm-mfa | trained here on the training split; no published checkpoint | test | 897 | 87.88 [87.10, 88.61] | 71.53 [70.77, 72.20] | 87.85 | 0.021 | 862 | 87.73 | 71.41 |
| hf-whisper-small | `openai/whisper-small` at `973afd24965f` | test | 897 | 29.60 [27.88, 31.63] | 9.14 [7.68, 11.06] | 28.63 | 0.141 | 862 | 29.11 | 8.70 |
| parakeet-tdt-0.6b-v3 | `nvidia/parakeet-tdt-0.6b-v3` at `541d1f99c6b0` | test | 897 | 21.90 [10.65, 45.98] | 9.74 [3.74, 21.21] | 21.65 | 0.020 | 862 | 21.92 | 9.62 |
| qwen3-asr-1.7b | `Qwen/Qwen3-ASR-1.7B-hf` at `bcd2b5b7f32b` | test | 897 | 28.02 [24.54, 32.91] | 12.42 [8.91, 17.25] | 27.27 | 0.286 | 862 | 25.66 | 10.37 |
| wav2vec2-xlsr-53-hungarian | `jonatasgrosman/wav2vec2-large-xlsr-53-hungarian` at `2bd0786e71ca` | test | 897 | 46.87 [44.22, 49.50] | 11.70 [10.61, 12.85] | 45.16 | 0.006 | 862 | 46.42 | 11.29 |

Source: `results/baselines/<system>/<split>/scores.json`.
<!-- end generated -->

The table gives, for every published checkpoint, the repository and the
first characters of the revision that was loaded from the local cache;
`run.json` has the full revision and the SHA-256 of the weight file.

### Whisper base and small through faster-whisper

`faster-whisper-base` and `faster-whisper-small` are run with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), which loads
CTranslate2 conversions of the OpenAI checkpoints.

| System | Loaded repository | Converted from (stated on its model card) |
|---|---|---|
| `faster-whisper-base` | `Systran/faster-whisper-base` | `openai/whisper-base` |
| `faster-whisper-small` | `Systran/faster-whisper-small` | `openai/whisper-small` |

Settings, declared in `src/huasr/systems.py` and copied into every
`run.json`: CPU, weights quantised to 8-bit integers (`int8`), a fixed number
of threads, language `hu`, task `transcribe`, beam size 5, temperature 0
without fallback to higher temperatures, no conditioning on previous text,
everything else as faster-whisper sets it. What this means for reading the
table:

- The model cards name the OpenAI checkpoint but not its revision, and the
  weights are quantised here. The figures are those of this conversion with
  these settings; we have not compared them with the original implementation.
- The number of threads is fixed because the text depends on it: on a sample
  of clips the output was identical when the same number was used twice, and
  differed slightly between numbers.
- With the temperature fixed at 0 the decoder cannot retry a clip, so a
  repetition loop stays in the hypothesis and counts as insertions. The
  prediction files show such clips for both models.
- **The RTF column is a rough figure only.** The runs of these two systems
  were made on a laptop whose load was not controlled: the operating system
  moved the threads between fast and slow processor cores during the runs,
  and a short trial of the same system on the same clips ran much faster
  than the full run. Their RTF values should not be compared with each
  other, and not with the runs on the GPU. Speed was not measured again
  under a controlled load, so no RTF of this table is a result on speed.
- Whisper's training data is not published (audio with transcripts collected
  from the web; Radford et al., 2023). Whether it contains recordings of the
  European Parliament cannot be checked by us, so "zero-shot" means only that
  we did not train the model.

### Pretrained baselines run through `transformers`

`hf-whisper-small`, `wav2vec2-xlsr-53-hungarian`, `canary-1b-v2`,
`parakeet-tdt-0.6b-v3` and `qwen3-asr-1.7b` were run on a GPU, which their
`run.json` names. The main README describes how each is decoded, why
Qwen3-ASR is loaded from the repository `Qwen/Qwen3-ASR-1.7B-hf`, and what
each model card says about the training data
([Pretrained baselines](../README.md#pretrained-baselines-of-other-model-families)).
What limits the reading of their rows is under
[Discussion](../README.md#discussion) there. Two things show in the table
itself:

- The WER interval of `parakeet-tdt-0.6b-v3` is the widest of all systems on
  both splits, by far on the test split. The intervals resample speakers,
  and on the test split its word errors are spread very unevenly over the
  speakers.
- Some systems wrote nothing for a few clips, `parakeet-tdt-0.6b-v3` most
  often. `scores.json` counts such clips as `empty_hypotheses`, and every
  word of their reference is a deletion.

### GMM-HMM baseline

`gmm-hmm-mfa` is trained and run by `src/huasr/gmm_hmm.py` with the Montreal
Forced Aligner (Kaldi): MFCC features without pitch; monophone, triphone, LDA
and speaker-adapted training; a graphemic lexicon of the training words; a
trigram language model of the training transcripts; `mfa transcribe` with its
default settings. The main README describes the system and its limits. Its
`run.json` has the same fields as the other runs and, in addition, the
versions and builds of MFA and Kaldi, the training recipe, the training time,
the words of the references that are not in the lexicon (`out_of_vocabulary`)
and what was done to the output of MFA (`output`).

<!-- generated: results-gmm-hmm -->
| Split | Training clips | Words in the lexicon | Reference words not in the lexicon | Clips with such a word | Word errors: substituted / deleted / inserted | Empty hypotheses | Unknown-word placeholders removed | Acoustic training (min) | Decoding (min) |
|---|---|---|---|---|---|---|---|---|---|
| validation | 4,597 (14.00 h, 20 speakers) | 20,021 | 2,933 of 18,145 (16.16%) | 754 of 826 | 7,654 / 8,042 / 63 | 2 of 826 | 0 in 0 clips | 73.2 | 2.9 |
| test | 4,597 (14.00 h, 20 speakers) | 20,021 | 2,836 of 17,406 (16.29%) | 806 of 897 | 7,987 / 7,193 / 116 | 0 of 897 | 0 in 0 clips | 73.2 | 3.1 |

Source: `results/baselines/gmm-hmm-mfa/<split>/run.json` and `scores.json`.
<!-- end generated -->

How to read this table and the row of `gmm-hmm-mfa` in the table above:

- **Reference words not in the lexicon** can never be written by this system,
  so their share is a lower bound for its WER. The other systems have no
  such bound.
- **Training clips** names the speakers the acoustic model has heard. Every
  speaker of the validation split and of the test split is new to it.
- **Word errors** shows how the WER is made up. A deletion is a spoken word
  for which the system wrote nothing. Deletions are about half of its word
  errors on both splits, far more than the words outside the lexicon
  account for. Why the decoder wrote nothing for so much speech has not been
  established.
- **Empty hypotheses** are clips for which nothing is left after the
  placeholders of MFA (such as `<unk>`) have been removed; every word of their
  reference counts as a deletion.
- **RTF** is not comparable with any other row. It is the wall time of
  `mfa transcribe` for the whole split (features, decoding graph, two decoding
  passes with speaker adaptation, rescoring) with several parallel jobs,
  divided by the audio duration. The number of jobs is in `run.json`.
- The system was given the speaker id of every clip (it adapts to the
  speaker); the other systems were not.
- Decoding used the default settings of MFA. Nothing was tuned on the
  validation split.
- The run was made on Windows 11 with the win-64 packages of conda-forge
  (versions and builds in `run.json`).

## Our own models

<!-- generated: results-own -->
| System | Checkpoint and revision | Split | Clips | WER % [95%] | CER % [95%] | WER % without diacritics | RTF | Clips without English number words | their WER % | their CER % |
|---|---|---|---|---|---|---|---|---|---|---|
| hf-whisper-small-finetuned | `openai/whisper-small` at `973afd24965f`, fine-tuned on the training split (step 1500 of 2000) | validation | 826 | 22.06 [17.89, 27.28] | 8.35 [5.11, 12.94] | 21.64 | 0.140 | 771 | 19.93 | 6.54 |
| hf-whisper-small-finetuned | `openai/whisper-small` at `973afd24965f`, fine-tuned on the training split (step 1500 of 2000) | test | 897 | 20.69 [19.00, 22.37] | 6.54 [5.49, 7.71] | 20.29 | 0.140 | 862 | 20.15 | 6.13 |

Source: `results/own/<system>/<split>/scores.json`.
<!-- end generated -->

### Whisper small, fine-tuned

Trained and run on 22 November 2026 on Google Colab; the GPU is named in the
table of runs above. `hf-whisper-small-finetuned` is `openai/whisper-small`
after `python -m huasr.finetune train`. The main README has the clips it is
trained on, the settings, the record of the training run and the commands
([Fine-tuning Whisper small](../README.md#fine-tuning-whisper-small-our-own-model)).
What its run leaves in this folder, and how it is to be read:

- **Before and after.** The checkpoint it starts from is run as the baseline
  `hf-whisper-small` (in the table of the baselines) with the same code and
  the same decoding: Hugging Face implementation, language `hu`, task
  `transcribe`, no timestamps, beam size 5, temperature 0. The distance
  between that row and the row of this model is the effect of the
  fine-tuning; the comparison table at the end of this section tests it.
  `faster-whisper-small` is the same published checkpoint in another
  implementation, with quantised weights and on the CPU.
- **`own/hf-whisper-small-finetuned/training.json`**, next to the split
  folders: the training record. Clips used and left out, settings, seed, loss
  and validation scores of every evaluated step, the kept step, SHA-256 of
  the kept weights, GPU and package versions. The validation scores in it are
  decoded greedily in batches, without the clips longer than 30 s, and only
  choose the checkpoint; the figures of the table come from the run folder.
- **`comparison_with_hf-whisper-small.json`** in the run folder, written by
  `python -m huasr.compare`: the difference between the two systems on the
  same clips for WER, CER and the WER without diacritics, each with a 95%
  interval and a one-sided p-value from the paired bootstrap over speakers.
- In the table, the column "Checkpoint and revision" names for this model the
  checkpoint it started from, its revision and the step that was kept;
  `run.json` has the SHA-256 of the weights that were evaluated.
- **The validation split chose the checkpoint**, so its figures for this
  model are development figures and may be optimistic. The test split is the
  unseen set.
- **RTF.** The fine-tuned model and `hf-whisper-small` ran on the same kind
  of GPU, named in their `run.json`, so their two RTF values can be compared
  with each other. They cannot be compared with the CPU runs.

The paired comparison of the fine-tuned model with `hf-whisper-small`, on
both splits (the main README explains the method under
[Comparing two systems](../README.md#comparing-two-systems) and discusses
the result on the test split):

<!-- generated: results-comparisons -->
| Our model | Baseline | Split | Clips | Metric | Baseline % | Our model % | Difference in points [95%] | p-value |
|---|---|---|---|---|---|---|---|---|
| hf-whisper-small-finetuned | hf-whisper-small | validation | all 826 (15 speakers) | WER | 33.65 | 22.06 | -11.58 [-15.60, -7.64] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | validation | all 826 (15 speakers) | CER | 12.04 | 8.35 | -3.69 [-7.22, -0.58] | 0.0063 |
| hf-whisper-small-finetuned | hf-whisper-small | validation | all 826 (15 speakers) | WER without diacritics | 32.73 | 21.64 | -11.09 [-15.21, -7.06] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | validation | 771 without English number words (15 speakers) | WER | 31.04 | 19.93 | -11.11 [-14.88, -7.41] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | validation | 771 without English number words (15 speakers) | CER | 9.78 | 6.54 | -3.24 [-6.35, -0.44] | 0.009 |
| hf-whisper-small-finetuned | hf-whisper-small | validation | 771 without English number words (15 speakers) | WER without diacritics | 30.10 | 19.46 | -10.64 [-14.51, -6.86] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | all 897 (34 speakers) | WER | 29.60 | 20.69 | -8.91 [-10.24, -7.70] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | all 897 (34 speakers) | CER | 9.14 | 6.54 | -2.60 [-4.30, -1.41] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | all 897 (34 speakers) | WER without diacritics | 28.63 | 20.29 | -8.34 [-9.68, -7.13] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | 862 without English number words (34 speakers) | WER | 29.11 | 20.15 | -8.96 [-10.37, -7.68] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | 862 without English number words (34 speakers) | CER | 8.70 | 6.13 | -2.57 [-4.35, -1.31] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | 862 without English number words (34 speakers) | WER without diacritics | 28.14 | 19.77 | -8.37 [-9.81, -7.09] | 0.0001 |

Difference: error rate of our model minus error rate of the baseline, in percentage points; a negative difference means fewer errors of our model. The interval is the 95% percentile interval of that difference and the p-value is one-sided (H0: our model is not better than the baseline); both come from a paired bootstrap that resamples speakers, with 10,000 resamples and seed 0.

Source: `results/own/<system>/<split>/comparison_with_<baseline>.json`.
<!-- end generated -->

The single-system intervals of the two models in the tables above overlap in
some rows where the paired interval excludes zero. The paired interval
resamples the same speakers for both systems, so it is the one to read for
the difference between two systems on the same clips.

## Reference

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C. and Sutskever, I.
(2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML
2023*. <https://proceedings.mlr.press/v202/radford23a.html>
