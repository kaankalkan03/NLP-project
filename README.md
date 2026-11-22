# Hungarian Automatic Speech Recognition on VoxPopuli

Transcribe Hungarian speech into Hungarian text, and measure how far
fine-tuning a small multilingual model on Hungarian European Parliament
speech (VoxPopuli) closes the gap to larger published models that are used
without further training, on speakers the model has never heard.

COE025 Natural Language Processing, Project 3 (automatic speech recognition).

> **Status (22 November 2026).** Data, baselines, our own model and the final
> evaluation are done; the benchmark is under
> [Results and discussion](#results-and-discussion).
>
> - **Data.** A VoxPopuli subset whose three splits share no speaker and no
>   recording session is in `data/raw/voxpopuli/`, and its repaired references
>   are in `data/clean/voxpopuli/`.
> - **Systems.** Every baseline listed under [Systems](#systems) and our own
>   model, Whisper small fine-tuned on our training split, have been run on
>   the validation split and, once, on the test split. The test runs were made
>   on 22 November 2026 with every setting frozen; `results/README.md` lists
>   every run with its date and machine.
> - **Not done.** The FLEURS test set is built and committed, but no system
>   has been scored on it. For Common Voice only the scripts exist: the
>   dataset has not been downloaded with them and no system has been scored
>   on it. No subword error rate is computed. HuBERT + CTC is not run, because
>   no Hungarian checkpoint was found. No cleaning rule removes clips.
>
> The section [Status](#status) has the full list. Every table of numbers in
> the READMEs is written by `python -m huasr.report` from files that the
> builders, the runs and the scorer generate; no such number is typed by hand.

## Task

Input: a short recording of Hungarian speech. Output: what was said, as
Hungarian text. Speech and reference transcript are in the same language;
there is no translation step.

Research question: how much of the gap between a small multilingual
checkpoint and larger published models is closed by fine-tuning the small one
on Hungarian parliament speech? Our splits share no speaker, so every figure
is measured on speakers the fine-tuned model has not heard. A second
question, whether the gain carries over to read speech recorded elsewhere, is
not answered: no system was scored on FLEURS or Common Voice.

Hungarian is agglutinative: one word carries a stem and several suffixes
(*házaimban*, "in my houses", is one word), and compounds are written
together. A single wrong suffix makes the whole word wrong, so the choice of
error unit matters more than in English. That is why this project reports
word and character error rates side by side, and the WER without diacritics
([Metrics](#metrics-and-why-they-fit-hungarian)).

## Language eligibility

- Language: **Hungarian (`hu`)**, registered for Group 8 in the course's
  dataset sheet.
- It is not English and not the mother tongue of any group member.
- No group member speaks or reads Hungarian.

Consequences for the method: nobody can judge a transcript by eye, so
references are never corrected by hand. A clip that looks wrong may only be
removed and logged, never relabelled, and the output of a model is never
used as a reference. The one repair of the references is rule-based and
proved by the corpus's own second transcript ([Data](#data)).

## Results and discussion

The benchmark of the project: every system on the **test split**, which was
scored once, with every setting frozen. All systems get the same clips and
the same references, and their output passes through the same scoring policy
and the same scorer ([Evaluation protocol](#evaluation-protocol)). Baselines
and our own model are in separate tables. The three blocks below are written
by `python -m huasr.report` from the result files. The same figures for the
validation split, the list of all runs with date and machine, and the notes
on each system are in [`results/README.md`](results/README.md).

### Baselines on the test split

Published checkpoints used without any training by us, and the GMM-HMM
system, which is a standard recipe trained on our training split
([Systems](#systems) says why it is a baseline and not our model).

<!-- generated: benchmark-baselines -->
| System | Checkpoint and revision | WER % [95%] | CER % [95%] | WER % without diacritics | Word errors: substituted / deleted / inserted | WER % / CER % without the clips with English number words | RTF | Run on |
|---|---|---|---|---|---|---|---|---|
| canary-1b-v2 | `nvidia/canary-1b-v2` at `d455706339a6` | 9.12 [8.16, 10.10] | 3.77 [3.04, 4.59] | 8.97 | 892 / 456 / 240 | 8.47 / 3.29 | 0.051 | GPU: NVIDIA L4 |
| parakeet-tdt-0.6b-v3 | `nvidia/parakeet-tdt-0.6b-v3` at `541d1f99c6b0` | 21.90 [10.65, 45.98] | 9.74 [3.74, 21.21] | 21.65 | 1,225 / 464 / 2,123 | 21.92 / 9.62 | 0.020 | GPU: NVIDIA L4 |
| qwen3-asr-1.7b | `Qwen/Qwen3-ASR-1.7B-hf` at `bcd2b5b7f32b` | 28.02 [24.54, 32.91] | 12.42 [8.91, 17.25] | 27.27 | 3,226 / 758 / 893 | 25.66 / 10.37 | 0.286 | GPU: NVIDIA L4 |
| faster-whisper-small | `Systran/faster-whisper-small` at `536b0662742c` | 29.43 [27.70, 31.31] | 8.53 [7.52, 9.65] | 28.54 | 3,655 / 846 / 621 | 28.91 / 7.99 | 0.279 | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 4 threads |
| hf-whisper-small | `openai/whisper-small` at `973afd24965f` | 29.60 [27.88, 31.63] | 9.14 [7.68, 11.06] | 28.63 | 3,653 / 820 / 680 | 29.11 / 8.70 | 0.141 | GPU: NVIDIA L4 |
| wav2vec2-xlsr-53-hungarian | `jonatasgrosman/wav2vec2-large-xlsr-53-hungarian` at `2bd0786e71ca` | 46.87 [44.22, 49.50] | 11.70 [10.61, 12.85] | 45.16 | 6,206 / 923 / 1,030 | 46.42 / 11.29 | 0.006 | GPU: NVIDIA L4 |
| faster-whisper-base | `Systran/faster-whisper-base` at `ebe41f70d5b6` | 52.05 [50.30, 53.98] | 15.03 [14.00, 16.11] | 50.47 | 6,965 / 952 / 1,143 | 51.66 / 14.58 | 0.209 | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 4 threads |
| gmm-hmm-mfa | trained here on the training split; no published checkpoint | 87.88 [87.10, 88.61] | 71.53 [70.77, 72.20] | 87.85 | 7,987 / 7,193 / 116 | 87.73 / 71.41 | 0.021 | CPU: 13th Gen Intel(R) Core(TM) i9-13900H, 8 parallel jobs |

Rows are sorted by WER, the lowest first. Every row: the same 897 clips of 34 speakers with 17,406 reference words, read from one clean manifest (SHA-256 `045cf81225a9`); clips with English number words in the reference: 35. Intervals: 95% percentile bootstrap over speakers, 10,000 resamples, seed 0. Runs made with `--final-test`: 8 of 8.

Source: `results/baselines/<system>/test/scores.json`.
<!-- end generated -->

### Our own model on the test split

`openai/whisper-small` fine-tuned on our training split
([Fine-tuning Whisper small](#fine-tuning-whisper-small-our-own-model)).

<!-- generated: benchmark-own -->
| System | Checkpoint and revision | WER % [95%] | CER % [95%] | WER % without diacritics | Word errors: substituted / deleted / inserted | WER % / CER % without the clips with English number words | RTF | Run on |
|---|---|---|---|---|---|---|---|---|
| hf-whisper-small-finetuned | `openai/whisper-small` at `973afd24965f`, fine-tuned on the training split (step 1500 of 2000) | 20.69 [19.00, 22.37] | 6.54 [5.49, 7.71] | 20.29 | 2,459 / 582 / 561 | 20.15 / 6.13 | 0.140 | GPU: NVIDIA L4 |

Rows are sorted by WER, the lowest first. Every row: the same 897 clips of 34 speakers with 17,406 reference words, read from one clean manifest (SHA-256 `045cf81225a9`); clips with English number words in the reference: 35. Intervals: 95% percentile bootstrap over speakers, 10,000 resamples, seed 0. Runs made with `--final-test`: 1 of 1.

Source: `results/own/<system>/test/scores.json`.
<!-- end generated -->

### Our model against the baseline it started from

`hf-whisper-small` is the checkpoint the fine-tuning starts from, run with
the same code and the same decoding as the fine-tuned model, so the
difference between the two is the effect of the fine-tuning. The method is
described under [Comparing two systems](#comparing-two-systems).

<!-- generated: benchmark-comparison -->
| Our model | Baseline | Split | Clips | Metric | Baseline % | Our model % | Difference in points [95%] | p-value |
|---|---|---|---|---|---|---|---|---|
| hf-whisper-small-finetuned | hf-whisper-small | test | all 897 (34 speakers) | WER | 29.60 | 20.69 | -8.91 [-10.24, -7.70] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | all 897 (34 speakers) | CER | 9.14 | 6.54 | -2.60 [-4.30, -1.41] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | all 897 (34 speakers) | WER without diacritics | 28.63 | 20.29 | -8.34 [-9.68, -7.13] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | 862 without English number words (34 speakers) | WER | 29.11 | 20.15 | -8.96 [-10.37, -7.68] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | 862 without English number words (34 speakers) | CER | 8.70 | 6.13 | -2.57 [-4.35, -1.31] | 0.0001 |
| hf-whisper-small-finetuned | hf-whisper-small | test | 862 without English number words (34 speakers) | WER without diacritics | 28.14 | 19.77 | -8.37 [-9.81, -7.09] | 0.0001 |

Difference: error rate of our model minus error rate of the baseline, in percentage points; a negative difference means fewer errors of our model. The interval is the 95% percentile interval of that difference and the p-value is one-sided (H0: our model is not better than the baseline); both come from a paired bootstrap that resamples speakers, with 10,000 resamples and seed 0.

Distance to the baseline with the lowest WER on the test split, `canary-1b-v2`, as a difference between point estimates on the same clips (no interval was computed for it): `hf-whisper-small` WER +20.48 points and CER +5.37 points; `hf-whisper-small-finetuned` WER +11.57 points and CER +2.77 points. Share of that distance closed by our model: WER 43.5%, CER 48.4%.

Source: `results/own/<system>/test/comparison_with_<baseline>.json` and `results/baselines/<system>/test/scores.json`.
<!-- end generated -->

### Discussion

This section reads the three tables above and holds no figure of its own.
The only numbers in it are model sizes taken from model cards.

**Which system is best.** `canary-1b-v2` has the lowest WER and the lowest
CER of all systems, baselines and our own. The margin is wide. Its WER
interval lies entirely below the WER interval of every other system, and the
system with the next lowest WER makes more than twice as many word errors
per reference word. Its CER interval lies below every other CER interval
except that of Parakeet, which is too wide to exclude anything.

**What the fine-tuning achieved.** Against `hf-whisper-small`, the checkpoint
it started from, our fine-tuned model has a lower WER, a lower CER and a
lower WER without diacritics. The 95% interval of each paired difference
lies entirely below zero, with and without the clips whose reference
contains English number words, and every one-sided p-value is at the
smallest value that the number of resamples allows. H0, "the fine-tuned
model is not better than its baseline", is rejected for the CER, which was
declared as the main comparison, and for the WER. Two things limit this
statement:

- The intervals of the two systems taken separately (first two tables) do
  not overlap for the WER, but they overlap at their ends for the CER. The
  paired interval is the one made for two systems on the same clips: it
  resamples the same speakers for both, so what the two systems have in
  common cancels out. Read without pairing, the CER result is weaker than
  the WER result.
- The fine-tuning was run once, with one seed. The intervals describe the
  uncertainty from the choice of test speakers, not from the training run.

The validation split shows the same direction for every metric
([`results/README.md`](results/README.md)); its figures are development
figures, because that split chose the checkpoint.

**How much of the gap is closed.** The fine-tuned model has the lowest WER
and CER after Canary, but it does not reach Canary: the intervals of the two
do not overlap, for the WER or for the CER. The lines below the comparison
table give the distance of both Whisper small systems to Canary. The
fine-tuning closed less than half of that distance, for the WER and for the
CER. Against the other baselines only the intervals can be read, because no
paired comparison was made with them. The WER interval of the fine-tuned
model lies below the WER intervals of both Whisper small runs, Qwen3-ASR,
wav2vec 2.0, Whisper base and the GMM-HMM system. It lies inside the
interval of Parakeet, so these two systems cannot be ranked.

**What has to be kept in mind when the rows are compared.**

- **Canary: size and training data.** Its
  [model card](https://huggingface.co/nvidia/canary-1b-v2) gives 978 million
  parameters; the [card of `openai/whisper-small`](https://huggingface.co/openai/whisper-small),
  the checkpoint we fine-tune, gives 244 million. The Canary card lists the
  Granary dataset and NeMo ASR Set 3.0 as training data and names VoxPopuli
  only in a benchmark table. We have not checked these corpora beyond the
  card and do not know whether they hold recordings that our test clips were
  cut from ([What these models have already seen](#what-these-models-have-already-seen)).
  The row of Canary and the row of our model therefore compare a model four
  times the size, whose training data we can only name, with a small model
  adapted on the training hours of the table under
  [Fine-tuning](#fine-tuning-whisper-small-our-own-model).
- **Parakeet: a very wide interval, and no language setting.** Its WER
  interval is by far the widest of the benchmark, and more than half of its
  word errors are insertions (column "Word errors"). In its prediction file
  these insertions come from very few clips, for which the output is many
  times longer than the reference. The intervals resample speakers, so a few
  such clips are enough to make the interval wide, and its point estimate
  should not be ranked against a system whose estimate lies inside that
  interval. Its [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
  says that the model detects the language of the audio by itself, and it
  takes no language setting. Its output language could therefore not be
  fixed to Hungarian, as it was for Whisper, Canary and Qwen3-ASR.
- **wav2vec 2.0: training domain.** Its
  [model card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-hungarian)
  says that it was fine-tuned for Hungarian on the train and validation
  splits of Common Voice 6.1 and on CSS10. It names no parliamentary speech,
  so our test set lies outside the data that the card names. Its row shows
  how this checkpoint transfers to another domain, decoded without a
  language model as on its card. It does not show what a wav2vec 2.0 model
  with a CTC head reaches when it is trained on VoxPopuli.
- **GMM-HMM: closed vocabulary, and a weakness we cannot explain.** This
  system can only write words that occur in the training transcripts. The
  table of the GMM-HMM baseline in [`results/README.md`](results/README.md)
  gives the share of reference words outside its lexicon, which is a floor
  for its WER. Its WER is far above that floor, and nearly half of its word
  errors are deletions. We have not found the cause. The recipe and the
  decoding are the defaults of the Montreal Forced Aligner and nothing was
  tuned, so the row shows what this recipe gives on our data, not what a
  GMM-HMM system can reach.
- **Qwen3-ASR: repository.** The row is the result of
  `Qwen/Qwen3-ASR-1.7B-hf`, not of `Qwen/Qwen3-ASR-1.7B`
  ([The repository of Qwen3-ASR](#the-repository-of-qwen3-asr)). Its card
  does not list its training data.
- **Real-time factor: different machines.** The column "Run on" names the
  machine of every run. The systems that run through `transformers` were
  run on an NVIDIA L4 GPU; faster-whisper and the GMM-HMM system were run on
  the CPU of a laptop. RTF values can only be compared between rows with
  the same entry in that column. The value of the GMM-HMM system cannot be
  compared with any other row: it is the wall time of decoding the whole
  split with parallel jobs, not a time per clip. The load of the laptop was
  not controlled ([`results/README.md`](results/README.md)).
- **English number words.** Some references contain numbers written out in
  English by the corpus ([Data](#data)), which no system can get right. The
  column without these clips gives the same order of the systems by WER.
- **The test set.** It is a part of the official VoxPopuli test split with
  few speakers ([Data](#data)). The figures are not comparable with
  published VoxPopuli results.

### Error analysis

Not written yet: see [issue #9](https://github.com/kaankalkan03/NLP-project/issues/9).

### Out-of-vocabulary analysis

Not written yet: see [issue #10](https://github.com/kaankalkan03/NLP-project/issues/10).

### Limitations

Not written yet: see [issue #11](https://github.com/kaankalkan03/NLP-project/issues/11).

## Data

| Corpus | Role | Licence | Where it lives |
|---|---|---|---|
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli), Hungarian transcribed set | training, development and main test data | CC0-1.0 | subset in `data/raw/voxpopuli/`, repaired references in `data/clean/voxpopuli/` |
| [FLEURS](https://huggingface.co/datasets/google/fleurs), `hu_hu`, test split | extra test set (read speech): built and committed, **not scored** | CC BY 4.0 (attribution below) | whole split in `data/raw/fleurs/` |
| [Common Voice Scripted Speech 27.0](https://mozilladatacollective.com/datasets/cmu5wq4u200dknq072xswzofb), Hungarian, test split | extra test set (read speech, many volunteers): scripts only, **not built and not scored** | CC0-1.0, re-hosting forbidden | **never in this repository**; the scripts fetch it into `external/` (ignored by git) |

Every result of this project is measured on VoxPopuli. The two read-speech
corpora were planned as extra test sets; the table says how far each got.

Every source file is pinned: `src/huasr/download.py` holds the dataset
revision, size and SHA-256 of each file and checks them after the download.
`python -m huasr.download voxpopuli --dry-run` (or `fleurs --dry-run`) prints
the list.

### VoxPopuli (training, development, main test)

Speeches from European Parliament plenary sessions, cut into utterances and
aligned with the official transcripts (Wang et al., 2021). The Hungarian part
comes as parquet files with these columns, read from the files' own metadata:
`audio_id`, `audio` (the bytes of one WAV file and its path), `raw_text`,
`normalized_text`, `gender`, `speaker_id`, plus `language`,
`is_gold_transcript` and `accent`, which this project does not use.

**Speakers and recording sessions.** Every clip has a speaker id (a few have
none) and comes from a recording session. VoxPopuli cuts all clips of one
`session_id` out of one recording file and names a clip
`<session_id>-<language>_<date>-<time of the speech>_<number>`
(`voxpopuli/get_asr_data.py` in the corpus's own code). The clip id up to the
first underscore, for example `20090202-0900-PLENARY-9-hu`, is therefore used
as the session. In the Hungarian data such a session is one numbered part of
a plenary sitting day: the parts of a day follow each other in time, and a
speech with all its clips lies inside one part.

**The official split is not disjoint.** The same speakers and the same
sessions occur in the official train, validation and test parts, and some
clips have no speaker id:

<!-- generated: voxpopuli-official-split -->
| Official split | Clips | Hours | Speakers | Sessions | Clips without speaker id |
|---|---|---|---|---|---|
| train | 18,120 | 54.09 | 47 | 1,537 | 601 |
| validation | 1,076 | 3.25 | 26 | 141 | 16 |
| test | 1,110 | 3.11 | 46 | 154 | 23 |

Speakers shared by two official splits: train and validation 21, train and test 27, validation and test 18. Sessions shared by two official splits: train and validation 90, train and test 93, validation and test 14.

Source: `data/raw/voxpopuli/summary.json`.
<!-- end generated -->

A model that is tested on a speaker or on a recording it was trained on looks
better than it is. `src/huasr/subset.py` therefore makes its own split, with a
fixed seed, under three rules:

1. Every speaker is in exactly one split.
2. Every recording session is in exactly one split.
3. A clip is only used in the official split it comes from. No official
   training clip becomes a test clip, because public models may have been
   trained on the official training part; and no official test clip is used
   for training.

A clip is kept if its official split, the split of its speaker and the split
of its session are the same. How speakers and sessions get their split:

- Clips without a speaker id are left out, since nobody can check them.
- Every speaker and every session goes to the split in which it keeps the
  most audio. Test and validation audio is scarce and training audio is
  plentiful, so a second of test audio and a second of validation audio count
  several times as much as a second of training audio (two weights). A
  speaker counts only the clips whose session is in the same split and a
  session only the clips whose speaker is, so sessions and speakers are
  assigned in turn until nothing changes.
- The two weights are searched on a fixed grid. The pair is taken that
  reaches, in this order, the minimum of test audio, the validation target
  and the training target, and then gives the most test audio.
- The test split keeps every clip that is left. Validation and training clips
  are drawn in a seeded order until the target of the split is reached, so
  every speaker is represented in proportion to their audio.

The build stops unless no speaker and no session is in two splits of the
result. What is in this repository, counted again from the stored manifest:

<!-- generated: voxpopuli-split-report -->
| Split | Speakers | Sessions | Utterances | Hours | References that occur word for word in the training split |
|---|---|---|---|---|---|
| train | 20 | 626 | 4,597 | 14.00 | - |
| validation | 15 | 112 | 826 | 2.50 | 6 of 826 |
| test | 34 | 104 | 897 | 2.41 | 7 of 897 |

Speakers in two splits: train and validation 0, train and test 0, validation and test 0. Sessions in two splits: train and validation 0, train and test 0, validation and test 0.

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

<!-- generated: voxpopuli-subset -->
| Split | Clips | Hours | Speakers | Sessions | Shards | Size (MB) |
|---|---|---|---|---|---|---|
| train | 4,597 | 14.00 | 20 | 626 | 11 | 987.8 |
| validation | 826 | 2.50 | 15 | 112 | 2 | 165.0 |
| test | 897 | 2.41 | 34 | 104 | 2 | 166.2 |
| total | 6,320 | 18.91 | 69 | 842 | 15 | 1319.0 |

Speakers shared by two splits of the subset: train and validation 0, train and test 0, validation and test 0. Sessions shared by two splits of the subset: train and validation 0, train and test 0, validation and test 0. Settings: training target 14 h, validation target 2.5 h, at least 2 h of test audio; weights taken: validation 40, test 36; seed 0.

Source: `data/raw/voxpopuli/summary.json`.
<!-- end generated -->

The targets are the defaults `DEFAULT_HOURS` and `DEFAULT_SEED` in
`src/huasr/subset.py`; the training and validation targets keep the
repository at a size GitHub accepts. [`data/raw/README.md`](data/raw/README.md)
shows the official split in more detail, what each rule removed, and what the
other weight pairs would have kept; `data/raw/voxpopuli/selection.csv` names
every source clip with its decision.

Two limits of this split. Different parts of the same sitting day can be in
different splits; the unit that is kept apart is the recording session of the
corpus, and `summary.json` counts the shared days
(`sitting_days_in_two_splits`). And because of rule 3 our test set is a part
of the official test set, but the scores are not comparable with published
VoxPopuli results, which use all of it.

**The reference text had to be repaired.** VoxPopuli gives two transcripts per
clip: `raw_text` as published, and `normalized_text`, which the corpus derived
from it and which is the usual reference. In the Hungarian part that
normalisation deleted every letter í (`raw_text` *javítani*,
`normalized_text` *javtani*). `src/huasr/repair.py` restores the letter word
by word, only where the aligned word of `raw_text` proves it; words that
cannot be aligned are left unchanged and counted, and no text is invented.
The result is the column `reference` of `data/clean/voxpopuli/manifest.csv`;
both original fields stay next to it, and every changed word is logged.

<!-- generated: voxpopuli-repair -->
| Reference repair | Count |
|---|---|
| Clips with at least one restored word | 2,666 |
| Restored words | 3,962 |
| Clips with nothing to restore | 3,507 |
| Clips without `raw_text` (cannot be checked) | 147 |
| Clips whose `normalized_text` is not in normalised form (left alone) | 0 |
| Words that could not be aligned with `raw_text` (left unchanged) | 787 in 325 clips |

| Letter missing from `normalized_text` | in `raw_text` | in `normalized_text` | in `reference` |
|---|---|---|---|
| `í` | 3,965 | 0 | 3,964 |
| `š` | 13 | 0 | 0 |
| `č` | 7 | 0 | 0 |
| `ş` | 4 | 0 | 0 |
| `ß` | 3 | 0 | 0 |
| `ä` | 2 | 0 | 0 |
| `º` | 1 | 0 | 0 |
| `è` | 1 | 0 | 0 |
| `ï` | 1 | 0 | 0 |
| `ć` | 1 | 0 | 0 |
| `ł` | 1 | 0 | 0 |
| `ň` | 1 | 0 | 0 |
| `ž` | 1 | 0 | 0 |
| `ș` | 1 | 0 | 0 |

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

The other letters in the second table are letters of other languages in
names; they were lost in the same way and are reported, not restored. Clips
without `raw_text` cannot be checked.

**Numbers are written out in English.** The corpus normalisation also turned
digits into English words inside the Hungarian text: `raw_text` *2008*,
`normalized_text` *two thousand and eight*. Nobody says that in the recording,
so every such number is an error for every system and a wrong target for
training. These references are flagged (`english_number_words` in the clean
manifest) and counted. No clip is removed from the data for it: the scorer
reports every figure with and without the flagged clips, and the fine-tuning
leaves the flagged clips out of its training clips.

<!-- generated: voxpopuli-number-words -->
| English number words | train | validation | test |
|---|---|---|---|
| References with English number words | 286 | 55 | 35 |
| of these: `raw_text` has the number in digits | 223 | 46 | 28 |
| of these: the clip has no `raw_text` | 62 | 9 | 6 |
| of these: `raw_text` has no digits (English words in the speech itself) | 1 | 0 | 1 |

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

Details of both problems: [`data/clean/README.md`](data/clean/README.md).

### FLEURS (extra test set, built but not scored)

Sentences read aloud by native speakers (Conneau et al., 2022). The whole
official Hungarian test split is stored in `data/raw/fleurs/`, built with
`python -m huasr.subset fleurs` and committed on 21 November 2026:

<!-- generated: fleurs-subset -->
| Split | Clips | Hours | Sentences | Shards | Size (MB) |
|---|---|---|---|---|---|
| test | 905 | 3.07 | 348 | 2 | 163.6 |

Source files: `google/fleurs` at revision `70bb2e84b976`. Encoding of the source audio: `wav_f32le` 905; stored as: `flac_s16` 905. Samples limited to the 16-bit range: 6. Clips that could not be read: 0; audio files without a transcript: 0; transcripts without audio: 0.

Source: `data/raw/fleurs/summary.json`.
<!-- end generated -->

**No system has been scored on FLEURS.** The runner reads the clean manifest
of VoxPopuli; for FLEURS there is no clean step and no run. FLEURS has no
speaker ids; several recordings share a sentence, so the manifest keeps the
sentence id, which would be the cluster for resampling scores.

Attribution: FLEURS by Conneau et al. (Google), obtained from
<https://huggingface.co/datasets/google/fleurs>, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made by
us: the WAV files of the Hungarian test split are stored as 16-bit FLAC files
in tar archives (see below), with the transcripts in a CSV file. No text was
changed.

### Common Voice (extra test set, scripts only, not stored here)

Volunteers read sentences aloud; other volunteers vote on each recording. The
dataset page states the licence as CC0-1.0 and adds, under "Forbidden Usage"
(read on 16 November 2026): "It is forbidden to attempt to determine the
identity of speakers in the Common Voice datasets. It is forbidden to re-host
or re-share this dataset." Common Voice can therefore only be a test set that
everyone fetches for themselves, and this repository holds only the scripts
for that:

- `python -m huasr.download commonvoice` writes to `external/common_voice/`,
  and `python -m huasr.subset commonvoice` writes its list of test clips next
  to the extracted release. Both refuse any target inside the repository
  other than `external/` (`require_untracked` in `src/huasr/download.py`), and
  tests check this.
- `.gitignore` excludes `external/`, extracted release folders, the archive,
  the clip list and every MP3 file.

**Common Voice has not been built and no system has been scored on it.** The
scripts are tested on generated files, and `commonvoice --info` was run
against the real service; the dataset itself has not been downloaded with
them. No result of this project uses Common Voice.

A download needs an account at the Mozilla Data Collective and the acceptance
of the dataset's terms on its web page; a script cannot do that. The manual
steps are listed under [Reproduce](#reproduce). We do not try to identify any
speaker in any corpus; speakers appear only as the ids the corpora publish.

### How audio is stored

VoxPopuli and FLEURS ship uncompressed WAV files, far too large for a GitHub
repository. `src/huasr/audio.py` reads the format of every clip from its own
header and stores it as a mono 16-bit FLAC file with the same sample rate:

- 16-bit PCM input is stored losslessly.
- 32-bit float input is quantised to 16 bits (sample × 32768, rounded,
  limited to the 16-bit range). This is the only change made to the audio; the
  number of limited samples is logged per clip.
- Any other encoding is refused, not converted. Input that is already
  compressed would be kept byte for byte. Nothing is resampled.

The manifest records, per clip, the encoding, size and SHA-256 of the source
file and the size and SHA-256 of the stored file, so a stored clip can be
traced back to the exact source bytes. The clips are packed into tar shards
below GitHub's file-size limit. Details and file formats:
[`data/raw/README.md`](data/raw/README.md).

### Cleaning

What the clean step does (`src/huasr/clean.py`, outputs in
`data/clean/voxpopuli/`): the reference repair above with a log of every
changed word; a log of every source clip that is not in the data with the
rule that left it out; and three flags per clip: English number words in the
reference, a validation or test reference that also occurs word for word in
the training split, and the number of characters per second, a signal for
transcripts that do not belong to their audio.

What it does not do: **no cleaning rule removes a clip.** Removal rules for
transcripts that do not match their audio, for clips that are too long for
the models and for transcripts with foreign letters were planned and are not
implemented, and no threshold was chosen. The flags are used downstream
instead: the English number words by the scorer and the fine-tuning (above),
and clips longer than the window of a model are cut by the systems
themselves ([Systems](#systems)). See
[`data/clean/README.md`](data/clean/README.md).

## Systems

Every system sits behind one small contract (`src/huasr/systems.py`): a name,
a group (`baselines` for checkpoints used as published, `own` for models
trained in this project), a description for the run manifest, and
`transcribe(samples, sample_rate) -> text`. The runner gives every system the
same clips in the same form and knows nothing else about it.

Every system below has been run on the whole validation split and, once, on
the whole test split. The revision of each checkpoint is in the tables under
[Results and discussion](#results-and-discussion) and, in full, in the
`run.json` of each run.

| System | Group | What it is | Checkpoint that is loaded | Described in |
|---|---|---|---|---|
| `gmm-hmm-mfa` | `baselines` | MFCC features, GMM-HMM acoustic model, n-gram language model: the classical recogniser, trained on our training split with a standard recipe | none published | [GMM-HMM baseline](#gmm-hmm-baseline-gmm-hmm-mfa) |
| `faster-whisper-base` | `baselines` | Whisper base (attention encoder-decoder), as published | `Systran/faster-whisper-base`, a conversion of `openai/whisper-base` | below |
| `faster-whisper-small` | `baselines` | Whisper small (attention encoder-decoder), as published | `Systran/faster-whisper-small`, a conversion of `openai/whisper-small` | below |
| `hf-whisper-small` | `baselines` | Whisper small through `transformers`, as published; the checkpoint our model starts from | `openai/whisper-small` | [Fine-tuning Whisper small](#fine-tuning-whisper-small-our-own-model) |
| `wav2vec2-xlsr-53-hungarian` | `baselines` | wav2vec 2.0 encoder with a CTC head, fine-tuned for Hungarian by its publisher | `jonatasgrosman/wav2vec2-large-xlsr-53-hungarian` | [Pretrained baselines](#pretrained-baselines-of-other-model-families) |
| `canary-1b-v2` | `baselines` | attention encoder-decoder, as published | `nvidia/canary-1b-v2` | [Pretrained baselines](#pretrained-baselines-of-other-model-families) |
| `parakeet-tdt-0.6b-v3` | `baselines` | transducer (TDT), as published | `nvidia/parakeet-tdt-0.6b-v3` | [Pretrained baselines](#pretrained-baselines-of-other-model-families) |
| `qwen3-asr-1.7b` | `baselines` | audio encoder with a language model as decoder, as published | `Qwen/Qwen3-ASR-1.7B-hf` | [Pretrained baselines](#pretrained-baselines-of-other-model-families) |
| `hf-whisper-small-finetuned` | `own` | Whisper small fine-tuned on our training split: our contribution | `openai/whisper-small` after our fine-tuning | [Fine-tuning Whisper small](#fine-tuning-whisper-small-our-own-model) |

How the baselines that the course names for this project are covered:

| Baseline named by the course | In this repository |
|---|---|
| MFCC + GMM-HMM | `gmm-hmm-mfa`, trained here |
| CTC encoder; wav2vec 2.0 + CTC fine-tuned for the language | `wav2vec2-xlsr-53-hungarian` |
| HuBERT + CTC fine-tuned for the language | not run: no Hungarian checkpoint was found ([details](#hubert--ctc-is-not-run)) |
| Attention-based encoder-decoder; multilingual Whisper | `faster-whisper-base`, `faster-whisper-small`, `hf-whisper-small` |
| `nvidia/canary-1b-v2` | `canary-1b-v2` |
| Transducer; `nvidia/parakeet-tdt-0.6b-v3` | `parakeet-tdt-0.6b-v3` |
| `Qwen/Qwen3-ASR-1.7B` or `Qwen/Qwen3-ASR-0.6B` | `qwen3-asr-1.7b`, run from the repository `Qwen/Qwen3-ASR-1.7B-hf` ([why](#the-repository-of-qwen3-asr)) |

For the three neural families (CTC, attention encoder-decoder, transducer)
the baselines are published checkpoints of that family. We trained no model
of these families from scratch; the GMM-HMM system is the only baseline that
is trained here, and the fine-tuned Whisper small is our own model.

### Whisper base and small through faster-whisper

`faster-whisper-base` and `faster-whisper-small` are run with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), which loads
CTranslate2 conversions of the OpenAI checkpoints; the model cards of the two
repositories name the OpenAI checkpoint each was converted from. They run on
the CPU with weights quantised to 8-bit integers and a fixed number of
threads, with language `hu`, task `transcribe`, beam size 5, temperature 0
and no conditioning on previous text. The revision that was loaded, the
SHA-256 of its weight file and all settings are in the `run.json` of each
run; what limits their figures is in
[`results/README.md`](results/README.md). `faster-whisper-small` and
`hf-whisper-small` are the same published checkpoint in two
implementations; the fine-tuned model is compared with `hf-whisper-small`,
which shares its code and decoding.

### GMM-HMM baseline (`gmm-hmm-mfa`)

The classical recogniser: hidden Markov models whose states are Gaussian
mixtures over MFCC features, a pronunciation lexicon and an n-gram language
model. There is no published checkpoint to take, so this baseline is trained
from scratch on our training split with a standard recipe. It is listed with
the baselines, not as our model: nothing in it is our own design.
`src/huasr/gmm_hmm.py` runs it with the
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/)
(MFA), which drives Kaldi. MFA works on a folder of clips, not on one clip,
so this system does not sit behind the contract above; see
[Evaluation protocol](#evaluation-protocol).

- **Features:** MFCC without pitch; everything else as MFA sets it.
- **Acoustic model:** the first four blocks of MFA's default recipe with
  their default settings (`TRAIN_CONFIG`): monophone, triphone, LDA, and
  speaker-adapted training (SAT). The later blocks of that recipe are left
  out.
- **Lexicon:** graphemic. Every word of the training transcripts is spelled
  as its letters, one letter per unit (*szép* is `s z é p`); digraphs are not
  merged. No pronunciation dictionary and no G2P model is used.
- **Language model:** trigram with Kneser-Ney smoothing, estimated by
  `mfa train_lm` from the training transcripts only. Its vocabulary is the
  lexicon.
- **Text:** the `reference` column after the scoring policy below, for the
  acoustic model and for the language model.
- **Decoding:** `mfa transcribe` with the search settings MFA uses by default
  (`DECODE_CONFIG`); nothing was tuned. The clips of the split are given to
  MFA grouped by speaker id, because the model adapts to each speaker; no
  reference text of the split is written to the folder MFA reads.

What limits it, and what has to be kept in mind when it is compared with the
other systems:

- **Closed vocabulary.** It can only write words that occur in the training
  transcripts. Every other word of a reference is an error before any audio
  is heard, which weighs heavily in a language that builds a new word form
  with every suffix. `collect` counts these words; the share is in
  [`results/README.md`](results/README.md) next to the error rates.
- **Unknown-word placeholders.** MFA has an unknown-word model and can write
  a placeholder such as `<unk>` for it. Any placeholder is removed from the
  hypothesis and counted in `run.json`.
- **Few training speakers.** The acoustic model has heard only the speakers
  of the training split, and every speaker it is evaluated on is new to it.
- **Letters as sound units.** Hungarian spelling is close to the sounds, but
  a letter is not always one sound (*sz*, *cs*, *gy*, *dzs*, long consonants);
  the context-dependent models have to absorb that.
- **Speaker ids are used.** The Whisper baselines get one clip at a time and
  nothing else. This system also gets the speaker id of every clip.
- **Speed is not comparable.** MFA decodes the whole split with several
  parallel jobs; the processing time in `run.json` is the wall time of
  `mfa transcribe` shared out over the clips, not a per-clip time.
- **Platform.** The run in this repository was made on Windows 11 with the
  win-64 packages of conda-forge, without WSL. The steps are plain Python
  entry points and the environment file pins the same versions for every
  platform, but they have not been run on Linux or Colab.

Decoding settings: the Whisper systems use the settings declared in
`src/huasr/systems.py` and `src/huasr/whisper_hf.py`; the GMM-HMM system and
the pretrained baselines use the defaults of their toolkit or checkpoint,
with the exceptions named in their sections. The validation split decided
one thing only: which checkpoint of the fine-tuning is kept. No setting was
chosen on the test split. What each published model's card says about its
training data is under
[What these models have already seen](#what-these-models-have-already-seen).

## Fine-tuning Whisper small (our own model)

> **Status (22 November 2026).** Done. The fine-tuning was run once, on 22
> November 2026 on Google Colab; the zero-shot checkpoint and the fine-tuned
> model were then run on the validation split and, once, on the test split.
> The table "Fine-tuning run" below is written from the training record of
> that run.

Fine-tuning is this project's own contribution. It is reported apart from
the baselines, under [Results and discussion](#results-and-discussion) for
the test split and under "Our own models" in
[`results/README.md`](results/README.md) for both splits, on the same clips,
references, normalisation and scoring as every baseline.

**Two systems, one piece of code.** `src/huasr/whisper_hf.py` runs Whisper
small through the Hugging Face implementation (`transformers`), behind the
system contract:

| System | Group | What is loaded |
|---|---|---|
| `hf-whisper-small` | `baselines` | `openai/whisper-small` at revision `973afd24965f72e36ca33b3055d56a652f456b4d`, as published (zero-shot) |
| `hf-whisper-small-finetuned` | `own` | that checkpoint after `python -m huasr.finetune train`, read from a local folder |

The zero-shot system is measured first, then the checkpoint is fine-tuned,
then the kept checkpoint is run on the same clips. Both systems decode in the
same way: language Hungarian, task transcribe, no timestamps, beam size 5 and
temperature 0 as for the faster-whisper baselines, 32-bit floats, on the GPU
if there is one (the run manifest names the device). Whisper reads at most 30
seconds at once and cannot move on through a longer recording without
timestamps, so a longer clip is cut into the fewest equal parts that fit and
the texts are joined. A word at a cut can come out wrong, for both systems at
the same place. `python -m huasr.compare` compares the two prediction files
with the paired bootstrap of [Comparing two systems](#comparing-two-systems)
and refuses runs that do not hold the same clips and references.

**Fixed before the first run** (`src/huasr/finetune.py`). The validation
split decides one thing only: which checkpoint is kept.

- *Target text:* the column `reference` after the scoring policy, so the
  model learns to write what the scorer counts.
- *Training clips with English number words are left out.* Their text is a
  wrong target (see [Data](#data)). The validation clips keep them, as every
  system is scored on them, and the scorer gives the figures with and without.
- *Clips longer than Whisper's window are left out* of the training and of
  the checkpoint selection: the end of the audio would be cut off while the
  text went on. The evaluation of the kept checkpoint covers every clip of
  the validation split and of the test split.
- *Nothing else is changed:* no augmentation, no frozen layers, the
  tokenizer and the feature extractor of the checkpoint as published. The
  settings were not tried in several variants.

<!-- generated: finetune-plan -->
| Split | Used for | Clips in the split | Left out: English number words in the reference | Left out: longer than 30 s, of all such clips | Left out: empty target | Clips used | Hours used | Speakers |
|---|---|---|---|---|---|---|---|---|
| train | training | 4,597 | 286 | 18 of 29 | 0 | 4,293 | 12.51 | 20 |
| validation | choosing the checkpoint | 826 | kept | 7 of 7 | 0 | 819 | 2.42 | 15 |

A clip is counted under the first rule that leaves it out; the other training clips longer than 30 s are already left out for their number words. Settings: 2,000 updates of 16 clips (about 7.5 passes over the training clips used), learning rate 1e-05 after a linear warm-up of 200 updates, then linear decay to zero; optimizer `adamw_torch`, weight decay 0, gradient norm limited to 1; mixed precision `fp16`; validation and a checkpoint every 250 updates; seed 0. Checkpoints are compared by validation WER, decoded with beam size 1 and at most 225 tokens.

Source: `data/clean/voxpopuli/manifest.csv` and the constants of `src/huasr/finetune.py` (`python -m huasr.finetune plan`).
<!-- end generated -->

**The run that was made.** The training record
`results/own/hf-whisper-small-finetuned/training.json` states the counts
above for the run that was really made, together with the loss and the
validation scores of every evaluated step, the kept step, the GPU and the
package versions:

<!-- generated: finetune-run -->
| Fine-tuning run | As recorded |
|---|---|
| Started and finished (UTC) | 2026-11-22 02:38 to 2026-11-22 03:42 |
| GPU | NVIDIA L4, 22.0 GiB of memory |
| Peak GPU memory | 17.1 GiB with 16 clips per update (batches of 16) |
| Training clips read | 4,293 (12.51 h, 20 speakers) |
| Updates | 2,000; continued after an interruption: no |
| Time | 63.6 min in all: 30.5 min of updates and 29.9 min for 9 validation passes |
| Checkpoint kept | update 1,500, the lowest validation WER of the evaluated checkpoints |
| Packages | torch 2.14.1+cu130, transformers 5.23.0, accelerate 1.16.0 |

| Update | Epoch, as the trainer counts it | Validation loss | Validation WER % | Validation CER % |  |
|---|---|---|---|---|---|
| 0 | 0.00 | 0.7760 | 34.76 | 11.22 |  |
| 250 | 0.93 | 0.3678 | 25.91 | 8.02 |  |
| 500 | 1.86 | 0.3232 | 24.25 | 7.96 |  |
| 750 | 2.79 | 0.3186 | 23.22 | 7.69 |  |
| 1,000 | 3.72 | 0.3272 | 21.98 | 7.17 |  |
| 1,250 | 4.65 | 0.3365 | 21.11 | 6.80 |  |
| 1,500 | 5.58 | 0.3452 | 20.87 | 6.60 | kept |
| 1,750 | 6.51 | 0.3521 | 21.19 | 6.77 |  |
| 2,000 | 7.43 | 0.3545 | 20.97 | 6.55 |  |

The validation figures of this table only choose the checkpoint: they are decoded in batches with beam size 1, on the 819 validation clips that fit the window of the model. The results of the kept checkpoint are in the result tables.

Source: `results/own/hf-whisper-small-finetuned/training.json`.
<!-- end generated -->

The run is seeded, but GPU arithmetic is not repeatable bit for bit, so a
second run can differ slightly; the fine-tuning was run once. The
fine-tuning code refuses the test split. Weights are never stored in this
repository: they go to `scratch/`, which is not tracked, and the run
manifest of the evaluation records their SHA-256 and that of the training
record.

**How to run it.** On Google Colab with a GPU that has more memory than the
peak in the table above (the committed run used the GPU named there), open
[`notebooks/finetune_whisper_small.ipynb`](notebooks/finetune_whisper_small.ipynb)
and run all cells. It clones this repository, so nothing is uploaded by hand,
it never uses the test split, and it ends with an archive of the small result
files. The cells call the commands below, which work on any machine with a
GPU:

```bash
pip install -r requirements.txt -r requirements-train.txt
pip install --no-deps -e .
python -m huasr.subset unpack --raw data/raw/voxpopuli    # audio files -> scratch/audio/voxpopuli/
python -m huasr.whisper_hf fetch                          # the pinned checkpoint -> Hugging Face cache
python -m huasr.finetune plan                             # clips and settings; needs no model
python -m huasr.finetune train --smoke                    # a few updates on a few clips: checks the whole path
python -m huasr.evaluate --system hf-whisper-small        # before: zero-shot, validation split
python -m huasr.finetune train                            # the real run; --resume continues an interrupted one
python -m huasr.evaluate --system hf-whisper-small-finetuned    # after: the kept checkpoint, same clips
python -m huasr.score results/baselines/hf-whisper-small/validation results/own/hf-whisper-small-finetuned/validation
python -m huasr.compare results/own/hf-whisper-small-finetuned/validation results/baselines/hf-whisper-small/validation
python -m huasr.report
```

`requirements-train.txt` pins the packages that only this part needs (torch,
transformers, accelerate, tokenizers, safetensors, huggingface_hub) at the
versions of the Colab GPU runtime of 20 November 2026, so that Colab keeps
its GPU build of torch; `requirements.txt` is unchanged. `fetch` is the only
command that downloads a model, and it checks the weight file against a
pinned SHA-256. If the clips of an update do not fit in the GPU memory,
`train --batch-size 8` splits every update into two batches and the record
says so. The peak GPU memory of the committed run (table above) is more than
a Colab T4 has; on such a card the smaller batch is needed, and that is a
different setting from the one reported here. The smoke run prints a
projection of the time, measured on the machine.

**The final evaluation on the test split** is not part of the notebook. It
was made once, on 22 November 2026, with the commands below; the run
manifests name the machine:

```bash
python -m huasr.evaluate --system hf-whisper-small --split test --final-test
python -m huasr.evaluate --system hf-whisper-small-finetuned --split test --final-test
python -m huasr.score results/baselines/hf-whisper-small/test results/own/hf-whisper-small-finetuned/test
python -m huasr.compare results/own/hf-whisper-small-finetuned/test results/baselines/hf-whisper-small/test
python -m huasr.report
```

`hf-whisper-small-finetuned` reads its weights from
`scratch/finetune/hf-whisper-small/best`, where the fine-tuning leaves them,
or from the folder named in the environment variable `HUASR_FINETUNED_DIR`.
Because the weights are not in this repository, running the fine-tuned model
again needs the fine-tuning to be run again, or the weights kept from it.
The files of this part are listed under
[Repository layout](#repository-layout).

## Pretrained baselines of other model families

> **Status (22 November 2026).** Done. The four systems of this section were
> run on 22 November 2026 on Google Colab, each on the whole validation split
> and once on the whole test split. Their rows are in the baseline tables
> under [Results and discussion](#results-and-discussion) and in
> [`results/README.md`](results/README.md).

The course lists these pretrained models as baselines. Each one is a
published checkpoint that is run the way its model card and the
documentation of `transformers` describe it. Nothing is trained and no
setting was tuned on our data, so every one of them is a baseline and never
our own model. The few settings that had to be made by us are named below
and recorded in every `run.json`: the language, the cut of long clips for
Canary and the token limit for Qwen3-ASR. `src/huasr/pretrained.py` puts the
four systems behind the system contract; they get the same clips,
references, normalisation and scoring as every other system.

| System | Family | Checkpoint and revision | Licence | Language setting |
|---|---|---|---|---|
| `wav2vec2-xlsr-53-hungarian` | self-supervised encoder with a CTC head | `jonatasgrosman/wav2vec2-large-xlsr-53-hungarian` at `2bd0786e71cad097c4e6ce9b3e45b40c74ad8f37` | Apache-2.0 | none to set: the checkpoint is Hungarian only |
| `canary-1b-v2` | attention encoder-decoder | `nvidia/canary-1b-v2` at `d455706339a6b32e1aa40f82c713a482a0c938e2` | CC BY 4.0 | source `hu` and target `hu`: transcription, not translation |
| `parakeet-tdt-0.6b-v3` | transducer (TDT) | `nvidia/parakeet-tdt-0.6b-v3` at `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | CC BY 4.0 | none exists: the model detects the language itself |
| `qwen3-asr-1.7b` | audio encoder with a language model as decoder | `Qwen/Qwen3-ASR-1.7B-hf` at `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` | Apache-2.0 | Hungarian, forced |

Model cards:
[wav2vec 2.0](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-hungarian),
[Canary](https://huggingface.co/nvidia/canary-1b-v2),
[Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3),
[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf). All three
multilingual cards list Hungarian among the supported languages.

**How they are run.** `transformers` 5.23.0 implements all four model
families, and the cards of the four checkpoints show this way of running
them. All four therefore use one set of pinned packages,
`requirements-pretrained.txt`, with the versions of the fine-tuning runs;
nothing conflicts. The cards of Canary and Parakeet also show the NVIDIA
toolkit `nemo_toolkit`, and the card of `Qwen/Qwen3-ASR-1.7B` shows the
package `qwen-asr`. Neither is used. `qwen-asr` 0.0.6 requires
`transformers==4.57.6` and `accelerate==1.12.0`
([PyPI](https://pypi.org/project/qwen-asr/)), which cannot be installed next
to our pinned versions. `nemo_toolkit` is not needed, because the cards show
the `transformers` way as well; its `asr` extra would add a training stack
with version bounds of its own (`lightning`, `hydra-core`, `omegaconf` in
release 3.0.0; [PyPI](https://pypi.org/project/nemo-toolkit/)). For
Qwen3-ASR this decides the repository; see the next section.

### The repository of Qwen3-ASR

The course list names `Qwen/Qwen3-ASR-1.7B`. Our system `qwen3-asr-1.7b`
loads `Qwen/Qwen3-ASR-1.7B-hf`, and every table names that repository. What
the two model cards and the repository metadata say (read on 22 November
2026):

- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B): the
  card installs the package `qwen-asr` (`pip install -U qwen-asr`) and loads
  the model with `from qwen_asr import Qwen3ASRModel`. It shows no way of
  loading the model with `transformers` alone, and it does not mention a
  repository with the suffix `-hf`.
- [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf):
  a repository of the same organisation, `Qwen`. Its card is headed
  "Qwen3-ASR (Transformers native)", its metadata has
  `library_name: transformers`, and it states that Qwen3-ASR is supported
  natively in Transformers from version 5.13.0. Its examples load the model
  under the id `Qwen/Qwen3-ASR-1.7B-hf` with the `transformers` classes
  `AutoProcessor` and `AutoModelForMultimodalLM`. It does not mention
  `qwen-asr`. It describes the same model family as the other card
  (Qwen3-ASR-1.7B and Qwen3-ASR-0.6B) and lists Hungarian among the
  languages.

So `-hf` marks the repository whose files are in the format that the
Qwen3-ASR implementation inside the `transformers` library loads, published
by the same organisation as the original. **Neither card says how the two
repositories relate.** In particular, neither says that the weights are the
same, so we do not claim it: our row is the result of
`Qwen/Qwen3-ASR-1.7B-hf` at the pinned revision, and we have not run
`Qwen/Qwen3-ASR-1.7B` through `qwen-asr` to compare the outputs.

Why the `-hf` repository is used: every system that runs through
`transformers` in this project uses one pinned set of packages, with
`transformers` 5.23.0. The only way of running `Qwen/Qwen3-ASR-1.7B` that
its card shows is `qwen-asr`, whose latest release requires
`transformers==4.57.6`. That cannot be installed next to our pinned
versions, so this one system would have needed its own environment and
could not have gone through the same installation as the others. The
weight file of the 1.7B model has about 4 GB (`weights_bytes` in its
`run.json`) and it ran in bfloat16 on the GPU of the other runs, so the
smaller 0.6B model was not needed.

### How each pretrained system is decoded

- **wav2vec 2.0 + CTC.** Its card: `facebook/wav2vec2-large-xlsr-53`
  fine-tuned for Hungarian on the train and validation splits of Common
  Voice 6.1 and on CSS10, for input at 16 kHz. It is decoded as in the
  script of the card: the best token of every frame, repeats merged, blank
  removed, and no language model. The model writes upper-case letters
  without punctuation; case is left to the scoring policy.
- **Canary.** The task is given by the decoder prompt: a source and a target
  language, and equal languages mean transcription. Both are set to `hu`,
  and before every clip the code checks that the prompt holds the Hungarian
  token twice and no other language token. Decoding is greedy, as the
  checkpoint sets it, with punctuation as the processor requests it by
  default.
- **Parakeet TDT.** Greedy transducer decoding, as the checkpoint sets it.
  Its card says that the model detects the language of the audio by itself
  and needs no prompt. There is no language setting that could be wrong or
  missing; the run manifest records this.
- **Qwen3-ASR.** Hungarian is forced by prefilling the beginning of the
  answer, `language Hungarian<asr_text>`, which is how its card says the
  language can be forced; before every clip the code checks that the prompt
  ends with it. Decoding is greedy with at most 1,024 new tokens. The
  generation config of the checkpoint sets 512 and the examples of the card
  pass 256; we set a higher limit so that the text of a long clip is not
  cut off. No context or hotwords are given.

**Clips longer than a model reads at once.** Canary is used on at most 40
seconds: its card describes that its own toolkit cuts audio longer than 40
seconds into chunks. A longer clip is cut into the fewest equal parts that
fit and the texts are joined, as for Whisper; a word at a cut can come out
wrong. The other three are given every clip whole: wav2vec 2.0 has no
window, the card of Parakeet gives 24 minutes for full attention and the
card of Qwen3-ASR says that the model handles long audio, and our longest
clip is under a minute.

**Text.** The hypothesis is the decoded token sequence without special
tokens. Nothing else is done to it for any system: no language model on top
of the CTC output, and not the clean-up of repeated characters and patterns
that the Qwen3-ASR processor offers. Case and punctuation are handled by the
scoring policy, which is the same for every system.

### What these models have already seen

What the model card of each published checkpoint says about its training
data (cards read on 22 November 2026). We have read the cards and nothing
beyond them: which recordings, speakers and languages the named corpora
contain has not been checked.

- **Canary** ([card](https://huggingface.co/nvidia/canary-1b-v2)): the
  Granary dataset, described as pseudo-labelled and filtered versions of the
  corpora YTC, MOSEL and YODAS, and NeMo ASR Set 3.0, described as
  human-labelled transcriptions from Multilingual LibriSpeech, Mozilla Common
  Voice 7.0, AMI, FLEURS, LibriSpeech, Fisher, National Speech Corpus Part 1,
  VCTK and Europarl-ASR. VoxPopuli is named only in a benchmark table.
- **Parakeet** ([card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)):
  initialised from a checkpoint pretrained on Granary, then trained on
  pseudo-labelled data from Granary (YTC, MOSEL, YODAS) and on
  human-transcribed data from NeMo ASR Set 3.0 (the corpora named for Canary
  without FLEURS). VoxPopuli is named only in benchmark tables, FLEURS only
  as evaluation data.
- **What follows, and what does not.** Neither card names VoxPopuli as
  training data. We have not checked whether Granary or any corpus of NeMo
  ASR Set 3.0 holds European Parliament recordings, so we claim neither that
  these two models have seen recordings our clips were cut from nor that
  they have not. FLEURS and Common Voice 7.0 are named as training data; that
  would matter for a score on those corpora, which this project does not
  have.
- **wav2vec 2.0** ([card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-hungarian)):
  the train and validation splits of Common Voice 6.1, and CSS10. The card
  names no parliamentary speech.
- **Qwen3-ASR** ([card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)): the
  card speaks of large-scale speech training data and does not list it.
- **Whisper** ([card](https://huggingface.co/openai/whisper-small)): audio
  with transcripts collected from the internet; the data is not published.
  Whether it holds recordings of the European Parliament cannot be checked
  by us.

For every published model, "used as published" therefore means only that we
did not train it. It does not mean that the model has never met this kind of
audio.

### The test split and the pretrained baselines

These four systems are run on the validation split and once on the test
split by the same notebook, which is the one exception to the rule that
notebooks never touch the test split. They are published checkpoints for
which nothing was tuned on any of our data, so a look at the test split had
nothing to tune, and the test evaluation of the other systems had already
taken place earlier on the same day (22 November 2026; the table of runs in
[`results/README.md`](results/README.md) has the times). The runner still
demands `--final-test` and refuses to replace an existing test run unless
`--force` is given.

### HuBERT + CTC is not run

No checkpoint for Hungarian was found. The course also lists a HuBERT
encoder with a CTC head, fine-tuned for the language. On 22 November 2026
the Hugging Face Hub had no such checkpoint for Hungarian:

- [speech recognition models tagged Hungarian with "hubert" in the name](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&language=hu&search=hubert):
  no result;
- the models tagged both `hubert` and `hu`
  (<https://huggingface.co/api/models?filter=hubert&filter=hu>): 15 models,
  none of them a speech recogniser. They are Hungarian text models whose
  name contains "huBERT", and pretrained speech encoders without a CTC head
  such as [`utter-project/mHuBERT-147`](https://huggingface.co/utter-project/mHuBERT-147)
  and [`coml/hubert-base-vp20`](https://huggingface.co/coml/hubert-base-vp20).

A bare encoder cannot write text and is not a baseline result. The HuBERT
checkpoints with a CTC head that its authors published, for example
[`facebook/hubert-large-ls960-ft`](https://huggingface.co/facebook/hubert-large-ls960-ft),
are fine-tuned on English LibriSpeech; running one on Hungarian would be
running a model in the wrong language, so it is not done. The wav2vec 2.0
checkpoint above is the baseline of this family for Hungarian.

### How to run the pretrained baselines

On Google Colab with a GPU (the committed runs used an NVIDIA L4, as their
`run.json` records), open
[`notebooks/baselines_pretrained.ipynb`](notebooks/baselines_pretrained.ipynb),
run its setup section and then only the section of one system. The section
installs the packages, fetches the checkpoint, runs and scores the validation
split, runs and scores the test split once, and ends with an archive of that
system's result folders. The cells call these commands, shown for one system:

```bash
pip install -r requirements.txt -r requirements-pretrained.txt
pip install --no-deps -e .
python -m huasr.pretrained list                           # the four systems: checkpoint, revision, settings
python -m huasr.pretrained fetch canary-1b-v2             # the pinned checkpoint -> Hugging Face cache
python -m huasr.evaluate --system canary-1b-v2            # validation split
python -m huasr.evaluate --system canary-1b-v2 --split test --final-test    # test split, once
python -m huasr.score results/baselines/canary-1b-v2/validation results/baselines/canary-1b-v2/test
python -m huasr.report                                    # their rows in results/README.md
```

`fetch` is the only command that downloads a model; it takes the files that
are needed, not the other formats of the weights in the same repository, and
checks the weight file against a pinned size and SHA-256. A model whose
weights are not all found in the checkpoint is refused, so that no layer can
run with random values. What is committed after a run: the folders
`results/baselines/<system>/validation/` and `results/baselines/<system>/test/`
with `run.json`, `predictions.csv` and `scores.json`, and the READMEs after
`python -m huasr.report`. The files of this part are listed under
[Repository layout](#repository-layout).

## Metrics and why they fit Hungarian

All error rates come from one edit-distance alignment between reference and
system output (`src/huasr/metrics.py`):

error rate = (substitutions + deletions + insertions) / reference units

- **WER** (units: words) is the standard ASR metric and is what other work
  reports, so it is needed for comparison. Insertions are in the numerator
  only, so WER can exceed 100%. For Hungarian it is a harsh unit: a word with
  one wrong suffix, a compound written as two words, or one missing accent
  counts as a full error, exactly like a word that is completely wrong.
- **CER** (units: characters, including the space between words) gives partial
  credit. A wrong suffix costs one or two characters and a wrong word boundary
  costs one. It shows how much of the text is wrong, which WER hides in a
  language with long words. Its weakness is the reverse: a one-character error
  can change the meaning (*kor* "age", *kór* "illness", *kör* "circle"), and
  CER makes such an error look small.
- **Subword error rate** (units: tokens of one declared tokenizer; **not
  computed in this project**) lies between the two: a wrong suffix changes
  one token, not the whole word. Its value depends on the tokenizer, so the
  tokenizer is part of the scoring policy and the rate cannot be compared
  across tokenizers. `error_rate` in `metrics.py` accepts any tokenizer, but
  no tokenizer was chosen, so no run has this rate.

Reading WER and CER together separates "the model got the stem but not the
ending" (WER high, CER low) from "the model produced something else" (both
high). No single one of them is enough for an agglutinative language.

**Corpus-level rates.** Edit counts are summed over all clips and divided by
the summed reference units. The mean of per-clip percentages is not used: it
would give a one-word clip the same weight as a twenty-word clip, and a single
short clip with insertions can have a rate far above 100%.

**Diacritic-insensitive WER** is reported next to the WER
(`strip_diacritics=True`). The difference between the two shows how many
errors are accent errors only.

**Speed.** Real-time factor = processing time / audio duration, summed over
the clips of a split. Clips are processed one at a time and loading the model
is not counted; the GMM-HMM system is the exception, because its toolkit
decodes the whole split at once ([GMM-HMM baseline](#gmm-hmm-baseline-gmm-hmm-mfa)).
The machine is named in the run manifest and in the result tables. The runs
were made on different machines, so the real-time factors of two systems can
only be compared if both ran on the same one.

`src/huasr/score.py` computes all of this from the prediction file of a run:
WER, CER and diacritic-insensitive WER with their substitution, deletion and
insertion counts and 95% intervals, and the real-time factor, once for all
clips and once without the clips whose reference contains English number
words.

## Scoring policy

Declared before any system is scored, and applied identically to references
and system outputs by `normalize` in `src/huasr/text.py`:

1. Unicode NFC.
2. Lower case.
3. Every punctuation character and every symbol (Unicode categories P and S)
   is replaced by a space, so a hyphenated word counts as two words.
4. Control and format characters (soft hyphen, zero-width space) are deleted.
5. Whitespace is collapsed to single spaces.
6. Diacritics are kept: *a* and *á*, *o*, *ö* and *ő* are different letters.
7. Digits are not rewritten as words.

Reference text: the column `reference` of
`data/clean/voxpopuli/manifest.csv`, the repaired `normalized_text` of
VoxPopuli. It passes through the same `normalize` as the system output. For
the two corpora that were not scored, the declared reference columns were
`normalized_text` for FLEURS and `raw_text` (the sentence that was read) for
Common Voice.

A clip whose reference is empty after normalisation has no reference units,
so it cannot enter an error rate. The scorer counts such clips
(`clips_with_empty_reference` in `scores.json`) and leaves them out of the
rates; they stay in the prediction file.

## Comparing two systems

Two systems are compared on the same clips with a paired bootstrap that
resamples speakers (`paired_bootstrap` in `src/huasr/metrics.py`;
`python -m huasr.compare` applies it to two runs and refuses runs that do not
hold the same clips and references). Clips of one speaker share a voice and
a microphone, so they are not independent; resampling clips would make the
intervals too narrow. Each resample draws speakers with replacement, takes
all of their clips for both systems and recomputes the corpus-level
difference in error rate. The number of test speakers is small in VoxPopuli,
which widens the intervals; it is reported with every comparison.

One comparison was made: our fine-tuned model against `hf-whisper-small`,
the checkpoint it started from, on the validation split and on the test
split. Its result on the test split is under
[Results and discussion](#results-and-discussion); both splits are in
[`results/README.md`](results/README.md).

- H0: the new system is not better than the baseline (its error rate is not
  lower). The comparison declared as the main one is the CER of the
  fine-tuned model against the CER of the same checkpoint as published; WER
  and the WER without diacritics are tested the same way.
- Reported: the difference, its 95% percentile interval, and a one-sided
  p-value, the share of resamples in which the new system is not better
  (computed as (1 + count) / (1 + resamples), so it is never exactly zero).
- The number of resamples and the seed are the defaults `DEFAULT_RESAMPLES`
  and `DEFAULT_SEED` in `metrics.py`; the result is the same on every run.
- `bootstrap_interval` gives the interval of a single system's error rate in
  the same way.

Bootstrap intervals for ASR error rates follow Bisani and Ney (2004); paired
bootstrap resampling for comparing two systems follows Koehn (2004).

## Evaluation protocol

- One runner for every system (`src/huasr/evaluate.py`): it reads the clips
  of a split from the shards, checks them against the manifest, and passes
  the audio to the system. Baselines and our own models get the same clips.
  The GMM-HMM baseline is the one exception, because MFA decodes a folder at
  once: `src/huasr/gmm_hmm.py` writes the same clips of the same manifest to
  a folder, with the same check, and turns the output of MFA into the same
  two run files. It is scored by the same scorer.
- Every run leaves a run manifest and a prediction file with clip ids in
  `results/`; scores are computed from the prediction file by a separate
  script. [`results/README.md`](results/README.md) explains the folder and
  holds the result tables for both splits, with baselines and our own models
  in separate sections, and a table of all runs.
- Same clips, same references: every run records the SHA-256 of the raw and
  of the clean manifest it read. The note below each benchmark table states
  whether all its rows hold the same clips, speakers, reference words and
  clean manifest.
- Nothing is chosen on the test split. The validation split chose the
  checkpoint of the fine-tuning; the decoding settings are the ones declared
  in the code ([Systems](#systems)).
- **The test split was used once, after all settings were frozen.** Every
  system was run on it once, on 22 November 2026. The runner refuses
  `--split test` unless the flag `--final-test` is given and refuses to
  replace an existing run unless `--force` is given; the run manifest
  records the flag, and the table of runs in
  [`results/README.md`](results/README.md) shows it for every run.

## Reproduce

Tested with Python 3.13 on Windows. Python 3.12 or newer is required.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                  # makes the huasr package importable
python -m unittest                # unit tests (also run without the line above)
```

Raw data from VoxPopuli and FLEURS (no account needed):

```bash
python -m huasr.download voxpopuli --dry-run     # list the pinned files and their sizes
python -m huasr.download voxpopuli               # parquet files -> scratch/voxpopuli/
python -m huasr.download fleurs                  # test.tsv, test.tar.gz -> scratch/fleurs/
python -m huasr.subset voxpopuli                 # -> data/raw/voxpopuli/
python -m huasr.subset search                    # -> data/raw/voxpopuli/split_search.json, from selection.csv alone
python -m huasr.clean voxpopuli                  # -> data/clean/voxpopuli/
python -m huasr.subset fleurs                    # -> data/raw/fleurs/
python -m huasr.report                           # rewrite the generated tables of the READMEs
python -m huasr.subset unpack --raw data/raw/voxpopuli    # audio files -> scratch/audio/voxpopuli/
```

Evaluation (needs only the files of this repository and the model). The two
faster-whisper baselines on the validation split:

```bash
python -m huasr.evaluate --system faster-whisper-base     # validation split -> results/baselines/faster-whisper-base/validation/
python -m huasr.evaluate --system faster-whisper-small
python -m huasr.score --all                               # scores.json next to every run
python -m huasr.report                                    # result tables of the READMEs
```

The other systems are run in the same way, each with its own packages:
the GMM-HMM baseline below, the two Whisper small systems of the Hugging
Face implementation and the fine-tuning under
[Fine-tuning Whisper small](#fine-tuning-whisper-small-our-own-model), the
four pretrained baselines under
[How to run the pretrained baselines](#how-to-run-the-pretrained-baselines).

The final evaluation on the test split was made once, on 22 November 2026.
An existing test run is not replaced unless `--force` is given, so these
commands only work in a clone without the test results:

```bash
python -m huasr.evaluate --system faster-whisper-base --split test --final-test
python -m huasr.evaluate --system faster-whisper-small --split test --final-test
python -m huasr.gmm_hmm decode --split test --final-test --jobs 8    # in the conda environment of the GMM-HMM baseline
python -m huasr.gmm_hmm collect --split test                         # -> results/baselines/gmm-hmm-mfa/test/
python -m huasr.score --all
python -m huasr.report
```

The test commands of the other systems are in the two sections named above.

`evaluate` runs the validation split unless `--split` says otherwise, and it
refuses `--split test` without `--final-test`: the test split is for the
final evaluation only. `--limit N` makes a trial run on the first N clips,
which is marked as incomplete and kept out of the tables; an existing run is
only replaced with `--force`. The faster-whisper systems take their model
from the local Hugging Face cache, at the revision pinned in `REVISIONS` in
`src/huasr/systems.py`, and download nothing. If the model is not there,
fetch it once (`base` or `small`):

```bash
python -c "from faster_whisper import download_model; from huasr.systems import REVISIONS; download_model('small', revision=REVISIONS['small'])"
```

GMM-HMM baseline. It needs its own conda environment with MFA and Kaldi; the
steps are Python entry points and are the same on Windows, Linux and macOS:

```bash
conda env create -f environment-gmm-hmm.yml      # MFA 3.4.3 and the Kaldi build of the committed run
conda activate huasr-gmm-hmm
# The package must be importable in this environment: set the environment variable PYTHONPATH
# to src (as the committed run did), or run "pip install -e . --no-deps".
python -m huasr.gmm_hmm prepare                  # training corpus, lexicon, settings -> scratch/gmm_hmm/
python -m huasr.gmm_hmm train --jobs 8           # language model, then acoustic model
python -m huasr.gmm_hmm decode --jobs 8          # the audio of the validation split -> transcripts
python -m huasr.gmm_hmm collect                  # -> results/baselines/gmm-hmm-mfa/validation/
python -m huasr.score results/baselines/gmm-hmm-mfa/validation
python -m huasr.report
```

`prepare` and `decode` read the shards and the clean manifest of this
repository and download nothing. `--work` puts the work folder (several GB:
the corpus as MFA reads it, the models, the temporary files of MFA) somewhere
else; the committed run used a folder outside the repository. `prepare` and
`train` are run once. `decode` and `collect` take `--split`, so that the same
trained models can be run on the test split at the end; like `evaluate`,
`decode` refuses `--split test` without `--final-test`. `prepare
--train-speakers ID ...` and `decode --limit N` make a trial run on a few
speakers and the first N clips, which is marked as incomplete and kept out of
the tables. The time the committed run took is in the table of
[`results/README.md`](results/README.md).

The committed data was built with exactly these commands and their defaults.
`clean`, `search`, `report` and `unpack` need only the files of this
repository; `subset voxpopuli` needs the downloaded source files and also
writes `split_search.json` itself.

An interrupted download continues on the next run. `subset voxpopuli` takes
`--train-hours`, `--validation-hours`, `--test-hours` and `--seed`; both
builders refuse to replace existing data unless `--force` is given, and
building twice from the same files gives the same manifest and summary. The
builders read the source one parquet row group at a time.

Common Voice test set (kept outside the tracked files). Manual steps, once per
person:

1. Create an account at <https://mozilladatacollective.com> and complete the
   profile.
2. Open the
   [dataset page](https://mozilladatacollective.com/datasets/cmu5wq4u200dknq072xswzofb),
   read the licence and the conditions of use, and accept them there.
3. Either download `common-voice-scripted-speech-27-0-hungar-9f5b5529.tar.gz`
   in the browser, or create an API key in the profile settings and put it in
   the environment variable `MDC_API_KEY`. Never write the key into a file of
   this repository.

```bash
python -m huasr.download commonvoice --info                               # published size, checksum, licence
python -m huasr.download commonvoice --archive <downloaded .tar.gz> --extract
python -m huasr.download commonvoice --extract                            # instead: uses MDC_API_KEY
python -m huasr.subset commonvoice --release external/common_voice/cv-corpus-27.0-2026-09-11/hu
```

The download checks the size and SHA-256 of the archive, unpacks it into
`external/common_voice/` and prints the release folder to pass to
`subset commonvoice`. The Mozilla Data Collective limits the number of dataset
downloads per day
([API reference](https://mozilladatacollective.com/api-reference/docs)); a
finished download is kept and not requested again.

## Repository layout

Every tracked file and folder of the repository:

```text
README.md                    this file
LICENSE                      MIT licence for the code
requirements.txt             pinned versions of the packages the code imports (the last two only
                             for the faster-whisper systems)
requirements-train.txt       pinned packages of the fine-tuning and of the two Whisper small systems
                             of the Hugging Face implementation, on top of requirements.txt
requirements-pretrained.txt  pinned packages of the four pretrained baselines, on top of
                             requirements.txt (the versions of requirements-train.txt)
environment-gmm-hmm.yml      conda environment of the GMM-HMM baseline: MFA and the Kaldi build of
                             the committed run
pyproject.toml               lets "pip install -e ." install the package from src/
.gitignore                   Python caches, local environment, scratch/, and everything of Common Voice
.gitattributes               LF line endings for text; tar, FLAC, MP3 and PNG files are binary
src/huasr/                   the package
  __init__.py                package marker, language code
  text.py                    scoring policy: text normalisation, word and character units
  metrics.py                 edit-distance alignment, corpus-level WER/CER, bootstrap interval,
                             paired bootstrap over speakers
  download.py                pinned source files of VoxPopuli and FLEURS with size and SHA-256 check;
                             Common Voice archive into external/ only
  audio.py                   how audio is stored: WAV inspection, 16-bit FLAC, pass-through of
                             compressed input
  shards.py                  tar shards below the file-size limit, manifest, unpacking with hash check
  subset.py                  VoxPopuli: seeded subset whose splits share no speaker and no recording
                             session, selection log, search record; FLEURS: test split; Common
                             Voice: list of test clips outside the tracked files
  repair.py                  restores the letter that the corpus normalisation deleted from the
                             reference, only where raw_text proves it
  clean.py                   data/raw -> data/clean: repaired references, repair log, removal log,
                             flags (English number words, repeated references, characters per
                             second), the numbers to report for the split
  systems.py                 the contract every system fulfils (name, group, description, transcribe)
                             and the list of all systems; Whisper base and small through
                             faster-whisper, at pinned revisions
  whisper_hf.py              Whisper small through transformers, as published (baseline) and
                             fine-tuned (our model): pinned checkpoint, fixed decoding, clips longer
                             than the window; the command fetch
  pretrained.py              the four pretrained baselines (wav2vec 2.0 + CTC, Canary, Parakeet TDT,
                             Qwen3-ASR): pinned checkpoints, language settings, decoding, long
                             clips; the commands list and fetch
  gmm_hmm.py                 GMM-HMM baseline with the Montreal Forced Aligner: corpus folders,
                             graphemic lexicon, training and decoding with MFA, its output as
                             run.json and predictions.csv; refuses the test split without --final-test
  finetune.py                our own model: choice of the clips, target text, dataset, collator,
                             selection metric, training with Seq2SeqTrainer, best checkpoint,
                             training record; refuses the test split
  evaluate.py                runs one system on one split: run manifest and prediction file in
                             results/; refuses the test split without --final-test
  score.py                   prediction file -> scores.json: WER, CER, diacritic-insensitive WER with
                             edit counts and speaker-level intervals, real-time factor, with and
                             without the clips flagged for English number words
  compare.py                 two runs on the same clips: paired bootstrap over speakers ->
                             comparison_with_<baseline>.json
  report.py                  writes every generated block of the READMEs from the summary, run,
                             score, comparison and training files: data tables, result tables for
                             both splits, the benchmark on the test split, the paired comparison,
                             the list of runs, the fine-tuning plan and run
notebooks/                   Google Colab notebooks, stored without output; they clone this
                             repository and call the commands of the package
  finetune_whisper_small.ipynb   zero-shot baseline, fine-tuning, evaluation and comparison on the
                                 validation split, archive of the result files; never the test split
  baselines_pretrained.ipynb     shared setup, then one self-contained section per pretrained
                                 baseline: validation split, test split once, archive of the results
tests/                       unit tests (python -m unittest)
  __init__.py                puts src/ on the import path; skips tests whose package is missing
  fixtures.py                generated signals and WAV files used by the tests
  fake_transformers.py       stand-in for the parts of transformers that the fine-tuning calls
  test_text.py               normalisation rules
  test_metrics.py            alignment, WER/CER worked examples, bootstrap
  test_audio.py              WAV formats, 16-bit conversion, FLAC round trip, pass-through
  test_shards.py             shard size limit, identical rebuild, unpacking and tamper detection
  test_subset.py             sessions, assignment of speakers and sessions, leak check, builders for
                             VoxPopuli (rows and parquet files), FLEURS and the Common Voice list,
                             on generated sources
  test_download.py           pinned files, checksum check, resume, Common Voice kept out of the
                             tracked files; no network
  test_repair.py             reference repair: what is restored, what is left alone
  test_clean.py              clean manifest, flags, repair and removal logs, summary
  test_report.py             generated data tables, the FLEURS table and the table of the
                             fine-tuning run; the READMEs of this repository are up to date, name
                             every file, and their discussion agrees with the committed results
  test_systems.py            checkpoint id and revision from the cache path, declared settings; no
                             model is loaded
  test_whisper_hf.py         the two Whisper small systems: registration, pinned checkpoint,
                             decoding settings, long clips; stand-in model
  test_pretrained.py         the four pretrained baselines: registration, pinned checkpoints,
                             language checks, decoding calls, missing weights, runner and scorer;
                             stand-ins for processors and models
  test_gmm_hmm.py            GMM-HMM baseline without MFA: graphemic lexicon, corpus folders, the
                             conversion of MFA output into the run files, its tables
  test_finetune.py           fine-tuning without a model: target text, clips left out, settings,
                             dataset, collator, metric, best checkpoint, the plan table
  test_finetune_run.py       the training command from start to end with the stand-in for
                             transformers
  test_evaluate.py           runner with a stand-in system on generated data: prediction file, run
                             manifest, refusal of the test split without the flag
  test_score.py              scorer on hand-made prediction files (rates, edit counts, intervals,
                             real-time factor, flagged clips); the result tables, the benchmark
                             tables and the table of runs
  test_compare.py            the paired comparison on hand-made prediction files; the comparison
                             table and the distance to the best baseline
  test_notebooks.py          the notebooks as files: well-formed, without output, calling only what
                             exists; only the pretrained-baselines notebook runs the test split,
                             once per system
data/raw/                    raw data, explained in data/raw/README.md
  README.md                  what is stored, the split, the file formats, sources and licences
  voxpopuli/                 clips/*.tar (FLAC audio in shards), manifest.csv, summary.json,
                             selection.csv, split_search.json
  fleurs/                    clips/*.tar (FLAC audio in shards), manifest.csv, summary.json: the
                             FLEURS test split; no system has been scored on it
data/clean/                  clean data, explained in data/clean/README.md
  README.md                  the reported split, the reference repair, the flags, what is not done
  voxpopuli/                 manifest.csv (with the repaired reference), repair_log.csv,
                             removal_log.csv, summary.json
results/                     runs and result tables, explained in results/README.md
  README.md                  what a run leaves behind, all runs, result tables for both splits
  baselines/<system>/<split>/    one run of one baseline on one split (validation, test):
                                 run.json, predictions.csv, scores.json. <system> is one of
                                 faster-whisper-base, faster-whisper-small, hf-whisper-small,
                                 gmm-hmm-mfa, wav2vec2-xlsr-53-hungarian, canary-1b-v2,
                                 parakeet-tdt-0.6b-v3, qwen3-asr-1.7b
  own/hf-whisper-small-finetuned/    our fine-tuned model
    training.json                    the training record of the fine-tuning run
    <split>/                         run.json, predictions.csv, scores.json and
                                     comparison_with_hf-whisper-small.json (validation, test)
contributions/               one file per team member, written by that member:
                             <NAME>_<STUDENT_ID>_<GITHUB_USERNAME>.md
```

Not tracked (see `.gitignore`): `scratch/` for downloaded source files,
unpacked audio, the fine-tuned weights and the work folder of the GMM-HMM
baseline; `external/` for Common Voice. No model weights, caches or
temporary files are tracked.

## Status

State of the repository on 22 November 2026.

Implemented and tested (`python -m unittest`; no test loads a model, uses
the network or needs a GPU):

- Scoring policy (`src/huasr/text.py`).
- Edit-distance alignment, corpus-level WER and CER with substitution,
  deletion and insertion counts, error rate over any token unit, bootstrap
  interval and paired bootstrap over speakers (`src/huasr/metrics.py`).
- Audio storage rules, shards and unpacking (`src/huasr/audio.py`,
  `src/huasr/shards.py`).
- Builders for VoxPopuli, FLEURS and the Common Voice clip list
  (`src/huasr/subset.py`), tested on generated sources.
- Reference repair, clean step and README tables (`src/huasr/repair.py`,
  `src/huasr/clean.py`, `src/huasr/report.py`).
- Download script (`src/huasr/download.py`), tested without network.
- Evaluation: system contract (`src/huasr/systems.py`), runner with the
  guard on the test split (`src/huasr/evaluate.py`), scorer
  (`src/huasr/score.py`), paired comparison (`src/huasr/compare.py`) and
  result tables (`src/huasr/report.py`), tested with stand-in systems.
- The systems: Whisper through faster-whisper (`src/huasr/systems.py`) and
  through `transformers` (`src/huasr/whisper_hf.py`), the four pretrained
  baselines (`src/huasr/pretrained.py`), the GMM-HMM baseline
  (`src/huasr/gmm_hmm.py`) and the fine-tuning (`src/huasr/finetune.py`),
  tested with stand-ins for the models, for `transformers` and for MFA.

Done on the real data:

- **Data.** All six VoxPopuli files were downloaded with `huasr.download`
  and matched their pinned sizes and SHA-256 values (16 November 2026). The
  subset was rebuilt with the default settings so that its splits share no
  speaker and no session (18 November 2026); `unpack` restored every clip
  from the shards with a matching SHA-256; the clean step and the report
  were run. The FLEURS test split was built from the real files and
  committed (21 November 2026).
- **Baselines.** Whisper base and small through faster-whisper (validation
  split on 18 November 2026); the GMM-HMM baseline, trained on the whole
  training split with MFA on Windows (21 November 2026); Whisper small
  through `transformers`, wav2vec 2.0 + CTC, Canary, Parakeet TDT and
  Qwen3-ASR on Google Colab (22 November 2026).
- **Our own model.** Whisper small was fine-tuned once on the training split
  (22 November 2026) and compared with the checkpoint it started from.
- **Final evaluation.** Every system above was run once on the whole test
  split on 22 November 2026 and scored. The date, the machine and the
  `--final-test` flag of every run are in the table of runs in
  [`results/README.md`](results/README.md).

Not done:

- **No system has been scored on FLEURS**, although its test split is in the
  repository.
- **Common Voice has not been downloaded with the scripts, built or
  scored.** Only `commonvoice --info` was run against the real service.
- **No subword error rate**: no tokenizer was chosen.
- **HuBERT + CTC**: not run, because no Hungarian checkpoint was found
  ([details](#hubert--ctc-is-not-run)).
- **No cleaning rule removes clips.** The references with English number
  words are flagged and reported separately, not removed or corrected.
- **No paired comparison other than** the fine-tuned model against
  `hf-whisper-small`; the fine-tuning was run once, with one seed.
- **No controlled measurement of speed**: the real-time factors come from
  different machines.
- The error analysis, the out-of-vocabulary analysis and the limitations
  section are not written yet (see the end of
  [Results and discussion](#results-and-discussion)).

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). The licence covers the code only; the data
keeps the licences and conditions listed under [Data](#data) and in
[`data/raw/README.md`](data/raw/README.md).

- Wang, C., Riviere, M., Lee, A., Wu, A., Talnikar, C., Haziza, D.,
  Williamson, M., Pino, J. and Dupoux, E. (2021). VoxPopuli: A Large-Scale
  Multilingual Speech Corpus for Representation Learning, Semi-Supervised
  Learning and Interpretation. *ACL 2021*, 993-1003.
  <https://aclanthology.org/2021.acl-long.80>. The recordings come from the
  European Parliament, which we thank for creating and sharing them
  ([legal notice](https://www.europarl.europa.eu/legal-notice/en/)).
- Conneau, A., Ma, M., Khanuja, S., Zhang, Y., Axelrod, V., Dalmia, S.,
  Riesa, J., Rivera, C. and Bapna, A. (2022). FLEURS: Few-shot Learning
  Evaluation of Universal Representations of Speech.
  <https://arxiv.org/abs/2205.12446>
- Ardila, R., Branson, M., Davis, K., Henretty, M., Kohler, M., Meyer, J.,
  Morais, R., Saunders, L., Tyers, F. M. and Weber, G. (2020). Common Voice: A
  Massively-Multilingual Speech Corpus. *Proceedings of LREC 2020*,
  4211-4215. <https://arxiv.org/abs/1912.06670>
- Bisani, M. and Ney, H. (2004). Bootstrap estimates for confidence intervals
  in ASR performance evaluation. *ICASSP 2004*.
  <https://doi.org/10.1109/ICASSP.2004.1326009>
- Koehn, P. (2004). Statistical significance tests for machine translation
  evaluation. *EMNLP 2004*. <https://aclanthology.org/W04-3250/>

## Team

| Member | Student ID | GitHub |
| --- | --- | --- |
| Berkay Hakan Sadi | 2309011065 | [hakanstarlord](https://github.com/hakanstarlord) |
| Fatih Safran Dündar | 2309011133 | [fatihsafran](https://github.com/fatihsafran) |
| Aamer mashar sanjer | 2309015705 | [aamersanjer0-a11y](https://github.com/aamersanjer0-a11y) |
| Kaan Kalkan | 220901594 | [kaankalkan03](https://github.com/kaankalkan03) |
| hilal ibrahimoglu | 229920269 | [hilalibrahimoglu34](https://github.com/hilalibrahimoglu34) |
| MOOAZ SADIK | 2309015405 | [moathkhaled2k5](https://github.com/moathkhaled2k5) |

Each member's own record is in [`contributions/`](contributions/).
