# data/raw

Raw data of the project: the audio and transcripts as obtained from the
sources, selected and packed by `python -m huasr.subset`. Nothing in this
folder is edited by hand. The tables below are written by
`python -m huasr.report` from the JSON files the builder generates.

| Folder | Corpus | Role | Licence | State |
|---|---|---|---|---|
| `voxpopuli/` | VoxPopuli, Hungarian transcribed set: a subset whose splits share no speaker and no recording session | training, development and test data of every result of the project | CC0-1.0 | built; every system has been scored on it |
| `fleurs/` | FLEURS, `hu_hu`: the whole official test split | extra test set | CC BY 4.0 | built; no system has been scored on it |

Common Voice was planned as a third test set, but **it is never stored
here**. Its dataset page forbids re-hosting or re-sharing the dataset, so its
files would stay in `external/` on each member's machine (ignored by git).
The scripts refuse to write Common Voice files into this folder. Common
Voice has not been downloaded with them, and no result uses it.

## VoxPopuli: what is in `voxpopuli/`

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

Every clip of the subset comes from a mono 32-bit float WAV file and is
stored as 16-bit FLAC with the sample rate of the source; `summary.json`
lists the encodings found (`audio`) and how many samples had to be limited to
the 16-bit range.

### Speakers and sessions

A speaker is the `speaker_id` of the corpus. A recording session is the
`session_id` of the corpus: VoxPopuli cuts all clips of one session id out of
one recording file, and a clip id is
`<session_id>-<language>_<date>-<time of the speech>_<number>`. The clip id up
to the first underscore is stored as `session_id` in the manifest. In the
Hungarian data a session is one numbered part of a plenary sitting day.

### The official split, and why it is not used as it is

<!-- generated: voxpopuli-official-split -->
| Official split | Clips | Hours | Speakers | Sessions | Clips without speaker id |
|---|---|---|---|---|---|
| train | 18,120 | 54.09 | 47 | 1,537 | 601 |
| validation | 1,076 | 3.25 | 26 | 141 | 16 |
| test | 1,110 | 3.11 | 46 | 154 | 23 |

Speakers shared by two official splits: train and validation 21, train and test 27, validation and test 18. Sessions shared by two official splits: train and validation 90, train and test 93, validation and test 14.

Source: `data/raw/voxpopuli/summary.json`.
<!-- end generated -->

The same speakers and the same sessions occur in several official splits,
and some clips have no speaker id. The builder therefore makes its own split:
every speaker and every session is in exactly one split, and a clip is only
used in the official split it comes from. The rules are written out in the
main README and in the module text of `src/huasr/subset.py`. The build stops
if a speaker or a session is left in two splits.

### What the rules left out

<!-- generated: voxpopuli-removed -->
| Rule that left clips out | official train | official validation | official test |
|---|---|---|---|
| unknown speaker | 601 (1.51 h) | 16 (0.08 h) | 23 (0.07 h) |
| speaker assigned to other split | 10,579 (31.35 h) | 172 (0.51 h) | 155 (0.52 h) |
| session assigned to other split | 756 (2.30 h) | 24 (0.06 h) | 35 (0.10 h) |
| beyond validation target | 0 (0.00 h) | 38 (0.11 h) | 0 (0.00 h) |
| beyond train target | 1,587 (4.92 h) | 0 (0.00 h) | 0 (0.00 h) |

Source: `data/raw/voxpopuli/summary.json`.
<!-- end generated -->

`selection.csv` lists every source clip with its decision, so each of these
clips can be named.

### What other weights would keep

Speakers and sessions go to the split in which they keep the most audio; a
second of test audio and a second of validation audio count several times as
much as a second of training audio. The two weights are searched on a fixed
grid. A higher test weight gives a larger test split and leaves less audio for
training. The pair is taken that leaves the minimum of test audio, the
validation target and the training target, and among those gives the most
test audio. `split_search.json` has the result of every pair;
`python -m huasr.subset search` rewrites it from `selection.csv` alone.

<!-- generated: voxpopuli-search -->
| Test weight | Validation weight | Test hours (speakers, sessions) | Validation hours (speakers) | Training hours left (speakers) | All targets met |  |
|---|---|---|---|---|---|---|
| 36 | 40 | 2.41 (34, 104) | 2.61 (15) | 18.92 (20) | yes | taken |
| 32 | 36 | 2.37 (33, 102) | 2.53 (14) | 22.85 (22) | yes |  |
| 32 | 40 | 2.37 (33, 102) | 2.61 (15) | 20.20 (21) | yes |  |
| 32 | 44 | 2.27 (32, 97) | 2.82 (17) | 15.09 (20) | yes |  |
| 28 | 36 | 2.23 (31, 90) | 2.53 (14) | 26.92 (24) | yes |  |
| 32 | 56 | 2.16 (31, 90) | 2.89 (18) | 15.10 (20) | yes |  |
| 24 | 36 | 2.12 (30, 85) | 2.61 (15) | 26.94 (24) | yes |  |
| 28 | 40 | 2.12 (30, 85) | 2.69 (16) | 24.27 (23) | yes |  |
| 28 | 44 | 2.12 (30, 85) | 2.82 (17) | 19.02 (22) | yes |  |
| 24 | 40 | 2.09 (30, 84) | 2.71 (16) | 24.27 (23) | yes |  |
| 24 | 44 | 2.09 (30, 84) | 2.84 (17) | 19.02 (22) | yes |  |
| 28 | 56 | 2.09 (30, 84) | 2.89 (18) | 17.09 (21) | yes |  |

625 weight pairs were tried (weights 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 80, 96, 128); 16 of them leave at least 2 h of test audio, 2.5 h of validation audio and 14 h of training audio. The table shows the different results among these, and the pair that was taken.

Source: `data/raw/voxpopuli/split_search.json`.
<!-- end generated -->

## Files in each folder

| File | Content |
|---|---|
| `clips/<split>-NNN.tar` | The clips as FLAC files (mono, 16-bit, sample rate of the source), packed into uncompressed tar shards. Every shard stays below `MAX_SHARD_BYTES` (`src/huasr/shards.py`), which is below GitHub's file-size limit. `<split>` is `train`, `validation` or `test`. |
| `manifest.csv` | One row per clip, columns below. |
| `summary.json` | VoxPopuli: source (repository, revision, files with size and SHA-256), seed, targets and the weights taken; clips, hours, speakers and sessions per split for the official split and for the subset; the speakers, sessions and sitting days shared by each pair of splits; the audio removed by each rule; the audio encodings found; clips that could not be read. FLEURS: source, the clips, hours, sentences and shards of the test split, the audio encodings found, clips that could not be read, and audio and transcripts without a partner. |
| `selection.csv` (VoxPopuli) | One row per clip of the source: `clip_id`, `source_split`, `speaker_id`, `session_id`, `duration_ms`, `decision` (`kept`, or the rule that left the clip out). |
| `split_search.json` (VoxPopuli) | The targets, the weights that were tried, and what every weight pair keeps in each split. |

Manifest columns:

| Column | Meaning |
|---|---|
| `clip_id` | Id of the clip in the source (`audio_id` in VoxPopuli, the file name without extension in FLEURS). |
| `split` | `train`, `validation` or `test` in this project. |
| `speaker_id` | Speaker id of the source; empty for FLEURS, which has none. |
| `session_id` | Recording session of the clip (VoxPopuli); empty for FLEURS, where every clip is its own recording. |
| `sentence_id` | Sentence id of FLEURS (several recordings share a sentence); empty for VoxPopuli. |
| `gender` | As given by the source. |
| `raw_text`, `normalized_text` | The two transcript fields of the source, unchanged. For VoxPopuli the reference used for scoring is the repaired column `reference` of `data/clean/voxpopuli/manifest.csv`, because `normalized_text` has lost a letter (see `data/clean/README.md`). |
| `duration_ms` | Length of the audio, from the number of samples. |
| `file`, `shard` | Name of the FLAC file and the tar shard that holds it. |
| `bytes`, `sha256` | Size and SHA-256 of the stored FLAC file. |
| `source_split` | Official split the clip comes from (always equal to `split`). |
| `source_encoding`, `source_bytes`, `source_sha256` | Encoding, size and SHA-256 of the source audio file, so that a stored clip can be traced to the exact source bytes. |
| `limited_samples` | Number of samples that were outside the 16-bit range and had to be limited. |

`python -m huasr.subset unpack --raw data/raw/<folder>` restores the FLAC
files into `scratch/audio/<folder>/<split>/` (not tracked) and checks every
SHA-256 against the manifest.

## Audio format

The sources ship uncompressed WAV files. They are stored here as FLAC, the
standard lossless format for 16-bit audio, by `src/huasr/audio.py`:

- 16-bit PCM is stored without any loss.
- 32-bit float is quantised to 16 bits first: sample × 32768, rounded, limited
  to [-32768, 32767]. This is the only change made to the audio.
- Other encodings are refused and listed in `summary.json`; nothing is
  resampled, and input that is already compressed would be kept unchanged.

Every FLAC file carries the MD5 of its decoded samples in its header.

## VoxPopuli: source and licence

- Source: <https://huggingface.co/datasets/facebook/voxpopuli>, folder `hu/`
  (parquet files); revision, sizes and SHA-256 values are pinned in
  `src/huasr/download.py` and repeated in `summary.json`. Corpus: Wang et al.
  (2021), <https://aclanthology.org/2021.acl-long.80>.
- Licence: CC0-1.0, as stated on the dataset card, which also points to the
  [legal notice of the European Parliament](https://www.europarl.europa.eu/legal-notice/en/)
  for the original recordings. The speakers are members of parliament
  speaking in public sessions.

## FLEURS: what is in `fleurs/`

Built with `python -m huasr.subset fleurs` from the pinned source files and
committed on 21 November 2026. It was meant as an extra test set of read
speech; no system has been scored on it, and it has no clean step.

<!-- generated: fleurs-subset -->
| Split | Clips | Hours | Sentences | Shards | Size (MB) |
|---|---|---|---|---|---|
| test | 905 | 3.07 | 348 | 2 | 163.6 |

Source files: `google/fleurs` at revision `70bb2e84b976`. Encoding of the source audio: `wav_f32le` 905; stored as: `flac_s16` 905. Samples limited to the 16-bit range: 6. Clips that could not be read: 0; audio files without a transcript: 0; transcripts without audio: 0.

Source: `data/raw/fleurs/summary.json`.
<!-- end generated -->

FLEURS has no speaker ids and no sessions, so `speaker_id` and `session_id`
are empty in its manifest; `sentence_id` names the sentence a clip reads.

- Source: <https://huggingface.co/datasets/google/fleurs>, files
  `data/hu_hu/test.tsv` and `data/hu_hu/audio/test.tar.gz`; revision, sizes
  and SHA-256 values are pinned in `src/huasr/download.py`. Corpus: Conneau
  et al. (2022), <https://arxiv.org/abs/2205.12446>.
- Licence and attribution: FLEURS by Conneau et al. (Google), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made by
  us: WAV audio stored as 16-bit FLAC in tar archives, transcripts copied
  into `manifest.csv`. No text is changed. Anyone who reuses `fleurs/` must
  keep this attribution.
- Selection: none; every clip of the official test split that has a
  transcript is kept. Clips without transcript and transcripts without audio
  are counted in `summary.json` and in the note below the table above.

## Reproduce

```bash
python -m huasr.download voxpopuli      # -> scratch/voxpopuli/
python -m huasr.subset voxpopuli        # -> data/raw/voxpopuli/ (shards, manifest, summary, selection, search)
python -m huasr.subset search           # split_search.json again, from selection.csv alone
python -m huasr.report                  # tables of the READMEs
python -m huasr.download fleurs         # -> scratch/fleurs/
python -m huasr.subset fleurs           # -> data/raw/fleurs/
```

GitHub rejects single files of 100 MB or more and pushes above 2 GB, so the
shards are committed in several pushes.
