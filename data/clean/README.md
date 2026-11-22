# data/clean

Clean data: the clips of `data/raw/voxpopuli/` with a repaired reference
transcript, flags for known problems, and logs of everything that was changed
or left out. All of it is written by `python -m huasr.clean voxpopuli` from
`data/raw/voxpopuli/` alone; nothing is edited by hand. The tables below are
written by `python -m huasr.report` from `voxpopuli/summary.json`.

The audio is not copied. A clean clip is found through its `file` and `shard`
in the shards of `data/raw/voxpopuli/clips/`. Common Voice is never stored in
this repository, clean or raw.

## The split, as it has to be reported

Counted from the manifest of this folder, not taken over from the builder:

<!-- generated: voxpopuli-split-report -->
| Split | Speakers | Sessions | Utterances | Hours | References that occur word for word in the training split |
|---|---|---|---|---|---|
| train | 20 | 626 | 4,597 | 14.00 | - |
| validation | 15 | 112 | 826 | 2.50 | 6 of 826 |
| test | 34 | 104 | 897 | 2.41 | 7 of 897 |

Speakers in two splits: train and validation 0, train and test 0, validation and test 0. Sessions in two splits: train and validation 0, train and test 0, validation and test 0.

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

A speaker is the `speaker_id` of the corpus and a session its `session_id`
(one recording file that the corpus cut into clips; see `data/raw/README.md`).
The last column compares references after the scoring policy; such a clip has
the flag `repeats_training_reference`.

## Files in `voxpopuli/`

| File | Content |
|---|---|
| `manifest.csv` | One row per clip: every column of `data/raw/voxpopuli/manifest.csv`, unchanged, followed by `reference` (the repaired `normalized_text`), `reference_status` (`restored`, `unchanged`, `no_raw_text` or `unusual_text`), `restored_words`, `unaligned_words`, `english_number_words` (how many English number words the reference contains), `repeats_training_reference` (validation and test clips: 1 if the same reference occurs in the training split, else 0; empty for training clips) and `chars_per_second`. |
| `repair_log.csv` | One row per restored word: `clip_id`, `split`, `word` (position of the word in `normalized_text`, counted from 0), `before`, `after`. |
| `removal_log.csv` | One row per clip of the source that is not in the data: `clip_id`, `source_split`, `speaker_id`, `session_id`, `duration_ms`, `stage`, `reason`. Every row has the stage `selection`: the rules of the split in `data/raw/README.md` (no speaker id, speaker or session in another split, beyond a target). |
| `summary.json` | The numbers of the table above, the counts of the repair, the letters that `normalized_text` lacks, the flags per split, characters per second per split, and the clips removed by each rule. |

## Reference repair

VoxPopuli gives two transcripts per clip. `raw_text` is the text as
published; `normalized_text` was derived from it by the corpus (lower case,
no punctuation, numbers written out as words) and is the usual reference. In
the Hungarian part that normalisation deleted every letter **í**: `raw_text`
has *javítani* where `normalized_text` has *javtani*. The letter is ordinary
Hungarian spelling. A reference without it counts every correct í of a system
as an error, and a model trained on it would learn to drop the letter.

`src/huasr/repair.py` restores the letter only where `raw_text` proves it:

1. Both texts are split into words by the scoring policy and aligned by edit
   distance; two words count as equal when they are equal after í is removed
   from both.
2. A word of `normalized_text` is replaced by its partner from `raw_text`
   only if it is exactly that partner without its í. Everything else in the
   string is kept.
3. Words without such a partner are left unchanged and counted: numbers that
   `raw_text` writes in digits, words that differ for another reason, and
   every word of a clip that has no `raw_text`.

No text is invented: for every clip the code checks that the reference
without í equals `normalized_text` without í.

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

Read with care:

- The other letters in the second table were lost in the same way. They are
  letters of other languages in names (the first columns show how rare they
  are). They are reported and **not** restored: the rule above is limited to
  the one letter that belongs to the Hungarian alphabet.
- A clip without `raw_text` cannot be checked, so its reference may still
  lack the letter (`reference_status` is `no_raw_text`). These clips are kept
  as they are, in the data and in every result.
- A word that could not be aligned may also lack the letter, for example a
  number written out as a word. Nothing proves it, so it is left as it is.

## Numbers written out in English

The corpus normalisation turned digits into English words inside the
Hungarian text: `raw_text` *2008*, `normalized_text` *two thousand and eight*.
The speaker says the number in Hungarian, so such a reference is wrong for
every system and a wrong target for training. `english_number_words` counts
the English number words (*zero* to *nineteen*, the tens, *hundred*,
*thousand*, *million*, *billion*) of each reference. The table separates the
cases: `raw_text` has digits (the normalisation did it), the clip has no
`raw_text` (cannot be checked), or `raw_text` has the English word itself
(for example an English name that was really spoken).

<!-- generated: voxpopuli-number-words -->
| English number words | train | validation | test |
|---|---|---|---|
| References with English number words | 286 | 55 | 35 |
| of these: `raw_text` has the number in digits | 223 | 46 | 28 |
| of these: the clip has no `raw_text` | 62 | 9 | 6 |
| of these: `raw_text` has no digits (English words in the speech itself) | 1 | 0 | 1 |

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

The references are flagged only; no clip is removed from the data and no
reference is rewritten. The flag is used in two places. The scorer reports
every figure once for all clips and once without the flagged clips
(`results/README.md`). The fine-tuning leaves the flagged clips out of its
training clips, because their text is a wrong target; the validation and
test clips keep them.

## Characters per second

A transcript that does not belong to its audio tends to have too many or too
few characters for the length of the clip. `chars_per_second` is the number
of characters of the reference after the scoring policy (spaces included)
divided by the duration.

<!-- generated: voxpopuli-chars-per-second -->
| Split | Clips | min | 1% | 5% | 25% | median | 75% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| train | 4597 | 2.29 | 9.53 | 11.15 | 13.03 | 14.35 | 15.77 | 17.92 | 20.25 | 32.14 |
| validation | 826 | 3.62 | 9.11 | 11.67 | 13.56 | 15.06 | 16.56 | 18.79 | 21.06 | 28.85 |
| test | 897 | 6.78 | 9.63 | 11.6 | 13.92 | 15.29 | 16.58 | 18.5 | 20.41 | 25.0 |

Source: `data/clean/voxpopuli/summary.json`.
<!-- end generated -->

No rule uses this number: no threshold was chosen, and no clip is removed for
it. The table is a description of the data.

## Not applied

No clip is removed by a cleaning rule (`summary.json`,
`removed_by_cleaning_rules`). The clips that are not in the data were left
out by the rules of the split, which `removal_log.csv` lists. These removal
rules were planned and are not implemented:

- Removing validation or test clips whose reference also occurs in the
  training split. They are flagged (`repeats_training_reference`) and
  counted in the first table, and they stay in every result.
- Removing clips whose transcript does not match the audio (characters per
  second, or another signal chosen on training and validation data).
- Removing clips that are too long for the models. The systems cut such
  clips themselves, and the fine-tuning leaves them out of its training
  clips (main README).
- Removing transcripts that contain letters outside the Hungarian alphabet.

No transcript is corrected beyond the repair above. Nobody in the group reads
Hungarian, so a clip that looks wrong may only be removed, never relabelled,
and a model's output is never used as a reference. In this project no clip
was removed for that reason.

## Reproduce

```bash
python -m huasr.clean voxpopuli     # data/raw/voxpopuli/ -> data/clean/voxpopuli/
python -m huasr.report              # tables of the READMEs
```

Running the first command again gives the same files.
