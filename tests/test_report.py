"""Tests for the generated tables of the READMEs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from huasr import report
from tests import REPOSITORY

ZERO = {"train-validation": 0, "train-test": 0, "validation-test": 0}
RAW = {
    "seed": 0,
    "target_hours": {"train": 14.0, "validation": 2.5, "test": 2.0},
    "weights": {"validation": 40, "test": 36},
    "official_split": {
        "train": {"clips": 12000, "hours": 30.5, "speakers": 40, "sessions": 1200, "clips_without_speaker_id": 300},
        "validation": {"clips": 800, "hours": 2.25, "speakers": 20, "sessions": 150, "clips_without_speaker_id": 10},
        "test": {"clips": 900, "hours": 2.5, "speakers": 30, "sessions": 160, "clips_without_speaker_id": 20},
        "speakers_in_two_splits": {"train-validation": 15, "train-test": 21, "validation-test": 12},
        "sessions_in_two_splits": {"train-validation": 60, "train-test": 70, "validation-test": 9},
    },
    "subset": {
        "train": {"clips": 5000, "hours": 14.0, "speakers": 25, "sessions": 480, "shards": 11,
                  "shard_bytes": 957_600_000},
        "validation": {"clips": 300, "hours": 0.875, "speakers": 5, "sessions": 40, "shards": 1,
                       "shard_bytes": 59_850_000},
        "test": {"clips": 700, "hours": 2.125, "speakers": 22, "sessions": 90, "shards": 2,
                 "shard_bytes": 145_350_000},
        "speakers_in_two_splits": ZERO, "sessions_in_two_splits": ZERO,
    },
    "removed": {"unknown_speaker": {"train": {"clips": 300, "hours": 0.75}, "validation": {"clips": 10, "hours": 0.03},
                                    "test": {"clips": 20, "hours": 0.06}}},
}


def search_row(test_weight, validation_weight, test, validation, train, met, taken=False):
    def part(hours, speakers, sessions):
        return {"clips": 100, "hours": hours, "speakers": speakers, "sessions": sessions}

    return {"test_weight": test_weight, "validation_weight": validation_weight, "test": part(*test),
            "validation": part(*validation), "train": part(*train), "targets_met": met, "taken": taken}


SEARCH = {
    "seed": 0, "minimum_test_hours": 2.0, "validation_target_hours": 2.5, "training_target_hours": 14.0,
    "weights_tried": [1, 2, 4], "taken": {"validation": 4, "test": 2},
    "rows": [
        search_row(1, 1, (0.5, 5, 20), (1.0, 4, 30), (40.0, 30, 900), False),
        search_row(2, 2, (2.125, 22, 90), (2.75, 15, 100), (25.5, 25, 600), True),
        search_row(2, 4, (2.25, 23, 95), (2.875, 16, 110), (15.25, 20, 480), True, taken=True),
        search_row(4, 2, (2.25, 23, 95), (2.875, 16, 110), (15.25, 20, 480), True),   # same result as the row above
        search_row(4, 4, (2.75, 30, 120), (2.0, 12, 80), (9.5, 15, 300), False),
    ],
}
CLEAN = {
    "splits": {"train": {"clips": 5000, "hours": 14.0, "speakers": 25, "sessions": 480},
               "validation": {"clips": 300, "hours": 0.875, "speakers": 5, "sessions": 40},
               "test": {"clips": 700, "hours": 2.125, "speakers": 22, "sessions": 90}},
    "speakers_in_two_splits": ZERO, "sessions_in_two_splits": ZERO,
    "references_also_in_training": {"validation": {"clips": 3, "of": 300}, "test": {"clips": 1, "of": 700}},
    "english_number_words": {
        "train": {"clips": 250, "raw_text_has_digits": 200, "no_raw_text": 48, "raw_text_without_digits": 2},
        "validation": {"clips": 20, "raw_text_has_digits": 15, "no_raw_text": 5, "raw_text_without_digits": 0},
        "test": {"clips": 30, "raw_text_has_digits": 26, "no_raw_text": 3, "raw_text_without_digits": 1}},
    "reference_repair": {"clips": {"restored": 2100, "unchanged": 3850, "no_raw_text": 50},
                         "restored_words": 3456, "restored_letters": 3500,
                         "clips_with_words_that_could_not_be_aligned": 400, "words_that_could_not_be_aligned": 1200},
    "letters_of_raw_text_missing_from_normalized_text": {"í": {"raw_text": 3600, "normalized_text": 0,
                                                               "reference": 3500}},
    "chars_per_second": {split: {"clips": 10, "min": 2.5, "p1": 3.0, "p5": 5.0, "p25": 10.0, "p50": 14.0, "p75": 16.0,
                                 "p95": 19.0, "p99": 22.0, "max": 40.0, "mean": 13.5} for split in report.SPLITS},
}
FLEURS = {
    "source": {"repository": "google/fleurs", "revision": "0123456789abcdef0123"},
    "subset": {"test": {"clips": 1200, "hours": 3.5, "sentences": 400, "shards": 2, "shard_bytes": 150_250_000}},
    "audio": {"source_encodings": {"wav_f32le": 1190, "wav_s16le": 10}, "stored_encodings": {"flac_s16": 1200},
              "samples_limited_to_16_bit_range": 6, "unreadable_clips": {"clip-7": "truncated"},
              "audio_without_transcript": 2, "length_differs_from_tsv": 0, "transcripts_without_audio": 1},
}
TRAINING = {
    "system": "hf-whisper-small-finetuned", "group": "own", "complete": True, "smoke": False, "resumed": False,
    "data": {"train": {"clips": 4000, "hours": 11.5, "speakers": 20}, "validation": {"clips": 800}},
    "hyperparameters": {"max_steps": 1000, "batch_size": 8, "gradient_accumulation_steps": 2},
    "selection": {"metric": "wer", "decoding": {"num_beams": 1}},
    "evaluations": [{"step": 0, "epoch": 0.0, "loss": 0.75, "wer": 40.125, "cer": 12.5},
                    {"step": 500, "epoch": 2.0, "loss": 0.3125, "wer": 25.0, "cer": 8.25},
                    {"step": 1000, "epoch": 4.0, "loss": 0.375, "wer": 26.5, "cer": 8.0}],
    "best": {"step": 500},
    "seconds": {"total": 3600.0, "training_steps": 1800.0, "evaluations": 1500.0},
    "hardware": {"gpu": "Some GPU", "gpu_memory_bytes": 24 * 2 ** 30, "peak_gpu_memory_bytes": 18 * 2 ** 30},
    "packages": {"python": "3.13.0", "torch": "2.14.1", "transformers": "5.23.0", "accelerate": "1.16.0"},
    "started_utc": "2026-11-22T02:38:29Z", "finished_utc": "2026-11-22T03:38:29Z",
}
README = """# Title

Text before.

<!-- generated: voxpopuli-subset -->
old table
<!-- end generated -->

Text between.

<!-- generated: voxpopuli-repair -->
<!-- end generated -->

Text after.
"""


class ReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in report.READMES:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(README, encoding="utf-8", newline="\n")

    def write_summaries(self, raw=RAW, clean=CLEAN):
        for name, summary in ((report.RAW_SUMMARY, raw), (report.CLEAN_SUMMARY, clean), (report.SEARCH, SEARCH)):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(json.dumps(summary), encoding="utf-8")

    def run_main(self, *arguments):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = report.main(list(arguments), root=self.root)
        return code, out.getvalue() + err.getvalue()

    def test_tables_come_from_the_summary_files(self):
        self.write_summaries()
        text = report.render(README, self.root)
        self.assertIn("| train | 5,000 | 14.00 | 25 | 480 | 11 | 957.6 |", text)
        self.assertIn("| total | 6,000 | 17.00 | 52 | 610 | 14 | 1162.8 |", text)
        self.assertIn("Sessions shared by two splits of the subset: train and validation 0, train and test 0, "
                      "validation and test 0.", text)
        self.assertIn("training target 14 h, validation target 2.5 h, at least 2 h of test audio; weights taken: "
                      "validation 40, test 36; seed 0.", text)
        self.assertIn("| Restored words | 3,456 |", text)
        self.assertIn("| `í` | 3,600 | 0 | 3,500 |", text)
        self.assertIn("1,200 in 400 clips", text)
        self.assertIn(f"Source: `{report.RAW_SUMMARY}`.", text)
        self.assertNotIn("old table", text)
        self.assertTrue(text.startswith("# Title\n\nText before.\n\n<!-- generated: voxpopuli-subset -->\n| Split |"))
        self.assertTrue(text.endswith("<!-- end generated -->\n\nText after.\n"))

    def test_official_split_and_removal_tables(self):
        official = report.table_official(RAW)
        self.assertIn("| train | 12,000 | 30.50 | 40 | 1,200 | 300 |", official)
        self.assertIn("Speakers shared by two official splits: train and validation 15, train and test 21, "
                      "validation and test 12.", official)
        self.assertIn("Sessions shared by two official splits: train and validation 60, train and test 70, "
                      "validation and test 9.", official)
        self.assertIn("| unknown speaker | 300 (0.75 h) | 10 (0.03 h) | 20 (0.06 h) |", report.table_removed(RAW))
        self.assertIn("| validation | 10 | 2.5 | 3.0 | 5.0 | 10.0 | 14.0 | 16.0 | 19.0 | 22.0 | 40.0 |",
                      report.table_rate(CLEAN))

    def test_split_report_has_what_must_be_reported_for_a_split(self):
        table = report.table_split_report(CLEAN)
        self.assertIn("| Split | Speakers | Sessions | Utterances | Hours | References that occur word for word in "
                      "the training split |", table)
        self.assertIn("| train | 25 | 480 | 5,000 | 14.00 | - |", table)
        self.assertIn("| validation | 5 | 40 | 300 | 0.88 | 3 of 300 |", table)
        self.assertIn("| test | 22 | 90 | 700 | 2.12 | 1 of 700 |", table)
        self.assertIn("Speakers in two splits: train and validation 0, train and test 0, validation and test 0.", table)
        self.assertIn("Sessions in two splits: train and validation 0, train and test 0, validation and test 0.", table)

    def test_number_word_table(self):
        table = report.table_number_words(CLEAN)
        self.assertIn("| English number words | train | validation | test |", table)
        self.assertIn("| References with English number words | 250 | 20 | 30 |", table)
        self.assertIn("| of these: `raw_text` has the number in digits | 200 | 15 | 26 |", table)
        self.assertIn("| of these: the clip has no `raw_text` | 48 | 5 | 3 |", table)

    def test_search_table_shows_the_results_that_meet_the_targets_and_the_one_taken(self):
        lines = report.table_search(SEARCH).split("\n")
        self.assertEqual(lines[2], "| 2 | 4 | 2.25 (23, 95) | 2.88 (16) | 15.25 (20) | yes | taken |")
        self.assertEqual(lines[3], "| 2 | 2 | 2.12 (22, 90) | 2.75 (15) | 25.50 (25) | yes |  |")
        self.assertEqual(len([line for line in lines if line.startswith("| ")]), 3)  # header and two results
        self.assertIn("5 weight pairs were tried (weights 1, 2, 4); 3 of them leave at least 2 h of test audio, "
                      "2.5 h of validation audio and 14 h of training audio.", lines[-1])

    def test_fleurs_table_shows_the_stored_split_and_what_did_not_match(self):
        table = report.table_fleurs(FLEURS)
        self.assertIn("| Split | Clips | Hours | Sentences | Shards | Size (MB) |", table)
        self.assertIn("| test | 1,200 | 3.50 | 400 | 2 | 150.2 |", table)
        self.assertIn("Source files: `google/fleurs` at revision `0123456789ab`.", table)
        self.assertIn("Encoding of the source audio: `wav_f32le` 1,190, `wav_s16le` 10; stored as: `flac_s16` 1,200.",
                      table)
        self.assertIn("Samples limited to the 16-bit range: 6. Clips that could not be read: 1; audio files without "
                      "a transcript: 2; transcripts without audio: 1.", table)
        block = "<!-- generated: fleurs-subset -->\n<!-- end generated -->\n"
        self.assertIn(f"_Not built yet: `{report.FLEURS_SUMMARY}` does not exist._", report.render(block, self.root))
        (self.root / report.FLEURS_SUMMARY).parent.mkdir(parents=True)
        (self.root / report.FLEURS_SUMMARY).write_text(json.dumps(FLEURS), encoding="utf-8")
        self.assertIn("| test | 1,200 | 3.50 | 400 | 2 | 150.2 |\n\nSource files:", report.render(block, self.root))
        self.assertIn(f"Source: `{report.FLEURS_SUMMARY}`.", report.render(block, self.root))

    def test_fine_tuning_run_is_reported_from_its_training_record(self):
        block = "<!-- generated: finetune-run -->\nold\n<!-- end generated -->\n"
        path = self.root / report.RESULTS / report.TRAINING_RECORD
        self.assertIn(f"_Not run yet: `results/{report.TRAINING_RECORD}` does not exist._",
                      report.render(block, self.root))
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({**TRAINING, "smoke": True}), encoding="utf-8")
        self.assertIn("is the record of a trial run or of a run that did not finish", report.render(block, self.root))
        path.write_text(json.dumps(TRAINING), encoding="utf-8")
        text = report.render(block, self.root)
        for row in ("| Started and finished (UTC) | 2026-11-22 02:38 to 2026-11-22 03:38 |",
                    "| GPU | Some GPU, 24.0 GiB of memory |",
                    "| Peak GPU memory | 18.0 GiB with 16 clips per update (batches of 8) |",
                    "| Training clips read | 4,000 (11.50 h, 20 speakers) |",
                    "| Updates | 1,000; continued after an interruption: no |",
                    "| Time | 60.0 min in all: 30.0 min of updates and 25.0 min for 3 validation passes |",
                    "| Checkpoint kept | update 500, the lowest validation WER of the evaluated checkpoints |",
                    "| Packages | torch 2.14.1, transformers 5.23.0, accelerate 1.16.0 |",
                    "| 0 | 0.00 | 0.7500 | 40.12 | 12.50 |  |", "| 500 | 2.00 | 0.3125 | 25.00 | 8.25 | kept |",
                    "| 1,000 | 4.00 | 0.3750 | 26.50 | 8.00 |  |",
                    "decoded in batches with beam size 1, on the 800 validation clips",
                    f"Source: `results/{report.TRAINING_RECORD}`."):
            self.assertIn(row, text)
        self.assertNotIn("\nold\n", text)
        self.assertEqual(report.render(text, self.root), text)

    def test_rendering_twice_changes_nothing(self):
        self.write_summaries()
        once = report.render(README, self.root)
        self.assertEqual(report.render(once, self.root), once)

    def test_missing_summary_is_stated_instead_of_a_table(self):
        text = report.render(README, self.root)
        self.assertIn(f"_Not built yet: `{report.RAW_SUMMARY}` does not exist._", text)
        self.assertNotIn("| Split |", text)

    def test_unknown_block_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown generated block"):
            report.render("<!-- generated: something-else -->\n<!-- end generated -->\n", self.root)

    def test_command_rewrites_and_check_detects_stale_blocks(self):
        self.write_summaries()
        self.assertEqual(self.run_main("--check")[0], 1)
        self.assertEqual((self.root / "README.md").read_text(encoding="utf-8"), README)  # --check changes nothing
        code, printed = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("data/raw/README.md", printed)
        self.assertEqual(self.run_main("--check")[0], 0)
        changed = json.loads(json.dumps(RAW))
        changed["subset"]["train"]["clips"] += 1
        self.write_summaries(raw=changed)
        code, printed = self.run_main("--check")
        self.assertEqual(code, 1)
        self.assertIn("out of date", printed)

    def test_readmes_of_this_repository_are_up_to_date(self):
        for name in report.READMES:
            text = (REPOSITORY / name).read_text(encoding="utf-8")
            self.assertEqual(report.render(text, REPOSITORY), text, f"{name}: run python -m huasr.report")


class MainReadmeTests(unittest.TestCase):
    """What the main README says in prose, checked against the files of this repository."""

    @classmethod
    def setUpClass(cls):
        cls.text = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        cls.scores = {}
        for path in sorted((REPOSITORY / report.RESULTS).glob(f"*/*/{report.BENCHMARK_SPLIT}/scores.json")):
            scores = json.loads(path.read_text(encoding="utf-8"))
            cls.scores[scores["system"]] = scores
        own = REPOSITORY / report.RESULTS / "own" / "hf-whisper-small-finetuned" / report.BENCHMARK_SPLIT
        cls.comparison = json.loads((own / "comparison_with_hf-whisper-small.json").read_text(encoding="utf-8"))

    def rate(self, system, metric="wer", part="all_clips"):
        return self.scores[system][part][metric]["rate"]

    def interval(self, system, metric="wer"):
        return self.scores[system]["all_clips"][metric]["ci95"]

    def test_the_benchmark_is_in_the_main_readme_with_baselines_and_our_model_apart(self):
        blocks = [f"<!-- generated: {name} -->" for name in ("benchmark-baselines", "benchmark-own",
                                                              "benchmark-comparison")]
        places = [self.text.index(block) for block in blocks]
        self.assertEqual(places, sorted(places))
        baselines, own = (self.text[place:].split("<!-- end generated -->")[0] for place in places[:2])
        for system, scores in self.scores.items():
            self.assertEqual(f"\n| {system} |" in baselines, scores["group"] == "baselines", system)
            self.assertEqual(f"\n| {system} |" in own, scores["group"] == "own", system)

    def test_the_layout_names_every_file_of_the_code_and_every_system_folder(self):
        layout = self.text.split("## Repository layout")[1].split("\n## ")[0]
        names = [path.name for folder, pattern in (("src/huasr", "*.py"), ("tests", "*.py"), ("notebooks", "*.ipynb"),
                                                   (".", "requirements*.txt"), (".", "*.yml"), (".", "*.toml"),
                                                   ("results/baselines", "*"), ("results/own", "*"))
                 for path in sorted((REPOSITORY / folder).glob(pattern))]
        names += ["LICENSE", ".gitignore", ".gitattributes", "contributions/", "data/raw/", "data/clean/",
                  "voxpopuli/", "fleurs/", "training.json", "comparison_with_hf-whisper-small.json"]
        self.assertGreater(len(names), 50)
        self.assertEqual([name for name in names if name not in layout], [], "not explained in the layout")

    def test_every_run_of_the_benchmark_is_a_final_evaluation_on_the_same_clips(self):
        self.assertEqual(len({(scores["all_clips"]["clips"], scores["clean_manifest_sha256"])
                              for scores in self.scores.values()}), 1)
        self.assertTrue(all(scores["final_test"] and scores["complete"] for scores in self.scores.values()))

    def test_what_the_discussion_says_about_the_best_system_holds(self):
        best = min(self.scores, key=self.rate)
        self.assertEqual((best, min(self.scores, key=lambda system: self.rate(system, "cer"))),
                         ("canary-1b-v2", "canary-1b-v2"))
        others = [system for system in self.scores if system != best]
        self.assertTrue(all(self.interval(best)[1] < self.interval(system)[0] for system in others))
        self.assertGreater(min(self.rate(system) for system in others), 2 * self.rate(best))
        overlapping = [system for system in others if self.interval(system, "cer")[0] <= self.interval(best, "cer")[1]]
        self.assertEqual(overlapping, ["parakeet-tdt-0.6b-v3"])

    def test_what_the_discussion_says_about_the_fine_tuning_holds(self):
        new, baseline = "hf-whisper-small-finetuned", "hf-whisper-small"
        self.assertEqual((self.comparison["new"]["system"], self.comparison["baseline"]["system"]), (new, baseline))
        smallest = round(1 / (1 + self.comparison["scoring"]["resamples"]), 5)
        for part in ("all_clips", "without_english_number_words"):
            for metric, _ in report.METRICS:
                values = self.comparison[part][metric]
                self.assertLess(values["ci95"][1], 0, f"{part} {metric}")
                self.assertEqual(values["p_value"], smallest, f"{part} {metric}")
        self.assertLess(self.interval(new)[1], self.interval(baseline)[0])                # WER: apart
        self.assertGreater(self.interval(new, "cer")[1], self.interval(baseline, "cer")[0])  # CER: overlapping ends
        for metric in ("wer", "cer"):                              # second after Canary, and not reaching it
            ranked = sorted(self.scores, key=lambda system: self.rate(system, metric))
            self.assertEqual(ranked[:2], ["canary-1b-v2", new])
            self.assertLess(self.interval("canary-1b-v2", metric)[1], self.interval(new, metric)[0])
            before = self.rate(baseline, metric) - self.rate("canary-1b-v2", metric)
            after = self.rate(new, metric) - self.rate("canary-1b-v2", metric)
            self.assertLess((before - after) / before, 0.5)        # less than half of the distance closed
        below = ("faster-whisper-small", "hf-whisper-small", "qwen3-asr-1.7b", "wav2vec2-xlsr-53-hungarian",
                 "faster-whisper-base", "gmm-hmm-mfa")
        self.assertTrue(all(self.interval(new)[1] < self.interval(system)[0] for system in below))
        parakeet = self.interval("parakeet-tdt-0.6b-v3")
        self.assertTrue(parakeet[0] < self.interval(new)[0] and self.interval(new)[1] < parakeet[1])

    def test_what_the_discussion_says_about_the_caveats_holds(self):
        def width(system):
            return self.interval(system)[1] - self.interval(system)[0]

        def share(system, kind):
            wer = self.scores[system]["all_clips"]["wer"]
            return wer[kind] / (wer["substitutions"] + wer["deletions"] + wer["insertions"])

        parakeet = "parakeet-tdt-0.6b-v3"
        self.assertGreater(width(parakeet), 2 * max(width(system) for system in self.scores if system != parakeet))
        self.assertGreater(share(parakeet, "insertions"), 0.5)
        self.assertTrue(0.4 < share("gmm-hmm-mfa", "deletions") < 0.5)
        on_gpu = {system for system, scores in self.scores.items() if scores["settings"].get("device") == "cuda"}
        self.assertEqual(set(self.scores) - on_gpu, {"faster-whisper-base", "faster-whisper-small", "gmm-hmm-mfa"})
        self.assertEqual({self.scores[system]["settings"]["gpu"] for system in on_gpu}, {"NVIDIA L4"})
        self.assertEqual(sorted(self.scores, key=self.rate),       # the same order without the flagged clips
                         sorted(self.scores, key=lambda system: self.rate(system, part="without_english_number_words")))


if __name__ == "__main__":
    unittest.main()
