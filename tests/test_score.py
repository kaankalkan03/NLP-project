"""Tests for the scorer on hand-made prediction files, and for the result tables."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from huasr import evaluate, report, score
from huasr.shards import write_manifest

# clip id, speaker, duration in ms, flag for English number words, reference, hypothesis, processing seconds
PREDICTIONS = [
    ("c1", "s1", 2000, 0, "the cat sat on the mat", "the cat sit on the", 0.5),   # 1 substitution, 1 deletion
    ("c2", "s1", 1000, 0, "go", "please go now today", 0.25),                     # 3 insertions
    ("c3", "s2", 3000, 1, "a kör two", "a kor two", 0.75),                        # 1 substitution (a diacritic)
    ("c4", "s3", 2000, 0, "jó napot", "Jó napot!", 0.5),                          # no error after the policy
]


def write_run(folder: Path, predictions=PREDICTIONS, system="fake-system", group="baselines", split="validation",
              complete=True) -> Path:
    folder.mkdir(parents=True)
    rows = [{"clip_id": clip_id, "speaker_id": speaker, "session_id": f"session-{speaker}", "duration_ms": duration,
             "english_number_words": flag, "reference": reference, "hypothesis": hypothesis,
             "processing_seconds": seconds}
            for clip_id, speaker, duration, flag, reference, hypothesis, seconds in predictions]
    write_manifest(folder / "predictions.csv", evaluate.PREDICTION_COLUMNS, rows)
    run = {"system": system, "group": group, "split": split, "final_test": split == "test", "complete": complete,
           "clips": len(rows), "checkpoint": {"repository": "nobody/fake", "revision": "0123456789abcdef0123"},
           "settings": {"beam_size": 1}, "hardware": {"processor": "test processor"},
           "data": {"clean_manifest_sha256": "f" * 64}}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return folder


class TemporaryFolderTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)


class ScoreTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        self.folder = write_run(self.root / "results" / "baselines" / "fake-system" / "validation")
        self.scores = score.score_run(self.folder)

    def test_corpus_level_rates_with_their_edit_counts(self):
        everything = self.scores["all_clips"]
        self.assertEqual((everything["clips"], everything["speakers"], everything["clips_with_empty_reference"]), (4, 3, 0))
        wer = everything["wer"]
        self.assertEqual((wer["substitutions"], wer["deletions"], wer["insertions"], wer["reference_units"]),
                         (2, 1, 3, 12))
        self.assertEqual(wer["rate"], 0.5)
        cer = everything["cer"]
        self.assertEqual((cer["substitutions"], cer["deletions"], cer["insertions"], cer["reference_units"]),
                         (2, 4, 17, 41))
        self.assertEqual(cer["rate"], round(23 / 41, 4))

    def test_diacritic_insensitive_rate(self):
        plain = self.scores["all_clips"]["wer_no_diacritics"]
        self.assertEqual((plain["substitutions"], plain["reference_units"], plain["rate"]), (1, 12, round(5 / 12, 4)))

    def test_real_time_factor(self):
        everything = self.scores["all_clips"]
        self.assertEqual((everything["audio_seconds"], everything["processing_seconds"], everything["rtf"]),
                         (8.0, 2.0, 0.25))

    def test_intervals_are_speaker_level_and_contain_the_rate(self):
        for name in ("wer", "cer", "wer_no_diacritics"):
            part = self.scores["all_clips"][name]
            self.assertLessEqual(part["ci95"][0], part["rate"])
            self.assertGreaterEqual(part["ci95"][1], part["rate"])
        self.assertEqual(self.scores["scoring"]["interval"], "95% percentile bootstrap over speakers")
        self.assertEqual((self.scores["scoring"]["resamples"], self.scores["scoring"]["seed"]), (10_000, 0))
        # one speaker only: every resample is the same, so the interval has no width
        single = score.score_clips([dict(zip(evaluate.PREDICTION_COLUMNS,
                                             ("c1", "s1", "", 2000, 0, "egy kettő", "egy három", 0.5)))])
        self.assertEqual(single["wer"]["ci95"], [0.5, 0.5])

    def test_figures_are_also_given_without_the_flagged_clips_and_nothing_is_dropped(self):
        rest = self.scores["without_english_number_words"]
        self.assertEqual(self.scores["clips_with_english_number_words"], 1)
        self.assertEqual((rest["clips"], rest["speakers"]), (3, 2))
        self.assertEqual((rest["wer"]["rate"], rest["wer"]["reference_units"]), (round(5 / 9, 4), 9))
        self.assertEqual((rest["cer"]["rate"], rest["cer"]["reference_units"]), (round(22 / 32, 4), 32))
        self.assertEqual(rest["rtf"], 0.25)
        self.assertEqual(self.scores["all_clips"]["clips"], 4)
        self.assertEqual(len(score.read_predictions(self.folder)), 4)

    def test_scores_carry_what_identifies_the_run(self):
        self.assertEqual(json.loads((self.folder / "scores.json").read_text(encoding="utf-8")), self.scores)
        self.assertEqual((self.scores["system"], self.scores["group"], self.scores["split"], self.scores["complete"]),
                         ("fake-system", "baselines", "validation", True))
        self.assertEqual(self.scores["checkpoint"]["revision"], "0123456789abcdef0123")
        self.assertEqual(self.scores["clean_manifest_sha256"], "f" * 64)

    def test_scoring_again_gives_the_same_file(self):
        first = (self.folder / "scores.json").read_bytes()
        score.score_run(self.folder)
        self.assertEqual((self.folder / "scores.json").read_bytes(), first)

    def test_clips_with_an_empty_reference_are_counted_and_left_out_of_the_rates(self):
        silent = [*PREDICTIONS, ("c5", "s3", 1000, 0, " ... ", "valami", 0.25)]
        scores = score.score_run(write_run(self.root / "silent", silent))
        self.assertEqual((scores["all_clips"]["clips"], scores["all_clips"]["clips_with_empty_reference"]), (4, 1))
        self.assertEqual(scores["all_clips"]["wer"], self.scores["all_clips"]["wer"])

    def test_empty_hypotheses_are_counted(self):
        scores = score.score_run(write_run(self.root / "quiet", [("c1", "s1", 1000, 0, "egy kettő", "", 0.1)]))
        self.assertEqual((scores["all_clips"]["empty_hypotheses"], scores["all_clips"]["wer"]["deletions"]), (1, 2))

    def test_prediction_file_that_does_not_match_the_run_manifest_is_an_error(self):
        run = json.loads((self.folder / "run.json").read_text(encoding="utf-8"))
        (self.folder / "run.json").write_text(json.dumps({**run, "clips": 5}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "5 clips"):
            score.score_run(self.folder)

    def test_command_line_scores_one_folder_or_all(self):
        other = write_run(self.root / "results" / "own" / "our-model" / "validation", system="our-model", group="own")
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(score.main([str(other)]), 0)
            self.assertEqual(score.main(["--all", "--results", str(self.root / "results")]), 0)
            self.assertEqual(score.main([str(self.root / "missing")]), 1)
        self.assertTrue((other / "scores.json").is_file())
        self.assertIn("WER 50.00%", out.getvalue())
        self.assertEqual(score.run_folders(self.root / "results"), sorted([self.folder, other]))


class ResultTableTests(TemporaryFolderTestCase):
    def setUp(self):
        super().setUp()
        results = self.root / "results"
        for system, split, complete in (("system-b", "validation", True), ("system-a", "validation", True),
                                        ("system-a", "test", True), ("system-c", "validation", False)):
            score.score_run(write_run(results / "baselines" / system / split, system=system, split=split,
                                      complete=complete))

    def test_one_row_per_complete_run_validation_first(self):
        lines = report.table_results(self.root, "baselines").split("\n")
        self.assertTrue(lines[0].startswith("| System | Checkpoint and revision | Split | Clips | WER % [95%] |"))
        self.assertEqual([line.split(" | ")[0] + " / " + line.split(" | ")[2] for line in lines[2:5]],
                         ["| system-a / validation", "| system-b / validation", "| system-a / test"])
        self.assertIn("| `nobody/fake` at `0123456789ab` | validation | 4 | 50.00 [", lines[2])
        self.assertTrue(lines[2].endswith("| 41.67 | 0.250 | 3 | 55.56 | 68.75 |"))
        self.assertIn("1 trial runs on a part of a split are not shown.", lines[-1])

    def test_group_without_runs_says_so(self):
        self.assertEqual(report.table_results(self.root, "own"), "_No complete run yet._")

    def test_run_table_lists_every_run_folder_with_date_machine_and_the_final_flag(self):
        lines = report.table_runs(self.root).split("\n")
        self.assertEqual(lines[0], "| System | Group | Split | Started (UTC) | Clips | Whole split | Made with "
                                   "`--final-test` | Scored | Run on |")
        self.assertEqual([line.split(" | ")[0] + " / " + line.split(" | ")[2] for line in lines[2:]],
                         ["| system-a / validation", "| system-b / validation", "| system-c / validation",
                          "| system-a / test"])  # every folder, also the trial run
        self.assertEqual(lines[2], "| system-a | baselines | validation | not recorded | 4 | yes | - | yes | "
                                   "CPU: test processor |")
        self.assertIn("| system-c | baselines | validation | not recorded | 4 | no | - | yes |", lines[4])
        self.assertIn("| system-a | baselines | test | not recorded | 4 | yes | yes | yes |", lines[5])
        self.assertEqual(report.table_runs(self.root / "nothing"), "_No run yet._")

    def test_blocks_of_the_results_readme_are_rendered(self):
        text = ("# results\n\n<!-- generated: results-baselines -->\n<!-- end generated -->\n\n"
                "<!-- generated: results-own -->\nold\n<!-- end generated -->\n")
        rendered = report.render(text, self.root)
        self.assertIn("| system-a | `nobody/fake` at `0123456789ab` | validation |", rendered)
        self.assertIn("_No complete run yet._\n\nSource: `results/own/<system>/<split>/scores.json`.", rendered)
        self.assertNotIn("\nold\n", rendered)
        self.assertEqual(report.render(rendered, self.root), rendered)
        self.assertIn("results/README.md", report.READMES)


class BenchmarkTableTests(TemporaryFolderTestCase):
    """The tables of the main README: the test split only, the lowest WER first, baselines apart from our models."""

    BETTER = [row[:5] + (row[4],) + row[6:] for row in PREDICTIONS[:3]] + [PREDICTIONS[3]]  # hypothesis = reference

    def add(self, group, system, split, predictions=PREDICTIONS, complete=True, **changes):
        folder = write_run(self.root / "results" / group / system / split, predictions, system=system, group=group,
                           split=split, complete=complete)
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        (folder / "run.json").write_text(json.dumps({**run, **changes}), encoding="utf-8")
        return score.score_run(folder)

    def setUp(self):
        super().setUp()
        self.add("baselines", "system-a", "test", started_utc="2026-11-22T04:11:32Z",
                 settings={"device": "cpu", "cpu_threads": 4})
        self.add("baselines", "system-b", "test", self.BETTER, hardware={"processor": "some processor"},
                 settings={"device": "cuda", "gpu": "Some GPU"})
        self.add("baselines", "system-b", "validation")                  # another split: not in the benchmark
        self.add("baselines", "system-c", "test", complete=False)        # a trial run: not in the benchmark
        self.add("own", "our-model", "test", settings={"jobs": {"training": 2, "decoding": 8}})

    def test_only_complete_runs_on_the_test_split_with_the_lowest_wer_first(self):
        lines = report.table_benchmark(self.root, "baselines").split("\n")
        self.assertEqual(lines[0], "| System | Checkpoint and revision | WER % [95%] | CER % [95%] | WER % without "
                                   "diacritics | Word errors: substituted / deleted / inserted | WER % / CER % without "
                                   "the clips with English number words | RTF | Run on |")
        self.assertEqual([line.split(" | ")[0] for line in lines[2:4]], ["| system-b", "| system-a"])
        self.assertEqual(lines[4], "")  # two rows: the validation run and the trial run are not shown
        self.assertIn("| system-b | `nobody/fake` at `0123456789ab` | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 | "
                      "0 / 0 / 0 | 0.00 / 0.00 | 0.250 | GPU: Some GPU |", lines[2])
        self.assertIn("| 50.00 [", lines[3])
        self.assertTrue(lines[3].endswith("| 41.67 | 2 / 1 / 3 | 55.56 / 68.75 | 0.250 | CPU: test processor, "
                                          "4 threads |"))

    def test_the_note_says_whether_the_rows_can_be_compared(self):
        note = report.table_benchmark(self.root, "baselines").split("\n")[-1]
        self.assertEqual(note, "Rows are sorted by WER, the lowest first. Every row: the same 4 clips of 3 speakers "
                               "with 12 reference words, read from one clean manifest (SHA-256 `ffffffffffff`); "
                               "clips with English number words in the reference: 1. Intervals: 95% percentile "
                               "bootstrap over speakers, 10,000 resamples, seed 0. Runs made with `--final-test`: "
                               "2 of 2.")
        self.add("baselines", "system-d", "test", PREDICTIONS[:3], final_test=False)
        note = report.table_benchmark(self.root, "baselines").split("\n")[-1]
        self.assertIn("The rows do not all hold the same clips and references", note)
        self.assertIn("Runs made with `--final-test`: 2 of 3.", note)

    def test_our_own_models_have_their_own_table(self):
        own = report.table_benchmark(self.root, "own")
        self.assertIn("| our-model |", own)
        self.assertIn("CPU: test processor, 8 parallel jobs |", own)
        self.assertNotIn("system-a", own)
        self.assertNotIn("our-model", report.table_benchmark(self.root, "baselines"))
        self.assertEqual(report.table_benchmark(self.root / "nothing", "own"),
                         "_No complete run on the test split yet._")

    def test_run_table_reads_the_start_and_the_machine_from_the_run_manifest(self):
        table = report.table_runs(self.root)
        self.assertIn("| system-a | baselines | test | 2026-11-22 04:11 | 4 | yes | yes | yes | CPU: test processor, "
                      "4 threads |", table)
        self.assertIn("| system-b | baselines | test | not recorded | 4 | yes | yes | yes | GPU: Some GPU |", table)

    def test_blocks_of_the_main_readme_are_rendered(self):
        text = ("<!-- generated: benchmark-baselines -->\nold\n<!-- end generated -->\n\n"
                "<!-- generated: benchmark-own -->\n<!-- end generated -->\n\n"
                "<!-- generated: results-runs -->\n<!-- end generated -->\n")
        rendered = report.render(text, self.root)
        self.assertIn("| system-b | `nobody/fake` at `0123456789ab` | 0.00 [", rendered)
        self.assertIn("Source: `results/baselines/<system>/test/scores.json`.", rendered)
        self.assertIn("Source: `results/own/<system>/test/scores.json`.", rendered)
        self.assertIn("Source: `results/<group>/<system>/<split>/run.json`.", rendered)
        self.assertNotIn("\nold\n", rendered)
        self.assertEqual(report.render(rendered, self.root), rendered)


if __name__ == "__main__":
    unittest.main()
