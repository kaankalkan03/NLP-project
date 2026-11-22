"""Tests for the evaluation runner, with a stand-in system and generated data (no model, no network)."""

import contextlib
import csv
import hashlib
import io
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from huasr import audio, clean, evaluate, subset
from huasr.evaluate import FinalTestRequired
from tests import needs
from tests.test_subset import TARGETS, make_rows


class FakeSystem:
    """Answers with the number of samples it was given; remembers every call."""

    name, group = "fake-system", "baselines"

    def __init__(self):
        self.calls, self.loaded = [], 0

    def describe(self) -> dict:
        return {"checkpoint": {"repository": "nobody/fake", "revision": "abc123"}, "settings": {"beam_size": 1},
                "seed": 7, "packages": {"fake-package": "1.0"}}

    def load(self) -> None:
        self.loaded += 1

    def transcribe(self, samples, sample_rate) -> str:
        self.calls.append((samples, sample_rate))
        return f"{len(samples)} minta"


def ticking_clock(step: float = 0.5):
    """A clock that advances by `step` seconds every time it is read."""
    counter = itertools.count()
    return lambda: next(counter) * step


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@needs("av")
class RunnerTests(unittest.TestCase):
    """The data: the fixture of the subset tests, built and cleaned (15 train, 3 validation, 5 test clips)."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.data = Path(cls.temporary.name)
        rows = make_rows()
        subset.build_voxpopuli(None, cls.data / "raw", TARGETS, check_source=False,
                               rows_of=lambda split: iter(rows[split]))
        clean.clean_voxpopuli(cls.data / "raw", cls.data / "clean")
        cls.manifest = read_csv(cls.data / "clean" / "manifest.csv")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.results = Path(temporary.name) / "results"
        self.system = FakeSystem()

    def run_split(self, split="validation", **options):
        options = {"clock": ticking_clock(), **options}
        return evaluate.run(self.system, split, self.data / "raw", self.data / "clean", self.results, **options)

    def test_every_clip_of_the_split_is_paired_with_reference_and_hypothesis(self):
        folder = self.run_split()
        self.assertEqual(folder, self.results / "baselines" / "fake-system" / "validation")
        rows = read_csv(folder / "predictions.csv")
        self.assertEqual(tuple(rows[0]), evaluate.PREDICTION_COLUMNS)
        expected = [row for row in self.manifest if row["split"] == "validation"]
        self.assertEqual([row["clip_id"] for row in rows], [row["clip_id"] for row in expected])
        for row, source in zip(rows, expected):
            self.assertEqual((row["reference"], row["speaker_id"], row["session_id"], row["duration_ms"],
                              row["english_number_words"]),
                             (source["reference"], source["speaker_id"], source["session_id"], source["duration_ms"],
                              source["english_number_words"]))
            self.assertEqual(row["hypothesis"], "4000 minta")  # 250 ms at 16 kHz
            self.assertEqual(row["processing_seconds"], "0.5")

    def test_the_system_gets_the_stored_audio_as_floats(self):
        self.run_split()
        self.assertEqual(len(self.system.calls), 3)
        stored = {}
        subset.unpack(self.data / "raw", self.results / "audio")
        for row in self.manifest:
            if row["split"] == "validation":
                decoded, _ = audio.decode_flac((self.results / "audio" / "validation" / row["file"]).read_bytes())
                stored[decoded.tobytes()] = row["clip_id"]
        for samples, rate in self.system.calls:
            self.assertEqual((samples.dtype, samples.ndim, rate), (np.float32, 1, 16_000))
            self.assertLessEqual(float(np.abs(samples).max()), 1.0)
            self.assertIn(np.rint(samples * 32768).astype("<i2").tobytes(), stored)

    def test_run_manifest_says_what_was_run_on_what(self):
        folder = self.run_split()
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["system"], run["group"], run["split"], run["final_test"]),
                         ("fake-system", "baselines", "validation", False))
        self.assertEqual((run["checkpoint"], run["settings"], run["seed"]),
                         ({"repository": "nobody/fake", "revision": "abc123"}, {"beam_size": 1}, 7))
        self.assertEqual((run["clips"], run["limit"], run["complete"], run["data"]["clips_in_split"]), (3, None, True, 3))
        for name, folder_name in (("clean_manifest_sha256", "clean"), ("raw_manifest_sha256", "raw")):
            digest = hashlib.sha256((self.data / folder_name / "manifest.csv").read_bytes()).hexdigest()
            self.assertEqual(run["data"][name], digest)
        self.assertEqual(run["data"]["reference_column"], "reference")
        self.assertEqual((run["audio_seconds"], run["processing_seconds"], run["load_seconds"]), (0.75, 1.5, 0.5))
        self.assertEqual(set(run["hardware"]), {"processor", "machine", "logical_cpus", "system"})
        self.assertTrue(run["hardware"]["processor"])
        self.assertEqual(run["packages"]["fake-package"], "1.0")
        self.assertEqual(set(run["packages"]) - {"fake-package"}, {"python", "numpy", "av", "huasr"})
        for moment in (run["started_utc"], run["finished_utc"]):
            self.assertRegex(moment, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(self.system.loaded, 1)

    def test_the_test_split_is_refused_without_the_flag(self):
        with self.assertRaisesRegex(FinalTestRequired, "--final-test"):
            self.run_split("test")
        self.assertEqual(self.system.calls, [])
        self.assertFalse(self.results.exists())

    def test_the_test_split_runs_with_the_flag_and_the_manifest_records_it(self):
        folder = self.run_split("test", final_test=True)
        run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["split"], run["final_test"], run["clips"]), ("test", True, 5))
        self.assertEqual(len(read_csv(folder / "predictions.csv")), 5)

    def test_the_flag_changes_nothing_for_other_splits(self):
        run = json.loads((self.run_split("train") / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["split"], run["final_test"], run["clips"]), ("train", False, 15))

    def test_a_trial_run_on_a_few_clips_is_marked_as_incomplete(self):
        run = json.loads((self.run_split(limit=2) / "run.json").read_text(encoding="utf-8"))
        self.assertEqual((run["clips"], run["limit"], run["complete"], run["data"]["clips_in_split"]), (2, 2, False, 3))

    def test_an_existing_run_is_replaced_only_with_force(self):
        self.run_split()
        with self.assertRaises(FileExistsError):
            self.run_split()
        self.run_split(force=True)
        self.assertEqual(len(self.system.calls), 6)

    def test_unknown_split_or_group_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown split"):
            self.run_split("dev")
        self.system.group = "others"
        with self.assertRaisesRegex(ValueError, "unknown group"):
            self.run_split()

    def test_audio_that_differs_from_the_manifest_stops_the_run(self):
        raw = self.results.parent / "raw"
        (raw / "clips").mkdir(parents=True)
        for path in (self.data / "raw").iterdir():
            if path.is_file():
                (raw / path.name).write_bytes(path.read_bytes())
        for path in (self.data / "raw" / "clips").iterdir():
            data = bytearray(path.read_bytes())
            if path.name.startswith("validation"):
                data[600] ^= 0xFF
            (raw / "clips" / path.name).write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evaluate.run(self.system, "validation", raw, self.data / "clean", self.results)
        self.assertFalse(self.results.exists())

    def test_command_line(self):
        arguments = ["--system", "fake-system", "--raw", str(self.data / "raw"), "--clean", str(self.data / "clean"),
                     "--results", str(self.results)]
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(evaluate.SYSTEMS, {"fake-system": FakeSystem}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            self.assertEqual(evaluate.main(arguments), 0)                                  # validation by default
            self.assertEqual(evaluate.main([*arguments, "--split", "test"]), 1)
            self.assertEqual(evaluate.main([*arguments, "--split", "test", "--final-test"]), 0)
            self.assertEqual(evaluate.main(arguments), 1)                                  # already there
            self.assertEqual(evaluate.main([*arguments, "--force", "--limit", "1"]), 0)
        self.assertIn("--final-test", err.getvalue())
        self.assertIn("already holds a run", err.getvalue())
        self.assertTrue((self.results / "baselines" / "fake-system" / "validation" / "run.json").is_file())
        self.assertTrue((self.results / "baselines" / "fake-system" / "test" / "predictions.csv").is_file())


@needs("av")  # evaluate.environment() reports the version of the audio decoder it loads
class EnvironmentTests(unittest.TestCase):
    def test_processor_name_and_versions_are_reported(self):
        environment = evaluate.environment()
        self.assertTrue(evaluate.cpu_name())
        self.assertEqual(environment["hardware"]["processor"], evaluate.cpu_name())
        self.assertEqual(environment["packages"]["numpy"], np.__version__)


if __name__ == "__main__":
    unittest.main()
