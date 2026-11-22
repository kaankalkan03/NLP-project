"""MFCC + GMM-HMM baseline, trained with the Montreal Forced Aligner (MFA, which runs Kaldi).

    python -m huasr.gmm_hmm prepare [--work scratch/gmm_hmm] [--train-speakers ID ...]
    python -m huasr.gmm_hmm train   [--work scratch/gmm_hmm] [--jobs N]
    python -m huasr.gmm_hmm decode  [--work scratch/gmm_hmm] [--split validation] [--limit N] [--jobs N]
    python -m huasr.gmm_hmm collect [--work scratch/gmm_hmm] [--split validation] [--results results]

Run the four steps in this order inside the conda environment of
`environment-gmm-hmm.yml`; the scores are then computed by `huasr.score` like
those of every other system. Every step is a Python entry point, so the same
commands work on Windows, Linux and Colab.

What the system is:

- Acoustic model: MFCC features without pitch; monophone, triphone, LDA and
  speaker-adapted (SAT) training, in this order (TRAIN_CONFIG), on the
  training split.
- Lexicon: graphemic. Every word of the training transcripts is spelled as its
  letters, one letter per unit; digraphs such as "sz" stay two units. No
  pronunciation dictionary and no G2P model is used.
- Language model: trigram (LM_CONFIG), estimated by `mfa train_lm` from the
  training transcripts only.
- Text: the column `reference` of the clean manifest after the scoring policy
  of `huasr.text` (lower case, no punctuation, diacritics kept).
- Decoding: `mfa transcribe` with the search settings MFA uses by default
  (DECODE_CONFIG). The audio of the split is grouped by speaker id (MFA
  adapts to each speaker) and nothing else is given to MFA: no reference
  text of that split is written to the corpus folder.

The system can only write words of its lexicon. `collect` therefore records
the share of reference words that are not in the lexicon, which is a lower
bound for its word error rate. MFA writes a placeholder such as `<unk>` where
it decoded the unknown-word model; placeholders are removed from the
hypothesis and counted in the run manifest.

MFA decodes a whole folder at once, so this system does not go through the
per-clip runner `huasr.evaluate`. `collect` writes the same two files
(`run.json`, `predictions.csv`); the processing time of a clip is the wall
time of `mfa transcribe` shared out by clip duration.

`prepare` and `train` are run once; `decode` and `collect` once per split, with
the same trained models. `decode` refuses the test split unless `--final-test`
is given, as `huasr.evaluate` does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from .evaluate import (CLEAN, PREDICTION_COLUMNS, RAW, RESULTS, SPLITS, FinalTestRequired, _utc_now, cpu_name,
                       read_split, sha256_of)
from .shards import write_manifest
from .text import normalize

SYSTEM, GROUP = "gmm-hmm-mfa", "baselines"
WORK = Path("scratch/gmm_hmm")
WORK_FOLDERS = ("corpus", "output", "mfa_root", "logs")
PACKAGES = ("montreal-forced-aligner", "kaldi", "kalpy", "openfst", "ngram")   # recorded with version and build
PLACEHOLDER = re.compile(r"^(<[^<>\s]*>|\[[^\[\]\s]*\])$")   # <unk>, <eps>, [bracketed]: tokens that are not words
# The recipe given to `mfa train`: the first four blocks of MFA's own default recipe with their settings.
# Each block starts from the alignments of the block before; `subset` is the number of clips a block may use.
TRAIN_CONFIG = {
    "feature_type": "mfcc",
    "use_pitch": False,
    "training": [
        {"monophone": {"subset": 10000, "boost_silence": 1.25}},
        {"triphone": {"subset": 20000, "boost_silence": 1.25, "num_leaves": 2000, "max_gaussians": 10000}},
        {"lda": {"subset": 20000, "num_leaves": 2500, "max_gaussians": 15000}},
        {"sat": {"subset": 20000, "num_leaves": 2500, "max_gaussians": 15000}},
    ],
}
LM_CONFIG = {"order": 3, "method": "kneser_ney"}
# The search settings given to `mfa transcribe`. They are the defaults of MFA 3.4.3, written out so that the run
# manifest states them: a first pass without speaker adaptation, then a second pass with it.
DECODE_CONFIG = {"first_beam": 10, "first_max_active": 2000, "beam": 10, "max_active": 2500, "lattice_beam": 6,
                 "acoustic_scale": 0.083333}


def transcript(row: dict[str, str]) -> str:
    """The text of a clip as the system is trained on it: the reference after the scoring policy."""
    return normalize(row["reference"])


def build_lexicon(transcripts: list[str]) -> dict[str, tuple[str, ...]]:
    """Graphemic lexicon: every word of `transcripts` spelled as its letters, one letter per unit.

    Digraphs are not merged. A word with a character that is not a letter is refused: MFA reads a
    number after the word as a pronunciation probability, so a digit cannot be a unit.
    """
    lexicon = {}
    for text in transcripts:
        for word in text.split():
            if not word.isalpha():
                raise ValueError(f"the word {word!r} has a character that is not a letter")
            lexicon[word] = tuple(word)
    return dict(sorted(lexicon.items()))


def write_lexicon(path: Path, lexicon: dict[str, tuple[str, ...]]) -> None:
    lines = [f"{word}\t{' '.join(units)}\n" for word, units in lexicon.items()]
    Path(path).write_text("".join(lines), encoding="utf-8", newline="\n")


def read_lexicon(path: Path) -> dict[str, tuple[str, ...]]:
    entries = (line.split("\t") for line in Path(path).read_text(encoding="utf-8").split("\n") if line)
    return {word: tuple(units.split()) for word, units in entries}


def out_of_vocabulary(lexicon, references: list[str]) -> dict:
    """How many words of `references` (after the scoring policy) the lexicon does not have."""
    clips = [normalize(reference).split() for reference in references]
    tokens = [word for words in clips for word in words]
    missing = [word for word in tokens if word not in lexicon]
    return {"reference_words": len(tokens), "not_in_lexicon": len(missing),
            "rate": round(len(missing) / len(tokens), 4) if tokens else 0.0,
            "distinct_reference_words": len(set(tokens)), "distinct_not_in_lexicon": len(set(missing)),
            "clips_with_a_word_not_in_lexicon": sum(any(word not in lexicon for word in words) for words in clips)}


def export_corpus(raw: Path, rows: list[dict[str, str]], folder: Path, with_text: bool) -> int:
    """Write the clips of `rows` as `folder/<speaker>/<file>`, the layout MFA reads; return their number.

    With `with_text` every clip gets a `.lab` file with its transcript next to it. The stored audio
    is copied byte for byte after its SHA-256 has been checked against the manifest.
    """
    by_shard: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_shard.setdefault(row["shard"], {})[row["file"]] = row
    written = 0
    for shard, wanted in sorted(by_shard.items()):
        found = 0
        with tarfile.open(Path(raw) / shard, "r:") as tar:
            for member in tar:
                row = wanted.get(member.name)
                if row is None:
                    continue
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError(f"{shard}: {member.name} does not match the SHA-256 in the manifest")
                target = Path(folder) / row["speaker_id"] / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if with_text:
                    target.with_suffix(".lab").write_text(transcript(row) + "\n", encoding="utf-8", newline="\n")
                found += 1
        if found != len(wanted):
            raise ValueError(f"{shard}: {len(wanted) - found} clips of the manifest are missing")
        written += found
    return written


def clean_hypothesis(text: str) -> tuple[str, int]:
    """The words of an MFA transcript without placeholders such as `<unk>`; and how many were removed."""
    tokens = text.split()
    words = [token for token in tokens if not PLACEHOLDER.match(token)]
    return " ".join(words), len(tokens) - len(words)


def read_transcripts(folder: Path) -> dict[str, str]:
    """What `mfa transcribe --output_type transcription` wrote: file name without extension -> text."""
    return {path.stem: " ".join(path.read_text(encoding="utf-8").split()) for path in sorted(Path(folder).rglob("*.lab"))}


def predictions(rows: list[dict[str, str]], transcripts: dict[str, str], seconds: float) -> tuple[list[dict], dict]:
    """Prediction rows in the order of `rows`, and what was done to the output of MFA.

    `seconds`, the wall time of the decoding, is shared out over the clips by their duration.
    """
    total_ms = sum(int(row["duration_ms"]) for row in rows)
    result, counts = [], {"clips_without_output_file": 0, "clips_with_empty_output": 0, "placeholders_removed": 0,
                          "clips_with_placeholders": 0, "clips_with_placeholders_only": 0}
    for row in rows:
        text = transcripts.get(Path(row["file"]).stem)
        hypothesis, removed = clean_hypothesis(text or "")
        counts["clips_without_output_file"] += text is None
        counts["clips_with_empty_output"] += text == ""
        counts["placeholders_removed"] += removed
        counts["clips_with_placeholders"] += removed > 0
        counts["clips_with_placeholders_only"] += removed > 0 and not hypothesis
        result.append({"clip_id": row["clip_id"], "speaker_id": row["speaker_id"], "session_id": row["session_id"],
                       "duration_ms": row["duration_ms"], "english_number_words": row["english_number_words"],
                       "reference": row["reference"], "hypothesis": hypothesis,
                       "processing_seconds": round(seconds * int(row["duration_ms"]) / total_ms, 3)})
    return result, counts


def installed(prefix: Path = Path(sys.prefix)) -> dict[str, str]:
    """Version and build of the conda packages in PACKAGES, read from the records of the environment."""
    found = {}
    for path in sorted((Path(prefix) / "conda-meta").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("name") in PACKAGES:
            found[record["name"]] = f"{record['version']} {record['build']}"
    return found


def mfa(work: Path, name: str, arguments: list) -> float:
    """Run one `mfa` command with its temporary files below `work`; return the seconds it took."""
    executable = shutil.which("mfa")
    if executable is None:
        raise FileNotFoundError("the command `mfa` was not found; activate the environment of environment-gmm-hmm.yml")
    (Path(work) / "logs").mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "MFA_ROOT_DIR": str(Path(work).resolve() / "mfa_root"), "PYTHONUTF8": "1"}
    before = time.perf_counter()
    with (Path(work) / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
        code = subprocess.run([executable, name, *map(str, arguments)], env=environment, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    if code != 0:
        raise RuntimeError(f"mfa {name} failed with exit code {code}; see {Path(work) / 'logs' / (name + '.log')}")
    return round(time.perf_counter() - before, 1)


def _state(work: Path, **changes) -> dict:
    """What the steps so far recorded in `work/state.json`, after adding `changes`."""
    path = Path(work) / "state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if changes:
        state.update(changes)
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return state


def prepare(work: Path = WORK, raw: Path = RAW, clean: Path = CLEAN, train_speakers: list[str] | None = None,
            force: bool = False) -> dict:
    """Write the training corpus, the lexicon and the settings of the two models to `work`."""
    work = Path(work)
    if (work / "state.json").exists() and not force:
        raise FileExistsError(f"{work} already holds a run; use --force to replace it")
    training = [row for row in read_split(clean, "train")
                if (not train_speakers or row["speaker_id"] in train_speakers) and transcript(row)]
    if not training:
        raise ValueError("no training clip is left")
    for name in WORK_FOLDERS:                                    # only what the steps of this module write
        shutil.rmtree(work / name, ignore_errors=True)
    (work / "state.json").unlink(missing_ok=True)
    export_corpus(raw, training, work / "corpus" / "train", with_text=True)
    lexicon = build_lexicon([transcript(row) for row in training])
    write_lexicon(work / "lexicon.txt", lexicon)
    for name, content in (("train_config.json", TRAIN_CONFIG), ("lm_config.json", LM_CONFIG)):
        (work / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8", newline="\n")
    return _state(work, training={"clips": len(training), "speakers": len({row["speaker_id"] for row in training}),
                                  "audio_seconds": round(sum(int(row["duration_ms"]) for row in training) / 1000, 3),
                                  "words": sum(len(transcript(row).split()) for row in training),
                                  "only_speakers": sorted(train_speakers) if train_speakers else None},
                  lexicon={"words": len(lexicon), "units": sorted({unit for units in lexicon.values() for unit in units})})


def train(work: Path = WORK, jobs: int = 4, run=mfa) -> dict:
    """Estimate the language model and train the acoustic model on `work/corpus/train`."""
    work = Path(work)
    corpus, lexicon = work / "corpus" / "train", work / "lexicon.txt"
    started = _utc_now()
    lm_seconds = run(work, "train_lm", [corpus, work / "language_model.zip", "--dictionary_path", lexicon,
                                        "--config_path", work / "lm_config.json", "-j", jobs, "--clean", "--overwrite"])
    seconds = run(work, "train", [corpus, lexicon, work / "acoustic_model.zip", "--config_path",
                                  work / "train_config.json", "-j", jobs, "--clean", "--overwrite"])
    return _state(work, train={"started_utc": started, "finished_utc": _utc_now(), "jobs": jobs,
                               "language_model_seconds": lm_seconds, "acoustic_model_seconds": seconds})


def decode(work: Path = WORK, raw: Path = RAW, clean: Path = CLEAN, split: str = "validation",
           limit: int | None = None, final_test: bool = False, jobs: int = 4, run=mfa) -> dict:
    """Write the audio of `split` to `work/corpus/<split>` and transcribe it with the trained models."""
    if split not in SPLITS[1:]:
        raise ValueError(f"the split to decode must be one of {SPLITS[1:]}")
    if split == "test" and not final_test:
        raise FinalTestRequired("the test split is only for the final evaluation after everything is frozen; "
                                "pass --final-test to run it")
    work = Path(work)
    state = _state(work)
    if "train" not in state:
        raise ValueError(f"{work} has no trained models; run the steps prepare and train first")
    rows = read_split(clean, split)[:limit]
    for folder in (work / "corpus" / split, work / "output" / split):
        shutil.rmtree(folder, ignore_errors=True)
    export_corpus(raw, rows, work / "corpus" / split, with_text=False)
    (work / "decode_config.json").write_text(json.dumps(DECODE_CONFIG, indent=2) + "\n", encoding="utf-8", newline="\n")
    started = _utc_now()
    seconds = run(work, "transcribe", [work / "corpus" / split, work / "lexicon.txt", work / "acoustic_model.zip",
                                       work / "language_model.zip", work / "output" / split, "--config_path",
                                       work / "decode_config.json", "--output_type", "transcription", "-j", jobs,
                                       "--clean", "--overwrite"])
    done = {"limit": limit, "final_test": bool(final_test), "started_utc": started, "finished_utc": _utc_now(),
            "jobs": jobs, "seconds": seconds, "config": DECODE_CONFIG}
    return _state(work, decode={**state.get("decode", {}), split: done})


def collect(work: Path = WORK, clean: Path = CLEAN, raw: Path = RAW, results: Path = RESULTS,
            split: str = "validation", force: bool = False) -> Path:
    """Write `run.json` and `predictions.csv` of a decoded split; return their folder."""
    work = Path(work)
    state = _state(work)
    if split not in state.get("decode", {}):
        raise ValueError(f"{work} has no output for the split {split!r}; run the step decode first")
    decoded = state["decode"][split]
    limit = decoded["limit"]
    folder = Path(results) / GROUP / SYSTEM / split
    if (folder / "run.json").exists() and not force:
        raise FileExistsError(f"{folder} already holds a run; use --force to replace it")
    in_split = read_split(clean, split)
    rows = in_split[:limit]
    lexicon = read_lexicon(work / "lexicon.txt")
    ordered, output = predictions(rows, read_transcripts(work / "output" / split), decoded["seconds"])
    output["empty_hypotheses"] = sum(not row["hypothesis"] for row in ordered)
    folder.mkdir(parents=True, exist_ok=True)
    write_manifest(folder / "predictions.csv", PREDICTION_COLUMNS, ordered)
    packages = installed()
    manifest = {
        "system": SYSTEM,
        "group": GROUP,
        "checkpoint": {
            "description": "trained here on the training split; no published checkpoint",
            "toolkit": "Montreal Forced Aligner " + packages.get("montreal-forced-aligner", "not recorded"),
            "kaldi": packages.get("kaldi", "not recorded"),
            "training_data": state["training"],
            "acoustic_model_sha256": sha256_of(work / "acoustic_model.zip"),
            "language_model_sha256": sha256_of(work / "language_model.zip"),
        },
        "settings": {
            "features": "MFCC without pitch, MFA defaults otherwise",
            "training_recipe": TRAIN_CONFIG,
            "lexicon": {"kind": "graphemic: every word of the training transcripts as its letters, one letter per "
                                "unit; no pronunciation dictionary, no G2P model", **state["lexicon"]},
            "language_model": {**LM_CONFIG, "tool": "mfa train_lm", "text": "training transcripts only",
                               "vocabulary": "the lexicon"},
            "text": "column reference after huasr.text.normalize",
            "decoding": {"command": "mfa transcribe --output_type transcription", **decoded["config"],
                         "other_options": "MFA defaults",
                         "speaker_adaptation": "clips grouped by speaker id; no reference text given to MFA"},
            "hypothesis": "output of MFA with placeholder tokens such as <unk> removed",
            "processing_seconds": "wall time of mfa transcribe (features, decoding graph, decoding, rescoring) "
                                  "shared out over the clips by duration; not one clip at a time",
            "jobs": {"training": state["train"]["jobs"], "decoding": decoded["jobs"]},
            "fine_tuned": False,
        },
        "seed": 0,  # every run record has this field; no seed is given to MFA
        "split": split,
        "final_test": decoded["final_test"],
        "data": {"clean_manifest": (Path(clean) / "manifest.csv").as_posix(),
                 "clean_manifest_sha256": sha256_of(Path(clean) / "manifest.csv"),
                 "raw_manifest_sha256": sha256_of(Path(raw) / "manifest.csv"),
                 "reference_column": "reference", "clips_in_split": len(in_split)},
        "clips": len(ordered),
        "limit": limit,
        "complete": len(ordered) == len(in_split) and state["training"]["only_speakers"] is None,
        "audio_seconds": round(sum(int(row["duration_ms"]) for row in ordered) / 1000, 3),
        "processing_seconds": round(sum(row["processing_seconds"] for row in ordered), 3),
        "load_seconds": 0.0,
        "training": state["train"],
        "out_of_vocabulary": out_of_vocabulary(lexicon, [row["reference"] for row in rows]),
        "output": output,
        "hardware": {"processor": cpu_name(), "machine": platform.machine(), "logical_cpus": os.cpu_count(),
                     "system": platform.platform()},
        "packages": {"python": platform.python_version(), **packages},
        "started_utc": decoded["started_utc"],
        "finished_utc": decoded["finished_utc"],
    }
    (folder / "run.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
                                     newline="\n")
    return folder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "decode", "collect"):
        command = commands.add_parser(name)
        command.add_argument("--work", type=Path, default=WORK, help="folder for the corpus, the models and the "
                             "temporary files of MFA; several GB, not tracked")
        if name != "train":
            command.add_argument("--raw", type=Path, default=RAW)
            command.add_argument("--clean", type=Path, default=CLEAN)
        if name in ("prepare", "collect"):
            command.add_argument("--force", action="store_true", help="replace what is already there")
        if name in ("train", "decode"):
            command.add_argument("--jobs", type=int, default=4, help="parallel jobs of MFA")
        if name in ("decode", "collect"):
            command.add_argument("--split", default="validation", choices=SPLITS[1:])
    commands.choices["prepare"].add_argument("--train-speakers", nargs="+",
                                             help="train on these speakers only (trial run)")
    decoder = commands.choices["decode"]
    decoder.add_argument("--final-test", action="store_true",
                         help="needed for the test split: the final evaluation, after everything is frozen")
    decoder.add_argument("--limit", type=int, help="decode only the first N clips (trial run)")
    commands.choices["collect"].add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            state = prepare(args.work, args.raw, args.clean, args.train_speakers, args.force)
            print(f"training clips: {state['training']['clips']}, lexicon: {state['lexicon']['words']} words")
        elif args.command == "train":
            print(json.dumps(train(args.work, args.jobs)["train"]))
        elif args.command == "decode":
            state = decode(args.work, args.raw, args.clean, args.split, args.limit, args.final_test, args.jobs)
            print(json.dumps(state["decode"][args.split]))
        else:
            folder = collect(args.work, args.clean, args.raw, args.results, args.split, args.force)
            print(f"wrote {folder / 'run.json'} and {folder / 'predictions.csv'}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
