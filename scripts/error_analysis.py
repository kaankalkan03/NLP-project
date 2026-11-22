"""Analyze saved Hungarian ASR predictions without running any models."""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path

from huasr.metrics import EditCounts, align, edit_counts
from huasr.text import normalize, to_chars, to_words

SPLITS = ("validation", "test")
GROUPS = ("baselines", "own")
BASELINE = ("baselines", "hf-whisper-small")
TUNED = ("own", "hf-whisper-small-finetuned")
KINDS = ("accent-only", "suffix-only", "other substitution", "deletion")
LONG_MIN = 12
LOOP_MIN_EXTRA = 5
LOOP_MAX_PERIOD = 8
LOOP_MIN_COPIES = 3


@dataclass(frozen=True)
class Prediction:
    clip_id: str
    reference: str
    hypothesis: str


@dataclass
class ClipAnalysis:
    prediction: Prediction
    word: EditCounts
    char: EditCounts
    ref_words: list[str]
    hyp_words: list[str]
    ref_labels: list[str]
    loop: tuple[tuple[str, ...], int] | None
    overlong: bool


def read_predictions(path: Path) -> dict[str, Prediction]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = csv.DictReader(stream)
        required = {"clip_id", "reference", "hypothesis"}
        if not required.issubset(records.fieldnames or []):
            raise ValueError(f"{path}: expected columns {sorted(required)}")
        by_id = {}
        for line, row in enumerate(records, start=2):
            if any(row[field] is None for field in required) or None in row:
                raise ValueError(f"{path}:{line}: malformed CSV row")
            clip = Prediction(*(row[k] for k in ("clip_id", "reference", "hypothesis")))
            if not clip.clip_id.strip() or clip.clip_id in by_id:
                raise ValueError(f"{path}:{line}: blank or duplicate clip_id")
            by_id[clip.clip_id] = clip
    if not by_id:
        raise ValueError(f"{path}: no predictions")
    return by_id


def discover_runs(results_dir: Path) -> dict[str, dict[tuple[str, str], Path]]:
    """Discover prediction CSVs and reject systems with a missing split."""
    results_dir = Path(results_dir)
    runs = {split: {} for split in SPLITS}
    for group in GROUPS:
        for path in sorted((results_dir / group).glob("*/*/predictions.csv")):
            if path.parent.name in runs:
                runs[path.parent.name][group, path.parent.parent.name] = path
    if not runs["validation"] or runs["validation"].keys() != runs["test"].keys():
        raise ValueError("Expected the same nonempty set of systems in validation and test")
    return runs


def substitution_type(ref: str, hyp: str) -> str:
    """Classify two distinct normalized tokens; accent and suffix are disjoint."""
    if ref == hyp:
        raise ValueError("Equal tokens are hits, not substitutions")
    if normalize(ref, strip_diacritics=True) == normalize(hyp, strip_diacritics=True):
        return "accent-only"
    prefix = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(ref, hyp)))
    # Conservative surface-form criterion, not a Hungarian morphological parser.
    if ref.isalpha() and hyp.isalpha() and prefix >= 4 and 2 * prefix >= min(len(ref), len(hyp)):
        if max(len(ref) - prefix, len(hyp) - prefix) <= 5:
            return "suffix-only"
    return "other substitution"


def _repeated_run(words: list[str], phrase: tuple[str, ...]) -> int:
    width = len(phrase)
    counts = [0] * (len(words) + 1)
    for i in range(len(words) - width, -1, -1):
        if tuple(words[i:i + width]) == phrase:
            counts[i] = 1 + counts[i + width]
    return max(counts)


def repetition_loop(ref: list[str], hyp: list[str]) -> tuple[bool, tuple[tuple[str, ...], int] | None]:
    """Flag substantial excess length supported by an adjacent word/phrase run."""
    overlong = len(hyp) - len(ref) >= LOOP_MIN_EXTRA and 2 * len(hyp) >= 3 * len(ref)
    if not ref or not overlong:
        return False, None
    candidates = []
    seen = set()
    for width in range(1, min(LOOP_MAX_PERIOD, len(hyp) // LOOP_MIN_COPIES) + 1):
        for i in range(len(hyp) - LOOP_MIN_COPIES * width + 1):
            phrase = tuple(hyp[i:i + width])
            if tuple(hyp[i:i + LOOP_MIN_COPIES * width]) != phrase * LOOP_MIN_COPIES:
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            copies = _repeated_run(hyp, phrase)
            # At least two more adjacent copies than found in the reference.
            if copies - _repeated_run(ref, phrase) >= 2:
                candidates.append((width * copies, copies, -width, phrase))
    if not candidates:
        return True, None
    _, copies, _, phrase = max(candidates)
    return True, (phrase, copies)


def analyze_clip(prediction: Prediction) -> ClipAnalysis:
    ref_words = to_words(prediction.reference)
    hyp_words = to_words(prediction.hypothesis)
    labels = []
    operations = Counter()
    for op, ref, hyp in align(ref_words, hyp_words):
        operations[op] += 1
        if op == "sub":
            labels.append(substitution_type(ref, hyp))
        elif op != "ins":
            labels.append("deletion" if op == "del" else "hit")
    word = EditCounts(operations["sub"], operations["del"], operations["ins"], operations["hit"])
    char = edit_counts(to_chars(prediction.reference), to_chars(prediction.hypothesis))
    overlong, loop = repetition_loop(ref_words, hyp_words)
    return ClipAnalysis(prediction, word, char, ref_words, hyp_words, labels, loop, overlong)


def analyze_run(path: Path) -> dict[str, ClipAnalysis]:
    return {key: analyze_clip(p) for key, p in read_predictions(path).items()}


def length_stats(clips: dict[str, ClipAnalysis]) -> dict[str, Counter]:
    stats = {"short": Counter(), "long": Counter()}
    for clip in clips.values():
        for word, label in zip(clip.ref_words, clip.ref_labels):
            size = "long" if sum(ch.isalpha() for ch in word) >= LONG_MIN else "short"
            stats[size]["total"] += 1
            stats[size]["substitutions"] += label not in ("hit", "deletion")
            stats[size]["deletions"] += label == "deletion"
    return stats


def paired_stats(baseline: dict[str, ClipAnalysis], tuned: dict[str, ClipAnalysis]) -> dict:
    """Match by clip ID, then by position in the identical reference transcript."""
    if not baseline or baseline.keys() != tuned.keys():
        raise ValueError("Paired comparison requires identical nonempty clip-ID sets")
    transitions = Counter()
    inserts = Counter()
    loop_states = Counter()
    for clip_id in sorted(baseline):
        old, new = baseline[clip_id], tuned[clip_id]
        if old.ref_words != new.ref_words:
            raise ValueError(f"Reference mismatch for paired clip {clip_id!r}")
        transitions.update(zip(old.ref_labels, new.ref_labels))
        inserts["baseline"] += old.word.insertions
        inserts["tuned"] += new.word.insertions
        inserts["clips fewer"] += new.word.insertions < old.word.insertions
        inserts["clips unchanged"] += new.word.insertions == old.word.insertions
        inserts["clips more"] += new.word.insertions > old.word.insertions
        loop_states[bool(old.loop), bool(new.loop)] += 1
    return {"transitions": transitions, "insertions": inserts, "loops": loop_states, "clips": len(baseline)}


def _cell(value: object) -> str:
    # Escape source transcript content, not the Markdown structural elements.
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(
        "\\", "&#92;").replace("|", "&#124;").replace("`", "&#96;").replace(
        "*", "&#42;").replace("_", "&#95;").replace("\r", " ").replace("\n", " ")


def _table(headers, rows) -> str:
    return "\n".join(
        "| " + " | ".join(_cell(cell) for cell in row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def _pct(num, denom) -> str:
    return f"{100 * num / denom:.2f}%" if denom else "n/a"


def _name(run: tuple[str, str]) -> str:
    return "/".join(run)


def _paired_report(value: dict, baseline: tuple[str, str], tuned: tuple[str, str]) -> list[str]:
    trs, ins, loops = (value[k] for k in ("transitions", "insertions", "loops"))
    labels = ("hit", *KINDS)
    rows = []
    for kind in KINDS:
        had = sum(trs[kind, new] for new in labels)
        has = sum(trs[old, kind] for old in labels)
        stayed = trs[kind, kind]
        fixed = trs[kind, "hit"]
        rows.append((kind, had, has, has - had, fixed, stayed, had - stayed - fixed,
                     trs["hit", kind]))
    return [
        "### 6. Paired Whisper small comparison",
        f"Baseline: `{_name(baseline)}`. Fine-tuned: `{_name(tuned)}`. "
        f"Matched clips: {value['clips']}. These are the project’s paired Hugging Face runs.",
        "A fix means that the word at the same reference position became a hit. "
        "A changed label is still an error, even if the overall count for that type fell.",
        _table(("Type", "Baseline", "Fine-tuned", "Delta", "Fixed", "Same type",
                "Changed to another error", "New after a hit"), rows),
        "Full paired transition counts (rows are baseline, columns are fine-tuned):",
        _table(("Baseline → fine-tuned", *labels),
               [(old, *(trs[old, new] for new in labels)) for old in labels]),
        f"Insertions (not tied to a reference position): {ins['baseline']} → {ins['tuned']}. "
        f"Per clip: fewer {ins['clips fewer']}, unchanged {ins['clips unchanged']}, more {ins['clips more']}.",
        f"Loop candidates: cleared {loops[True, False]}, remained {loops[True, True]}, "
        f"new {loops[False, True]}, neither {loops[False, False]}.",
    ]


def render_report(runs: dict[str, dict[tuple[str, str], dict[str, ClipAnalysis]]], *,
                  top_n: int = 20, baseline=BASELINE, tuned=TUNED) -> str:
    """Generate deterministic Markdown without changing files or training systems."""
    if top_n < 1:
        raise ValueError("top_n must be positive")
    sections = [
        "# Error analysis",
        "Analysis of saved predictions only. Validation and test are reported separately.",
        "All scoring uses `huasr.text` and `huasr.metrics` with accents retained. "
        "WER and CER are total edits divided by total reference words and characters, respectively. "
        "Empty-reference rows, if any, are listed separately rather than included in rates.",
        "Long means at least 12 Unicode letters in the normalized reference token. "
        "The word-length error rate is (substitutions + deletions) / reference tokens of that length; "
        "insertions do not belong to a reference length bin.",
        "Accent-only means distinct aligned words become identical after the project's accent stripping. "
        "After excluding those, suffix-only means two alphabetic words share an exact initial prefix "
        "of at least four letters and at least half the shorter word, with no more than five "
        "letters remaining on either side. Empty tails are allowed. This is an ending-only "
        "surface heuristic, not a morphological analysis.",
        "An overlong hypothesis is at least 1.5 times the reference length and has at least five "
        "extra normalized words. A loop candidate also repeats the same 1–8-word phrase at least "
        "three times adjacently and at least twice more than the longest reference run. "
        "These flags suggest repetition, but are not judged by listening.",
    ]
    for split in SPLITS:
        systems = runs[split]
        sections.append(f"## {split.capitalize()}")
        totals, lengths, substitutions, loops = [], [], [], []
        ranking_sections = []
        long_higher = []
        loop_examples = []
        for key, clips in sorted(systems.items()):
            scored = [c for c in clips.values() if c.ref_words]
            omissions = len(clips) - len(scored)
            labels = Counter(label for c in scored for label in c.ref_labels)
            word = sum((c.word for c in scored), EditCounts())
            char = sum((c.char for c in scored), EditCounts())
            name = _name(key)
            totals.append((name, len(clips), omissions, _pct(word.errors, word.reference_units),
                           _pct(char.errors, char.reference_units)))
            stats = length_stats(clips)
            short, long = stats["short"], stats["long"]
            short_wrong = short["substitutions"] + short["deletions"]
            long_wrong = long["substitutions"] + long["deletions"]
            if short["total"] and long["total"] and long_wrong * short["total"] > short_wrong * long["total"]:
                long_higher.append(name)
            for size, counts in stats.items():
                wrong = counts["substitutions"] + counts["deletions"]
                lengths.append((name, size, counts["total"], counts["substitutions"],
                                counts["deletions"], wrong, _pct(wrong, counts["total"])))
            substitutions.append((name, labels["suffix-only"], labels["accent-only"],
                                  labels["other substitution"], word.substitutions,
                                  _pct(labels["suffix-only"], word.substitutions),
                                  _pct(labels["accent-only"], word.substitutions)))
            loop_clips = sorted((c for c in clips.values() if c.loop), key=lambda c: (
                -(len(c.hyp_words) - len(c.ref_words)), c.prediction.clip_id))
            loops.append((name, sum(c.overlong for c in clips.values()), len(loop_clips),
                          _pct(len(loop_clips), len(scored))))
            for c in loop_clips[:5]:
                phrase, copies = c.loop
                loop_examples.append((name, c.prediction.clip_id, len(c.ref_words), len(c.hyp_words),
                                      " ".join(phrase), copies))
            top = sorted(scored, key=lambda c: (-c.word.errors, -c.word.rate, c.prediction.clip_id))[:top_n]
            ranking_sections.extend([
                f"#### {name}",
                _table(("#", "Clip ID", "Word edits", "WER", "CER", "Reference", "Hypothesis"),
                       [(i, c.prediction.clip_id, c.word.errors, _pct(c.word.errors, c.word.reference_units),
                         _pct(c.char.errors, c.char.reference_units), c.prediction.reference,
                         c.prediction.hypothesis) for i, c in enumerate(top, 1)]),
            ])
            if omissions:
                ranking_sections.extend(["Empty-reference rows (WER/CER undefined):",
                    _table(("Clip ID", "Insertions", "Hypothesis"),
                           [(c.prediction.clip_id, c.word.insertions, c.prediction.hypothesis)
                            for c in sorted(clips.values(), key=lambda c: c.prediction.clip_id)
                            if not c.ref_words])])
        sections.extend([
            _table(("System", "Clips", "Empty refs", "WER", "CER"), totals),
            f"### 1. Top {top_n} clips per system",
            "Ranked by count of word edits, then higher WER, then stable clip ID. "
            "Transcripts are the original saved strings, not the normalized scoring strings.",
            *ranking_sections,
            "### 2. Reference word length",
            f"Long reference tokens had a higher error share in {len(long_higher)} of "
            f"{len(systems)} systems on this split. This is descriptive, not a significance test.",
            _table(("System", "Length", "Ref tokens", "Substituted", "Deleted", "Wrong", "Wrong rate"), lengths),
            "### 3–4. Ending-only and accent-only substitutions",
            _table(("System", "Suffix-only", "Accent-only", "Other substitutions",
                    "Total substitutions", "Suffix-only share", "Accent-only share"), substitutions),
            "### 5. Repetition loops",
            _table(("System", "Overlong clips", "Loop candidates", "Share of nonempty refs"), loops),
        ])
        if loop_examples:
            sections.extend(["Up to five examples per system, ranked by excess words:",
                             _table(("System", "Clip ID", "Ref words", "Hyp words", "Repeated phrase", "Copies"),
                                    loop_examples)])
        sections.extend(_paired_report(paired_stats(systems[baseline], systems[tuned]), baseline, tuned))
    return "\n\n".join(sections) + "\n"


def analyze_results(results_dir: Path) -> dict:
    runs = {}
    for split, paths in discover_runs(results_dir).items():
        systems = {}
        shared_refs = None
        for key, path in sorted(paths.items()):
            clips = analyze_run(path)
            refs = {clip_id: clip.prediction.reference for clip_id, clip in clips.items()}
            if shared_refs is not None and refs != shared_refs:
                raise ValueError(f"{path}: clip IDs or reference transcripts differ within {split}")
            shared_refs = refs
            systems[key] = clips
        runs[split] = systems
    return runs


def build_report(repo: Path | str) -> str:
    """Analyze a repository's saved runs and return the Markdown report."""
    return render_report(analyze_results(Path(repo) / "results"))
