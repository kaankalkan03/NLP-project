"""Reproducible lexical coverage and OOV analysis for VoxPopuli test runs."""
from __future__ import annotations
import csv, re, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from huasr.text import to_words
from huasr.metrics import wer, cer
MANIFEST = ROOT / "data/clean/voxpopuli/manifest.csv"
PREDICTIONS = sorted((ROOT / "results/baselines").glob("*/test/predictions.csv")) + sorted((ROOT / "results/own").glob("*/test/predictions.csv"))
OUT = ROOT / "results/oov_analysis.md"
TOKEN_RE = re.compile(r"^\w+$", re.UNICODE)
def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))
def pct(n, d): return f"{100*n/d:.2f}%" if d else "n/a"
def main():
    if not MANIFEST.exists(): raise FileNotFoundError(MANIFEST)
    manifest = read_csv(MANIFEST)
    required = {"clip_id", "split", "reference", "speaker_id"}
    if not manifest or not required.issubset(manifest[0]): raise ValueError(f"Manifest needs columns {sorted(required)}")
    train = [r for r in manifest if r["split"] == "train"]
    test = [r for r in manifest if r["split"] == "test"]
    if not train or not test: raise ValueError("Non-empty train and test splits are required")
    vocab = {w for r in train for w in to_words(r["reference"]) if TOKEN_RE.match(w)}
    refs = {r["clip_id"]: r for r in test}
    if len(refs) != len(test): raise ValueError("Duplicate test clip_id in manifest")
    details = {cid: (to_words(r["reference"])) for cid, r in refs.items()}
    n_ref = sum(map(len, details.values()))
    n_oov = sum(w not in vocab for words in details.values() for w in words)
    lines = ["# Lexical coverage and OOV analysis", "",
      "Training vocabulary is the set of normalized word types in clean VoxPopuli train references, using the project scoring tokenizer. OOV means absent from that vocabulary.", "",
      "Per-clip hypothesis coverage is multiset overlap: each reference token occurrence receives a match only up to the number of identical hypothesis occurrences in the same clip. This is a descriptive proxy, not a causal measure of OOV effects.", "",
      "## Test reference vocabulary", "",
      f"- Training vocabulary types: **{len(vocab):,}**",
      f"- Test clips: **{len(test):,}**",
      f"- Test reference tokens: **{n_ref:,}**",
      f"- OOV reference tokens: **{n_oov:,} ({pct(n_oov,n_ref)})**",
      f"- IV reference tokens: **{n_ref-n_oov:,} ({pct(n_ref-n_oov,n_ref)})**", "",
      "## System comparison", "",
      "| System | Clips | WER | CER | IV tokens | IV matched | IV coverage | OOV tokens | OOV matched | OOV coverage |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    n_systems = 0
    for path in PREDICTIONS:
        rows = read_csv(path)
        if not rows: continue
        pred = {r["clip_id"]: r for r in rows}
        if len(pred) != len(rows): raise ValueError(f"Duplicate clip_id in {path}")
        if set(pred) != set(refs): raise ValueError(f"{path}: prediction clip IDs do not exactly match test manifest")
        ids = sorted(refs)
        rtexts = [refs[c]["reference"] for c in ids]
        htexts = [pred[c]["hypothesis"] for c in ids]
        wc, cc = wer(rtexts, htexts), cer(rtexts, htexts)
        iv_n = iv_m = oov_n = oov_m = 0
        for cid in ids:
            rc, hc = Counter(details[cid]), Counter(to_words(pred[cid]["hypothesis"]))
            for word, count in rc.items():
                matched = min(count, hc[word])
                if word in vocab: iv_n += count; iv_m += matched
                else: oov_n += count; oov_m += matched
        system = path.parts[-3]
        lines.append(f"| {system} | {len(ids):,} | {wc.rate:.4f} | {cc.rate:.4f} | {iv_n:,} | {iv_m:,} | {pct(iv_m,iv_n)} | {oov_n:,} | {oov_m:,} | {pct(oov_m,oov_n)} |")
        n_systems += 1
    if not n_systems: lines.append("| No test predictions found | - | - | - | - | - | - | - | - | - |")
    lines += ["", "## Limitations", "",
      "- Exact token overlap is a descriptive measure, not edit-distance attribution and not causal evidence that OOV status produced an error.",
      "- Acoustic difficulty, morphology, named entities, and language-model behavior also affect recognition.",
      "- OOV refers to word types absent from training references, not tokenizer-level subword OOV.",
      "- WER and CER are recomputed with the repository scoring code.", "",
      "## Reproduction", "", "`python scripts/oov_analysis.py`", "",
      f"Prediction files included: **{n_systems}**. All metrics are generated from committed input files, not hard-coded.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
if __name__ == "__main__": main()