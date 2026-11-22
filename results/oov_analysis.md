# Lexical coverage and OOV analysis

Training vocabulary is the set of normalized word types in clean VoxPopuli train references, using the project scoring tokenizer. OOV means absent from that vocabulary.

Per-clip hypothesis coverage is multiset overlap: each reference token occurrence receives a match only up to the number of identical hypothesis occurrences in the same clip. This is a descriptive proxy, not a causal measure of OOV effects.

## Test reference vocabulary

- Training vocabulary types: **20,021**
- Test clips: **897**
- Test reference tokens: **17,406**
- OOV reference tokens: **2,836 (16.29%)**
- IV reference tokens: **14,570 (83.71%)**

## System comparison

| System | Clips | WER | CER | IV tokens | IV matched | IV coverage | OOV tokens | OOV matched | OOV coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canary-1b-v2 | 897 | 0.0912 | 0.0377 | 14,570 | 13,709 | 94.09% | 2,836 | 2,381 | 83.96% |
| faster-whisper-base | 897 | 0.5205 | 0.1503 | 14,570 | 8,903 | 61.11% | 2,836 | 811 | 28.60% |
| faster-whisper-small | 897 | 0.2943 | 0.0853 | 14,570 | 11,485 | 78.83% | 2,836 | 1,489 | 52.50% |
| gmm-hmm-mfa | 897 | 0.8788 | 0.7153 | 14,570 | 2,863 | 19.65% | 2,836 | 0 | 0.00% |
| hf-whisper-small | 897 | 0.2960 | 0.0914 | 14,570 | 11,507 | 78.98% | 2,836 | 1,501 | 52.93% |
| parakeet-tdt-0.6b-v3 | 897 | 0.2190 | 0.0974 | 14,570 | 13,462 | 92.40% | 2,836 | 2,281 | 80.43% |
| qwen3-asr-1.7b | 897 | 0.2802 | 0.1242 | 14,570 | 11,905 | 81.71% | 2,836 | 1,585 | 55.89% |
| wav2vec2-xlsr-53-hungarian | 897 | 0.4687 | 0.1170 | 14,570 | 9,334 | 64.06% | 2,836 | 1,116 | 39.35% |
| hf-whisper-small-finetuned | 897 | 0.2069 | 0.0654 | 14,570 | 12,661 | 86.90% | 2,836 | 1,747 | 61.60% |

## Limitations

- Exact token overlap is a descriptive measure, not edit-distance attribution and not causal evidence that OOV status produced an error.
- Acoustic difficulty, morphology, named entities, and language-model behavior also affect recognition.
- OOV refers to word types absent from training references, not tokenizer-level subword OOV.
- WER and CER are recomputed with the repository scoring code.

## Reproduction

`python scripts/oov_analysis.py`

Prediction files included: **9**. All metrics are generated from committed input files, not hard-coded.
