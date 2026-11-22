"""Published checkpoints of four more model families, run as baselines through `transformers`.

    python -m huasr.pretrained list            the systems: checkpoint, revision, settings
    python -m huasr.pretrained fetch SYSTEM    download the pinned checkpoint of one system (once)

Four systems behind the contract of `huasr.systems`, all in the group
"baselines" and run like every other system with
`python -m huasr.evaluate --system <name>`. Nothing is trained or tuned here.

    wav2vec2-xlsr-53-hungarian   wav2vec 2.0 encoder with a CTC head, fine-tuned for Hungarian by its publisher
    canary-1b-v2                 FastConformer encoder with an attention decoder (multitask, 25 languages)
    parakeet-tdt-0.6b-v3         FastConformer encoder with a transducer (TDT) decoder (25 languages)
    qwen3-asr-1.7b               audio encoder with a Qwen3 language model as decoder (30 languages)

How each one is run is taken from its model card and from the documentation
of `transformers`, which implements all four in the pinned release, so one
set of packages serves them all (`requirements-pretrained.txt`). The packages
that two cards name first, `nemo_toolkit` and `qwen-asr`, are not used: they
pin other versions of the packages this project runs on. For Qwen3-ASR this
means the checkpoint that Qwen publishes for the Transformers library,
`Qwen/Qwen3-ASR-1.7B-hf`; `Qwen/Qwen3-ASR-1.7B` itself can only be loaded
with `qwen-asr`. That the two hold identical weights has not been checked.

Language. A model is never run with a wrong or a missing language setting:

- wav2vec 2.0: the checkpoint is Hungarian only; there is nothing to set.
- Canary: source and target language are both set to Hungarian, which is how
  the model is told to transcribe and not to translate. Before every clip the
  decoder prompt is checked to hold the Hungarian token twice and no other
  language token.
- Parakeet: the model detects the language itself and takes no language
  setting (its model card). This is recorded in the run manifest.
- Qwen3-ASR: Hungarian is forced by the prefix the model was trained with,
  `language Hungarian<asr_text>`; before every clip the prompt is checked to
  end with it.

Clips longer than what a model reads at once (`window_seconds`). Canary is
used on at most 40 seconds: beyond that its own toolkit cuts the audio too
(model card). A longer clip is cut into the fewest equal parts that fit, as
for Whisper (`huasr.whisper_hf.windows`), and the texts are joined; a word at
a cut can come out wrong. The other three take the whole clip: wav2vec 2.0
has no window, Parakeet reads up to 24 minutes with full attention and
Qwen3-ASR handles long audio (model cards); the longest clip here is under a
minute.

Text. The hypothesis is the decoded token sequence without special tokens,
and nothing else is done to it: no language model on top of the CTC output,
and not the repetition clean-up that the Qwen3-ASR processor offers. Case and
punctuation are left to the scoring policy, which is the same for every
system.

Checkpoints are taken from the local Hugging Face cache only, at the pinned
revision, and the weight file must have the pinned size and SHA-256; `fetch`
is the one command that downloads. A model whose weights are not all found in
the checkpoint is refused, so a layer can never run with random values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import LANGUAGE
from .systems import BASELINES, cached_checkpoint, sha256_of
from .whisper_hf import SAMPLE_RATE, accelerator, versions, windows

LANGUAGE_NAME = "Hungarian"
TEXT = "decoded tokens without special tokens; no other change"


@dataclass(frozen=True)
class Checkpoint:
    """A published checkpoint at one revision: its weight file as the repository lists it, and what else is read."""

    repository: str
    revision: str
    weights: str
    weights_bytes: int
    weights_sha256: str
    other_files: tuple[str, ...]

    @property
    def files(self) -> tuple[str, ...]:
        """Everything `fetch` downloads. The repositories hold more (other formats of the same weights)."""
        return (*self.other_files, self.weights)


WAV2VEC2 = Checkpoint(
    "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian", "2bd0786e71cad097c4e6ce9b3e45b40c74ad8f37",
    "pytorch_model.bin", 1_262_101_911, "9ab6510ff7c1c59ff751c63047eba527956df889284321f87a73cbf322012932",
    ("config.json", "preprocessor_config.json", "special_tokens_map.json", "vocab.json"))
CANARY = Checkpoint(
    "nvidia/canary-1b-v2", "d455706339a6b32e1aa40f82c713a482a0c938e2",
    "model.safetensors", 3_916_173_816, "4f6d381f6a939b95e9f8516212148b701e9b3805d26462c4c3e9e34f3d89b6f8",
    ("config.json", "generation_config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json",
     "tokenizer_config.json"))
PARAKEET = Checkpoint(
    "nvidia/parakeet-tdt-0.6b-v3", "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    "model.safetensors", 2_508_311_120, "3a2026366188c8c68598edbbff92f8d11590a08e0ae2e6775544e7b07d6a5e11",
    ("config.json", "generation_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json"))
QWEN = Checkpoint(
    "Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    "model.safetensors", 4_076_193_080, "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1",
    ("config.json", "generation_config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json",
     "tokenizer_config.json"))


def snapshot(checkpoint: Checkpoint, download: bool = False) -> Path:
    """Folder of `checkpoint` in the Hugging Face cache, at its pinned revision.

    Nothing is downloaded unless `download` is set (the `fetch` command); a
    checkpoint that is not in the cache is an error.
    """
    if not download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import snapshot_download

    try:
        folder = Path(snapshot_download(checkpoint.repository, revision=checkpoint.revision,
                                        allow_patterns=list(checkpoint.files), local_files_only=not download))
    except Exception as error:  # the library has its own error types; offline, each of them means "not there"
        if download:
            raise
        raise FileNotFoundError(f"{checkpoint.repository} at {checkpoint.revision} is not in the local Hugging Face "
                                f"cache; run python -m huasr.pretrained fetch <system>") from error
    missing = [name for name in checkpoint.files if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{folder} lacks {', '.join(missing)}; run python -m huasr.pretrained fetch <system>")
    return folder


def published_weights(checkpoint: Checkpoint, folder: Path) -> dict:
    """Size and SHA-256 of the weight file in `folder`; an error unless they are the pinned ones."""
    weights = Path(folder) / checkpoint.weights
    size = weights.stat().st_size
    found = {"weights_file": checkpoint.weights, "weights_bytes": size,
             "weights_sha256": sha256_of(weights) if size == checkpoint.weights_bytes else "not computed"}
    if (found["weights_bytes"], found["weights_sha256"]) != (checkpoint.weights_bytes, checkpoint.weights_sha256):
        raise ValueError(f"{weights} is not the weight file of {checkpoint.repository} at {checkpoint.revision}: "
                         f"{size} bytes, SHA-256 {found['weights_sha256']}")
    return found


def prompt_languages(tokens: list[str]) -> list[str]:
    """The language tokens (`<|xx|>`) of a Canary decoder prompt, in their order: source language, target language."""
    return [token for token in tokens if re.fullmatch(r"<\|[a-z]{2}\|>", token)]


class Pretrained:
    """A published checkpoint behind the system contract.

    A subclass names its checkpoint and its settings and says how the model is loaded (`_load`) and how one
    part of audio is turned into text (`_transcribe`).
    """

    group = BASELINES
    checkpoint: Checkpoint
    implementation: str                 # the class of `transformers` that runs the checkpoint
    dtype = "float32"                   # the precision the checkpoint is published in, and is run in
    window_seconds: int | None = None   # the most audio the model is given at once; None: the whole clip
    settings: dict = {}                 # language and decoding, for the run manifest
    unused_weights: tuple[str, ...] = ()  # weights of the model class that play no part in recognition

    def __init__(self, name: str, seed: int = 0):
        self.name, self.seed = name, seed
        self._folder = self._model = self._processor = self._device = None

    def folder(self) -> Path:
        """Folder the model is loaded from; fails if the checkpoint is not in the cache at its revision."""
        if self._folder is None:
            self._folder = snapshot(self.checkpoint)
        return self._folder

    def describe(self) -> dict:
        folder = self.folder()
        cached = cached_checkpoint(folder)
        if (cached["repository"], cached["revision"]) != (self.checkpoint.repository, self.checkpoint.revision):
            raise ValueError(f"{folder} is not {self.checkpoint.repository} at {self.checkpoint.revision}")
        longer = ("the whole clip is given to the model at once" if self.window_seconds is None
                  else "cut into the fewest equal parts of at most one window; texts joined")
        return {
            "checkpoint": {**cached, **published_weights(self.checkpoint, folder)},
            "settings": {"implementation": self.implementation, **accelerator(), "dtype": self.dtype, **self.settings,
                         "window_seconds": self.window_seconds, "longer_audio": longer, "text": TEXT,
                         "fine_tuned": False},
            "seed": self.seed,
            "packages": versions(),
        }

    def load(self) -> None:
        if self._model is not None:
            return
        folder = str(self.folder())
        import torch

        torch.manual_seed(self.seed)
        self._device = accelerator()["device"]
        self._load(folder)

    def weights_into(self, model_class, folder: str):
        """`model_class` with the weights of the checkpoint, ready to run; refused if a weight was not found."""
        import torch

        model, report = model_class.from_pretrained(folder, local_files_only=True, dtype=getattr(torch, self.dtype),
                                                    output_loading_info=True)
        absent = sorted(report["missing_keys"]) + sorted(str(entry[0]) for entry in report["mismatched_keys"])
        absent = [key for key in absent if not any(name in key for name in self.unused_weights)]
        if absent:
            raise ValueError(f"{self.name}: {len(absent)} weights of {model_class.__name__} were not found in the "
                             f"checkpoint and would be random, for example {', '.join(absent[:3])}")
        return model.to(self._device).eval()

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"{self.name} expects audio at 16 kHz, got {sample_rate} Hz")
        self.load()
        parts = [samples] if self.window_seconds is None else windows(samples, self.window_seconds * SAMPLE_RATE)
        texts = [self._transcribe(np.ascontiguousarray(part, dtype=np.float32)).strip() for part in parts]
        return " ".join(text for text in texts if text)

    def _load(self, folder: str) -> None:
        raise NotImplementedError

    def _transcribe(self, samples: np.ndarray) -> str:
        raise NotImplementedError


class Wav2Vec2CTC(Pretrained):
    """wav2vec 2.0 with a CTC head, decoded as in the script of its model card: the best token of every frame."""

    checkpoint = WAV2VEC2
    implementation = "transformers Wav2Vec2ForCTC"
    settings = {"language": f"{LANGUAGE} (the checkpoint is Hungarian only and takes no language setting)",
                "decoding": "greedy CTC: best token per frame, repeats merged, blank removed",
                "language_model": None}
    unused_weights = ("masked_spec_embed",)  # the mask vector of SpecAugment, used while training only

    def _load(self, folder: str) -> None:
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(folder, local_files_only=True)
        self._tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(folder, local_files_only=True)
        self._model = self.weights_into(Wav2Vec2ForCTC, folder)

    def _transcribe(self, samples: np.ndarray) -> str:
        import torch

        inputs = self._processor(samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self._model(inputs["input_values"].to(self._device),
                                 attention_mask=inputs["attention_mask"].to(self._device)).logits
        return self._tokenizer.batch_decode(torch.argmax(logits, dim=-1))[0]


class Canary(Pretrained):
    """Canary: the task is given by the decoder prompt; equal source and target language mean transcription."""

    checkpoint = CANARY
    implementation = "transformers CanaryForConditionalGeneration"
    window_seconds = 40
    request = {"source_language": LANGUAGE, "target_language": LANGUAGE, "punctuation": True}
    settings = {**request, "task": "transcription (source and target language are the same)",
                "decoding": "greedy, as the generation config of the checkpoint sets it",
                "max_new_tokens": "the positions of the decoder that the prompt leaves free"}

    def _load(self, folder: str) -> None:
        from transformers import AutoProcessor, CanaryForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self._model = self.weights_into(CanaryForConditionalGeneration, folder)

    def _transcribe(self, samples: np.ndarray) -> str:
        import torch

        inputs = self._processor.apply_transcription_request(audio=samples, **self.request).to(self._device)
        prompt = inputs["decoder_input_ids"]
        languages = prompt_languages(self._processor.tokenizer.convert_ids_to_tokens(prompt[0].tolist()))
        if languages != [f"<|{LANGUAGE}|>"] * 2:
            raise ValueError(f"{self.name}: the decoder prompt names the languages {languages}, not Hungarian twice")
        room = self._model.config.decoder_config.max_position_embeddings - prompt.shape[-1]
        with torch.no_grad():
            tokens = self._model.generate(**inputs, max_new_tokens=room)
        return self._processor.decode(tokens, skip_special_tokens=True)[0]


class ParakeetTDT(Pretrained):
    """Parakeet TDT v3: greedy transducer decoding. The model finds the language itself; there is no setting."""

    checkpoint = PARAKEET
    implementation = "transformers ParakeetForTDT"
    settings = {"language": "detected by the model; it takes no language setting",
                "decoding": "greedy TDT, as the checkpoint sets it"}

    def _load(self, folder: str) -> None:
        from transformers import AutoProcessor, ParakeetForTDT

        self._processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self._model = self.weights_into(ParakeetForTDT, folder)

    def _transcribe(self, samples: np.ndarray) -> str:
        import torch

        inputs = self._processor(samples, sampling_rate=SAMPLE_RATE).to(self._device, dtype=self._model.dtype)
        with torch.no_grad():
            output = self._model.generate(**inputs, return_dict_in_generate=True)
        return self._processor.decode(output.sequences, skip_special_tokens=True)[0]


class Qwen3ASR(Pretrained):
    """Qwen3-ASR: the language is forced by the prefix of the answer; the model then writes the transcript only."""

    checkpoint = QWEN
    implementation = "transformers Qwen3ASRForConditionalGeneration"
    dtype = "bfloat16"
    prefix = f"language {LANGUAGE_NAME}<asr_text>"
    max_new_tokens = 1024  # the checkpoint sets 512; raised so that long clips are not cut off (fixed before any run)
    settings = {"language": LANGUAGE_NAME, "language_forced_by": prefix, "context": None,
                "decoding": "greedy, as the generation config of the checkpoint sets it",
                "max_new_tokens": max_new_tokens}

    def _load(self, folder: str) -> None:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        self._processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
        self._model = self.weights_into(Qwen3ASRForConditionalGeneration, folder)

    def _transcribe(self, samples: np.ndarray) -> str:
        import torch

        inputs = self._processor.apply_transcription_request(audio=samples, language=LANGUAGE)
        ending = "".join(self._processor.tokenizer.decode(inputs["input_ids"][0][-16:]).split())
        if not ending.endswith("".join(self.prefix.split())):
            raise ValueError(f"{self.name}: the prompt does not end with {self.prefix!r} but with {ending[-40:]!r}")
        inputs = inputs.to(self._device, self._model.dtype)
        with torch.no_grad():
            output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self._processor.decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


CLASSES = {"wav2vec2-xlsr-53-hungarian": Wav2Vec2CTC, "canary-1b-v2": Canary, "parakeet-tdt-0.6b-v3": ParakeetTDT,
           "qwen3-asr-1.7b": Qwen3ASR}


def build(name: str) -> Pretrained:
    """The system registered as `name` in `huasr.systems.SYSTEMS`."""
    if name not in CLASSES:
        raise ValueError(f"unknown system {name!r}")
    return CLASSES[name](name)


def listing() -> dict:
    """What every system is, without touching a model."""
    return {name: {"group": cls.group, "repository": cls.checkpoint.repository, "revision": cls.checkpoint.revision,
                   "weights_file": cls.checkpoint.weights, "weights_bytes": cls.checkpoint.weights_bytes,
                   "weights_sha256": cls.checkpoint.weights_sha256, "implementation": cls.implementation,
                   "dtype": cls.dtype, "window_seconds": cls.window_seconds, **cls.settings}
            for name, cls in CLASSES.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="print the systems with checkpoint, revision and settings")
    fetcher = commands.add_parser("fetch", help="download the pinned checkpoint of one system (once)")
    fetcher.add_argument("system", choices=sorted(CLASSES))
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print(json.dumps(listing(), indent=2))
            return 0
        checkpoint = CLASSES[args.system].checkpoint
        folder = snapshot(checkpoint, download=True)
        weights = published_weights(checkpoint, folder)
        print(f"{checkpoint.repository} at {checkpoint.revision}\n{folder}\n{checkpoint.weights}: "
              f"{weights['weights_bytes']:,} bytes, SHA-256 {weights['weights_sha256']}")
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
