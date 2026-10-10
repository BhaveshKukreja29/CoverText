"""Config-driven encode / decode / detect loop.

Cover prompts are split once, with ``seed``. Stego is produced only from those
prompts. The negative class is ``clean_contexts`` when that list is set (the
held-out human corpus), and the prompts themselves otherwise. Either pool is
split with the same seed, and the detector sees only its train half:

- a detector with ``fit`` (CLS) is trained on train-clean versus train-stego,
  separately for each payload size, because a head fit at 8 bits is not the
  detector for 64 bits;
- a detector with ``calibrate`` and no ``fit`` (PPL) is calibrated on train
  clean text only;
- a detector with neither (the stub) is used as it stands.

AUROC uses the held-out clean texts and the held-out stego. Decode accuracy,
capacity, and fluency are computed on that stego. Fluency is the perplexity
of the stego text minus the perplexity of its prompt when a perplexity
function is supplied, and null otherwise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from covertext.common.interfaces import Detector, Encoder
from covertext.common.model import MODEL_ID
from covertext.common.schema import StegoRecord
from covertext.eval.results import write_results
from covertext.eval.samples import (
    TokenCounter,
    count_tokens,
    encode_payloads,
    records_from_samples,
    split_train_eval,
    summarize_samples,
)


BUILTIN_CONTEXTS: list[str] = [
    "The museum opened its north gallery in the spring.",
    "A river cuts through the valley below the ridge.",
    "Historians still argue about the treaty's wording.",
    "The telescope was repaired after the storm.",
]


@dataclass
class ExperimentConfig:
    encoder_name: str
    detector_name: str
    payload_sizes: list[int]
    n_samples: int = 4
    seed: int = 0
    train_fraction: float = 0.5
    model_name: str = "none"
    output_path: str | None = None
    contexts: list[str] = field(default_factory=lambda: list(BUILTIN_CONTEXTS))
    clean_contexts: list[str] | None = None

    def validate(self) -> None:
        if not self.payload_sizes or any(size < 1 for size in self.payload_sizes):
            raise ValueError("payload_sizes must be positive integers")
        if self.n_samples < 1:
            raise ValueError("n_samples must be positive")
        if len(self.contexts) < 2:
            raise ValueError("contexts must contain at least two strings")
        if self.clean_contexts is not None and len(self.clean_contexts) < 2:
            raise ValueError("clean_contexts must contain at least two strings")


def run_experiment(
    config: ExperimentConfig,
    encoder: Encoder,
    detector: Detector,
    token_counter: TokenCounter | None = None,
    perplexity_fn: Callable[[str], float] | None = None,
) -> dict[str, Any]:
    """Run one encoder against one detector across ``config.payload_sizes``."""
    config.validate()
    train_contexts, eval_contexts = split_train_eval(
        config.contexts, config.train_fraction, config.seed
    )
    if config.clean_contexts is None:
        clean_train, clean_eval = train_contexts, eval_contexts
    else:
        clean_train, clean_eval = split_train_eval(
            config.clean_contexts, config.train_fraction, config.seed
        )
    if _calibrates(detector):
        detector.calibrate(clean_train)

    summary: list[dict[str, Any]] = []
    records: list[StegoRecord] = []
    for size_index, size in enumerate(config.payload_sizes):
        cell_seed = config.seed + 10007 * size_index
        if _fits(detector):
            train_samples = encode_payloads(
                encoder,
                train_contexts,
                size,
                config.n_samples,
                seed=cell_seed,
                token_counter=token_counter,
                perplexity_fn=None,
            )
            cover_texts = _negative_texts(config, clean_train, train_samples)
            detector.fit(
                cover_texts + [sample.stego_text for sample in train_samples],
                [0] * len(train_samples) + [1] * len(train_samples),
            )
        eval_samples = encode_payloads(
            encoder,
            eval_contexts,
            size,
            config.n_samples,
            seed=cell_seed + 500_000,
            token_counter=token_counter,
            perplexity_fn=perplexity_fn,
        )
        clean_scores = [
            detector.score(text) for text in _negative_texts(config, clean_eval, eval_samples)
        ]
        stego_scores = [detector.score(sample.stego_text) for sample in eval_samples]
        row = summarize_samples(
            eval_samples,
            encoder.name,
            detector.name,
            size,
            clean_scores,
            stego_scores,
        )
        summary.append(row)
        records.extend(
            records_from_samples(eval_samples, encoder.name, config.model_name, row)
        )
    result = {"summary": summary, "records": records, "config": asdict(config)}
    if config.output_path is not None:
        write_results(config.output_path, summary, records)
    return result


def _negative_texts(config: ExperimentConfig, clean_pool: list[str], samples) -> list[str]:
    """Cover prompts, unless a separate clean corpus was provided."""
    if config.clean_contexts is None:
        return [sample.context for sample in samples]
    return [clean_pool[index % len(clean_pool)] for index in range(len(samples))]


def _fits(detector: Detector) -> bool:
    return callable(getattr(detector, "fit", None))


def _calibrates(detector: Detector) -> bool:
    return callable(getattr(detector, "calibrate", None)) and not _fits(detector)


def config_from_mapping(payload: dict[str, Any]) -> ExperimentConfig:
    known = {name for name in ExperimentConfig.__dataclass_fields__}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"unknown experiment settings: {unknown}")
    config = ExperimentConfig(**payload)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run one encoder/detector experiment")
    parser.add_argument("--config", help="JSON file of ExperimentConfig fields")
    parser.add_argument("--encoder", default="stub")
    parser.add_argument("--detector", default="stub")
    parser.add_argument("--payload-sizes", default="8,16,32,64")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contexts", help="JSON file containing a list of cover strings")
    args = parser.parse_args(argv)
    if args.config:
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        payload["output_path"] = args.output
        config = config_from_mapping(payload)
    else:
        contexts = list(BUILTIN_CONTEXTS)
        if args.contexts:
            contexts = json.loads(Path(args.contexts).read_text(encoding="utf-8"))
        config = ExperimentConfig(
            encoder_name=args.encoder,
            detector_name=args.detector,
            payload_sizes=[int(part) for part in args.payload_sizes.split(",") if part],
            n_samples=args.n_samples,
            seed=args.seed,
            model_name=MODEL_ID if args.encoder.lower() != "stub" else "none",
            output_path=args.output,
            contexts=contexts,
        )
    from covertext.common.registry import build_detector, build_encoder

    model = tokenizer = None
    if config.encoder_name.lower() != "stub" or config.detector_name.lower() != "stub":
        from covertext.common.model import load_model

        model, tokenizer = load_model()
        config.model_name = MODEL_ID
    encoder = build_encoder(config.encoder_name, model, tokenizer)
    detector_kwargs = {}
    if config.detector_name.lower() in {"stub", "cls"}:
        detector_kwargs["seed"] = config.seed
    detector = build_detector(
        config.detector_name, model, tokenizer, **detector_kwargs
    )
    counter = None
    perplexity = None
    if tokenizer is not None and model is not None:
        from covertext.detector.ppl_detector import PPLDetector

        def counter(text: str, _tokenizer=tokenizer) -> int:
            return count_tokens(text, _tokenizer)

        perplexity = PPLDetector(model, tokenizer).compute_ppl
    run_experiment(
        config, encoder, detector, token_counter=counter, perplexity_fn=perplexity
    )


if __name__ == "__main__":
    main()
