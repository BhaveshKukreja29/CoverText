"""Encoder fluency benchmark and the real-encoder detector matrix.

Encoder benchmark (issue 15): for each encoder and payload size, log bits per
token, bit and exact decode accuracy, and the perplexity delta into the stego
record schema.

Detector benchmark (issue 16): the same generation from encoder prompts, with
the held-out human corpus as the negative class. AUROC is stored per encoder,
detector, and payload size. The matrix is the experiment runner applied to
every pair, so the train/eval split is not reimplemented here.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from covertext.common.interfaces import Detector, Encoder
from covertext.common.model import MODEL_ID
from covertext.common.schema import StegoRecord
from covertext.eval.results import write_results
from covertext.eval.runner import ExperimentConfig, run_experiment
from covertext.eval.samples import (
    TokenCounter,
    encode_payloads,
    records_from_samples,
    summarize_samples,
)

PAYLOAD_SIZES: tuple[int, ...] = (8, 16, 32, 64)


def run_encoder_benchmark(
    encoders: Mapping[str, Encoder],
    contexts: Sequence[str],
    payload_sizes: Sequence[int] = PAYLOAD_SIZES,
    n_per_size: int = 8,
    seed: int = 0,
    token_counter: TokenCounter | None = None,
    perplexity_fn: Callable[[str], float] | None = None,
    model_name: str = MODEL_ID,
) -> dict[str, Any]:
    """Generate stego for every encoder and payload size and aggregate metrics."""
    if not encoders:
        raise ValueError("encoders must be non-empty")
    if not contexts:
        raise ValueError("contexts must be non-empty")
    if n_per_size < 1:
        raise ValueError("n_per_size must be positive")
    if not payload_sizes or any(size < 1 for size in payload_sizes):
        raise ValueError("payload_sizes must be positive integers")

    summary: list[dict[str, Any]] = []
    records: list[StegoRecord] = []
    cover = list(contexts)
    for encoder_index, encoder in enumerate(encoders.values()):
        for size_index, size in enumerate(payload_sizes):
            samples = encode_payloads(
                encoder,
                cover,
                size,
                n_per_size,
                seed=seed + 10007 * encoder_index + 1009 * size_index,
                token_counter=token_counter,
                perplexity_fn=perplexity_fn,
            )
            row = summarize_samples(samples, encoder.name, None, size, None, None)
            summary.append(row)
            records.extend(records_from_samples(samples, encoder.name, model_name, row))
    return {"summary": summary, "records": records}


def run_detector_benchmark(
    encoders: Mapping[str, Encoder],
    detectors: Mapping[str, Detector],
    contexts: Sequence[str],
    payload_sizes: Sequence[int] = PAYLOAD_SIZES,
    n_samples: int = 4,
    seed: int = 0,
    train_fraction: float = 0.5,
    token_counter: TokenCounter | None = None,
    perplexity_fn: Callable[[str], float] | None = None,
    model_name: str = MODEL_ID,
    clean_contexts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """AUROC of each detector against each encoder at each payload size.

    Each pair is one ``run_experiment`` call. Detectors are not shared across
    pairs without being refit or recalibrated inside that call.
    """
    if not encoders or not detectors:
        raise ValueError("encoders and detectors must be non-empty")
    summary: list[dict[str, Any]] = []
    records: list[StegoRecord] = []
    for encoder in encoders.values():
        for detector in detectors.values():
            config = ExperimentConfig(
                encoder_name=encoder.name,
                detector_name=detector.name,
                payload_sizes=list(payload_sizes),
                n_samples=n_samples,
                seed=seed,
                train_fraction=train_fraction,
                model_name=model_name,
                contexts=list(contexts),
                clean_contexts=list(clean_contexts) if clean_contexts is not None else None,
            )
            result = run_experiment(
                config,
                encoder,
                detector,
                token_counter=token_counter,
                perplexity_fn=perplexity_fn,
            )
            summary.extend(result["summary"])
            records.extend(result["records"])
    return {"summary": summary, "records": records}


def _encoder_kwargs(name: str, seed: int) -> dict:
    """MEC's within-bin sampler is seeded. AC is deterministic given the model."""
    if name.strip().lower() == "mec":
        return {"seed": seed}
    return {}


def _parse_sizes(text: str) -> list[int]:
    sizes = [int(part) for part in text.split(",") if part]
    if not sizes:
        raise ValueError("payload sizes must be non-empty")
    return sizes


def main_encoders(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark AC and MEC encoders")
    parser.add_argument("--encoders", default="ac,mec")
    parser.add_argument("--payload-sizes", default="8,16,32,64")
    parser.add_argument("--n-per-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contexts", type=int, default=8, help="How many train-split covers to use")
    args = parser.parse_args(argv)
    from covertext.common.model import load_model
    from covertext.common.registry import build_encoder
    from covertext.detector.corpus import get_encoder_contexts
    from covertext.detector.ppl_detector import PPLDetector
    from covertext.eval.samples import count_tokens

    model, tokenizer = load_model()
    names = [part.strip() for part in args.encoders.split(",") if part.strip()]
    encoders = {
        name: build_encoder(name, model, tokenizer, **_encoder_kwargs(name, args.seed))
        for name in names
    }
    contexts = get_encoder_contexts(tokenizer=tokenizer)[: args.contexts]
    perplexity = PPLDetector(model, tokenizer).compute_ppl

    def counter(text: str) -> int:
        return count_tokens(text, tokenizer)

    result = run_encoder_benchmark(
        encoders,
        contexts,
        payload_sizes=_parse_sizes(args.payload_sizes),
        n_per_size=args.n_per_size,
        seed=args.seed,
        token_counter=counter,
        perplexity_fn=perplexity,
        model_name=MODEL_ID,
    )
    write_results(args.output, result["summary"], result["records"])


def main_detectors(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AUROC of PPL and CLS on AC and MEC")
    parser.add_argument("--encoders", default="ac,mec")
    parser.add_argument("--detectors", default="ppl,cls")
    parser.add_argument("--payload-sizes", default="8,16,32,64")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contexts", type=int, default=8)
    args = parser.parse_args(argv)
    from covertext.common.model import load_model
    from covertext.common.registry import build_detector, build_encoder
    from covertext.detector.corpus import build_clean_corpus, get_encoder_contexts
    from covertext.detector.ppl_detector import PPLDetector
    from covertext.eval.samples import count_tokens

    model, tokenizer = load_model()
    encoder_names = [part.strip() for part in args.encoders.split(",") if part.strip()]
    detector_names = [part.strip() for part in args.detectors.split(",") if part.strip()]
    encoders = {
        name: build_encoder(name, model, tokenizer, **_encoder_kwargs(name, args.seed))
        for name in encoder_names
    }
    detectors = {}
    for name in detector_names:
        kwargs = {"seed": args.seed} if name.lower() == "cls" else {}
        detectors[name] = build_detector(name, model, tokenizer, **kwargs)
    contexts = get_encoder_contexts(tokenizer=tokenizer)[: args.contexts]
    clean = build_clean_corpus(tokenizer=tokenizer)[: args.contexts]
    perplexity = PPLDetector(model, tokenizer).compute_ppl

    def counter(text: str) -> int:
        return count_tokens(text, tokenizer)

    result = run_detector_benchmark(
        encoders,
        detectors,
        contexts,
        payload_sizes=_parse_sizes(args.payload_sizes),
        n_samples=args.n_samples,
        seed=args.seed,
        token_counter=counter,
        perplexity_fn=perplexity,
        model_name=MODEL_ID,
        clean_contexts=clean,
    )
    write_results(args.output, result["summary"], result["records"])
