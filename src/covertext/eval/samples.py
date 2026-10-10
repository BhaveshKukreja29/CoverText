"""Encode payloads and reduce them to the shared evaluation metrics."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Any, Callable

from covertext.common.interfaces import Encoder
from covertext.common.payload import generate_payload
from covertext.common.schema import StegoRecord
from covertext.eval.metrics import (
    capacity_bpt,
    compute_auroc,
    decode_accuracy_bit,
    decode_accuracy_exact,
    fluency_loss,
)


TokenCounter = Callable[[str], int]
PerplexityFn = Callable[[str], float]


def count_tokens(text: str, tokenizer=None) -> int:
    """Token count used for bits per token.

    With no tokenizer, whitespace-separated words are the tokens. An empty
    tokenization is an error: capacity would be undefined.
    """
    if tokenizer is None:
        count = len(text.split())
    else:
        count = len(tokenizer.encode(text, add_special_tokens=False))
    if count <= 0:
        raise ValueError("text tokenized to zero tokens")
    return count


def split_train_eval(
    items: list[str], train_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministic disjoint split. Both sides are non-empty."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if len(items) < 2:
        raise ValueError("need at least two items to hold out an eval set")
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_train = int(len(items) * train_fraction)
    n_train = min(max(n_train, 1), len(items) - 1)
    train = [items[index] for index in order[:n_train]]
    eval_items = [items[index] for index in order[n_train:]]
    return train, eval_items


@dataclass(frozen=True)
class EncodedSample:
    context: str
    stego_text: str
    payload_bits: str
    recovered_bits: str
    stego_tokens: int
    clean_ppl: float | None
    stego_ppl: float | None
    decode_error: str | None = None


def encode_payloads(
    encoder: Encoder,
    contexts: list[str],
    num_bits: int,
    n: int,
    seed: int,
    token_counter: TokenCounter | None = None,
    perplexity_fn: PerplexityFn | None = None,
) -> list[EncodedSample]:
    """Encode ``n`` payloads. Contexts cycle if ``n`` exceeds ``len(contexts)``.

    Payload seed ``seed + i`` is unique per sample. A decode exception is
    recorded as a total miss (empty recovery) so one bad sample does not
    drop the rest of the benchmark.
    """
    if num_bits < 1:
        raise ValueError("num_bits must be positive")
    if n < 1:
        raise ValueError("n must be positive")
    if not contexts:
        raise ValueError("contexts must be non-empty")
    counter = token_counter or (lambda text: count_tokens(text))
    samples: list[EncodedSample] = []
    for index in range(n):
        payload = generate_payload(num_bits, seed=seed + index)
        context = contexts[index % len(contexts)]
        stego = encoder.encode(payload, context)
        error = None
        try:
            recovered = encoder.decode(stego, context, num_bits)
        except Exception as exc:
            recovered = ""
            error = f"{type(exc).__name__}: {exc}"
        clean_ppl = stego_ppl = None
        if perplexity_fn is not None:
            clean_ppl = perplexity_fn(context)
            stego_ppl = perplexity_fn(stego) if stego else float("inf")
        samples.append(
            EncodedSample(
                context=context,
                stego_text=stego,
                payload_bits=payload,
                recovered_bits=recovered,
                stego_tokens=counter(stego),
                clean_ppl=clean_ppl,
                stego_ppl=stego_ppl,
                decode_error=error,
            )
        )
    return samples


def sample_metrics(sample: EncodedSample) -> dict[str, Any]:
    """Per-sample capacity, bit accuracy, exact match, and fluency delta."""
    metrics: dict[str, Any] = {
        "capacity_bpt": capacity_bpt(len(sample.payload_bits), sample.stego_tokens),
        "decode_accuracy_bit": decode_accuracy_bit(
            sample.payload_bits, sample.recovered_bits
        ),
        "decode_accuracy_exact": decode_accuracy_exact(
            sample.payload_bits, sample.recovered_bits
        ),
        "fluency_loss": _fluency(sample),
    }
    if sample.decode_error is not None:
        metrics["decode_error"] = sample.decode_error
    return metrics


def summarize_samples(
    samples: list[EncodedSample],
    encoder_name: str,
    detector_name: str | None,
    payload_bits: int,
    clean_scores: list[float] | None,
    stego_scores: list[float] | None,
) -> dict[str, Any]:
    """Mean metrics. AUROC is defined only when both score lists are present."""
    if not samples:
        raise ValueError("samples must be non-empty")
    per_sample = [sample_metrics(sample) for sample in samples]
    row: dict[str, Any] = {
        "encoder": encoder_name,
        "detector": detector_name,
        "payload_bits": payload_bits,
        "n": len(samples),
        "capacity_bpt": statistics.fmean(item["capacity_bpt"] for item in per_sample),
        "decode_accuracy_bit": statistics.fmean(
            item["decode_accuracy_bit"] for item in per_sample
        ),
        "decode_accuracy_exact": statistics.fmean(
            float(item["decode_accuracy_exact"]) for item in per_sample
        ),
        "fluency_loss": _mean_finite(item["fluency_loss"] for item in per_sample),
        "auroc": None,
    }
    if clean_scores is not None and stego_scores is not None:
        if len(clean_scores) != len(samples) or len(stego_scores) != len(samples):
            raise ValueError("score lists must align with samples")
        row["auroc"] = compute_auroc(
            list(clean_scores) + list(stego_scores),
            [0] * len(samples) + [1] * len(samples),
        )
    return row


def records_from_samples(
    samples: list[EncodedSample],
    encoder_name: str,
    model_name: str,
    summary: dict[str, Any],
) -> list[StegoRecord]:
    """One schema record per sample. Group metrics are copied onto each record."""
    records: list[StegoRecord] = []
    for sample in samples:
        settings = sample_metrics(sample)
        settings.update(
            {
                "encoder": encoder_name,
                "detector": summary.get("detector"),
                "payload_bits": summary["payload_bits"],
                "auroc": summary.get("auroc"),
            }
        )
        records.append(
            StegoRecord(
                cover_text=sample.context,
                stego_text=sample.stego_text,
                payload_bits=sample.payload_bits,
                method_name=encoder_name,
                model_name=model_name,
                generation_settings=settings,
            )
        )
    return records


def _fluency(sample: EncodedSample) -> float | None:
    if sample.clean_ppl is None or sample.stego_ppl is None:
        return None
    if not math.isfinite(sample.clean_ppl) or not math.isfinite(sample.stego_ppl):
        return None
    return fluency_loss(sample.stego_ppl, sample.clean_ppl)


def _mean_finite(values) -> float | None:
    numbers = list(values)
    if any(value is None or not math.isfinite(value) for value in numbers):
        return None
    return statistics.fmean(numbers)
