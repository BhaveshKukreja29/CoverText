"""Encode, decode, then detect. The round trip must recover every bit."""

from __future__ import annotations

import argparse

from covertext.common.model import MODEL_ID, load_model
from covertext.common.registry import build_detector, build_encoder
from covertext.common.seed import seed_everything
from covertext.demo.cli import DEFAULT_CONTEXT, decode_message, detect_text, encode_message


def run_smoke(
    encoder_name: str = "stub",
    message: str = "hi",
    context: str = DEFAULT_CONTEXT,
    seed: int = 0,
) -> dict:
    """Return the encode, decode, and detect payloads after checking the bits.

    The stub detector always scores. When the encoder needs the language model,
    the perplexity detector is calibrated on two fixed cover lines and scored
    as well. A classifier head is not trained here. That happens in the sweep.
    """
    seed_everything(seed)
    model = tokenizer = None
    if encoder_name.strip().lower() != "stub":
        model, tokenizer = load_model()
    encoder_kwargs = {"seed": seed} if encoder_name.strip().lower() == "mec" else {}
    encoder = build_encoder(encoder_name, model, tokenizer, **encoder_kwargs)
    encoded = encode_message(encoder, message, context)
    decoded = decode_message(encoder, encoded["stego_text"], context, encoded["num_bits"])
    if decoded["payload_bits"] != encoded["payload_bits"]:
        raise AssertionError(
            f"{encoder.name} recovered {decoded['payload_bits']}, sent {encoded['payload_bits']}"
        )
    detectors = [build_detector("stub", seed=seed)]
    if model is not None and tokenizer is not None:
        perplexity = build_detector("ppl", model, tokenizer)
        perplexity.calibrate([context, "A river cuts through the valley below the ridge."])
        detectors.append(perplexity)
    detected = detect_text(encoded["stego_text"], detectors)
    for name, score in detected["scores"].items():
        if not isinstance(score, float) or not 0.0 <= score <= 1.0:
            raise AssertionError(f"{name} score {score} is outside [0, 1]")
    return {
        "encoder": encoder.name,
        "model": MODEL_ID if model is not None else "none",
        "encoded": encoded,
        "decoded": decoded,
        "detected": detected,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Encode, decode, and detect one message")
    parser.add_argument("--encoder", default="stub")
    parser.add_argument("--message", default="hi")
    parser.add_argument("--context", default=DEFAULT_CONTEXT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_smoke(args.encoder, args.message, args.context, args.seed)
    bits = result["encoded"]["payload_bits"]
    scores = ", ".join(f"{name}={score:.4f}" for name, score in result["detected"]["scores"].items())
    print(f"ok encoder={result['encoder']} bits={bits} scores={scores}")
