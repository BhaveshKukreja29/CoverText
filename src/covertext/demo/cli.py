"""Encode a message, decode it, or score text with one or more detectors.

``--encoder`` and ``--detectors`` select implementations by name. ``stub``
runs with no model. ``ac``, ``mec``, ``ppl``, and ``cls`` load
Qwen2.5-0.5B-Instruct. The flags stay the same when a stub is swapped for a
real method.

PPL must be given ``--clean-file`` (one cover string per line) so it can
calibrate. CLS must be given ``--state`` written by ``CLSDetector.save``.
"""

from __future__ import annotations

import argparse
import json
import sys

from covertext.common.interfaces import Detector, Encoder
from covertext.common.payload import bits_to_text, text_to_bits
from covertext.common.registry import build_detector, build_encoder

DEFAULT_CONTEXT = "The following is a short note."


def encode_message(
    encoder: Encoder,
    message: str,
    context: str,
    *,
    payload_bits: str | None = None,
    max_tokens: int = 100,
) -> dict:
    bits = payload_bits if payload_bits is not None else text_to_bits(message)
    if any(char not in "01" for char in bits) or bits == "":
        raise ValueError("payload bits must be a non-empty string of 0 and 1")
    stego = encoder.encode(bits, context, max_tokens=max_tokens)
    return {
        "encoder": encoder.name,
        "context": context,
        "payload_bits": bits,
        "num_bits": len(bits),
        "stego_text": stego,
    }


def decode_message(
    encoder: Encoder,
    stego_text: str,
    context: str,
    num_bits: int,
) -> dict:
    if num_bits < 1:
        raise ValueError("num_bits must be positive")
    bits = encoder.decode(stego_text, context, num_bits)
    message = None
    if len(bits) % 8 == 0 and bits:
        try:
            message = bits_to_text(bits)
        except UnicodeDecodeError:
            message = None
    return {
        "encoder": encoder.name,
        "payload_bits": bits,
        "num_bits": num_bits,
        "message": message,
    }


def detect_text(text: str, detectors: list[Detector]) -> dict:
    """Score ``text`` with every detector. Higher means more likely stego."""
    if not detectors:
        raise ValueError("at least one detector is required")
    if not text:
        raise ValueError("text must be non-empty")
    names = [detector.name for detector in detectors]
    if len(names) != len(set(names)):
        raise ValueError("detector names must be unique")
    return {"scores": {detector.name: detector.score(text) for detector in detectors}}


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="covertext")
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="Hide a message in text")
    encode.add_argument("--message", help="Unicode message. Mutually exclusive with --payload")
    encode.add_argument("--payload", help="Raw bitstring of 0 and 1")
    encode.add_argument("--context", default=DEFAULT_CONTEXT)
    encode.add_argument("--encoder", default="stub")
    encode.add_argument("--max-tokens", type=int, default=100)
    encode.set_defaults(func=_cmd_encode)

    decode = sub.add_parser("decode", help="Recover a payload from stego text")
    decode.add_argument("--text", required=True)
    decode.add_argument("--context", default=DEFAULT_CONTEXT)
    decode.add_argument("--num-bits", type=int, required=True)
    decode.add_argument("--encoder", default="stub")
    decode.set_defaults(func=_cmd_decode)

    detect = sub.add_parser("detect", help="Score text with each named detector")
    detect.add_argument("--text", required=True)
    detect.add_argument(
        "--detectors",
        default="stub",
        help="Comma-separated names: stub, ppl, cls",
    )
    detect.add_argument("--clean-file", help="Cover lines used to calibrate PPL")
    detect.add_argument("--state", help="CLS head checkpoint from CLSDetector.save")
    detect.add_argument("--seed", type=int, default=42)
    detect.set_defaults(func=_cmd_detect)
    return parser


def _cmd_encode(args: argparse.Namespace) -> None:
    if (args.message is None) == (args.payload is None):
        raise ValueError("pass exactly one of --message or --payload")
    encoder = _load_encoder(args.encoder)
    result = encode_message(
        encoder,
        args.message or "",
        args.context,
        payload_bits=args.payload,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(result))


def _cmd_decode(args: argparse.Namespace) -> None:
    encoder = _load_encoder(args.encoder)
    print(json.dumps(decode_message(encoder, args.text, args.context, args.num_bits)))


def _cmd_detect(args: argparse.Namespace) -> None:
    names = [part.strip() for part in args.detectors.split(",") if part.strip()]
    detectors = _load_detectors(names, args.clean_file, args.state, args.seed)
    print(json.dumps(detect_text(args.text, detectors)))


def _load_encoder(name: str) -> Encoder:
    model = tokenizer = None
    if name.strip().lower() != "stub":
        model, tokenizer = load_shared_model()
    return build_encoder(name, model, tokenizer)


def _load_detectors(
    names: list[str], clean_file: str | None, state: str | None, seed: int
) -> list[Detector]:
    needs_model = any(name.lower() != "stub" for name in names)
    model = tokenizer = None
    if needs_model:
        model, tokenizer = load_shared_model()
    detectors: list[Detector] = []
    for name in names:
        kwargs = {"seed": seed} if name.lower() in {"stub", "cls"} else {}
        detector = build_detector(name, model, tokenizer, **kwargs)
        if name.lower() == "ppl":
            if not clean_file:
                raise ValueError("PPL requires --clean-file for calibration")
            detector.calibrate(_read_lines(clean_file))
        if name.lower() == "cls":
            if not state:
                raise ValueError("CLS requires --state")
            detector.load(state)
        detectors.append(detector)
    return detectors


def _read_lines(path: str) -> list[str]:
    from pathlib import Path

    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("clean file is empty")
    return lines


_MODEL = None


def load_shared_model():
    global _MODEL
    if _MODEL is None:
        from covertext.common.model import load_model

        _MODEL = load_model()
    return _MODEL
